//! fgsea-style statistics on the weighted KS enrichment score: the exact score from hit
//! positions alone, the normalized score (NES), and the multilevel p-value.
//!
//! Korotkevich, Sukhov, Budin, Shpak, Artyomov & Sergushichev, "Fast gene set enrichment
//! analysis", bioRxiv 060012 (2021).
//!
//! # The score from hit positions
//!
//! [`crate::es::weighted_ks_es`] walks every gene. It need not: between two hits the running sum
//! only falls, so its extremes sit right after a hit (the peaks) and right before one (the
//! troughs). With the hits at ranks `r_1 < … < r_m`, weights `w_j`, `W = Σ w_j` and `n_miss`
//! non-hits,
//!
//! ```text
//! after_j  = (w_1 + … + w_j) / W  −  (r_j − (j − 1)) / n_miss
//! before_j = (w_1 + … + w_{j−1}) / W  −  (r_j − (j − 1)) / n_miss
//! ```
//!
//! and the score is whichever of these has the largest magnitude, taken in walk order, exactly as
//! the walk takes it. That is `O(m log m)` per gene set instead of `O(G)`, which is what makes a
//! null of thousands of draws, and the multilevel sampler, cheap.
//!
//! # NES
//!
//! `NES = ES / mean(null ES ≥ 0)` for a positive score (and against the negative side's mean for a
//! negative one). Dividing by the same-sign null mean, not its SD, is fgsea's normalization: the
//! null of a KS maximum is skewed and differs in shape between its two sides.
//!
//! # The multilevel p-value
//!
//! A null of `B` draws cannot resolve a p below `1 / (B + 1)`. The multilevel estimate splits the
//! tail into levels: keep the null sets scoring above the median, refill the sample from them,
//! move each by MCMC (swap one gene for another, accepting only moves that stay above the level),
//! and repeat until the median passes the observed score. Each level halves the probability, so
//! `p ≈ ∏ (fraction kept) × (fraction ≥ observed)`, conditional on `ES ≥ 0` as the simple
//! sign-aware p is. Swaps stay within the gene's abundance stratum and keep the slot's weight, so
//! every set the sampler visits is matched as the null draws are ([`crate::gene_strata`]).

use crate::gene_strata::GeneStrata;
use rand::rngs::SmallRng;
use rand::RngExt;
use special::Gamma;

#[cfg(test)]
mod tests;

/// The weighted KS score of hits at rank `positions` (0 = the top of the ranking) with weights,
/// among `n_genes`. Equal to [`crate::es::weighted_ks_es`] on the same set; hits with a
/// non-positive weight are not hits. Sorts `hits` in place.
#[must_use]
pub fn es_from_hits(hits: &mut [(u32, f32)], n_genes: usize) -> f32 {
    let mut total: f64 = 0.0;
    let mut n_hit = 0usize;
    for &(_, w) in hits.iter() {
        if w > 0.0 {
            total += f64::from(w);
            n_hit += 1;
        }
    }
    let n_miss = n_genes.saturating_sub(n_hit) as f64;
    if total <= 0.0 || n_miss <= 0.0 {
        return 0.0;
    }
    hits.sort_unstable_by_key(|h| h.0);
    let (hit_norm, miss_norm) = (1.0 / total, 1.0 / n_miss);
    let (mut cum, mut j) = (0.0f64, 0usize);
    let (mut max_abs, mut max_dev) = (0.0f64, 0.0f64);
    let take = |v: f64, max_abs: &mut f64, max_dev: &mut f64| {
        if v.abs() > *max_abs {
            *max_abs = v.abs();
            *max_dev = v;
        }
    };
    for &(r, w) in hits.iter() {
        if w <= 0.0 {
            continue;
        }
        let misses = f64::from(r) - j as f64;
        // The trough before this hit. With no miss since the last hit it equals that peak, and
        // the strict `>` leaves it be.
        if misses > 0.0 {
            take(
                cum * hit_norm - misses * miss_norm,
                &mut max_abs,
                &mut max_dev,
            );
        }
        cum += f64::from(w);
        j += 1;
        take(
            cum * hit_norm - misses * miss_norm,
            &mut max_abs,
            &mut max_dev,
        );
    }
    max_dev as f32
}

/// A gene set as the sampler moves it: each slot's gene. Every set of one call shares the slots'
/// weights and strata, which [`GeneStrata::draw_matched`] fixes by the profile and swaps keep.
#[derive(Clone)]
struct Set {
    genes: Vec<u32>,
    es: f32,
}

/// What every set of one call shares: the ranking, each slot's weight and stratum.
struct Slots<'a> {
    pos: &'a [u32],
    weights: Vec<f32>,
    strata: Vec<usize>,
}

impl Slots<'_> {
    fn score(&self, genes: &[u32], buf: &mut Vec<(u32, f32)>) -> f32 {
        buf.clear();
        buf.extend(
            genes
                .iter()
                .zip(&self.weights)
                .map(|(&g, &w)| (self.pos[g as usize], w)),
        );
        es_from_hits(buf, self.pos.len())
    }
}

/// Settings of [`multilevel_p`].
#[derive(Debug, Clone, Copy)]
pub struct Multilevel {
    /// Gene sets per level (fgsea's `sampleSize`). Each level's split is estimated from this many.
    pub sample_size: usize,
    /// The smallest p reported; below it the estimate stops (fgsea's `eps`). The default is the
    /// smallest normal `f32`, the floor of an `f32` p-value (fgsea's own default, 1e-50, is not).
    pub eps: f64,
}

impl Default for Multilevel {
    fn default() -> Self {
        Self {
            sample_size: 101,
            eps: f64::from(f32::MIN_POSITIVE),
        }
    }
}

/// A multilevel p-value and fgsea's estimate of its error (the SD of `log2 p`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultilevelP {
    pub p: f64,
    pub log2err: f64,
    pub levels: usize,
}

/// `P(ES ≥ obs | ES ≥ 0)` under the matched gene-set null, for a positive `obs`: null sets are
/// drawn with [`GeneStrata::draw_matched`] against `profile` (the panel's per-stratum weights)
/// and scored against the ranking `pos` (gene → rank). `None` when the null cannot be sampled:
/// `20 × sample_size` draws do not yield `sample_size` scoring `≥ 0` (about 5% of draws or fewer).
#[must_use]
pub fn multilevel_p(
    obs: f32,
    pos: &[u32],
    strata: &GeneStrata,
    profile: &[Vec<f32>],
    settings: &Multilevel,
    rng: &mut SmallRng,
) -> Option<MultilevelP> {
    let n = settings.sample_size.max(3);
    let mut scratch = strata.scratch();
    let mut drawn = Vec::new();
    let mut buf = Vec::new();

    // Level 0: the null conditional on ES ≥ 0. Every draw fills the same slots.
    strata.draw_matched(profile, &mut scratch, &mut drawn, rng);
    if drawn.is_empty() {
        return None;
    }
    let slots = Slots {
        pos,
        weights: drawn.iter().map(|d| d.1).collect(),
        strata: drawn.iter().map(|d| strata.stratum_of(d.0)).collect(),
    };
    let mut sets: Vec<Set> = Vec::with_capacity(n);
    for _ in 0..20 * n {
        let genes: Vec<u32> = drawn.iter().map(|d| d.0).collect();
        let es = slots.score(&genes, &mut buf);
        if es >= 0.0 {
            sets.push(Set { genes, es });
            if sets.len() == n {
                break;
            }
        }
        strata.draw_matched(profile, &mut scratch, &mut drawn, rng);
    }
    if sets.len() < n {
        return None;
    }
    let m = drawn.len();

    let mut log_p = 0.0f64;
    let mut levels = 0usize;
    loop {
        let mut es: Vec<f32> = sets.iter().map(|s| s.es).collect();
        es.sort_by(f32::total_cmp);
        let threshold = es[n / 2];
        if threshold >= obs {
            let hit = sets.iter().filter(|s| s.es >= obs).count();
            log_p += (hit.max(1) as f64 / n as f64).ln();
            break;
        }
        let survivors: Vec<usize> = (0..n).filter(|&i| sets[i].es > threshold).collect();
        if survivors.is_empty() {
            // A plateau the sampler cannot climb: report the bound reached so far.
            log_p += (1.0 / n as f64).ln();
            break;
        }
        log_p += (survivors.len() as f64 / n as f64).ln();
        levels += 1;
        if log_p <= settings.eps.ln() {
            log_p = settings.eps.ln();
            break;
        }
        // Refill from the survivors, then move every set within the level.
        let next: Vec<Set> = (0..n)
            .map(|i| {
                if i < survivors.len() {
                    sets[survivors[i]].clone()
                } else {
                    sets[survivors[rng.random_range(0..survivors.len())]].clone()
                }
            })
            .collect();
        sets = next;
        for s in &mut sets {
            perturb(s, threshold, m, &slots, strata, &mut buf, rng);
        }
    }
    let p = log_p.exp().clamp(settings.eps, 1.0);
    Some(MultilevelP {
        p,
        log2err: log2err(p, n),
        levels,
    })
}

/// `moves` Metropolis steps on the sets scoring above `threshold`: swap a slot's gene for another
/// gene of its stratum not in the set; keep the swap only if the score stays above.
fn perturb(
    s: &mut Set,
    threshold: f32,
    moves: usize,
    slots: &Slots,
    strata: &GeneStrata,
    buf: &mut Vec<(u32, f32)>,
    rng: &mut SmallRng,
) {
    for _ in 0..moves {
        let slot = rng.random_range(0..s.genes.len());
        let pool = strata.members(slots.strata[slot]);
        let candidate = pool[rng.random_range(0..pool.len())];
        if s.genes.contains(&candidate) {
            continue;
        }
        let old = s.genes[slot];
        s.genes[slot] = candidate;
        let es = slots.score(&s.genes, buf);
        if es > threshold {
            s.es = es;
        } else {
            s.genes[slot] = old;
        }
    }
}

/// fgsea's error of a multilevel p: the SD of `log2 p` from its number of halvings and the
/// sample size, `sqrt(⌊−log2 p + 1⌋ · (ψ₁((n+1)/2) − ψ₁(n+1))) / ln 2`.
fn log2err(p: f64, n: usize) -> f64 {
    let halvings = (-p.log2() + 1.0).floor().max(0.0);
    let n = n as f64;
    (halvings * (((n + 1.0) / 2.0).trigamma() - (n + 1.0).trigamma()))
        .max(0.0)
        .sqrt()
        / std::f64::consts::LN_2
}
