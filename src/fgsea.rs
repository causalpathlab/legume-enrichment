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
//!
//! The details follow fgseaMultilevel's `EsRuler`: scores are tie-broken by a hash of the set's
//! genes; each level's log fraction is the posterior mean `ψ(h + 1) − ψ(n + 1)` for `h` of `n`
//! kept, and the last stage's `ψ(hit) − ψ(n + 1)`, with `log2err` summed from their `ψ₁`
//! variances; and each level's MCMC runs until `n·m/2` swaps are accepted, then as many sweeps
//! again. Where the estimate is only a bound its `log2err` is NaN (see [`MultilevelP`]).

use crate::gene_strata::GeneStrata;
use legume_numeric::matrix::rand_util::mix_seed;
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
    key: Key,
}

/// A set's score, tie-broken by a hash of its genes as fgsea does, so equal scores still split
/// at the median and a level never stalls on a plateau.
/// Ordered by `f32::total_cmp` then the hash, with equality to match.
#[derive(Clone, Copy)]
struct Key {
    es: f32,
    hash: u64,
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.es
            .total_cmp(&other.es)
            .then(self.hash.cmp(&other.hash))
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Key {}

/// What every set of one call shares: the ranking, each slot's weight and stratum, the salt of
/// the genes' hashes, and the slots that can move at all (a slot whose stratum the panel takes
/// whole has no gene to swap in).
struct Slots<'a> {
    pos: &'a [u32],
    weights: Vec<f32>,
    strata: Vec<usize>,
    salt: u64,
    movable: Vec<usize>,
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

    /// A gene's hash (splitmix64 of the salted id); a set's is the wrapping sum of its genes', so
    /// a swap updates it in O(1) and it does not depend on slot order.
    fn gene_hash(&self, g: u32) -> u64 {
        mix_seed(self.salt, u64::from(g))
    }

    fn key(&self, genes: &[u32], buf: &mut Vec<(u32, f32)>) -> Key {
        Key {
            es: self.score(genes, buf),
            hash: genes
                .iter()
                .fold(0u64, |h, &g| h.wrapping_add(self.gene_hash(g))),
        }
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

/// A multilevel p-value and its error, the SD of `log2 p` (fgsea's `log2err`). `log2err` is NaN,
/// as fgsea's is NA, where it is unknown and `p` is only a bound: the estimate reached `eps` or its
/// level cap, no sampled set reached the score, or a level's MCMC hit its sweep cap.
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
///
/// As fgseaMultilevel (`EsRuler`): each level keeps the sets above the sample's median (ties
/// broken by a gene hash), estimates the kept probability as the posterior mean of its log,
/// `ψ(h + 1) − ψ(n + 1)` for `h` of `n` kept, with variance `ψ₁(h + 1) − ψ₁(n + 1)`, and moves
/// the refilled sample by MCMC until `n·m/2` swaps are accepted, then as many sweeps again.
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
    // A stratum's slots can move only if it holds genes the panel does not take.
    let slot_strata: Vec<usize> = drawn.iter().map(|d| strata.stratum_of(d.0)).collect();
    let movable = (0..slot_strata.len())
        .filter(|&i| {
            let st = slot_strata[i];
            strata.members(st).len() > profile[st].len()
        })
        .collect();
    let slots = Slots {
        pos,
        weights: drawn.iter().map(|d| d.1).collect(),
        strata: slot_strata,
        salt: rng.random(),
        movable,
    };
    let mut sets: Vec<Set> = Vec::with_capacity(n);
    for _ in 0..20 * n {
        let genes: Vec<u32> = drawn.iter().map(|d| d.0).collect();
        let key = slots.key(&genes, &mut buf);
        if key.es >= 0.0 {
            sets.push(Set { genes, key });
            if sets.len() == n {
                break;
            }
        }
        strata.draw_matched(profile, &mut scratch, &mut drawn, rng);
    }
    if sets.len() < n {
        return None;
    }
    // `ψ(a) − ψ(n + 1)` and `ψ₁(a) − ψ₁(n + 1)`: fgsea's `betaMeanLog(a, n)` and
    // `getVarPerLevel(a, n)`, the posterior mean and variance of a log fraction. A level passes
    // `a = h + 1` for `h` kept above its bound; the last stage passes the raw count `hit` of sets
    // at least `obs`, as fgsea's `getPvalue` does.
    let (psi_n, psi1_n) = ((n as f64 + 1.0).digamma(), (n as f64 + 1.0).trigamma());
    let level = |a: f64| (a.digamma() - psi_n, a.trigamma() - psi1_n);
    let ln_eps = settings.eps.ln();
    // Each level halves p or so; four times the halvings to `eps` is far past any healthy run.
    let max_levels = (4.0 * (1.0 / settings.eps).log2().max(1.0)).ceil() as usize;
    let (mut log_p, mut var) = (0.0f64, 0.0f64);
    let mut levels = 0usize;
    let (p, log2err) = loop {
        sets.sort_unstable_by_key(|s| s.key);
        let central = sets[n / 2].key;
        let mut start = sets.partition_point(|s| s.key < central);
        if start == 0 {
            // The median is the minimum: split above the lowest key instead.
            let low = sets[0].key;
            start = sets.partition_point(|s| s.key == low);
        }
        if start == n || obs <= sets[start - 1].key.es {
            // The last stage. A sample of identical sets is fully correlated, and a count of zero
            // is one pseudo-count: either way the p is only a bound.
            let hit = n - sets.partition_point(|s| s.key.es < obs);
            let (dm, dv) = level(hit.max(1) as f64);
            let lp = log_p + dm;
            let err = if start == n || hit == 0 || lp < ln_eps {
                f64::NAN
            } else {
                (var + dv).sqrt() / std::f64::consts::LN_2
            };
            break (lp.exp(), err);
        }
        let bound = sets[start - 1].key;
        let kept = n - start;
        let (dm, dv) = level(kept as f64 + 1.0);
        log_p += dm;
        var += dv;
        levels += 1;
        if log_p < ln_eps || levels >= max_levels {
            // Past `eps` (fgsea reports `eps` with no error), or out of levels: a bound.
            break (log_p.exp(), f64::NAN);
        }
        // Refill the dropped from the kept, in place, then move every set within the level.
        let (dropped, kept_sets) = sets.split_at_mut(start);
        for d in dropped {
            let k = &kept_sets[rng.random_range(0..kept)];
            d.genes.clone_from(&k.genes);
            d.key = k.key;
        }
        if !mix(&mut sets, bound, &slots, strata, &mut buf, rng) {
            // The sample cannot mix, so no further level can be trusted. What is estimated so far
            // is `P(ES > bound)` with `bound < obs`: a conservative bound on the p.
            break (log_p.exp(), f64::NAN);
        }
    };
    Some(MultilevelP {
        p: p.clamp(settings.eps, 1.0),
        log2err,
        levels,
    })
}

/// fgsea's MCMC schedule for one level: sweeps of `max(1, m/10)` swap attempts per set until
/// `n·m/2` swaps are accepted (`m` counting only the slots that can move), then as many sweeps
/// again. `false`, with the second pass skipped, when the sets cannot move or `MAX_SWEEPS` cut the
/// first pass short.
fn mix(
    sets: &mut [Set],
    bound: Key,
    slots: &Slots,
    strata: &GeneStrata,
    buf: &mut Vec<(u32, f32)>,
    rng: &mut SmallRng,
) -> bool {
    const MAX_SWEEPS: usize = 1000;
    let m = slots.movable.len();
    if m == 0 {
        return false;
    }
    let attempts = (m / 10).max(1);
    let need = sets.len() * m / 2;
    let mut sweep = || -> usize {
        sets.iter_mut()
            .map(|s| perturb(s, bound, attempts, slots, strata, buf, rng))
            .sum()
    };
    let (mut accepted, mut sweeps) = (0usize, 0usize);
    while accepted < need {
        if sweeps == MAX_SWEEPS {
            return false;
        }
        accepted += sweep();
        sweeps += 1;
    }
    for _ in 0..sweeps {
        sweep();
    }
    true
}

/// `moves` Metropolis steps on a set above `bound`: swap a movable slot's gene for another gene of
/// its stratum not in the set; keep the swap only if the set stays above. Returns the swaps kept.
fn perturb(
    s: &mut Set,
    bound: Key,
    moves: usize,
    slots: &Slots,
    strata: &GeneStrata,
    buf: &mut Vec<(u32, f32)>,
    rng: &mut SmallRng,
) -> usize {
    let mut kept = 0;
    for _ in 0..moves {
        let slot = slots.movable[rng.random_range(0..slots.movable.len())];
        let pool = strata.members(slots.strata[slot]);
        let candidate = pool[rng.random_range(0..pool.len())];
        if s.genes.contains(&candidate) {
            continue;
        }
        let old = s.genes[slot];
        s.genes[slot] = candidate;
        let key = Key {
            es: slots.score(&s.genes, buf),
            hash: s
                .key
                .hash
                .wrapping_sub(slots.gene_hash(old))
                .wrapping_add(slots.gene_hash(candidate)),
        };
        if key > bound {
            s.key = key;
            kept += 1;
        } else {
            s.genes[slot] = old;
        }
    }
    kept
}
