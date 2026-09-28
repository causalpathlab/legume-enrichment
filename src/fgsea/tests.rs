//! [`super`] against the walk it replaces and against brute force.

use super::*;
use crate::es::{positions, rank_descending, weighted_ks_es};
use rand::SeedableRng;

/// A random ranking of `g` genes (`pos[gene]` = rank) and its gene order.
fn ranking(g: usize, rng: &mut SmallRng) -> (Vec<u32>, Vec<u32>) {
    let scores: Vec<f32> = (0..g).map(|_| rng.random::<f32>()).collect();
    let order = rank_descending(&scores);
    (positions(&order), order)
}

#[test]
fn the_score_from_hit_positions_is_the_walks_score() {
    let mut rng = SmallRng::seed_from_u64(1);
    for trial in 0..500 {
        let g = 20 + trial % 200;
        let (pos, order) = ranking(g, &mut rng);
        let m = 1 + rng.random_range(0..g / 2);
        let mut weights = vec![0.0f32; g];
        let mut hits = Vec::new();
        for _ in 0..m {
            let gene = rng.random_range(0..g);
            if weights[gene] == 0.0 {
                let w = if trial % 3 == 0 {
                    1.0
                } else {
                    0.1 + rng.random::<f32>()
                };
                weights[gene] = w;
                hits.push((pos[gene], w));
            }
        }
        let walk = weighted_ks_es(&order, &weights);
        let fast = es_from_hits(&mut hits, g);
        assert!(
            (walk - fast).abs() < 1e-5,
            "trial {trial}: walk {walk}, fast {fast}"
        );
    }
}

#[test]
fn hits_at_the_edges_score_as_the_walk_does() {
    let order: Vec<u32> = (0..10).collect();
    for set in [vec![0u32, 1, 2], vec![7, 8, 9], vec![0, 9], vec![5]] {
        let mut weights = vec![0.0f32; 10];
        let mut hits = Vec::new();
        for &gene in &set {
            weights[gene as usize] = 1.0;
            hits.push((gene, 1.0));
        }
        let (walk, fast) = (
            weighted_ks_es(&order, &weights),
            es_from_hits(&mut hits, 10),
        );
        assert!(
            (walk - fast).abs() < 1e-6,
            "{set:?}: walk {walk}, fast {fast}"
        );
    }
}

/// Brute force `P(ES ≥ obs | ES ≥ 0)` from `draws` matched null sets, and the count behind it.
fn brute_force(
    obs: f32,
    pos: &[u32],
    strata: &GeneStrata,
    profile: &[Vec<f32>],
    draws: usize,
    rng: &mut SmallRng,
) -> (f64, usize) {
    let (mut scratch, mut drawn) = (strata.scratch(), Vec::new());
    let (mut ge, mut pos_n) = (0usize, 0usize);
    for _ in 0..draws {
        strata.draw_matched(profile, &mut scratch, &mut drawn, rng);
        let mut hits: Vec<(u32, f32)> =
            drawn.iter().map(|&(gi, w)| (pos[gi as usize], w)).collect();
        let es = es_from_hits(&mut hits, pos.len());
        if es >= 0.0 {
            pos_n += 1;
            ge += usize::from(es >= obs);
        }
    }
    (ge as f64 / pos_n as f64, ge)
}

#[test]
fn multilevel_agrees_with_brute_force_and_reaches_below_it() {
    let g = 2000;
    let mut rng = SmallRng::seed_from_u64(7);
    let (pos, _) = ranking(g, &mut rng);
    let strata = GeneStrata::unstratified(g);
    let profile = vec![vec![1.0f32; 30]];
    let settings = Multilevel::default();

    // A tail brute force can still see.
    let obs = 0.25;
    let (truth, truth_hits) = brute_force(obs, &pos, &strata, &profile, 200_000, &mut rng);
    assert!(
        truth > 1e-3 && truth < 0.05,
        "pick a resolvable tail: {truth}"
    );
    // Unbiased in log: the mean of log2 p over independent runs sits on the truth within the
    // runs' own reported error, shrunk by √runs, plus the brute force's binomial error.
    let runs = 40;
    let mut mean = 0.0;
    let mut err2 = 0.0;
    let mut ml = None;
    for _ in 0..runs {
        let r = multilevel_p(obs, &pos, &strata, &profile, &settings, &mut rng).unwrap();
        assert!(r.log2err.is_finite() && r.log2err > 0.0, "{r:?}");
        mean += r.p.log2() / runs as f64;
        err2 += r.log2err.powi(2) / runs as f64;
        ml = Some(r);
    }
    let ml = ml.unwrap();
    let off = (mean - truth.log2()).abs();
    let truth_err = (1.0 / truth_hits as f64).sqrt() / std::f64::consts::LN_2;
    let tol = 3.0 * (err2 / runs as f64 + truth_err.powi(2)).sqrt();
    assert!(
        off < tol,
        "mean log2 p {mean} vs brute force {} (off {off}, tol {tol})",
        truth.log2()
    );

    // Far past any feasible number of draws: still a finite, smaller p.
    let deep = multilevel_p(0.6, &pos, &strata, &profile, &settings, &mut rng).unwrap();
    assert!(deep.p < ml.p * 1e-3 && deep.p >= settings.eps, "{deep:?}");
    assert!(deep.levels > ml.levels);
    assert!(deep.log2err.is_finite(), "{deep:?}");
}

#[test]
fn the_most_extreme_set_is_reported_as_a_bound_at_eps() {
    // The panel is the top 30 genes: ES = 1, true p ≈ 1 / C(490, 30), below any f32 p.
    let g = 490;
    let mut rng = SmallRng::seed_from_u64(3);
    let pos: Vec<u32> = (0..g as u32).collect();
    let strata = GeneStrata::unstratified(g);
    let profile = vec![vec![1.0f32; 30]];
    let mut hits: Vec<(u32, f32)> = (0..30).map(|r| (r, 1.0)).collect();
    let obs = es_from_hits(&mut hits, g);
    assert_eq!(obs, 1.0);
    let settings = Multilevel::default();
    let r = multilevel_p(obs, &pos, &strata, &profile, &settings, &mut rng).unwrap();
    assert_eq!(r.p, settings.eps, "{r:?}");
    assert!(r.log2err.is_nan(), "a bound has no error: {r:?}");
}
