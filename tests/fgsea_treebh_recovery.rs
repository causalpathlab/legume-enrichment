//! `annotate` end to end on the gene-set null path (`num_sample_perm == 0`): the multilevel
//! p-value, TreeBH over a cell-type tree, and cell types dropped by `min_markers`.

use enrichment::fgsea::Multilevel;
use enrichment::treebh::TypeTree;
use enrichment::{annotate, AnnotateConfig, AnnotateOutputs, GroupInputs, Mat, SpecificityMode};

const K: usize = 3;
const BLOCK: usize = 30;
const NOISE: usize = 400;
const B: usize = 200;

/// Topic `k` is loaded on gene block `k`; cell type `c < K` is marked by block `c`. `extra`
/// appends more cell types, each by its marker genes.
fn fixture(extra: &[Vec<usize>]) -> (GroupInputs, Mat, Vec<Box<str>>) {
    let g = K * BLOCK + NOISE;
    // Spread abundance so the stratified null has non-markers to draw from (see
    // synthetic_recovery.rs); it leaves specificity, and so every ES, untouched.
    let scale = |gi: usize| -> f32 {
        let h = (gi as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x51ED);
        0.2 * 25f32.powf(((h >> 40) as f32) / ((1u64 << 24) as f32))
    };
    let mut beta = Mat::zeros(g, K);
    for gi in 0..g {
        let block = gi / BLOCK;
        for kk in 0..K {
            // A graded load within the block, so the ranking is not one big tie.
            let load = if block == kk {
                1.0 - 0.01 * (gi % BLOCK) as f32
            } else {
                0.05
            };
            beta[(gi, kk)] = load * scale(gi);
        }
    }
    let c = K + extra.len();
    let mut markers = Mat::zeros(g, c);
    for cc in 0..K {
        for i in 0..BLOCK {
            markers[(cc * BLOCK + i, cc)] = 1.0;
        }
    }
    for (j, genes) in extra.iter().enumerate() {
        for &gi in genes {
            markers[(gi, K + j)] = 1.0;
        }
    }
    let p = 30;
    let mut pb_membership = Mat::zeros(p, K);
    let mut pb_gene = Mat::zeros(g, p);
    for pi in 0..p {
        pb_membership[(pi, pi % K)] = 1.0;
        for gi in 0..g {
            pb_gene[(gi, pi)] = beta[(gi, pi % K)] * 1000.0;
        }
    }
    let n = 60;
    let mut cell_membership = Mat::zeros(n, K);
    for i in 0..n {
        cell_membership[(i, i % K)] = 1.0;
    }
    let inputs = GroupInputs {
        profile_gk: beta,
        pb_gene_gp: pb_gene,
        pb_membership_pk: pb_membership,
        cell_membership_nk: cell_membership,
        gene_names: (0..g).map(|i| format!("gene_{i}").into()).collect(),
        cell_names: (0..n).map(|i| format!("cell_{i}").into()).collect(),
    };
    let names = (0..c).map(|i| format!("ct_{i}").into()).collect();
    (inputs, markers, names)
}

fn config() -> AnnotateConfig {
    AnnotateConfig {
        specificity: SpecificityMode::Simplex,
        num_row_randomization: B,
        num_sample_perm: 0,
        fdr_alpha: 0.05,
        seed: 11,
        min_markers: 3,
        bootstrap: None,
        multilevel: Some(Multilevel::default()),
        type_tree: None,
        ..AnnotateConfig::default()
    }
}

/// A null cell type: markers spread evenly over the whole gene range, so each topic's block and
/// the background hold about their share of them, as a random set's would. (Background alone
/// ranks above the other topics' blocks, so a background-only panel is genuinely enriched.)
fn null_panel() -> Vec<usize> {
    (0..BLOCK)
        .map(|i| 3 + i * (K * BLOCK + NOISE) / BLOCK)
        .collect()
}

fn run(extra: &[Vec<usize>], cfg: &AnnotateConfig) -> AnnotateOutputs {
    let (inputs, markers, names) = fixture(extra);
    annotate(&inputs, &markers, &names, cfg).unwrap()
}

#[test]
fn multilevel_resolves_planted_types_below_the_draws_floor() {
    let floor = 1.0 / (B as f32 + 1.0);
    let draws_only = run(
        &[],
        &AnnotateConfig {
            multilevel: None,
            ..config()
        },
    );
    let ml = run(&[], &config());
    for d in 0..K {
        assert!(
            draws_only.pvalue_kc[(d, d)] >= floor,
            "draws alone reach below 1/(B+1)"
        );
        assert!(
            ml.pvalue_kc[(d, d)] < floor / 10.0,
            "multilevel p[{d},{d}] = {} not below the draws' floor {floor}",
            ml.pvalue_kc[(d, d)]
        );
        assert!(ml.p_log2err_kc[(d, d)] > 0.0);
        assert!(
            ml.nes_kc[(d, d)] > 1.0,
            "NES[{d},{d}] = {}",
            ml.nes_kc[(d, d)]
        );
        for o in (0..K).filter(|&o| o != d) {
            // Depletion is not a call.
            assert_eq!(
                ml.pvalue_kc[(d, o)],
                1.0,
                "off-diagonal ({d},{o}) is depleted"
            );
        }
    }
}

#[test]
fn a_dropped_type_is_not_a_hypothesis() {
    // ct_3 has two markers, under `min_markers`; it is last so the others keep their seeds.
    let cfg = AnnotateConfig {
        multilevel: None,
        ..config()
    };
    let with = run(&[vec![K * BLOCK, K * BLOCK + 1]], &cfg);
    let without = run(&[], &cfg);
    for kk in 0..K {
        assert_eq!(with.pvalue_kc[(kk, K)], 1.0);
        assert_eq!(with.qvalue_kc[(kk, K)], 1.0);
        assert_eq!(with.q_kc[(kk, K)], 0.0);
        for cc in 0..K {
            assert_eq!(
                with.qvalue_kc[(kk, cc)],
                without.qvalue_kc[(kk, cc)],
                "the dropped type moved q[{kk},{cc}]"
            );
        }
    }
    // The same under TreeBH, whether or not the tree names the dropped type.
    for leaf3 in [None, Some(5)] {
        let tree = TypeTree {
            children: vec![vec![1, 4], vec![2, 3], vec![], vec![], vec![5], vec![]],
            root: 0,
            leaf: vec![Some(2), Some(3), None, leaf3],
        };
        let tree_cfg = AnnotateConfig {
            type_tree: Some(tree),
            ..cfg.clone()
        };
        let with = run(&[vec![K * BLOCK, K * BLOCK + 1]], &tree_cfg);
        let mut small = tree_cfg.type_tree.clone().unwrap();
        small.leaf.truncate(K);
        let without = run(
            &[],
            &AnnotateConfig {
                type_tree: Some(small),
                ..cfg.clone()
            },
        );
        for kk in 0..K {
            assert_eq!(with.qvalue_kc[(kk, K)], 1.0);
            for cc in 0..K {
                assert_eq!(with.qvalue_kc[(kk, cc)], without.qvalue_kc[(kk, cc)]);
            }
        }
    }
}

#[test]
fn treebh_calls_a_planted_type_through_its_class_and_not_a_null_sibling() {
    // root 0 → {class 1 → {ct_0 at 2, ct_3 (null) at 3}, class 4 → {ct_1 at 5, ct_2 at 6}}.
    let tree = TypeTree {
        children: vec![
            vec![1, 4],
            vec![2, 3],
            vec![],
            vec![],
            vec![5, 6],
            vec![],
            vec![],
        ],
        root: 0,
        leaf: vec![Some(2), Some(5), Some(6), Some(3)],
    };
    let extra = [null_panel()];
    let tree_run = run(
        &extra,
        &AnnotateConfig {
            type_tree: Some(tree),
            ..config()
        },
    );
    let alpha = config().fdr_alpha;
    for d in 0..K {
        let q = tree_run.qvalue_kc[(d, d)];
        assert!(
            q < alpha,
            "planted ({d},{d}) not called under TreeBH: q = {q}"
        );
        assert!(
            tree_run.q_kc[(d, d)] > 0.5,
            "Q[{d},{d}] = {}",
            tree_run.q_kc[(d, d)]
        );
    }
    for kk in 0..K {
        assert!(
            tree_run.qvalue_kc[(kk, K)] >= alpha,
            "null ct_3 called for topic {kk}: q = {}",
            tree_run.qvalue_kc[(kk, K)]
        );
    }
}

#[test]
fn q_rows_are_a_softmax_of_the_probit_z_over_the_fdr_survivors() {
    let cfg = AnnotateConfig {
        q_softmax_temperature: 0.7,
        ..config()
    };
    let out = run(&[null_panel()], &cfg);
    let c = out.pvalue_kc.ncols();
    let mut any = false;
    for kk in 0..K {
        let survives =
            |cc: usize| out.qvalue_kc[(kk, cc)] < cfg.fdr_alpha && out.z_kc[(kk, cc)] > 0.0;
        let total: f32 = (0..c)
            .filter(|&cc| survives(cc))
            .map(|cc| (out.z_kc[(kk, cc)] / cfg.q_softmax_temperature).exp())
            .sum();
        for cc in 0..c {
            let p = f64::from(out.pvalue_kc[(kk, cc)]);
            // Φ(z) = 1 − p, by statrs' own CDF.
            let normal = statrs::distribution::Normal::standard();
            let back = 1.0
                - statrs::distribution::ContinuousCDF::cdf(&normal, f64::from(out.z_kc[(kk, cc)]));
            assert!(
                (back - p).abs() <= 1e-5 * p.max(1e-3),
                "z[{kk},{cc}] = {} is not the probit of p = {p}",
                out.z_kc[(kk, cc)]
            );
            let want = if survives(cc) {
                any = true;
                (out.z_kc[(kk, cc)] / cfg.q_softmax_temperature).exp() / total
            } else {
                0.0
            };
            assert!(
                (out.q_kc[(kk, cc)] - want).abs() < 1e-5,
                "Q[{kk},{cc}] = {} vs softmax of z {want}",
                out.q_kc[(kk, cc)]
            );
        }
    }
    assert!(any, "no FDR survivor to check");
}
