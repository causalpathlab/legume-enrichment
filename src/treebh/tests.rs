//! Unit tests for the TreeBH primitives (`super` = [`crate::treebh`]).

use super::*;

/// Augmented tree mirroring an internal "T" label with CD8/CD4 children:
///   0 root → 1 A("T") → {2 A_self, 3 B_node("CD8")→4 B_self, 5 C_node("CD4")→6 C_self}
/// leaves (carrying data p) are the *_self nodes 2/4/6.
fn tree() -> (Vec<Vec<usize>>, Vec<usize>) {
    let children = vec![
        vec![1],       // 0 root
        vec![2, 3, 5], // 1 A (T)
        vec![],        // 2 A_self
        vec![4],       // 3 B_node (CD8)
        vec![],        // 4 B_self
        vec![6],       // 5 C_node (CD4)
        vec![],        // 6 C_self
    ];
    let postorder = vec![2, 4, 3, 6, 5, 1, 0];
    (children, postorder)
}

fn run(p_tself: f64, p_cd8: f64, p_cd4: f64) -> Vec<bool> {
    let (children, postorder) = tree();
    let mut leaf_p = vec![None; 7];
    leaf_p[2] = Some(p_tself);
    leaf_p[4] = Some(p_cd8);
    leaf_p[6] = Some(p_cd4);
    let cp = combine_bottom_up(&children, &postorder, &leaf_p);
    descend(&children, 0, &cp, 0.1, false)
}

#[test]
fn peaked_subtype_descends() {
    // Generic-T present AND CD8 strong, CD4 absent → resolve to CD8.
    let r = run(0.001, 0.001, 0.9);
    assert!(r[4], "CD8 self should be rejected");
    assert!(!r[6], "CD4 self should NOT be rejected");
    assert!(r[2], "T self rejected");
}

#[test]
fn flat_siblings_stop_at_parent() {
    // Generic-T strong but CD8≈CD4 weak → resolve to T, abstain on subtype.
    let r = run(0.001, 0.2, 0.2);
    assert!(r[2], "T self rejected (resolved at T)");
    assert!(!r[4] && !r[6], "neither subtype rejected (abstain)");
}

#[test]
fn all_weak_cannot_explain() {
    // Nothing significant anywhere → only the virtual root is 'rejected'.
    let r = run(0.9, 0.9, 0.9);
    assert!(r[0]);
    assert!(r.iter().skip(1).all(|&x| !x), "no real hypothesis rejected");
}

#[test]
fn simes_and_bh_basics() {
    assert!((simes(&[0.01, 0.5, 0.9]) - 0.03).abs() < 1e-9); // min(3*.01/1, 3*.5/2, 3*.9/3)
    assert_eq!(bh_reject(&[0.001, 0.2, 0.2], 0.1, false), vec![0]);
    assert!(bh_reject(&[0.9, 0.9], 0.1, false).is_empty());
}

#[test]
fn a_tree_q_is_the_smallest_target_that_rejects() {
    let (children, _) = tree();
    let mut leaf_p = vec![None; 7];
    leaf_p[2] = Some(0.001);
    leaf_p[4] = Some(0.01);
    leaf_p[6] = Some(0.9);
    let q = treebh_q(&children, 0, &leaf_p, false);
    assert_eq!(q[0], 0.0);
    for (v, &qv) in q.iter().enumerate().skip(1) {
        for alpha in [0.01, 0.03, 0.05, 0.1, 0.2, 0.5] {
            let order = postorder(&children, 0);
            let cp = combine_bottom_up(&children, &order, &leaf_p);
            let rejected = descend(&children, 0, &cp, alpha, false)[v];
            // Away from the threshold itself, q ≤ α exactly when TreeBH at α rejects.
            if (qv - alpha).abs() > 1e-9 {
                assert_eq!(qv <= alpha, rejected, "node {v} at {alpha}: q {qv}");
            }
        }
    }
    assert!(q[4] > q[2], "a subtype is gated by its parent family");
    assert!(
        q[6] >= 0.9 - 1e-9,
        "CD4 (p 0.9) is rejected only at a target that high: {}",
        q[6]
    );
}

#[test]
fn a_flat_tree_gives_the_bh_q_values() {
    // Root over five leaves: TreeBH is plain BH.
    let children = vec![vec![1, 2, 3, 4, 5], vec![], vec![], vec![], vec![], vec![]];
    let ps = [0.001, 0.008, 0.039, 0.041, 0.6];
    let mut leaf_p = vec![None];
    leaf_p.extend(ps.iter().map(|&p| Some(p)));
    let q = treebh_q(&children, 0, &leaf_p, false);
    // BH: q_(i) = min_{j ≥ i} m p_(j) / j.
    let bh = [0.005, 0.02, 0.05125, 0.05125, 0.6];
    for (i, &want) in bh.iter().enumerate() {
        assert!(
            (q[i + 1] - want).abs() < 1e-6,
            "leaf {i}: {} vs {want}",
            q[i + 1]
        );
    }
}

#[test]
fn types_missing_from_the_tree_hang_under_the_root() {
    let tree = TypeTree {
        children: vec![vec![1], vec![2], vec![]],
        root: 0,
        leaf: vec![Some(2), None],
    };
    let (children, leaf) = tree.completed(&[true; 3]).unwrap();
    assert_eq!(leaf, [Some(2), Some(3), Some(4)]);
    assert_eq!(children[0], [1, 3, 4]);
    let bad = TypeTree {
        children: vec![vec![1], vec![2], vec![]],
        root: 0,
        leaf: vec![Some(1)],
    };
    assert!(bad.completed(&[true]).is_err(), "a leaf with children");
    let shared = TypeTree {
        children: vec![vec![1], vec![]],
        root: 0,
        leaf: vec![Some(1), Some(1)],
    };
    assert!(
        shared.completed(&[true; 2]).is_err(),
        "two types on one leaf"
    );
}

#[test]
fn untested_types_and_their_empty_subtrees_leave_the_families() {
    // root 0 → {class 1 → {leaf 2, leaf 3}, class 4 → {leaf 5}}; type 2 is untested and type 3
    // has no leaf in the tree and is untested too.
    let tree = TypeTree {
        children: vec![vec![1, 4], vec![2, 3], vec![], vec![], vec![5], vec![]],
        root: 0,
        leaf: vec![Some(2), Some(3), Some(5), None],
    };
    let (children, leaf) = tree.completed(&[true, true, false, false]).unwrap();
    assert_eq!(leaf, [Some(2), Some(3), None, None]);
    assert_eq!(children[0], [1], "class 4 lost its only tested leaf");
    assert_eq!(children[1], [2, 3]);
    assert_eq!(children.len(), 6, "no leaf added for the untested type");
}
