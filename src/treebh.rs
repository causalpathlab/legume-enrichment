//! TreeBH — hierarchical, multi-resolution FDR testing on a tree.
//!
//! Bogomolov, Peterson, Benjamini & Sabatti, "Hypotheses on a tree: new error
//! rates and testing strategies", Biometrika 108(3):575–590, 2021
//! (DOI 10.1093/biomet/asaa086).
//!
//! Pure routines over a flat tree given as a children-adjacency list. The caller
//! designates a `root` that is treated as always-rejected; its children form
//! level 1. Hypotheses are tested top-down: Benjamini–Hochberg is applied within
//! each family (a node's children) at a working target shrunk by the proportion
//! of rejections along the ancestor path, and a child-family is tested only if
//! its parent was rejected. This controls the level-specific selective FDR at
//! every resolution simultaneously.

/// Simes' combined p-value for a family of p-values:
/// `min_i ( m · p_(i) / i )` over the sorted `p_(1) ≤ … ≤ p_(m)`.
#[must_use]
pub fn simes(ps: &[f64]) -> f64 {
    let m = ps.len();
    if m == 0 {
        return 1.0;
    }
    let mut s: Vec<f64> = ps.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut best = f64::INFINITY;
    for (rank, &p) in s.iter().enumerate() {
        let i = (rank + 1) as f64;
        best = best.min(m as f64 * p / i);
    }
    best.min(1.0)
}

/// Benjamini–Hochberg rejection set within a family at level `alpha`. With
/// `by = true`, applies the Benjamini–Yekutieli correction (`alpha /= H_m`,
/// `H_m = Σ 1/i`), valid under arbitrary dependence. Returns the indices into
/// `ps` that are rejected.
#[must_use]
pub fn bh_reject(ps: &[f64], alpha: f64, by: bool) -> Vec<usize> {
    let m = ps.len();
    if m == 0 {
        return Vec::new();
    }
    let alpha = if by {
        let hm: f64 = (1..=m).map(|i| 1.0 / i as f64).sum();
        alpha / hm
    } else {
        alpha
    };
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        ps[a]
            .partial_cmp(&ps[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Largest rank k with p_(k) ≤ (k/m)·alpha.
    let mut k_max: Option<usize> = None;
    for (rank, &idx) in order.iter().enumerate() {
        let i = (rank + 1) as f64;
        if ps[idx] <= (i / m as f64) * alpha {
            k_max = Some(rank);
        }
    }
    match k_max {
        Some(r) => order[..=r].to_vec(),
        None => Vec::new(),
    }
}

/// Bottom-up Simes combination. `leaf_p[i] = Some(p)` marks a leaf carrying a
/// data p-value; `None` marks an internal node whose p is combined from its
/// children. `postorder` must list nodes children-before-parents. Returns the
/// combined p-value per node.
#[must_use]
pub fn combine_bottom_up(
    children: &[Vec<usize>],
    postorder: &[usize],
    leaf_p: &[Option<f64>],
) -> Vec<f64> {
    let mut cp = vec![1.0f64; children.len()];
    for &node in postorder {
        if let Some(p) = leaf_p[node] {
            cp[node] = p.clamp(0.0, 1.0);
        } else {
            let kids = &children[node];
            cp[node] = if kids.is_empty() {
                1.0
            } else {
                simes(&kids.iter().map(|&c| cp[c]).collect::<Vec<_>>())
            };
        }
    }
    cp
}

/// Top-down TreeBH. `root` is treated as always-rejected; its children are the
/// level-1 family. Returns a per-node rejection mask (with `root` set true).
/// `q` is the per-level selective-FDR target; `by` toggles the BY correction
/// within families.
#[must_use]
pub fn descend(children: &[Vec<usize>], root: usize, cp: &[f64], q: f64, by: bool) -> Vec<bool> {
    let mut rejected = vec![false; children.len()];
    rejected[root] = true;
    // (node whose children-family to test, gamma = ∏ r_A/n_A over ancestors).
    let mut stack: Vec<(usize, f64)> = vec![(root, 1.0)];
    while let Some((v, gamma)) = stack.pop() {
        let fam = &children[v];
        if fam.is_empty() {
            continue;
        }
        let ps: Vec<f64> = fam.iter().map(|&c| cp[c]).collect();
        let rej = bh_reject(&ps, q * gamma, by);
        if rej.is_empty() {
            continue; // STOP — this node abstains over its children.
        }
        let child_gamma = gamma * (rej.len() as f64 / fam.len() as f64);
        for &i in &rej {
            let c = fam[i];
            rejected[c] = true;
            stack.push((c, child_gamma));
        }
    }
    rejected
}

/// A hypothesis tree over cell types for [`treebh_q`]: `children` per node, the `root`, and each
/// cell type's leaf. A leaf carries its type's p-value and must have no children; a type that is
/// also a coarser node (say `T cells` over `CD4 T cells`) gets its own leaf *under* that node, as
/// in the tests' augmented tree.
#[derive(Debug, Clone)]
pub struct TypeTree {
    pub children: Vec<Vec<usize>>,
    pub root: usize,
    /// `leaf[c]` = cell type `c`'s node; `None` hangs it under the root.
    pub leaf: Vec<Option<usize>>,
}

/// A completed [`TypeTree`]: the pruned `children`, and each cell type's leaf (`None` untested).
pub type CompletedTree = (Vec<Vec<usize>>, Vec<Option<usize>>);

impl TypeTree {
    /// The tree over the cell types with `tested[c]`, each on a leaf: those without one get a new
    /// leaf under the root. Untested types get no leaf (`None`), and every node with no tested
    /// leaf beneath it is cut from its parent, so neither enlarges a family. Errs when the nodes
    /// under the root are not a tree, or a type's leaf is the root, off the tree, has children,
    /// is shared by two types, or is out of range.
    pub fn completed(&self, tested: &[bool]) -> anyhow::Result<CompletedTree> {
        let mut children = self.children.clone();
        anyhow::ensure!(self.root < children.len(), "TreeBH root out of range");
        anyhow::ensure!(
            children.iter().flatten().all(|&v| v < children.len()),
            "TreeBH child out of range"
        );
        // A tree: every node reached from the root once, so no cycle or shared child.
        let mut reached = vec![false; children.len()];
        let mut stack = vec![self.root];
        while let Some(v) = stack.pop() {
            anyhow::ensure!(!reached[v], "TreeBH node {v} is reached twice: not a tree");
            reached[v] = true;
            stack.extend(&children[v]);
        }
        let mut leaf = Vec::with_capacity(tested.len());
        let mut owner: Vec<Option<usize>> = vec![None; children.len()];
        for (c, &t) in tested.iter().enumerate() {
            match self.leaf.get(c).copied().flatten() {
                Some(l) => {
                    anyhow::ensure!(l < children.len(), "cell type {c}'s leaf is out of range");
                    anyhow::ensure!(l != self.root, "cell type {c}'s leaf is the root");
                    anyhow::ensure!(reached[l], "cell type {c}'s leaf {l} is not under the root");
                    anyhow::ensure!(
                        children[l].is_empty(),
                        "cell type {c}'s leaf {l} has children; give the type its own leaf"
                    );
                    if let Some(prev) = owner[l].replace(c) {
                        anyhow::bail!("cell types {prev} and {c} share leaf {l}");
                    }
                    leaf.push(t.then_some(l));
                }
                None if t => {
                    children.push(Vec::new());
                    owner.push(Some(c));
                    let l = children.len() - 1;
                    children[self.root].push(l);
                    leaf.push(Some(l));
                }
                None => leaf.push(None),
            }
        }
        let mut keep = vec![false; children.len()];
        for &l in leaf.iter().flatten() {
            keep[l] = true;
        }
        for v in postorder(&children, self.root) {
            children[v].retain(|&ch| keep[ch]);
            keep[v] |= !children[v].is_empty();
        }
        Ok((children, leaf))
    }
}

/// Nodes under `root`, children before parents.
#[must_use]
pub fn postorder(children: &[Vec<usize>], root: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(children.len());
    let mut stack = vec![(root, false)];
    while let Some((v, expanded)) = stack.pop() {
        if expanded {
            out.push(v);
        } else {
            stack.push((v, true));
            stack.extend(children[v].iter().map(|&c| (c, false)));
        }
    }
    out
}

/// TreeBH-adjusted q per node: the smallest target at which [`descend`] rejects it, with internal
/// nodes' p-values combined by Simes from their leaves ([`combine_bottom_up`]). The analogue of a
/// BH q-value: a node has `q ≤ α` exactly when TreeBH at `α` rejects it. `1` for a node no target
/// up to 1 rejects; `0` for the root.
///
/// Rejection only grows with the target (BH within a family does, and so does every ancestor
/// path's rejection proportion), so each node's threshold is found by bisection.
#[must_use]
pub fn treebh_q(
    children: &[Vec<usize>],
    root: usize,
    leaf_p: &[Option<f64>],
    by: bool,
) -> Vec<f64> {
    let order = postorder(children, root);
    let cp = combine_bottom_up(children, &order, leaf_p);
    let mut q = vec![1.0f64; children.len()];
    q[root] = 0.0;
    let at_one = descend(children, root, &cp, 1.0, by);
    for &v in order.iter().filter(|&&v| v != root && at_one[v]) {
        // A node is rejected only at a target of at least its own p.
        let (mut lo, mut hi) = (cp[v], 1.0f64);
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if descend(children, root, &cp, mid, by)[v] {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        q[v] = hi;
    }
    q
}

#[cfg(test)]
mod tests;
