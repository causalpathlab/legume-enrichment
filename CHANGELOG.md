# Changelog

## 0.4.0

**Breaking** relative to 0.3: new public fields on `AnnotateConfig` and `AnnotateOutputs`
(exhaustive struct literals and destructuring must be updated), and changed p-value and Q
defaults.

### Added

- `fgsea` module: the exact weighted-KS score from hit ranks in `O(m log m)` (`es_from_hits`),
  and fgsea's multilevel p-value (`multilevel_p`, `Multilevel`) for tails the gene-set null's
  draws cannot resolve. It follows fgseaMultilevel's `EsRuler`: per-level posterior-mean log
  estimates `ψ(h + 1) − ψ(n + 1)`, `log2err` summed from per-level `ψ₁` variances, ties broken
  by a gene hash, and MCMC until `n·m/2` swaps are accepted, then as many sweeps again.
  `log2err` is NaN where the p is only a bound (it reached `eps`, no sampled set reached the
  score, or a level's MCMC hit its sweep cap).
- `treebh::TypeTree` and `treebh::treebh_q`: TreeBH-adjusted q-values over a tree of cell types.
- `AnnotateConfig::multilevel` (default on) and `AnnotateConfig::type_tree` (default `None`).
- `AnnotateOutputs::nes_kc` (fgsea's normalized enrichment score, the reported effect size),
  `z_kc` (`Φ⁻¹(1 − p)`) and `p_log2err_kc` (the multilevel estimate's error).
- `es::positions`, and `GeneStrata::stratum_of` / `GeneStrata::members`.

### Changed

- The gene-set-null p-value (`num_sample_perm == 0`) is fgsea's sign-aware
  `(#null ≥ ES + 1) / (#null ≥ 0 + 1)`, with `p = 1` for `ES ≤ 0`; below 10 draws past ES the
  multilevel estimate replaces it.
- q-values are TreeBH over `type_tree` when given, per topic row; flat BH otherwise.
- The Q matrix is a row softmax over `z = Φ⁻¹(1 − p)` among FDR survivors (was the
  restandardized ES).
- Cell types dropped by `min_markers`, or whose panel covers every gene, are no longer
  hypotheses: `p = q = 1` and they are left out of every BH / TreeBH family. Before, they could
  carry the floor p-value and make the other types' q anti-conservative.
- `TypeTree` validation rejects cycles, shared children, a type on the root or off the tree, and
  two types on one leaf; `annotate` fails on a malformed tree before any null is drawn.
- The gene-set null is scored from hit ranks in one pass instead of two walks over every gene.

The sample-permutation path (`num_sample_perm > 0`) and the marker bootstrap are unchanged.

## 0.3.9

Initial release, extracted from `legume-rs`.
