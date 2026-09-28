# legume-enrichment

Gene-set enrichment, ontology scoring, and marker bootstrap for single-cell /
spatial annotation. Extracted from
[`legume-rs`](https://github.com/causalpathlab/legume-rs).

The Rust library crate is named `enrichment`:

```toml
enrichment = { version = "0.4.0", package = "legume-enrichment" }
```

```sh
cargo add legume-enrichment
# then: use enrichment::...
```

## 0.4.0

fgsea-style statistics (NES, sign-aware and multilevel p-values), TreeBH q-values over a
cell-type tree, and a Q matrix softmaxed over the probit of p. This release is semver-breaking
relative to 0.3; see [CHANGELOG.md](CHANGELOG.md).

## License

MIT
