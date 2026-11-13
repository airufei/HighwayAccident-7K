# Results

`paper_tables/` contains the printed numeric/text values of all 16 manuscript tables,
in English and Chinese. These CSV files are tabular data, not images.

`recomputed/results.json` is generated locally by `03_evaluation/reproduce.py --check`
and is intentionally ignored by Git. Its expected numeric values are versioned in
`03_evaluation/reference/verified_metrics.json` so a clean checkout can verify them.

The original full-inventory and two-record exclusion analyses are both retained.
The primary predictor and the frozen-feature controls have distinct trained feature
pools/training histories. The released controls do not establish an isolated
equal-budget architecture comparison. See `03_evaluation/README.md` for which tables
are reconstructed and which supplementary summaries are archived only.
