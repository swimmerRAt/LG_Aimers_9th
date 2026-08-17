# CatBoost iteration stability source notes

## Diagnostic frame

- Question: whether replacing fold-specific early stopping with one robust CatBoost iteration count fixes the prior `75→208→1` collapse.
- Metric: row-level Brier Score; lower is better.
- Selection folds: 2022 and 2023, with each fold trained only on prior seasons.
- Reused diagnostic fold: 2024. It is not treated as a fresh one-shot validation.
- Submission baseline: official leaderboard score `852.1984993386`; local 2024 OOF Brier `0.2480239578589654` after the fixed `logit -0.0461` shift.

## Sources

- `data/train.csv`: target, season, game type, model features and row identifiers.
- `artifacts/temporal_ensemble/run_summary.json`: official model feature list.
- `artifacts/probability_refinement/final_comparison/oof_predictions.csv`: rolling-refined official OOF probability.
- `artifacts/catboost_iteration_stability/learning_curve_metrics.csv`: executed iteration-grid results.
- `artifacts/catboost_iteration_stability/fold_metrics.csv`: selected blend and baseline comparison.
- `artifacts/catboost_iteration_stability/run_summary.json`: selection rule, result and paired uncertainty.

## Chart map

| Report section | Analytical question | Family / type | Fields | Supported claim | Palette | Output |
|---|---|---|---|---|---|---|
| Experiment 7 | Does one CatBoost complexity work across seasons? | Trend / three vertically stacked line charts | iterations, refined_brier, validation_season | The season minima differ materially: 210, 10 and 190 trees | Blue line, gold minimum marker, charcoal selected-iteration reference | `learning_curve.png` |

Independent y-scales are used because the chart compares within-season curve shape and iteration minima, not absolute Brier levels across seasons. Exact cross-season Brier values are provided in the adjacent table. The chart was inspected at its exported size; labels, minima and the selected 80-tree reference are visible without clipping.
