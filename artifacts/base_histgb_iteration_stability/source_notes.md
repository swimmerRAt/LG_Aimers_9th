# Base HistGB iteration stability source notes

## Report contract

- Audience: technical.
- Primary report surface: the existing `docs/progress_2026-0817.md` progress report selected earlier in the project.
- Question: whether changing HistGB iterations inside the official temporal HistGB + ExtraTrees model improves season-forward Brier Score.
- Decision baseline: official leaderboard score `852.1984993386`; local shifted 2024 OOF Brier `0.2480239578589654`.
- Success condition: candidate selected only from 2022 and 2023, no development-fold degradation, then lower 2024 diagnostic Brier. Because 2024 is reused, an improvement would still require fresh confirmation.

## Technical report structure mapping

| Required role | Progress report section |
|---|---|
| Technical summary | Experiment 8 opening conclusion |
| Key findings | 2022·2023 candidate table and 2024 comparison |
| Scope and metric definitions | Fixed structure and changed parameter |
| Methodology / model specification | HistGB iteration grid, forward folds and post-processing |
| Robustness and limitations | Exact 300-iteration reconstruction and reused-2024 caveat |
| Recommended next step | Project-level next work list |
| Further question | Whether new information or explicit season-shift features generalize |

## Sources

- `data/train.csv`: row-level target, season, game type and official feature inputs.
- `artifacts/temporal_ensemble/run_summary.json`: official temporal component weights and feature list.
- `artifacts/probability_refinement/final_comparison/oof_predictions.csv`: official rolling-refined OOF probability.
- `artifacts/base_histgb_iteration_stability/development_curve_metrics.csv`: candidate-by-season executed results.
- `artifacts/base_histgb_iteration_stability/outer_diagnostic_metrics.csv`: selected candidate and official 2024 comparison.
- `artifacts/base_histgb_iteration_stability/run_summary.json`: equivalence, selection, uncertainty and adoption decision.

## Visual omission

No chart was added for this experiment. The evidence contains seven ordered hyperparameter candidates and two development folds, and the audit task requires exact Brier deltas and pass/fail status. A compact table is more precise than a trend chart because the x-axis is a discrete model-setting grid rather than elapsed time, and the selected result is determined by a fold constraint plus robust objective rather than curve shape alone.
