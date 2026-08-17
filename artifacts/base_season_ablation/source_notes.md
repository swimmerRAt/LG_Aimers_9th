# Base season ablation source notes

## Report contract

- Audience: technical.
- Primary surface: the existing `docs/progress_2026-0817.md` report selected earlier in the project.
- Question: whether removing the high-importance season feature improves season-forward generalization of the official temporal HistGB + ExtraTrees model.
- Baseline: official leaderboard score `852.1984993386`; local shifted 2024 OOF Brier `0.2480239578589654`.
- Selection rule: improve the robust 2022–2023 objective with no individual development-fold degradation. Treat 2024 as reused diagnostic only.

## Technical report structure mapping

| Required role | Progress report coverage |
|---|---|
| Technical summary | Experiment 9 opening conclusion |
| Key findings | Season-level and game-type cohort tables |
| Scope and definitions | One-feature ablation and fixed pipeline description |
| Methodology / model specification | Forward folds, unchanged temporal weights and post-processing |
| Robustness and limitations | Per-season gate, paired intervals and reused-2024 warning |
| Recommended next step | F/R-specific temporal modeling |
| Further question | Whether game-type-specific windows generalize without overfitting F's smaller cohort |

## Sources

- `data/train.csv`: row-level target, season, game type and official inputs.
- `artifacts/temporal_ensemble/run_summary.json`: feature list and temporal component weights.
- `artifacts/probability_refinement/final_comparison/oof_predictions.csv`: official rolling-refined OOF probability.
- `artifacts/base_season_ablation/season_metrics.csv`: season-level result.
- `artifacts/base_season_ablation/game_type_metrics.csv`: game-type driver decomposition.
- `artifacts/base_season_ablation/run_summary.json`: objectives, paired intervals and decision.

## Visual omission

No chart was added. The result has three discrete validation seasons and two game-type cohorts, and the decision depends on exact Brier deltas, prediction means and a pass/fail gate. Tables provide the required audit precision; a three-point trend would overstate continuity and does not add evidence beyond the exact cohort comparison.
