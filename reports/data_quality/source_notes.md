# Report source notes

Audience: technical. Delivery mode: portable HTML.

## Required structure mapping

- Title: `LG Aimers 데이터·기준선 준비 보고서`
- Technical summary: `## 기술 요약`
- Key findings with visual evidence: target-rate and forward-score sections
- Scope and definitions: `## 범위와 지표 정의`
- Methodology/model validation: `## 검사 및 검증 방법`
- Limitations and robustness: `## 아직 최종 모델을 확정할 수 없는 이유`
- Recommended next steps: `## 다음 실험 순서`
- Further questions: `## 확인이 필요한 질문`

## Chart map

| Section | Question | Family/type | Fields | Takeaway | Palette policy |
|---|---|---|---|---|---|
| Target drift | Does the event rate change by season? | Comparison / bar | season, target_rate | Random split is unsafe | single-root preferred |
| Forward validation | Are model gains stable by validation year? | Comparison / grouped bar | validation_season, competition_score, model | 2023 fails for all learned models | relaxed multi-category, 3 roots |

The six-season target-rate view uses bars instead of a line because six annual anchors are too
sparse for a trend line under the visualization contract. Brier differences are preserved in an
exact table; the chart uses competition score from zero so small absolute Brier differences are not
shown on a misleading truncated bar scale.

## Evidence lineage

- Data-quality rows originate from `audit_data.py` and `artifacts/data_quality/report.json`.
- Model rows originate from `train_baseline.py`, `artifacts/forward_cv_2022_2024/metrics.csv`, and
  the matching OOF predictions.
- The SQL files in `queries/` are executable SQLite projections of the reviewed bounded rows used
  in the report artifact.

