#!/usr/bin/env python3
"""Build the canonical report artifact for the season-importance diagnosis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
DIAGNOSTIC_DIR = ROOT / "artifacts" / "season_diagnostics"


def source(source_id: str, label: str, path: str, description: str, definitions: list[str]):
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "python",
            "language": "python",
            "description": description,
            "executed_at": "2026-08-16T10:30:00Z",
            "tables_used": ["data/train.csv", "model/final_model.pkl"],
            "filters": ["all 1,475,092 training pitches", "seasons 2019 through 2024"],
            "metric_definitions": definitions,
        },
    }


def main() -> None:
    season = pd.read_csv(DIAGNOSTIC_DIR / "season_summary.csv")
    game_type = pd.read_csv(DIAGNOSTIC_DIR / "game_type_by_season.csv")
    counterfactual = pd.read_csv(DIAGNOSTIC_DIR / "season_counterfactual.csv")
    diagnostics = json.loads((DIAGNOSTIC_DIR / "summary.json").read_text(encoding="utf-8"))

    rate_rows = []
    for row in season.itertuples(index=False):
        rate_rows.append({
            "season": str(row.season),
            "series": "Overall",
            "target_rate": row.target_rate,
            "rows": int(row.rows),
        })
    for row in game_type.itertuples(index=False):
        rate_rows.append({
            "season": str(row.season),
            "series": f"Game type {row.game_type}",
            "target_rate": row.target_rate,
            "rows": int(row.rows),
        })

    start = season.loc[season["season"] == 2019, "target_rate"].iloc[0]
    end = season.loc[season["season"] == 2024, "target_rate"].iloc[0]
    overall_change = end - start
    game_2019 = game_type[game_type["season"] == 2019].set_index("game_type")
    game_2024 = game_type[game_type["season"] == 2024].set_index("game_type")
    composition = ((game_2024["season_share"] - game_2019["season_share"]) * game_2019["target_rate"]).sum()
    within = (game_2024["season_share"] * (game_2024["target_rate"] - game_2019["target_rate"])).sum()
    cf_2019 = counterfactual.loc[counterfactual["forced_season"] == 2019, "change_vs_2024_pp"].iloc[0] / 100.0

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sources = [
        source(
            "season_summary_source",
            "Season-level target diagnostics",
            "artifacts/season_diagnostics/season_summary.csv",
            "Season-level aggregation produced by diagnose_season_importance.py.",
            [
                "target_rate = mean(control_success) within season",
                "target_rate_change_pp = year-over-year target-rate change in percentage points",
            ],
        ),
        source(
            "game_type_source",
            "Season and game-type diagnostics",
            "artifacts/season_diagnostics/game_type_by_season.csv",
            "Season-by-game_type aggregation produced by diagnose_season_importance.py.",
            [
                "season_share = rows in game type / all rows in season",
                "within-type effect holds 2024 game-type shares fixed while changing within-type target rates",
            ],
        ),
        source(
            "model_sensitivity_source",
            "Season counterfactual sensitivity",
            "artifacts/season_diagnostics/season_counterfactual.csv",
            "Prediction sensitivity from changing only season on the same 2024 feature rows.",
            [
                "prediction_mean = mean final-ensemble probability after forcing season",
                "change_vs_2024_pp = prediction_mean difference from forced season 2024",
            ],
        ),
        source(
            "tree_diagnostic_source",
            "ExtraTrees split diagnostics",
            "artifacts/season_diagnostics/summary.json",
            "Fitted ExtraTrees feature importance and split-use diagnostics.",
            [
                "importance = normalized impurity decrease in the ExtraTrees component",
                "split share = season splits / all internal tree splits",
            ],
        ),
    ]

    headline = [{
        "season_importance": diagnostics["extra_trees_season_importance"],
        "target_rate_change": overall_change,
        "trees_using_season": diagnostics["trees_using_season"],
        "counterfactual_effect": cf_2019,
    }]
    counterfactual_rows = [
        {
            "forced_season": str(int(row.forced_season)),
            "prediction_mean": row.prediction_mean,
            "prediction_std": row.prediction_std,
            "change_vs_2024_pp": row.change_vs_2024_pp,
        }
        for row in counterfactual.itertuples(index=False)
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "왜 season 중요도가 가장 높은가",
            "description": "LG Aimers 최종 앙상블의 season 중요도에 대한 기술 진단",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [
                {
                    "id": "importance_card",
                    "description": "ExtraTrees 구성요소의 정규화 impurity importance",
                    "dataset": "headline",
                    "sourceId": "tree_diagnostic_source",
                    "metrics": [{"label": "Season importance", "field": "season_importance", "format": "percent"}],
                },
                {
                    "id": "target_change_card",
                    "description": "2019년 대비 2024년 전체 제구 성공률 변화",
                    "dataset": "headline",
                    "sourceId": "season_summary_source",
                    "metrics": [{"label": "Target-rate change", "field": "target_rate_change", "format": "percent", "signed": True}],
                },
                {
                    "id": "tree_use_card",
                    "description": "최종 ExtraTrees 160개 중 season을 한 번 이상 사용한 트리",
                    "dataset": "headline",
                    "sourceId": "tree_diagnostic_source",
                    "metrics": [{"label": "Trees using season", "field": "trees_using_season", "format": "number"}],
                },
                {
                    "id": "sensitivity_card",
                    "description": "동일 2024 행에서 season만 2019로 바꾼 평균 예측 변화",
                    "dataset": "headline",
                    "sourceId": "model_sensitivity_source",
                    "metrics": [{"label": "Prediction shift", "field": "counterfactual_effect", "format": "percent", "signed": True}],
                },
            ],
            "charts": [
                {
                    "id": "season_rate_chart",
                    "title": "시즌·경기 유형별 제구 성공률",
                    "subtitle": "2019~2024, 투구 단위 평균; 전체와 game_type F/R 비교",
                    "type": "bar",
                    "dataset": "season_rate_series",
                    "sourceId": "game_type_source",
                    "valueFormat": "percent",
                    "options": {"grouping": "grouped"},
                    "encodings": {
                        "x": {"field": "season", "type": "nominal", "label": "Season"},
                        "y": {"field": "target_rate", "type": "quantitative", "label": "Control success rate", "format": "percent"},
                        "color": {"field": "series", "type": "nominal", "label": "Series"},
                        "tooltip": [
                            {"field": "rows", "type": "quantitative", "label": "Rows", "format": "number"}
                        ],
                    },
                }
            ],
            "tables": [
                {
                    "id": "counterfactual_table",
                    "title": "Season 단독 변경에 대한 모델 민감도",
                    "subtitle": "동일한 2024 검증 행에서 season 값만 변경; 인과효과가 아닌 모델 의존도",
                    "dataset": "counterfactual",
                    "sourceId": "model_sensitivity_source",
                    "defaultSort": {"field": "forced_season", "direction": "asc"},
                    "columns": [
                        {"field": "forced_season", "label": "Forced season", "type": "text"},
                        {"field": "prediction_mean", "label": "Prediction mean", "format": "percent"},
                        {"field": "prediction_std", "label": "Prediction std", "format": "percent"},
                        {"field": "change_vs_2024_pp", "label": "Change vs 2024, pp", "format": "number", "signed": True},
                    ],
                }
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# 왜 season 중요도가 가장 높은가"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": "## 기술 요약\n\n- **`season`은 직접적인 제구 원인이라기보다 시간 드리프트의 대리변수다.** 실제 성공률은 2019년 56.47%에서 2024년 48.61%로 7.86%p 하락했다.\n- **경기 유형의 비중 변화가 원인은 아니다.** 2019→2024 변화 중 구성 효과는 +0.14%p이고, 같은 유형 내부 변화가 -7.99%p였다.\n- **모델도 이 연도 구간을 적극 사용한다.** ExtraTrees 160개 모두 `season`을 사용했고, 동일 2024 행의 season만 2019로 바꾸면 평균 예측이 3.89%p 상승했다.\n- **중요도 16.23%는 인과성이나 2025 외삽 능력을 의미하지 않는다.** 2025는 학습 범위 밖이므로 트리에서 2024와 같은 말단 구간으로 이동한다."
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["importance_card", "target_change_card", "tree_use_card", "sensitivity_card"]},
                {
                    "id": "target_shift_finding",
                    "type": "markdown",
                    "sourceId": "season_summary_source",
                    "body": "## 연도별 기준선이 크게 달라 season이 강한 분할 기준이 된다\n\n전체 타깃률은 2019년 56.47%에서 2020년 53.27%로 3.20%p 떨어졌고, 2023년에 다시 2.90%p 하락한 뒤 2024년 48.61%까지 내려갔다. 각 시즌의 표본은 23.7만~25.4만 투구로 비슷하므로 작은 표본의 우연한 변동보다는 광범위한 분포 변화로 보는 편이 타당하다."
                },
                {"id": "season_rate_chart_block", "type": "chart", "chartId": "season_rate_chart"},
                {
                    "id": "within_type_finding",
                    "type": "markdown",
                    "sourceId": "game_type_source",
                    "body": f"## game_type 구성보다 같은 유형 내부 변화가 하락을 설명한다\n\n2019→2024 전체 성공률 변화는 {overall_change * 100:.2f}%p다. 2024 유형 비중을 적용한 분해에서 구성 효과는 {composition * 100:+.2f}%p에 불과했고, 유형 내부 성공률 변화는 {within * 100:+.2f}%p였다. 특히 F 유형은 68.92%에서 45.93%로 23.00%p 하락했다. 따라서 `season`은 단순히 F/R 비율 차이를 대신하는 것이 아니라 각 유형 안의 시대별 체계적 차이까지 포착한다."
                },
                {
                    "id": "model_sensitivity_finding",
                    "type": "markdown",
                    "sourceId": "model_sensitivity_source",
                    "body": "## 동일 행에서도 season 값이 예측 기준선을 이동시킨다\n\n2024의 253,507개 행에서 다른 피처를 고정하고 season만 바꾸면 평균 예측은 2019일 때 52.65%, 2023일 때 49.23%, 2024일 때 48.76%였다. 2025도 48.76%로 2024와 완전히 같았다. 이는 트리가 알려진 연도 구간을 선택하지만 숫자형 연도의 추세를 학습 범위 밖으로 외삽하지는 못한다는 뜻이다."
                },
                {"id": "counterfactual_table_block", "type": "table", "tableId": "counterfactual_table"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": "## 분석 범위와 중요도 정의\n\n- 분석 단위는 학습 데이터의 투구 1행이며 기간은 2019~2024년이다.\n- `target_rate`는 시즌 또는 시즌×game_type 안의 `control_success` 평균이다.\n- 피처 중요도는 최종 앙상블 중 ExtraTrees 구성요소의 정규화 impurity decrease다. 앙상블 전체의 인과 기여도나 permutation importance가 아니다.\n- 반사실 민감도는 동일 2024 행에서 `season` 값만 바꾼 예측 차이다. 실제 과거·미래 데이터 시나리오나 인과효과가 아니다."
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": "## 진단 방법\n\n연도별 타깃률과 game_type별 비중·타깃률을 전체 147.5만 행에서 집계했다. 2019→2024 차이는 game_type 구성 효과와 유형 내부 효과로 정확히 분해했다. 저장된 최종 ExtraTrees 160개의 분할 노드를 조사해 season 사용 트리 수와 깊이를 계산했고, 최종 모델에서 2024 행의 season만 2019~2025로 바꿔 예측 민감도를 측정했다."
                },
                {
                    "id": "model_specification",
                    "type": "markdown",
                    "sourceId": "tree_diagnostic_source",
                    "body": "## 트리 사용 패턴은 높은 중요도가 반복 분할에서 나온다는 점을 확인한다\n\nExtraTrees 160개 전부가 `season`을 사용했다. 전체 내부 분할 818,780개 중 season 분할은 29,866개(3.65%)였고 중앙 깊이는 12였다. 루트에서도 사용됐지만 대부분은 다른 상황 피처와 결합된 깊은 상호작용 분할이므로, 16.23% 중요도를 단순한 단변량 상관으로 해석해서는 안 된다. 실제 season과 타깃의 Pearson 상관은 -0.0483으로 작다."
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## 확인된 것은 시간 드리프트이며 구체적 원인은 아직 미확정이다\n\n데이터만으로는 성공률 하락이 리그 환경, 선수 구성, 측정 장비, 라벨 정의 또는 운영 규칙 중 무엇 때문에 발생했는지 식별할 수 없다. impurity importance는 상관 피처 사이에서 분산되거나 트리 구조에 의해 과대평가될 수 있다. 또한 2025 실제 정답률이 비공개이므로 season을 제거하거나 추세 보정하는 것이 공식 점수를 높인다고 아직 단정할 수 없다."
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## 1,000점 목표를 위한 다음 검증\n\n1. 현재 2024 holdout에서 `season` 포함/제거 모델을 같은 하이퍼파라미터로 다시 학습해 Brier 차이를 측정한다.\n2. `season`을 제거하는 대신 과거 시즌과의 상대값, 최근 2개 시즌 가중, 명시적 확률 intercept 보정을 비교한다.\n3. 단순 impurity importance가 아니라 2024 permutation importance와 grouped permutation을 계산해 상관 피처 묶음의 실제 기여를 평가한다.\n4. 위 변경은 한 번에 하나씩 ablation하고 2022·2023·2024 forward split을 모두 기록한다."
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## 추가 확인 질문\n\n- 2023년에 `control_success` 또는 game_type F의 생성·라벨 기준이 변경됐는가?\n- 2025 평가 데이터에서도 2023~2024의 성공률 하락이 계속된다는 사전 정보가 있는가?\n- 시즌별 장비·운영·규칙 변경을 설명할 수 있는 공식 메타데이터가 제공되는가?"
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "season_rate_series": rate_rows,
                "counterfactual": counterfactual_rows,
            },
        },
        "sources": sources,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(REPORT_DIR / "artifact.json")


if __name__ == "__main__":
    main()
