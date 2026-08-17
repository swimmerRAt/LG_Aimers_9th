# Optuna 점수 하락 진단 소스 노트

## 보고 목적과 독자

- 질문: 내부 Optuna objective가 개선됐는데 2024 점수는 왜 하락했는가?
- 독자: 모델링·검증 방법을 검토하는 기술 독자
- 비교 기준: 현재 45% HistGB + 55% ExtraTrees와 robust Optuna 후보
- 성공 기준: 방향 전환의 원인을 내부 시즌, 모델 구성, 2024 편향으로 분리

## 기술 보고서 구조 매핑

1. 제목: Optuna 내부 개선이 2024에서 뒤집힌 이유
2. 기술 요약: Optuna가 아니라 objective와 데이터 드리프트의 불일치
3. 주요 근거: 시즌별 Brier 표와 2024 paired 비교표
4. 범위·정의: 2021~2023 inner, 2024 one-shot outer, 정규화 Brier
5. 방법: recency-weighted mean + stability penalty, frozen selection
6. 한계·강건성: outer 재사용 금지, 평균 편향은 전체 악화의 7.2%만 설명
7. 다음 단계: 기존 모델 유지, 새 실험은 별도 사전등록 디렉터리 사용
8. 추가 질문: 2023 분포 변화가 2025에도 반복되는지 확인 가능한 메타데이터 존재 여부

## 근거 파일

- `artifacts/optuna_robust/selection.json`
- `artifacts/optuna_robust/inner_fold_metrics.csv`
- `artifacts/optuna_robust/outer_evaluation.json`
- `artifacts/optuna_robust/histgb_trials.csv`
- `artifacts/optuna_robust/extra_trees_trials.csv`
- `artifacts/optuna_robust/blend_trials.csv`
- `src/lg_aimers/metrics.py`

## 시각화 생략 이유

핵심 비교는 2021~2024 네 개 시즌과 두 후보의 정확한 Brier·점수 값이다. 네 점으로 추세를
강조하면 일반적인 시간 추세로 오해할 수 있어 기술 보고서에서는 비교표가 더 적합하다.

## 복잡한 모델로 교체하면 해결되는가

- 결론: 더 강한 모델은 개선 후보이지만, 검증 objective 불일치를 단독으로 해결하지 못한다.
- 현재 모델도 비선형 `HistGradientBoostingClassifier`와 `ExtraTreesClassifier`의 앙상블이다.
- HistGB 탐색 범위에는 최대 leaf nodes `63`, iteration `500`이 포함됐지만 선택값은 각각
  `7`, `150`이었다. ExtraTrees도 depth `10~22`를 탐색했지만 선택값은 `11`이었다.
- 따라서 이번 결과는 탐색기가 복잡한 후보를 전혀 보지 못한 결과라기보다, 2021~2023
  가중 objective가 더 단순하고 규제된 후보를 선호한 결과다.
- 더 표현력이 큰 LightGBM, XGBoost, CatBoost를 같은 objective에 연결하면 시간에 따라
  바뀌는 `season` 및 상황별 우연한 패턴까지 더 잘 학습할 수 있어 과적합 위험도 함께 커진다.
- 다음 모델 비교는 동일한 forward folds에서 수행하고, 평균 objective와 별도로 모든 시즌의
  Brier 악화 상한, 앙상블 다양성, 추론 시간 및 모델 크기를 통과 조건으로 둬야 한다.
- 2024 outer 결과는 이미 확인했으므로 이후 실험에서 다시 미관측 one-shot 검증으로 부를 수
  없다. 새 모델의 최종 일반화 주장은 새로운 외부 기간 또는 공식 블라인드 평가가 필요하다.
