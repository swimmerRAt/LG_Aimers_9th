# Optuna 과적합 진단 보고서 빌드 차단 기록

- 선택한 보고서 형태: 기술 독자용 portable HTML
- 진단 범위: 2024 단일 holdout Optuna 탐색과 실제 제출 산출물의 일치 여부
- 차단 원인: 현재 실행 환경에 `node`와 `npm`이 없어 Data Analytics 보고서의 필수
  `deliver_portable_artifact.mjs` 검증·패키징 명령을 실행할 수 없음
- 대체 MCP/Sites 보고서 렌더러: 현재 세션에 callable report artifact 도구가 없음
- 확인한 근거:
  - `artifacts/optuna_2024/best_params.json`
  - `artifacts/optuna_2024/oof_predictions.csv`
  - `artifacts/optuna_2024/temporal_validation.csv`
  - `artifacts/optimized_ensemble_2024/oof_predictions.csv`
  - `model/final_model*.pkl`
  - `artifacts/submissions/submit_*.zip`
- 핵심 진단:
  - Optuna는 2022가 아니라 2024 단일 holdout에 최적화됨
  - Optuna 원시 후보의 2024 Brier 개선량 `-0.00001069`의 paired 95% 구간은
    `[-0.00002881, +0.00000744]`로 0을 포함함
  - 현재 제출 ZIP의 모델 해시는 Optuna 이전 `model/final_model*.pkl`과 일치하며,
    Optuna validation model이 제출 ZIP에 들어간 증거가 없음
- 권장 수정: 2021~2023 inner time-series CV, 2024 one-shot outer holdout, fold별 정규화
  Brier의 recency-weighted robust objective, 동일 tree 수로 탐색·최종 학습, 보정 분리

## Robust 재설계 후 추가 증거

- 내부 objective는 `1.00466492 → 1.00202448`로 개선됐지만 시즌별 방향이 일치하지 않음
  - 2021 Brier: `-0.00018182` 개선
  - 2022 Brier: `+0.00031138` 악화
  - 2023 Brier: `-0.00104304` 개선
- 2023 가중치가 55%라 Optuna는 2022 악화를 감수하고 2023 개선이 큰 후보를 선택함
- 선택된 앙상블은 HistGB `99.987%`, ExtraTrees `0.013%`로 사실상 다양성을 제거함
- 2024 one-shot 결과:
  - 기존 모델 `700.25764점`, Brier `0.24805763`
  - Robust Optuna `494.85698점`, Brier `0.24857074`
  - Brier 차이 `+0.00051311`, paired 95% 구간 `[+0.00044383, +0.00058238]`
- 평균 예측 편향 제곱 증가는 전체 Brier 악화의 약 7.2%만 설명함. 나머지 약 92.8%는
  행별 확률의 분해능·순위·상황별 오차 구조가 나빠진 부분으로 해석됨
- 실험 결론: `rejected_keep_baseline`; 동일 artifact는 outer lock으로 추가 튜닝 금지

## 현재 보고서 전달 상태

정량 비교에는 시즌이 4개뿐이라 추세 차트보다 정확한 비교표가 적합하다. 하지만 portable
HTML은 필수 Node 패키저가 없어 생성·검증할 수 없으며, MCP/Sites 보고서 렌더러도 현재
세션에서 callable하지 않다. 따라서 이 파일이 `build-report` 절차의 구체적 차단 기록이다.

## 복잡도 가설 추가 진단

강한 부스팅 모델로 교체하는 것은 다음 실험 가치가 있지만 단독 해결책은 아니다. 기존 탐색도
HistGB leaf nodes 최대 `63`/iteration 최대 `500`, ExtraTrees depth 최대 `22`의 후보를
포함했으나 robust objective는 HistGB leaf nodes `7`/iteration `150`, ExtraTrees depth
`11`과 HistGB 비중 `99.987%`를 선택했다. 즉 이번 실패의 직접 원인은 사용할 수 있는 모델
복잡도의 부족보다는 시즌 간 손실을 상쇄할 수 있었던 선택 규칙과 외부 기간으로의 일반화 실패다.
