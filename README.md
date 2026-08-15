# LG Aimers DACON — 제구 성공 확률 예측

목표는 누수 없는 forward validation에서 Brier Score를 낮추고, 평가 서버에서 10분 안에
오프라인 추론되는 `submit.zip`을 만드는 것이다.

## 현재 데이터와 기준

- `train.csv`: 2019–2024, 1,475,092행, 타깃 포함
- `test.csv`: 2025 형식 확인용 5행; 실제 평가는 245,789행
- `trackman_history.csv`: 2019–2024, 1,793,078행; 메인 데이터와 1:1 조인 불가
- 검증 기본값: 2019–2023 학습 → 2024 검증
- 1차 지표: validation Brier Score

## 로컬 환경

평가 서버 버전과 맞춘 프로젝트 환경이 `.venv/`에 있다.

```bash
.venv/bin/python audit_data.py
.venv/bin/python -m unittest discover -s tests -v
```

## 기준선 학습

먼저 상수 모델과 제공 RandomForest를 2024 forward holdout에서 비교한다.

```bash
.venv/bin/python train_baseline.py \
  --models constant random_forest \
  --validation-seasons 2024
```

다음 실험에서는 선형 모델과 HistGradientBoosting을 같은 split으로 비교한다.

```bash
.venv/bin/python train_baseline.py \
  --models constant logistic random_forest histgb \
  --validation-seasons 2022 2023 2024
```

학습 결과는 `artifacts/baseline/`, 선택 모델은 `model/final_model.pkl`에 저장된다.

## 추론과 제출물

```bash
.venv/bin/python script.py
.venv/bin/python build_submit.py
unzip -l submit.zip
```

`script.py`는 `data/`를 우선 사용하고 없으면 `open/`을 탐색한다. ID 누락·중복,
피처 누락, NaN/무한대, `[0,1]` 밖의 확률이 있으면 placeholder를 남기지 않고 실패한다.

제출 ZIP 최상위에는 `model/`, `script.py`, `requirements.txt`만 포함된다.

## 실험 원칙

- 새 피처는 [누수 점검표](docs/leakage_checklist.md)를 먼저 통과한다.
- 가중치·보정·clipping은 리더보드가 아니라 고정 OOF Brier로 결정한다.
- 실제 test의 다른 행을 이용한 빈도, rolling, 분포 통계는 사용하지 않는다.
- 최종 후보는 실제 평가 행 수를 가정해 7–8분 이내 추론을 목표로 한다.

현재 실험 결과와 다음 작업은 [2026-08-15 진행 기록](docs/progress_2026-08-15.md), 데이터
품질·검증 근거는 [기술 보고서](reports/data_quality/report.html)에서 확인할 수 있다.
