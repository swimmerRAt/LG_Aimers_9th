# LG Aimers 제구 성공 확률 예측 모델

## 1. 프로젝트 개요

이 프로젝트는 KBO의 각 투구에 대해 **투구 직전까지 알 수 있는 정보만 사용하여 제구에
성공할 확률**을 예측한다. 데이터의 기본 단위는 투구 1개이며, 정답을 0 또는 1로 단순
분류하는 것보다 실제 결과에 잘 맞는 확률을 생성하는 것이 중요하다.

모델이 예측하는 `control_success`의 의미는 다음과 같다.

| 값 | 의미 |
|---:|---|
| `1` | 제구 성공 |
| `0` | 제구 실패 |
| `0~1` 사이의 예측값 | 해당 투구가 제구에 성공할 것으로 예측한 확률 |

## 2. 데이터

### 2.1 입력 데이터

현재 최종 모델 학습에는 `train.csv`만 사용한다. `trackman_history.csv`도 활용 가능성을
검토했지만, 신뢰할 수 있는 선수 매핑의 검증 행 커버리지가 충분하지 않아 최종 모델에는
포함하지 않았다.

| 파일 | 기간 | 크기 | 주요 내용 | 사용 목적 |
|---|---|---:|---|---|
| `data/train.csv` | 2019~2024년 | 1,475,092행 × 49컬럼 | `row_id`, 입력 피처 47개, 정답 `control_success` | 모델 학습과 로컬 검증 |
| `data/trackman_history.csv` | 2019~2024년 | 1,793,078행 × 30컬럼 | 구종, 구속, 회전수, 무브먼트, 릴리스 위치 등 | 매핑 가능성 검토, 최종 모델에서는 미사용 |

`train.csv`의 47개 입력 피처는 다음과 같이 구성된다.

| 피처 종류 | 대표 컬럼 | 설명 |
|---|---|---|
| 경기 정보 | `season`, `game_month`, `game_dayofweek`, `inning`, `top_bottom`, `game_type` | 시즌, 날짜 특성, 이닝 및 경기 유형 |
| 투구 직전 카운트 | `balls_before`, `strikes_before`, `outs_before` | 현재 투구가 시작되기 직전의 볼·스트라이크·아웃 수 |
| 점수 상황 | `run_top_before`, `run_bot_before`, `run_total_before`, `score_diff_home`, `score_diff_pitcher_team` | 투구 직전 점수와 팀 기준 점수 차이 |
| 주자 상황 | `runner_on_1b`, `runner_on_2b`, `runner_on_3b`, `num_runners_on`, `base_state` | 각 루의 주자 여부와 전체 베이스 상태 |
| 경기 중요도 | `home_win_expectancy`, `away_win_expectancy`, `li` | 기대 승률과 레버리지 지수 |
| 선수·팀 정보 | `pitcher_id`, `batter_id`, `pitcher_hand`, `batter_hand`, `pitcher_team_id`, `batter_team_id` | 투수·타자·소속 팀과 좌우 유형 |
| 공식 과거 이력 | `asof_pitcher_*`, `asof_batter_*` | 현재 투구 직전까지 계산된 누적 투구 수, 제구 성공률, 직전 경기 이력, 구종 구성 비율 |

`asof_*` 컬럼은 현재 행의 결과를 포함하지 않고 투구 직전까지의 과거 정보로 계산된
공식 피처다. 결측값은 주로 과거 표본이 없는 cold-start 선수에게 발생한다.

`trackman_history.csv`는 메인 데이터와 1:1로 연결되는 정답 테이블이 아니다. 현재 품질
점검에서는 메인 `pitcher_id`와 `pitcher_trackman_id`의 직접 교집합이 0개로 확인됐으므로,
공식 매핑이나 검증 가능한 연결 키 없이 선수 단위로 직접 조인하지 않는다.

### 2.2 최종 모델 입력 피처

원본 47개 입력 피처 중 33개를 최종 모델에 사용한다. 정수 ID의 크기에 순서 의미가 없는
`pitcher_id`, `batter_id`와 2024 forward validation에서 불안정하거나 저신호로 확인된
12개 피처를 제외했다.

| 구분 | 피처 |
|---|---|
| 최종 사용 33개 | `season`, `game_dayofweek`, `inning`, `top_bottom`, `game_type`, `balls_before`, `strikes_before`, `outs_before`, `runner_on_1b`, `runner_on_2b`, `runner_on_3b`, `base_state`, `home_win_expectancy`, `li`, `pitcher_hand`, `batter_hand`, `pitcher_team_id`, `batter_team_id`, `asof_pitcher_n`, `asof_pitcher_success_rate`, `asof_pitcher_reverse_rate`, `asof_pitcher_ball_rate`, `asof_pitcher_strike_rate`, `asof_pitcher_prev1_game_success_rate`, `asof_pitcher_prev3_game_success_rate`, `asof_pitcher_prev5_game_success_rate`, `asof_pitcher_prev1_game_middle_rate`, `asof_batter_n`, `asof_batter_success_rate`, `asof_batter_middle_rate`, `asof_pitcher_fastball_rate`, `asof_pitcher_breaking_rate`, `asof_pitcher_offspeed_rate` |
| 제외 14개 | `pitcher_id`, `batter_id`, `game_month`, `run_top_before`, `run_bot_before`, `run_total_before`, `score_diff_home`, `score_diff_pitcher_team`, `num_runners_on`, `away_win_expectancy`, `asof_pitcher_middle_rate`, `asof_pitcher_prev3_game_middle_rate`, `asof_pitcher_prev5_game_middle_rate`, `asof_pitcher_pitchmix_n` |

타깃 인코딩은 최종 모델에 사용하지 않는다. 초기 leave-one-out 방식에서 그룹 건수와
인코딩 값의 미세한 차이를 트리가 이용해 행의 정답을 역산할 수 있는 누수 문제가
확인됐다. 교차적합 방식으로 수정한 뒤에도 검증 성능이 개선되지 않아 제거했다.

### 2.3 테스트 데이터

| 파일 | 기간 | 크기 | 정답 포함 여부 | 사용 목적 |
|---|---|---:|---|---|
| 로컬 `data/test.csv` | 2025년 형식 | 5행 × 48컬럼 | 미포함 | 컬럼 구조와 추론 코드 확인 |
| 평가 서버 `data/test.csv` | 2025년 | 245,789행 × 48컬럼 | 미포함 | 실제 리더보드 평가 |
| `data/sample_submission.csv` | 테스트 데이터와 동일한 ID | 로컬 5행, 서버에서는 실제 테스트 행 수 | 예측값 placeholder만 포함 | 제출 컬럼과 `row_id` 순서의 기준 |

테스트 데이터에는 `row_id`와 원본 입력 피처 47개가 존재하지만 실제 정답인
`control_success`는 제공되지 않는다. 추론 코드는 저장된 `feature_columns`를 기준으로
33개 피처만 정확한 순서로 선택한다. 현재 로컬의 5행 `test.csv`는 형식 확인용 샘플이며,
평가 서버에서 245,789행의 실제 평가 파일로 교체된다.

평가 데이터의 각 행은 **독립적으로 예측**해야 한다. 다음과 같이 테스트 데이터의 다른
행을 이용해 만든 피처는 사용할 수 없다.

- 테스트 데이터 내부의 선수·팀·월별 빈도 또는 분포 통계
- 테스트 행 순서를 이용한 rolling 또는 expanding 피처
- 테스트 데이터 내부 target encoding
- 테스트 데이터 전체를 확인한 뒤 계산한 사후 보정값

## 3. 모델 출력

최종 출력은 각 투구의 제구 성공 확률을 담은 `output/submission.csv`다.

```csv
row_id,control_success
TEST_000001,0.462913
TEST_000017,0.447574
TEST_000213,0.485205
```

출력 파일은 다음 조건을 모두 만족해야 한다.

| 검증 항목 | 조건 |
|---|---|
| 컬럼 | 정확히 `row_id`, `control_success` 두 개 |
| 행 수 | 실제 테스트 데이터와 동일 |
| ID | 테스트 데이터와 동일하며 중복·누락 없음 |
| 행 순서 | `sample_submission.csv`의 `row_id` 순서와 동일 |
| 예측값 자료형 | 숫자형 실수 |
| 예측값 범위 | `0 ≤ control_success ≤ 1` |
| 유효성 | NaN과 무한대가 없어야 함 |
| 저장 위치 | `output/submission.csv` |

예측 확률을 0 또는 1로 반올림하지 않는다. 예를 들어 `0.8`은 모델이 해당 투구의 제구
성공 가능성을 80%로 판단했다는 뜻이다.

학습 과정에서는 다음 중간 산출물도 생성한다.

| 산출물 | 역할 |
|---|---|
| `model/final_model.pkl` | 보정하지 않은 기본 최종 앙상블 |
| `model/final_model_mild.pkl` | 2024 OOF 선형 보정을 50% 적용한 후보 |
| `model/final_model_calibrated.pkl` | 2024 OOF 최적 선형 보정을 모두 적용한 후보 |
| `artifacts/optimized_ensemble_2024/` | 검증 지표, OOF 예측, 실행 요약 |
| `artifacts/submissions/submit_*.zip` | 원본·절반 보정·완전 보정 제출 파일 |

## 4. 성능지표

### 4.1 Brier Score

기본 성능지표는 예측 확률과 실제 정답의 제곱 오차 평균인 Brier Score다.

\[
Brier = \frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2
\]

- \(p_i\): 모델이 예측한 제구 성공 확률
- \(y_i\): 실제 정답 0 또는 1
- \(n\): 평가 대상 투구 수

Brier Score는 낮을수록 좋고, 완벽한 예측은 0이다. 실제 정답이 1일 때의 예시는 다음과
같다.

| 예측 확률 | 제곱 오차 |
|---:|---:|
| `0.9` | `0.01` |
| `0.7` | `0.09` |
| `0.5` | `0.25` |
| `0.1` | `0.81` |

확신도가 높은 오답일수록 손실이 매우 커지므로 정확도나 ROC-AUC뿐 아니라 확률 보정이
중요하다.

### 4.2 대회 점수

대회에서는 Brier Score를 다음과 같이 정규화한다.

\[
Score = \max\left(0,\ 100000\left(1-\frac{Brier}{r(1-r)}\right)\right)
\]

여기서 \(r\)은 평가 데이터의 실제 평균 제구 성공률이다. \(r(1-r)\)은 모든 행에 동일한
평균 성공 확률 \(r\)만 제출했을 때의 기준 Brier Score다.

| 결과 | 대회 점수 해석 |
|---|---|
| `Brier = 0` | 완벽한 예측으로 100,000점 |
| `Brier = r(1-r)` | 평균 확률 기준선과 같아 0점 |
| `Brier > r(1-r)` | 기준선보다 나쁘며 최종 점수는 0점으로 절삭 |
| `0 < Brier < r(1-r)` | 기준선보다 좋은 모델이며 Brier가 낮을수록 높은 점수 |

참가자 모두에게 분모 \(r(1-r)\)이 동일하므로 모델 순위를 높이려면 사실상 Brier Score를
최소화해야 한다.

## 5. 로컬 검증 방법

미래 시즌을 예측하는 평가 상황을 재현하기 위해 무작위 행 분할 대신 forward
validation을 사용한다. 최종 모델 선택의 주 검증은 2019~2023년 학습, 2024년 검증이며,
2022년과 2023년 split은 연도 간 안정성 진단에 사용했다.

| 학습 기간 | 검증 기간 |
|---|---|
| 2019~2021년 | 2022년 |
| 2019~2022년 | 2023년 |
| 2019~2023년 | 2024년 |

각 split에서 다음 항목을 기록한다.

- Brier Score와 대회 환산 점수
- 실제 평균 성공률과 모델의 평균 예측 확률
- 학습·추론 시간
- 선수 cold-start와 결측 구간 성능

현재 데이터에서는 실제 성공률이 2019년 약 56.47%에서 2024년 약 48.61%까지 하락했다.
이와 같은 시간 드리프트 때문에 랜덤 분할 결과만으로 모델을 선택하지 않는다.

2023년은 이전 시즌으로 학습한 모든 모델의 점수가 0으로 절삭될 정도로 분포 변화가 컸다.
최종 모델은 이 변화 이후인 2023년과 2024년을 모두 포함해 2019~2024년 전체 데이터로
다시 학습한다.

## 6. 전체 처리 흐름

```text
2019~2024년 과거 투구 데이터
        ↓
원본 47개 중 최종 피처 33개 선택
        ↓
결측값 처리 및 범주형 인코딩
        ↓
HistGB 45% + ExtraTrees 55% 확률 앙상블
        ↓
2025년 각 투구의 제구 성공 확률 예측
        ↓
output/submission.csv 생성
        ↓
Brier Score 및 정규화 대회 점수 산출
```

## 7. 현재 모델과 성능 현황 (2026-08-15)

현재 제출 모델은 `model/ensemble.py`의 `OptimizedBaseballEnsemble`이며, 두 모델이 동일한
전처리 행렬을 공유한다.

### 7.1 전처리

| 피처 구분 | 처리 방식 |
|---|---|
| 범주형 `top_bottom`, `game_type`, `base_state` | 최빈값 대치 후 `OrdinalEncoder`; 미등록 범주는 `-1` |
| 나머지 숫자형 30개 | 학습 데이터 중앙값으로 결측 대치 |

전처리기와 두 학습기는 하나의 joblib 산출물에 함께 저장된다. 테스트 데이터의 다른 행을
참조하는 전처리는 없다.

### 7.2 앙상블 구성

| 구성 요소 | 설정 | 최종 가중치 |
|---|---|---:|
| HistGradientBoosting | learning rate 0.04, 31 leaf nodes, 300 iterations, min leaf 200, L2 규제 5, early stopping 미사용 | 45% |
| ExtraTrees | 160 trees, max depth 16, min leaf 100, max features 0.8 | 55% |

두 모델의 클래스 `1` 확률을 각각 계산한 뒤 가중 평균한다. 최종 확률은 필요에 따라 아래
선형 보정을 적용하고 `[0, 1]` 범위로 제한한다.

\[
p_{final}=clip(a(0.45p_{HistGB}+0.55p_{ExtraTrees})+b,0,1)
\]

| 후보 | 기울기 `a` | 절편 `b` |
|---|---:|---:|
| 원본 | 1.00000000 | 0.00000000 |
| 절반 보정 | 1.06535491 | -0.03681724 |
| 완전 보정 | 1.13070982 | -0.07363449 |

### 7.3 성능 현황

| 평가 | 모델/후보 | 점수 | Brier |
|---|---|---:|---:|
| 공식 리더보드 | 기존 제출 | 549.51193 | 정답 평균 비공개로 역산 불가 |
| 공식 리더보드 | 당시 1위 | 1,176.54904 | 정답 평균 비공개로 역산 불가 |
| 2024 forward validation | 기존 HistGB | 576.61745 | 0.24836650 |
| 2024 forward validation | 최적 앙상블 원본 | 700.25764 | 0.24805763 |
| 2024 forward validation | 절반 확률 보정 후보 | 731.63033 | 0.24797926 |
| 2024 forward validation | 완전 확률 보정 후보 | 742.08790 | 0.24795314 |

원본 앙상블의 2024 검증 예측 평균은 `0.49503365`, 실제 평균은 `0.48610492`였다. 학습은
약 134.73초, 253,507행 추론은 약 0.90초가 걸렸다. 2019~2024년 전체 최종 학습은 약
221.49초였으며, 기본 모델 파일은 약 51MB다. 제출 ZIP은 약 50~52MB이고 로컬 5행 격리
실행은 약 0.8초였다.

로컬 점수와 공식 점수는 평가 시즌과 정답 평균이 다르므로 동일한 숫자로 간주하지 않는다.
새 모델의 공식 점수는 실제 ZIP을 제출한 뒤에만 확정된다. 제출 후보는
`artifacts/submissions/` 아래에 원본, 절반 보정, 완전 보정 순으로 구분되어 있다.

## 8. 재현 및 제출 명령

최종 원본 모델을 다시 학습하고 2024 OOF 산출물을 생성한다.

```bash
.venv/bin/python train_optimized.py --n-estimators 160 --hist-weight 0.45
```

이 명령은 보정하지 않은 `model/final_model.pkl`을 생성한다. 현재 mild와 calibrated 파일은
같은 최종 모델의 `calibration_slope`, `calibration_intercept`만 위 표의 값으로 변경해 별도
저장한 후보이며, `train_optimized.py`가 자동으로 다시 생성하지는 않는다.

세 가지 제출 ZIP은 다음과 같이 만든다. `--model-path`로 선택한 모델도 ZIP 안에서는 평가
코드가 요구하는 `model/final_model.pkl` 이름으로 저장된다.

```bash
mkdir -p artifacts/submissions

.venv/bin/python build_submit.py \
  --output artifacts/submissions/submit_raw.zip \
  --model-path model/final_model.pkl

.venv/bin/python build_submit.py \
  --output artifacts/submissions/submit_mild.zip \
  --model-path model/final_model_mild.pkl

.venv/bin/python build_submit.py \
  --output artifacts/submissions/submit_calibrated.zip \
  --model-path model/final_model_calibrated.pkl
```

평가 서버의 `script.py`는 `data/` 또는 `open/`에서 `test.csv`와
`sample_submission.csv`를 찾고, 모델을 불러와 `output/submission.csv`를 생성한다. 제출
우선순위는 숨겨진 2025 분포에 대한 과보정 위험을 고려해 `raw → mild → calibrated`다.

최종 모델 학습에 기록된 환경은 Python 3.11.15, pandas 2.0.3, NumPy 1.26.4,
scikit-learn 1.8.0, joblib 1.5.3이다.
