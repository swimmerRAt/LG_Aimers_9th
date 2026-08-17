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
| `model/final_model.pkl` | 시간창 앙상블 + `game_type` 및 Rolling 전체 확률 보정 최종 모델 |
| `model/final_model_before_temporal_refinement.pkl` | 승격 전 단일 기간 HistGB+ExtraTrees 백업 |
| `model/final_model_mild.pkl` | 2024 OOF 선형 보정을 50% 적용한 후보 |
| `model/final_model_calibrated.pkl` | 2024 OOF 최적 선형 보정을 모두 적용한 후보 |
| `artifacts/optimized_ensemble_2024/` | 검증 지표, OOF 예측, 실행 요약 |
| `artifacts/submissions/submit_*.zip` | 원본·절반 보정·완전 보정 제출 파일 |
| `output/feature_importance.csv` | ExtraTrees 기준 전체 33개 피처 중요도와 순위 |
| `output/feature_importance.svg` | 중요도 상위 20개 가로 막대그래프 |

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
기간별 HistGB 45% + ExtraTrees 55% 모델 세 개
        ↓
전체·최근 3년·최근 2년 확률 앙상블
        ↓
game_type 축소 보정 및 Rolling 전체 확률 보정
        ↓
2025년 각 투구의 제구 성공 확률 예측
        ↓
output/submission.csv 생성
        ↓
Brier Score 및 정규화 대회 점수 산출
```

## 7. 현재 모델과 성능 현황 (2026-08-17)

현재 제출 모델은 `model/probability_refinement.py`의 `RefinedProbabilityClassifier`다.
내부의 `TemporalWindowEnsemble`이 기간별 `OptimizedBaseballEnsemble` 세 개를 결합하며,
각 구성요소 안에서 HistGB와 ExtraTrees가 동일한 전처리 행렬을 공유한다. 이후
`game_type`별 축소 로짓 절편과 전체 로짓 절편 보정을 차례대로 적용한다.

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
| 2024 forward validation | 시간창 앙상블 원본 | 701.28755 | 0.24805506 |
| 2024 forward validation | **현재 승격 모델 구조** | **723.33442** | **0.24799999** |
| 2024 forward validation | 절반 확률 보정 후보 | 731.63033 | 0.24797926 |
| 2024 forward validation | 완전 확률 보정 후보 | 742.08790 | 0.24795314 |

원본 앙상블의 2024 검증 예측 평균은 `0.49503365`, 실제 평균은 `0.48610492`였다. 학습은
약 134.73초, 253,507행 추론은 약 0.90초가 걸렸다. 2019~2024년 전체 최종 학습은 약
221.49초였으며, 기본 모델 파일은 약 51MB다. 제출 ZIP은 약 50~52MB이고 로컬 5행 격리
실행은 약 0.8초였다.

로컬 점수와 공식 점수는 평가 시즌과 정답 평균이 다르므로 동일한 숫자로 간주하지 않는다.
새 모델의 공식 점수는 실제 ZIP을 제출한 뒤에만 확정된다. 제출 후보는
`artifacts/submissions/` 아래에 원본, 절반 보정, 완전 보정 순으로 구분되어 있다.

### 7.4 Optuna 하이퍼파라미터 탐색

2024년 forward validation의 Brier Score를 목적함수로 HistGradientBoosting 20회,
ExtraTrees 10회, 앙상블 가중치와 선형 확률 보정 200회를 탐색했다. Optuna는 로컬 튜닝에만
사용하며 평가 서버의 제출 ZIP에는 포함하지 않는다.

| 구성 요소 | Optuna 최적값 |
|---|---|
| HistGradientBoosting | learning rate `0.04`, iterations `300`, leaf nodes `31`, min leaf `200`, L2 `5.0`, max bins `255` |
| ExtraTrees | trees `160`, max depth `21`, min leaf `185`, max features `0.7`, criterion `log_loss` |
| 보정 없는 앙상블 | HistGB `0.368`, ExtraTrees `0.632` |
| 2024 전용 완전 보정 | HistGB `0.37669218`, ExtraTrees `0.62330782`, slope `1.12054610`, intercept `-0.06608570` |

| 검증 시즌 | 후보 | 점수 | Brier | 평균 예측 확률 |
|---:|---|---:|---:|---:|
| 2022 | 현재 원본 | 2,291.69464 | 0.243454 | 0.529535 |
| 2022 | Optuna 원본 | **2,302.23385** | **0.243427** | 0.530064 |
| 2022 | Optuna 2024 완전 보정 | 2,277.74644 | 0.243488 | 0.527839 |
| 2023 | 현재 원본 | 0 | **0.254026** | 0.521528 |
| 2023 | Optuna 원본 | 0 | 0.254038 | 0.521933 |
| 2023 | Optuna 2024 완전 보정 | 0 | 0.254975 | 0.518725 |
| 2024 | 현재 원본 | 700.25764 | 0.24805763 | 0.495034 |
| 2024 | Optuna 원본 | **704.53533** | **0.24804695** | 0.495079 |
| 2024 | Optuna 2024 완전 보정 | **746.08757** | **0.24794315** | 0.488681 |

Optuna 원본은 2022년과 2024년에 개선됐지만 2023년에는 Brier가 `0.000012` 악화됐다.
반면 2024 정답에 직접 맞춘 완전 보정은 2022년과 2023년 모두 악화되어 숨겨진 2025년에
그대로 적용할 근거가 부족하다. 따라서 현재 결론은 **ExtraTrees 설정과 원본 앙상블
가중치는 신규 후보로 유지하고, 2024 완전 보정은 최종 모델에 자동 승격하지 않는다**이다.
Optuna만으로 얻은 2024 원본 점수 증가는 약 `+4.28점`이므로 목표 1,000점까지는 새로운
피처나 시간 드리프트 대응이 추가로 필요하다.

### 7.5 다중 시즌 robust Optuna 재설계

기존 탐색은 2024년 한 개 holdout을 반복 사용했기 때문에 선택 편향이 생길 수 있었다.
새 `optimize_hyperparameters_robust.py`는 파라미터 선택과 최종 검증을 다음처럼 분리한다.

| 역할 | 순방향 split | Optuna 접근 여부 |
|---|---|---|
| 내부 튜닝 | 2019~2020 → 2021 | 접근함 |
| 내부 튜닝 | 2019~2021 → 2022 | 접근함 |
| 내부 튜닝 | 2019~2022 → 2023 | 접근함 |
| 외부 최종 검증 | 2019~2023 → 2024 | 파라미터 동결 후 한 번만 접근 |

시즌마다 실제 성공률이 다르므로 단순 Brier 평균 대신 시즌별 상수확률 기준선으로
정규화한 손실을 사용한다.

\[
L_s=\frac{Brier_s}{r_s(1-r_s)}
\]

기본 robust 목적함수는 최근 시즌에 더 큰 비중을 주고 시즌 간 불안정성에 페널티를 준다.

\[
J=0.15L_{2021}+0.30L_{2022}+0.55L_{2023}
+0.25\,WeightedStd(L_{2021},L_{2022},L_{2023})
\]

HistGB와 ExtraTrees를 이 목적함수로 각각 탐색한 뒤, 동결된 두 모델의 내부 fold 예측으로
앙상블 가중치만 다시 최적화한다. 2024 정답에 직접 맞춘 선형 확률 보정은 사용하지 않는다.
ExtraTrees는 탐색과 최종 후보 평가에서 모두 160개 트리를 사용한다.

2024 outer 평가가 시작되면 `outer_lock.json`을 먼저 생성한다. 이후 같은 artifact
디렉터리에서는 trial을 추가하거나 파라미터를 다시 선택할 수 없다. 이미 생성된 outer
결과를 다시 요청해도 모델을 재학습하지 않고 저장된 결과만 반환한다.

후보는 다음 조건을 모두 만족해야 최종 전체 학습 대상으로 승인된다.

- 내부 robust objective가 현재 기준 모델보다 낮음
- 2024 Brier 개선량이 최소 `0.00002` 이상
- 기존 모델과의 paired Brier 차이 95% 신뢰구간 상한이 0보다 작음

### 7.6 Robust Optuna 내부 튜닝 결과 (2026-08-17)

2024 outer holdout을 열지 않은 상태에서 HistGB 20회, ExtraTrees 20회, 앙상블 비중
100회를 완료했다. 동결된 selection signature는 `827130b5ef3d866ec6d85ce1`이다.

| 구성 요소 | 동결된 최적값 |
|---|---|
| HistGradientBoosting | learning rate `0.02267387`, iterations `150`, leaf nodes `7`, min leaf `151`, L2 `0.10103797`, max bins `255` |
| ExtraTrees | trees `160`, max depth `11`, min leaf `388`, max features `0.4`, criterion `log_loss` |
| Robust 앙상블 | HistGB `0.99987375`, ExtraTrees `0.00012625` |

앙상블 탐색은 사실상 HistGB 단독을 선택했다. ExtraTrees 자체의 최적값은 기존보다 얕고
강하게 규제됐지만, 세 내부 시즌에서 HistGB의 오차를 안정적으로 상쇄하지 못했다.

| 검증 시즌 | 현재 기준 Brier | Robust Optuna Brier | Brier 변화 | 점수 변화 |
|---:|---:|---:|---:|---:|
| 2021 | 0.24593699 | **0.24575517** | -0.00018182 | +73.04 |
| 2022 | **0.24345354** | 0.24376492 | +0.00031138 | -124.97 |
| 2023 | 0.25402588 | **0.25298284** | -0.00104304 | 0점 절삭 유지 |

최근 시즌 비중이 가장 큰 2023년의 Brier가 크게 개선되어 전체 robust objective는
`1.00466492 → 1.00202448`로 `0.00264044` 감소했다. 반면 2022년은 악화됐으므로 모든
시즌에서 우월한 모델은 아니다. 다음 단계인 2024 one-shot outer 검증에서 실제 일반화와
paired 신뢰구간을 확인하기 전에는 최종 제출 모델로 승격하지 않는다.

2024 one-shot outer 검증 결과 후보는 최종 게이트를 통과하지 못했다.

| 2024 outer 후보 | 점수 | Brier | 평균 예측 확률 |
|---|---:|---:|---:|
| 현재 기준 모델 | **700.25764** | **0.24805763** | 0.49503365 |
| Robust Optuna | 494.85698 | 0.24857074 | 0.49690649 |

Robust 후보의 Brier는 기준 모델보다 `0.00051311` 악화됐고 점수는 `205.40066점` 낮았다.
paired Brier 차이의 95% 신뢰구간은 `[+0.00044383, +0.00058238]`로 전체가 악화 방향이다.
따라서 `outer_evaluation.json`의 상태는 `rejected_keep_baseline`이며, robust 후보를 최종
학습이나 제출 모델로 사용하지 않는다. 이 실험은 2024 결과를 확인했으므로 잠겨 있으며
동일 artifact 디렉터리에서 추가 trial이나 파라미터 변경을 하지 않는다.

### XGBoost 분류 모델 비교

Python 3.11 및 Apple Silicon과 호환되는 공식 `xgboost 3.2.0`을 사용해 현재 모델과 동일한
2024 forward holdout에서 비교했다. 현재 앙상블과 피처 차이로 인한 혼동을 줄이기 위해
동일한 33개 입력 피처와 동일한 학습·검증 행을 사용했다.

- 목적함수: `binary:logistic`
- tree method: CPU `hist`
- 범주형: `top_bottom`, `game_type`, `base_state`를 학습 구간 기준 OOV-safe 원-핫 인코딩
- 수치형: 학습 구간 중앙값 대치
- 클래스 가중치: 미사용
- early stopping: validation RMSE, patience 100. 이진 타깃에서는 `RMSE² = Brier`이므로
  Brier와 동일한 iteration을 선택
- 선택 설정: learning rate `0.03`, depth `6`, min child weight `100`, subsample `0.8`,
  column sample `0.8`, L1 `0.1`, L2 `15`, 최대 2,000 trees

깊이 4 기본 후보도 확인했으며 632.67736점이었다. 용량을 늘린 깊이 6 후보가 더 나았지만
현재 모델을 넘지는 못했다. 최종 비교 결과는 다음과 같다.

| 모델 | 선택 trees | Brier | 대회 환산 점수 | 현재 모델 대비 |
| --- | ---: | ---: | ---: | ---: |
| 현재 HistGB 45% + ExtraTrees 55% | - | **0.24805763** | **700.25764** | 기준 |
| XGBoost depth 6 | 212 | 0.24818884 | 647.73609 | -52.52155점 |
| 진단용 최적 블렌드 | - | 0.24805722 | 700.42557 | +0.16793점 |

XGBoost의 candidate-minus-baseline Brier 차이는 `+0.00013120`이고 paired 95% 신뢰구간은
`[+0.00008426, +0.00017815]`로 전체가 악화 방향이다. 2024에서 사후 계산한 진단용 최적
블렌드는 XGBoost 비중이 `5.3438%`에 불과하고 개선 폭도 0.17점뿐이다. 이는 별도 패키지와
모델을 제출물에 추가할 만큼 충분한 이득이 아니며 같은 2024 데이터에서 비중을 선택한
낙관 편향도 있다. 따라서 상태는 `rejected_keep_baseline`이고 `model/final_model.pkl`은
변경하지 않는다.

`xgboost`는 현재 공식 제출용 `requirements.txt`의 기본 패키지가 아니므로 이번 결과는 로컬
비교 후보로만 사용한다. 향후 성능상 채택할 이유가 생기더라도 먼저 평가 서버의 외부 패키지
설치 허용 여부를 확인해야 한다.

재현 명령과 결과 파일은 다음과 같다.

```bash
.venv/bin/pip install -r requirements-xgboost.txt
.venv/bin/python train_xgboost.py --skip-final-fit
```

- `artifacts/xgboost_2024/metrics.csv`
- `artifacts/xgboost_2024/run_summary.json`
- `artifacts/xgboost_2024/feature_importance.csv`
- `artifacts/xgboost_2024/feature_importance.svg`
- 대용량 `oof_predictions.csv`는 Git에서 제외

### 학습 기간을 달리한 시간창 앙상블 비교

동일한 HistGB 45% + ExtraTrees 55% 모델을 전체 기간, 최근 3년, 최근 2년의 세 방식으로
학습하는 `TemporalWindowEnsemble`을 구현했다. 각 검증 시즌에는 그보다 과거인 행만
사용했다. 초기 실험에서 시간가중 모델은 모든 개발 fold에서 열세였으므로 최종 비중 탐색과
학습에서 제외했다.

| 검증 시즌 | 전체 기간 | 최근 3년 | 최근 2년 |
| ---: | ---: | ---: | ---: |
| 2022 | 2291.69465 | 2291.69465 | 2280.82584 |
| 2023 | 0 | 0 | 0 |
| 2024 | **700.25764** | 674.67912 | 674.28650 |

비중은 5% 격자가 아니라 SLSQP 연속 최적화를 사용한다. 각 비중은 0 이상이고 합계는 정확히
1이 되도록 제한하며, 목적함수는 2022 40%·2023 60%의 정규화 Brier와 시즌 간 안정성
페널티를 합친 값이다. 2024를 열기 전에 선택한 바깥 검증용 비중은 다음과 같다.

| 구성요소 | 비중 |
| --- | ---: |
| 전체 기간 | 66.7342% |
| 최근 3년 | 12.8355% |
| 최근 2년 | 20.4303% |
| 시간가중 | 0% |

이 비중을 고정한 2024 outer 결과는 다음과 같다.

| 모델 | Brier | 대회 환산 점수 | 기존 대비 |
| --- | ---: | ---: | ---: |
| 기존 전체 기간 모델 | **0.24805763** | 700.25764 | 기준 |
| 연속 최적 시간창 앙상블 | 0.24805506 | 701.28755 | +1.02991점 |

Brier 개선은 `0.00000257`로 사전에 정한 최소 개선량 `0.00002000`의 약 12.9%에 불과하다.
paired 95% 신뢰구간도 `[-0.00001370, +0.00000855]`로 0을 포함해 개선이 통계적으로
확실하지 않다. 모델 크기와 추론량은 약 세 배가 되는데 이득은 약 1점뿐이다. 따라서 구현과
OOF 결과는 보존하지만 상태는
`rejected_keep_single_window`로 두고 `model/final_model.pkl`은 기존 전체 기간 모델로
유지한다. 이후 사용자의 명시적 결정에 따라 시간창 앙상블과 두 후단 보정을 결합한
`723.33442`점 후보가 별도 승격됐으며, 위 결론은 보정 전 시간창 앙상블만 비교했던 당시
결과를 설명한다.

2022~2024를 모두 사용한 2025용 진단 비중은 전체 76.0299%, 최근 3년 4.0732%, 최근 2년
19.8969%, 시간가중 0%였으며 같은 2024에서 사후 평가한 점수는 701.82899이다. 이 값은
2024 정답을 비중 선택에 사용했으므로 일반화 점수로 해석하지 않는다.

재현 명령과 결과 파일은 다음과 같다.

```bash
.venv/bin/python train_temporal_ensemble.py
```

- `artifacts/temporal_ensemble/component_metrics.csv`
- `artifacts/temporal_ensemble/blend_metrics.csv`
- `artifacts/temporal_ensemble/development_weight_optimization.csv`
- `artifacts/temporal_ensemble/final_weight_optimization.csv`
- `artifacts/temporal_ensemble/run_summary.json`
- 대용량 OOF와 구성요소 캐시는 Git에서 제외

### 시간창 앙상블 위의 확률 정제 실험

확률 정제는 단일 HistGB+ExtraTrees 모델로 되돌아가 적용하지 않았다. 바로 앞 절의
`TemporalWindowEnsemble`을 기준으로 전체 기간 66.7342%, 최근 3년 12.8355%, 최근 2년
20.4303%의 2024 outer용 비중을 고정한 뒤 다음 순서로 적용했다.

```text
전체·최근 3년·최근 2년 HistGB+ExtraTrees
        ↓ 기존 연속 최적 비중으로 결합
시간창 앙상블 확률
        ↓ 선택적으로 성공률 smoothing 피처 사용
game_type별 축소 로짓 절편 보정
        ↓
전체 확률 수준의 Rolling 로짓 절편 보정
```

성공률 smoothing은 입력 피처이므로 각 시간창 구성요소를 학습하기 전에 생성한다.
`asof_pitcher_n`, `asof_pitcher_success_rate`, `asof_batter_n`,
`asof_batter_success_rate`와 해당 학습 fold의 평균 성공률을 이용해 선수별 `λ=50, 200,
500, 1000` 피처를 추가한다. 검증 시즌의 정답이나 테스트 데이터 내부 통계는 사용하지
않는다.

후단 보정은 각 검증 시즌보다 이른 OOF만 사용한다. 오래된 시즌의 가중치는 시즌당 `0.6`
배로 감쇠한다. `game_type` 보정은 유형별 로짓 절편을 전체 절편 쪽으로 `100,000` 표본만큼
축소하고 10%만 적용한다. 전체 확률 보정도 로짓 절편의 25%만 적용해 시즌 드리프트에 대한
과도한 보정을 제한한다.

| 2024 forward 후보 | Brier | 대회 환산 점수 | 원본 시간창 대비 |
| --- | ---: | ---: | ---: |
| 원본 시간창 앙상블 | 0.24805506 | 701.28755 | 기준 |
| smoothing 시간창 앙상블 | 0.24808941 | 687.53616 | -13.75138점 |
| smoothing + game_type 보정 | 0.24808289 | 690.15169 | -11.13586점 |
| smoothing + game_type + Rolling 전체 보정 | 0.24805249 | 702.31513 | +1.02759점 |
| 원본 시간창 + game_type 보정 | 0.24804393 | 705.74630 | +4.45876점 |
| 원본 시간창 + game_type + Rolling 전체 보정 | **0.24799999** | **723.33442** | **+22.04687점** |

요청한 전체 순서인 `시간창 앙상블 → smoothing → game_type → Rolling 보정`은 원본보다
약 1.03점 높았지만 paired Brier 95% 신뢰구간이
`[-0.00004119, +0.00003606]`으로 0을 포함한다. smoothing 자체가 세 시간창 모두에서
악화됐고 이 손실을 후단 보정이 겨우 만회한 결과이므로 채택하지 않는다.

smoothing을 제외한 두 후단 보정은 약 22.05점 개선됐으며 paired Brier 95% 신뢰구간은
`[-0.00007272, -0.00003743]`으로 개선 방향이다. 다만 2024는 이전 모델 개발에서도 이미
여러 번 확인한 시즌이므로 완전히 새로운 one-shot outer 검증으로 볼 수 없다. 이 한계를
기록한 상태에서 사용자의 명시적 결정으로 해당 후보를 `model/final_model.pkl`에 승격했다.
승격 전 단일 기간 모델은 `model/final_model_before_temporal_refinement.pkl`로 보존한다.

2019~2024년 전체 최종 학습에는 약 287.72초가 걸렸고 모델 파일은 약 107MB, 제출 ZIP은
약 106MB다. 245,789행 반복 입력 스트레스 테스트에서 모델 로드·추론·출력 검증 핵심 시간은
약 3.38초, 전체 프로세스 실측은 약 8.07초, 최대 RSS는 약 1.46GB였다. 이는 평가 서버의
10분·28GB 제한 안에 충분히 들어온다.

재현 명령과 결과 파일은 다음과 같다.

```bash
.venv/bin/python train_temporal_ensemble.py \
  --artifact-dir artifacts/probability_refinement \
  --smoothing-lambdas 50 200 500 1000 \
  --fixed-development-weights 0.6673418251 0.1283550847 0.2043030902 \
  --fixed-final-weights 0.7602988964 0.0407317827 0.1989693209 \
  --skip-final-fit
.venv/bin/python train_probability_refinement.py --fit-final
```

- `artifacts/probability_refinement/blend_metrics.csv`
- `artifacts/probability_refinement/run_summary.json`
- `artifacts/probability_refinement/final_comparison/metrics.csv`
- `artifacts/probability_refinement/final_comparison/run_summary.json`
- 대용량 OOF와 학습 캐시는 Git에서 제외

## 8. 재현 및 제출 명령

기존 단일-2024 Optuna 탐색과 사후 다년 검증은 아래 명령으로 재현할 수 있지만, 신규 모델
선택에는 사용하지 않는다. 완료된 기본 모델 trial은
`artifacts/optuna_2024/optuna.db`에서 재개되며, 대용량 DB·OOF·검증 모델은 Git에서
제외된다.

```bash
.venv/bin/pip install -r requirements-tuning.txt
.venv/bin/python optimize_hyperparameters.py
.venv/bin/python validate_optuna_candidate.py
```

최적값과 전체 trial 기록은 `artifacts/optuna_2024/best_params.json`,
`histgb_trials.csv`, `extra_trees_trials.csv`, `ensemble_trials.csv`에 저장된다.

신규 robust 탐색은 먼저 `tune`만 실행한다. 기본 설정은 HistGB 20회, ExtraTrees 20회,
앙상블 가중치 100회이며 세 내부 fold를 모두 학습하므로 기존 탐색보다 오래 걸린다.

```bash
.venv/bin/python optimize_hyperparameters_robust.py tune \
  --artifact-dir artifacts/optuna_robust \
  --inner-seasons 2021 2022 2023 \
  --inner-weights 0.15 0.30 0.55 \
  --outer-season 2024 \
  --stability-penalty 0.25 \
  --hist-trials 20 \
  --extra-trials 20 \
  --blend-trials 100 \
  --n-estimators 160
```

`selection.json`과 `inner_fold_metrics.csv`를 검토하고 trial 수를 더 늘릴 필요가 있다면
**outer 평가 전에만** 같은 `tune` 명령의 총 trial 수를 높여 재개한다. 탐색을 완전히
종료한 뒤 다음 명령을 정확히 한 번 실행한다.

```bash
.venv/bin/python optimize_hyperparameters_robust.py evaluate-outer \
  --artifact-dir artifacts/optuna_robust \
  --min-brier-improvement 0.00002
```

결과는 `outer_evaluation.json`에 저장되고 이후 해당 디렉터리는 잠긴다. 새로운 가설이나
탐색 범위를 시험하려면 2024 결과에 맞춰 기존 실험을 수정하지 말고, 사전에 설정을 정한
새 artifact 디렉터리를 사용한다.

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
`sample_submission.csv`를 찾고, 모델을 불러와 `output/submission.csv`,
`output/feature_importance.csv`, `output/feature_importance.svg`를 생성한다. 중요도는 앙상블
전체의 인과적 기여도가 아니라 ExtraTrees 구성요소의 impurity importance다. 상관된 피처는
중요도를 나눠 가질 수 있으므로 중요도만 보고 가중치를 직접 조절하지 않고, 다음 단계에서
2024 검증 데이터의 permutation importance 및 피처 상관도와 함께 판단한다. 제출 우선순위는
숨겨진 2025 분포에 대한 과보정 위험을 고려해 `raw → mild → calibrated`다.

최종 모델 학습에 기록된 환경은 Python 3.11.15, pandas 2.0.3, NumPy 1.26.4,
scikit-learn 1.8.0, joblib 1.5.3이다.
