from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_percentage_error

from features import FEATURES, TARGET, make_features
from preprocess import clean_cells, fill_missing, load_cells, scale_features, split_holdout

SEED = 42
TARGET_MAPE = 9.1  # 논문 목표 성능 (%)
ALPHAS = np.logspace(-2, 3, 100)  # alpha 탐색 범위

ROOT = Path(__file__).resolve().parents[1]
BATCH1_PATH = ROOT / "data/archive/2017-05-12_batchdata_updated_struct_errorcorrect.mat"
BATCH2_PATH = ROOT / "data/archive/2018-02-20_batchdata_updated_struct_errorcorrect.mat"

# 데이터 불러오기 및 피처 생성
batch1 = make_features(clean_cells(load_cells(BATCH1_PATH)), "Batch1")
batch2 = make_features(clean_cells(load_cells(BATCH2_PATH)), "Batch2")

# Batch 1을 Train / Valid(Hold-out)로 분리, Batch 2는 Test
train, valid = split_holdout(batch1, seed=SEED)
test = batch2
print(f"셀 수 - Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

# X, y 분리
X_train, y_train = train[FEATURES], train[TARGET]
X_valid, y_valid = valid[FEATURES], valid[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# 결측 처리 및 스케일링 (Train 기준)
X_train, X_valid, X_test = fill_missing(X_train, X_valid, X_test)
X_train, X_valid, X_test = scale_features(X_train, X_valid, X_test)

# 모델 정의: 5-fold 교차검증으로 최적 alpha를 자동 탐색
model = RidgeCV(
    alphas=ALPHAS,                                  # 탐색 범위
    cv=5,                                           # 5-fold 교차검증
    scoring="neg_mean_absolute_percentage_error",   # 평가지표(MAPE)
)

# 모델 학습 (Hold-out 셀은 CV에 포함하지 않는다)
model.fit(X_train, y_train)
best_alpha = model.alpha_
print(f"최적 Alpha: {best_alpha:.4f}")

# 예측값 생성
valid_pred = model.predict(X_valid)
test_pred = model.predict(X_test)

# 모델 평가
# Train MAPE는 학습 셀을 다시 예측한 값이 아니라 CV 평균이다.
train_mape = -model.best_score_ * 100
valid_mape = mean_absolute_percentage_error(y_valid, valid_pred) * 100
test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100

print(f"\n=== Ridge (alpha={best_alpha:.4f}) ===")
print(f"Train MAPE : {train_mape:.3f}%")
print(f"Valid MAPE : {valid_mape:.3f}%")
print(f"Test  MAPE : {test_mape:.3f}%")

# 성능 결과 저장 (Gap은 양수일수록 뒤 단계에서 성능이 나빠짐)
performance = pd.DataFrame([
    ["Train (Batch 1 CV)", train_mape, "5-fold CV 평균"],
    ["Valid (Batch 1 Hold-out)", valid_mape, ""],
    ["Test (Batch 2)", test_mape, ""],
    ["Gap (Train-Valid)", valid_mape - train_mape, "(+) : 과적합 의심"],
    ["Gap (Valid-Test)", test_mape - valid_mape, "(+) : 배치간 일반화 저하 의심"],
    ["Gap (Target-Test)", test_mape - TARGET_MAPE, "Target : 원논문 9.1%"],
], columns=["구분", "MAPE (%)", "비고"]).round(3)
print()
print(performance.to_string(index=False))

performance.to_csv(ROOT / "results/model_performance.csv", index=False, encoding="utf-8-sig")

# 오류 분석용: Batch 2에서 오차가 큰 셀
errors = test[["cell_id", TARGET]].copy()
errors["prediction"] = test_pred.round(1)
errors["error_%"] = (abs(errors[TARGET] - test_pred) / errors[TARGET] * 100).round(1)
print("\n[Batch 2 오차 상위 5개 셀]")
print(errors.sort_values("error_%", ascending=False).head(5).to_string(index=False))
