"""초기 사이클 피처로 Ridge 배터리 수명 모델을 학습한다.

실행: python src/train.py
의존성: numpy pandas scipy h5py scikit-learn joblib

docs/report.md의 충전시간·평균 온도·ΔQ 피처 네 개를 표준화해 사용한다.
alpha 선택과 전처리는 Batch 1 내부에서 수행하며 Batch 2는 최종 평가에 쓴다.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline


if __package__:
    from .features import EARLY_CYCLES, FEATURES, VOLTAGE_GRID, load_features
    from .preprocess import make_preprocessor
else:
    from features import EARLY_CYCLES, FEATURES, VOLTAGE_GRID, load_features
    from preprocess import make_preprocessor


ROOT = Path(__file__).resolve().parents[1]


def evaluate(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": float(mean_absolute_percentage_error(actual, predicted) * 100),
        "r2": float(r2_score(actual, predicted)),
    }


def calculate_vif(values):
    """표준화한 학습 행렬에서 각 피처를 나머지 피처로 회귀해 VIF를 구한다."""
    results = []
    for index, feature in enumerate(FEATURES):
        target = values[:, index]
        others = np.delete(values, index, axis=1)
        design = np.column_stack([np.ones(len(values)), others])
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        total = np.sum((target - target.mean()) ** 2)
        residual = np.sum((target - design @ coefficients) ** 2)
        vif = float(total / residual) if total > 0 and residual > total * 1e-12 else np.inf
        results.append({"feature": feature, "vif": vif})
    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-mat", type=Path,
        default=ROOT / "data/archive/2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    )
    parser.add_argument(
        "--test-mat", type=Path,
        default=ROOT / "data/archive/2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/ridge")
    parser.add_argument("--train-batch-id", default="Batch1")
    parser.add_argument("--test-batch-id", default="Batch2")
    parser.add_argument("--early-cycles", type=int, choices=[EARLY_CYCLES], default=EARLY_CYCLES)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.early_cycles < 2 or args.cv_folds < 2:
        parser.error("early-cycles와 cv-folds는 2 이상이어야 합니다.")
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in args.alphas):
        parser.error("alphas는 양의 유한한 값이어야 합니다.")
    for path in (args.train_mat, args.test_mat):
        if not path.is_file():
            parser.error(f"MAT 파일이 없습니다: {path}")
    if args.train_mat.resolve() == args.test_mat.resolve():
        parser.error("학습과 평가에 서로 다른 MAT 파일을 지정하세요.")
    if args.train_batch_id == args.test_batch_id:
        parser.error("학습과 평가 배치의 식별자는 달라야 합니다.")
    return args


def main():
    args = parse_args()
    train, train_skipped, train_issues = load_features(
        args.train_mat, args.early_cycles, args.train_batch_id,
    )
    test, test_skipped, test_issues = load_features(
        args.test_mat, args.early_cycles, args.test_batch_id,
    )
    if len(train) < args.cv_folds or len(test) < 2:
        raise ValueError("교차검증 또는 평가에 사용할 셀 수가 부족합니다.")
    x_train, y_train = train[FEATURES], train["cycle_life"]
    x_test, y_test = test[FEATURES], test["cycle_life"]
    # CV의 각 학습 fold에서도 피처 전체가 결측이면 명확히 실패하도록 검사.
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    for train_indices, _ in cv.split(x_train):
        fold = x_train.iloc[train_indices]
        missing = fold.columns[fold.isna().all()]
        if len(missing):
            raise ValueError(f"학습 fold에서 모두 결측인 피처: {list(missing)}")
    pipeline = Pipeline([
        *make_preprocessor().steps,
        ("ridge", Ridge()),
    ])
    search = GridSearchCV(
        pipeline, {"ridge__alpha": args.alphas}, cv=cv,
        scoring="neg_mean_squared_error", error_score="raise",
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    metrics = {
        "train": evaluate(y_train, train_pred),
        "test": evaluate(y_test, test_pred),
        "selected_alpha_cv_rmse": float(np.sqrt(-search.best_score_)),
        "best_alpha": float(search.best_params_["ridge__alpha"]),
        "train_cells": len(train), "test_cells": len(test),
        "features": FEATURES, "early_cycles": args.early_cycles,
        "train_mat": str(args.train_mat.resolve()),
        "test_mat": str(args.test_mat.resolve()),
        "cv_folds": args.cv_folds, "seed": args.seed,
        "skipped_train": train_skipped, "skipped_test": test_skipped,
        "train_feature_issues": train_issues, "test_feature_issues": test_issues,
        "train_batch_id": args.train_batch_id, "test_batch_id": args.test_batch_id,
        "scaling": "StandardScaler", "target": "total_cycle_life",
        "delta_q_variance_ddof": 0,
        "zero_variance_policy": "missing_log_feature_then_train_median_imputation",
        "voltage_grid": {
            "min": float(VOLTAGE_GRID[0]), "max": float(VOLTAGE_GRID[-1]),
            "points": len(VOLTAGE_GRID), "interpolation": "linear_no_extrapolation",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "ridge_pipeline.joblib")
    train.to_csv(args.output_dir / "train_features.csv", index=False)
    test.to_csv(args.output_dir / "test_features.csv", index=False)
    predictions = test[["batch_id", "cell_id", "cycle_life"]].copy()
    predictions["prediction"] = test_pred
    predictions["residual"] = y_test.to_numpy() - test_pred
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(search.cv_results_).to_csv(args.output_dir / "cv_results.csv", index=False)
    pd.DataFrame({
        "feature": FEATURES, "standardized_coefficient": model["ridge"].coef_,
    }).to_csv(args.output_dir / "coefficients.csv", index=False)
    standardized = model[:-1].transform(x_train)
    calculate_vif(standardized).to_csv(args.output_dir / "vif.csv", index=False)
    metrics["intercept"] = float(model["ridge"].intercept_)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"저장 위치: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
