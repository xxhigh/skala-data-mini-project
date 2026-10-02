"""데이터 로딩과 전처리 함수 모음."""

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EARLY_CYCLES = 100  # 입력으로 사용하는 초기 사이클 수

# 정상 범위 (초과, 이하). 근거는 tests/check_valid_range.py 로 확인한다.
# - chargetime: 0은 미측정 값. 가장 느린 정책(3.6C)도 완충에 20분이 안 걸린다.
#   Batch 1 정상값은 8.8~13.5분이고 그다음 값은 약 419분이다.
# - Tavg: 0은 미측정 값. Batch 1 정상값은 29.8~34.9도이며 상한에 걸리는 값은 없다.
VALID_RANGE = {
    "chargetime": (0, 20),
    "Tavg": (0, np.inf),
}


def load_cells(path):
    """MAT 파일에서 셀별 수명, 초기 100사이클 요약값, Q10·Q100 곡선을 읽는다."""
    cells = []
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        for i in range(batch["summary"].shape[0]):
            summary = f[batch["summary"][i, 0]]
            qdlin = f[batch["cycles"][i, 0]]["Qdlin"]
            cells.append({
                "cell_id": i,
                "cycle_life": float(f[batch["cycle_life"][i, 0]][()].ravel()[0]),
                "chargetime": summary["chargetime"][()].ravel()[:EARLY_CYCLES],
                "Tavg": summary["Tavg"][()].ravel()[:EARLY_CYCLES],
                # 계획서 정의: Q10 = cycles[9], Q100 = cycles[99]
                # Batch 1은 인덱스 0이 빈 사이클이라 실제 9·99번째 사이클을 쓴다.
                # 두 곡선의 간격은 다른 배치와 같은 90사이클이다.
                # 한 칸 밀면 인덱스 10의 노이즈 곡선(셀 41 등)을 집게 되어 그대로 둔다.
                "q10": f[qdlin[9, 0]][()].ravel(),
                "q100": f[qdlin[99, 0]][()].ravel(),
            })
    return cells


def clean_cells(cells):
    """수명값이 없는 셀을 제외하고, 비정상 측정값을 결측(NaN)으로 바꾼다."""
    cleaned = []
    for cell in cells:
        # 수명(타깃)이 없는 셀 제외. 수명이 짧다는 이유로는 제외하지 않는다.
        if not np.isfinite(cell["cycle_life"]):
            continue
        cell = dict(cell)
        for key, (low, high) in VALID_RANGE.items():
            values = cell[key].astype(float)
            values[(values <= low) | (values > high)] = np.nan
            cell[key] = values
        cleaned.append(cell)
    return cleaned


def split_holdout(df, seed=42):
    """Batch 1을 셀 단위로 Train / Hold-out(Valid) [8:2]으로 나눈다."""
    return train_test_split(df, test_size=0.2, random_state=seed)


def fill_missing(X_train, *others):
    """Train 기준 중앙값으로 결측값을 대체한다."""
    median_values = X_train.median()
    return [X.fillna(median_values) for X in (X_train, *others)]


def scale_features(X_train, *others):
    """Train 기준으로 표준화(StandardScaler)한다."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return [scaler.transform(X) for X in (X_train, *others)]
