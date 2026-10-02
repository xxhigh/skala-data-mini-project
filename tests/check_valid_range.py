"""초기 100사이클의 충전시간·평균 온도 정상 범위를 그래프로 확인한다.

실행: python tests/check_valid_range.py
정상 범위(VALID_RANGE)는 Batch 1을 기준으로 정하고, Batch 2·3은 참고용으로 본다.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import VALID_RANGE, load_cells

BATCHES = {
    "Batch 1": "2017-05-12",
    "Batch 2": "2018-02-20",
    "Batch 3": "2018-04-12",
}
UNITS = {"chargetime": "min", "Tavg": "degC"}
NORMAL, ABNORMAL, BEFORE = "#2a78d6", "#d03b3b", "#898781"

# 데이터 불러오기 (수명값이 있는 셀만 사용)
data = {}
for name, date in BATCHES.items():
    path = ROOT / f"data/archive/{date}_batchdata_updated_struct_errorcorrect.mat"
    data[name] = [c for c in load_cells(path) if np.isfinite(c["cycle_life"])]


def is_normal(values, key):
    """정상 범위 안에 있는 값인지 표시한다."""
    low, high = VALID_RANGE[key]
    return (values > low) & (values <= high)


# 1. 값의 분포 요약: 정상값과 이상값 사이에 빈 구간이 있는지 확인
for key in VALID_RANGE:
    print(f"\n[{key}] 정상 범위: {VALID_RANGE[key][0]} 초과 ~ {VALID_RANGE[key][1]} 이하")
    for name, cells in data.items():
        values = np.concatenate([c[key] for c in cells])
        normal = is_normal(values, key)
        print(f"  {name}: 정상 {values[normal].min():.1f} ~ {values[normal].max():.1f}"
              f" | 이상값 {(~normal).sum()}개 / {values.size}개")
        # 3. 이상값이 어느 셀의 몇 번째 사이클에 있는지 확인 (0은 개수만 표시)
        print(f"     값이 0인 개수: {(values == 0).sum()}")
        for cell in cells:
            for cycle in np.where(~is_normal(cell[key], key) & (cell[key] != 0))[0]:
                print(f"     cell {cell['cell_id']:>2}, cycle {cycle + 1:>3}: {cell[key][cycle]:.1f}")

# 그래프 1. 히스토그램: 정상값 덩어리와 이상값 사이의 빈 구간 확인
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for row, key in enumerate(VALID_RANGE):
    for col, (name, cells) in enumerate(data.items()):
        ax = axes[row, col]
        values = np.concatenate([c[key] for c in cells])
        normal = is_normal(values, key)
        # 정상 범위의 경계선까지 보이도록 구간을 잡는다.
        low, high = VALID_RANGE[key]
        upper = max(values.max(), high) if np.isfinite(high) else values.max()
        bins = np.linspace(min(values.min(), low), upper, 60)
        ax.hist(values[normal], bins=bins, color=NORMAL, label="normal")
        ax.hist(values[~normal], bins=bins, color=ABNORMAL, label="abnormal")
        for limit in VALID_RANGE[key]:
            if np.isfinite(limit):
                ax.axvline(limit, color="black", linestyle="--", linewidth=1)
        ax.set_yscale("log")  # 개수가 적은 이상값도 보이도록 로그 축 사용
        ax.set_title(f"{name}: {key}")
        ax.set_xlabel(f"{key} ({UNITS[key]})")
        ax.set_ylabel("count (log)")
fig.suptitle("1. Histogram of early 100 cycles (dashed line = valid range)", fontweight="bold")
fig.legend(*ax.get_legend_handles_labels(), loc="upper right")
fig.tight_layout(rect=(0, 0, 1, 0.95))  # 범례 자리 확보

# 그래프 2. 사이클별 값: 이상값이 특정 사이클에 몰려 있는지 확인
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for row, key in enumerate(VALID_RANGE):
    for col, (name, cells) in enumerate(data.items()):
        ax = axes[row, col]
        cycles = np.tile(np.arange(1, 101), len(cells))
        values = np.concatenate([c[key] for c in cells])
        normal = is_normal(values, key)
        ax.scatter(cycles[normal], values[normal], s=6, color=NORMAL, alpha=0.4, label="normal")
        ax.scatter(cycles[~normal], values[~normal], s=60, color=ABNORMAL, marker="x", label="abnormal")
        if key == "chargetime" and (~normal).any():
            ax.set_yscale("symlog", linthresh=1)  # 0과 수천 분을 함께 보기 위한 축
        ax.set_title(f"{name}: {key}")
        ax.set_xlabel("cycle")
        ax.set_ylabel(f"{key} ({UNITS[key]})")
fig.suptitle("2. Value by cycle (where abnormal values occur)", fontweight="bold")
fig.legend(*ax.get_legend_handles_labels(), loc="upper right")
fig.tight_layout(rect=(0, 0, 1, 0.95))  # 범례 자리 확보

# 그래프 3. 처리 전후 비교: 이상값 제거가 셀별 평균 피처를 얼마나 바꾸는지 확인
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for row, key in enumerate(VALID_RANGE):
    for col, (name, cells) in enumerate(data.items()):
        ax = axes[row, col]
        cell_ids = [c["cell_id"] for c in cells]
        before = [c[key].mean() for c in cells]
        after = [c[key][is_normal(c[key], key)].mean() for c in cells]
        ax.scatter(cell_ids, before, s=40, color=BEFORE, marker="o", label="before cleaning")
        ax.scatter(cell_ids, after, s=40, color=NORMAL, marker="^", label="after cleaning")
        ax.set_title(f"{name}: mean_{key}")
        ax.set_xlabel("cell_id")
        ax.set_ylabel(f"mean {key} ({UNITS[key]})")
fig.suptitle("3. Cell mean before / after cleaning", fontweight="bold")
fig.legend(*ax.get_legend_handles_labels(), loc="upper right")
fig.tight_layout(rect=(0, 0, 1, 0.95))  # 범례 자리 확보

plt.show()
