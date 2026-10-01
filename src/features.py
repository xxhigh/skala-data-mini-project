"""보고서의 초기 100사이클 피처 네 개를 셀 단위로 추출한다."""

import numpy as np
import pandas as pd

if __package__:
    from .preprocess import iter_cells
else:
    from preprocess import iter_cells


EARLY_CYCLES = 100
SUMMARY_FIELDS = {"mean_chargetime": "chargetime", "mean_Tavg": "Tavg"}
FEATURES = ["mean_chargetime", "mean_Tavg", "log_delta_q_var", "delta_q_min"]
# 원자료 Vdlin의 2.0~3.5 V 범위와 1,000개 지점을 공통 격자로 사용한다.
# 고정된 물리 격자이므로 평가 배치에서 격자나 통계량을 학습하지 않는다.
VOLTAGE_GRID = np.linspace(2.0, 3.5, 1000)


def extract_delta_q(voltage, q10, q100):
    """공통 전압 격자에서 Q100-Q10의 최소값과 log10 모집단 분산을 계산한다.

    결측·무한대는 대응하는 전압/용량 쌍과 함께 제외한다. 정렬 후 공통 격자로
    선형 보간하되 외삽하지 않는다. 곡선을 사용할 수 없으면 두 피처를 결측으로,
    분산이 0이면 로그 피처만 결측으로 두어 학습 fold 중앙값으로 대체한다.
    """
    missing = {"delta_q_min": np.nan, "log_delta_q_var": np.nan}
    if q10 is None or q100 is None:
        return missing, "missing_q10_or_q100"
    voltage, q10, q100 = (
        np.asarray(values, dtype=float).reshape(-1)
        for values in (voltage, q10, q100)
    )
    if not (voltage.size == q10.size == q100.size):
        return missing, "voltage_capacity_length_mismatch"
    valid = np.isfinite(voltage) & np.isfinite(q10) & np.isfinite(q100)
    order = np.argsort(voltage[valid])
    v, first, last = voltage[valid][order], q10[valid][order], q100[valid][order]
    if v.size < 2:
        return missing, "insufficient_voltage_points"
    if np.any(np.diff(v) <= 0):
        return missing, "duplicate_voltage_points"
    if v[0] > VOLTAGE_GRID[0] or v[-1] < VOLTAGE_GRID[-1]:
        return missing, "incomplete_voltage_range"
    delta = np.interp(VOLTAGE_GRID, v, last) - np.interp(VOLTAGE_GRID, v, first)
    variance = float(np.var(delta, ddof=0))
    result = {
        "delta_q_min": float(delta.min()),
        "log_delta_q_var": float(np.log10(variance)) if variance > 0 else np.nan,
    }
    return result, "zero_delta_q_variance" if variance == 0 else None


def extract_record(summary, life, cell_id, early_cycles=EARLY_CYCLES):
    """100사이클 이후 관측값과 타깃은 피처에 포함하지 않는다."""
    if early_cycles != EARLY_CYCLES:
        raise ValueError("보고서의 피처 정의는 초기 100사이클을 사용합니다.")
    life = np.asarray(life, dtype=float).reshape(-1)
    if life.size != 1 or not np.isfinite(life[0]) or life[0] <= early_cycles:
        return None, "invalid_or_insufficient_cycle_life"
    cycles = np.asarray(summary["cycle"], dtype=float).reshape(-1)
    early = (cycles >= 1) & (cycles <= early_cycles)
    if not np.all(np.isin(np.arange(1, early_cycles + 1), cycles)):
        return None, "incomplete_early_cycles"
    if np.unique(cycles[early]).size != early.sum():
        raise ValueError(f"Cell {cell_id}: 초기 사이클 번호가 중복되었습니다.")

    record = {"cell_id": cell_id, "cycle_life": float(life[0])}
    for feature, field in SUMMARY_FIELDS.items():
        values = np.asarray(summary[field], dtype=float).reshape(-1)
        if values.size != cycles.size:
            raise ValueError(f"Cell {cell_id}: {field}와 cycle의 길이가 다릅니다.")
        values = values[early]
        values = values[np.isfinite(values)]
        record[feature] = float(values.mean()) if values.size else np.nan
    return record, None


def load_features(path, early_cycles=EARLY_CYCLES, batch_id=None):
    """식별자·피처·타깃 표, 제외 셀과 곡선 품질 내역을 반환한다.

    짧은 수명 자체를 이상치로 제거하지 않는다. 제외 대상은 타깃이 없거나
    첫 100사이클을 관측할 수 없는 셀이다. 곡선 품질 문제는 결측 처리한다.
    """
    records, skipped, issues = [], [], []
    if batch_id is None:
        raise ValueError("batch_id를 지정하세요.")
    fields = {*SUMMARY_FIELDS.values(), "cycle"}
    for cell_id, summary, life, voltage, q10, q100 in iter_cells(path, fields):
        record, reason = extract_record(summary, life, cell_id, early_cycles)
        if record is None:
            skipped.append({"batch_id": batch_id, "cell_id": cell_id, "reason": reason})
            continue
        curve_features, issue = extract_delta_q(voltage, q10, q100)
        record.update(curve_features)
        record["batch_id"] = batch_id
        if issue:
            issues.append({"batch_id": batch_id, "cell_id": cell_id, "reason": issue})
        records.append(record)
    if not records:
        raise ValueError(f"{path}: 학습 가능한 셀이 없습니다. 제외 내역: {skipped}")
    columns = ["batch_id", "cell_id", *FEATURES, "cycle_life"]
    return pd.DataFrame(records, columns=columns), skipped, issues
