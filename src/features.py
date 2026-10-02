import numpy as np
import pandas as pd

# 입력 피처(X)와 타깃(y)
FEATURES = ["mean_chargetime", "mean_Tavg", "log_delta_q_var", "delta_q_min"]
TARGET = "cycle_life"


def make_delta_q_features(q10, q100):
    """ΔQ(V) = Q100(V) - Q10(V)의 로그 분산과 최솟값을 계산한다."""
    delta_q = q100 - q10
    variance = np.var(delta_q)
    # 분산이 0이면 로그를 계산할 수 없으므로 결측으로 둔다.
    log_delta_q_var = np.log10(variance) if variance > 0 else np.nan
    return log_delta_q_var, delta_q.min()


def make_features(cells, batch_id):
    """셀 하나를 한 행으로 하는 피처 테이블을 만든다."""
    rows = []
    for cell in cells:
        log_delta_q_var, delta_q_min = make_delta_q_features(cell["q10"], cell["q100"])
        rows.append({
            "batch_id": batch_id,
            "cell_id": cell["cell_id"],
            # 초기 100사이클의 평균 충전시간·평균 온도 (결측 제외)
            "mean_chargetime": np.nanmean(cell["chargetime"]),
            "mean_Tavg": np.nanmean(cell["Tavg"]),
            "log_delta_q_var": log_delta_q_var,
            "delta_q_min": delta_q_min,
            "cycle_life": cell["cycle_life"],
        })
    return pd.DataFrame(rows)
