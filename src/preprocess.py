"""MAT 초기 사이클 로딩과 학습 데이터 기준 결측값 대체·표준화."""

import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def iter_cells(path, fields):
    """셀 ID, summary, 수명, 전압축, 10·100사이클 Qdlin을 반환한다.

    MATLAB v7.3에서는 필요한 summary와 두 Qdlin 곡선만 읽는다.
    이전 MAT 형식은 scipy의 simplify_cells로 읽는다.
    """
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            batch = handle["batch"]
            summary_refs = batch["summary"][()].reshape(-1)
            life_refs = batch["cycle_life"][()].reshape(-1)
            if summary_refs.size != life_refs.size:
                raise ValueError(f"{path}: summary와 cycle_life의 셀 수가 다릅니다.")
            for cell_id, (summary_ref, life_ref) in enumerate(zip(summary_refs, life_refs)):
                group = handle[summary_ref]
                summary = {field: group[field][()] for field in fields}
                cycles = handle[batch["cycles"][()].reshape(-1)[cell_id]]
                q_refs = cycles["Qdlin"][()].reshape(-1)
                curves = []
                for index in (9, 99):
                    ref = q_refs[index] if index < q_refs.size else None
                    curves.append(handle[ref][()] if ref else None)
                voltage_ref = batch["Vdlin"][()].reshape(-1)[cell_id]
                yield (
                    cell_id, summary, handle[life_ref][()],
                    handle[voltage_ref][()], *curves,
                )
    else:
        batch = loadmat(path, simplify_cells=True)["batch"]
        cells = [batch] if isinstance(batch, dict) else np.asarray(batch, dtype=object).ravel()
        for cell_id, cell in enumerate(cells):
            summary = {field: cell["summary"][field] for field in fields}
            cycles = cell["cycles"]
            if isinstance(cycles, dict):
                raw = cycles["Qdlin"]
                curves = [raw[i] if i < len(raw) else None for i in (9, 99)]
            else:
                cycles = np.asarray(cycles, dtype=object).reshape(-1)
                curves = [cycles[i]["Qdlin"] if i < len(cycles) else None for i in (9, 99)]
            yield cell_id, summary, cell["cycle_life"], cell["Vdlin"], *curves


def make_preprocessor():
    """아직 fit하지 않은 전처리 파이프라인을 반환한다.

    Ridge와 함께 교차검증 파이프라인에 넣어 각 학습 fold에서만 통계량을 계산한다.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
