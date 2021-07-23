from typing import Dict, Tuple, Union
from affine import Affine
import numpy as np


def timedelta64_to_uint(td64: Union[np.timedelta64, np.ndarray]):
    return td64.astype("timedelta64[ns]").astype("uint64")


def affine_to_transform_dict(transform: Affine) -> Dict[str, float]:
    return {
        "a": transform.a,
        "b": transform.b,
        "c": transform.c,
        "d": transform.d,
        "e": transform.e,
        "f": transform.f,
        "a_i": (~transform).a,
        "b_i": (~transform).b,
        "c_i": (~transform).c,
        "d_i": (~transform).d,
        "e_i": (~transform).e,
        "f_i": (~transform).f,
    }


def create_evt_to_lf_arr(evt_metadata) -> Tuple[np.ndarray, int]:
    evt_to_lf = {
        k: evt_metadata[evt_metadata.VALUE == k].EVT_LF.values[0]
        for k in evt_metadata.VALUE.values
        if k != -9999
    }
    evt_to_lf[3000] = "Nodata - Land"
    classes = list(evt_metadata.EVT_LF.unique()) + ["Nodata - Land"]
    classes.remove("Nodata")
    lf_to_int = {k: i for i, k in enumerate(classes)}

    print(lf_to_int)

    evt_to_lf_arr = np.zeros(max(evt_to_lf.keys()) - 3000 + 1, dtype=np.int16)

    for k, v in evt_to_lf.items():
        evt_to_lf_arr[k - 3000] = lf_to_int[v]

    return evt_to_lf_arr, len(np.unique(evt_to_lf_arr))
