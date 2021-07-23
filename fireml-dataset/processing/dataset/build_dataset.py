import numpy as np
import tqdm
import multiprocessing

from .create_data_point import create_data_point
from .land_cover import create_evt_to_lf_arr

USE_LAND_COVER = True
# USE_METEOROLOGY = True
USE_METEOROLOGY = False


def build_dataset(
    points,
    fire_dets,
    land_cover,
    land_cover_meta,
    meteorology,
    cell_size,
    window_size,
    time_window_lower,
    time_window_upper,
    time_offset,
    lags,
    method,
    num_workers=8,
    filter_data=None,
):
    """
    Build and update data structures used for building dataset, then build dataset.
    """
    xs: np.ndarray = []
    ys: np.ndarray = []

    # Extract x,y positions of geometry points
    xs, ys = extract_positions(fire_dets)
    search_table = create_search_table(fire_dets, xs, ys)

    if filter_data is not None:
        xs_, ys_ = extract_positions(filter_data)
        filter_data["x"] = xs_
        filter_data["y"] = ys_

    if USE_LAND_COVER:
        if "landfire_evt" in land_cover_meta:
            evt_to_lf_arr, num_evt_classes = create_evt_to_lf_arr(
                land_cover_meta["landfire_evt"][-1]
            )
            land_cover_meta["landfire_evt"] = land_cover_meta["landfire_evt"][:-1] + (
                (evt_to_lf_arr, num_evt_classes),
            )
    else:
        land_cover, land_cover_meta = None, None

    meteorology = meteorology if USE_METEOROLOGY else None

    # pool = multiprocessing.Pool(num_workers)
    res = map(
        # pool.starmap(
        lambda x: create_data_point(*x),
        # create_data_point,
        [
            (
                p,
                search_table,
                land_cover,
                land_cover_meta,
                meteorology,
                cell_size,
                window_size,
                time_window_lower,
                time_window_upper,
                time_offset,
                lags,
                method,
                filter_data,
            )
            for p in points
        ],
    )

    xs, ys, burned, land_cover, meteorology = (np.array(x) for x in zip(*res))

    return {
        "observed": xs,
        "target": ys,
        "burned": burned,
        "land_cover": land_cover,
        "meteorology": meteorology,
    }


def create_search_table(fire_dets, xs, ys):
    """
    Create sorted representation of detections to improve search speed

    Sort by time, then xs, then ys.

    Return a) ordered inds, b) corresponding times (always in sorted order), c) xs and ys (sorted, subject to time).
    """
    time = fire_dets.acq_datetime.values.astype(np.uint64)

    sort_inds = np.lexsort((ys, xs, time))

    return (
        np.argsort(sort_inds),
        time[sort_inds],
        np.hstack([xs[sort_inds, None], ys[sort_inds, None]]),
    )


def extract_positions(fire_dets):
    xs = fire_dets.geometry.apply(lambda x: x.x).values
    ys = fire_dets.geometry.apply(lambda x: x.y).values

    return xs, ys
