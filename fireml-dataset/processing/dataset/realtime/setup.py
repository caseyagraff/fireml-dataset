import numpy as np


def extract_positions(fire_dets):
    """Extract x and y positions from fire detection DataFrame

    Args:
        fire_dets (GeoDataFrame): fire detections to extract

    Returns:
        np.array: x positions
        np.array: y positions
    """
    xs = fire_dets.geometry.apply(lambda x: x.x).values
    ys = fire_dets.geometry.apply(lambda x: x.y).values

    return xs, ys


def create_search_table(fire_dets, xs, ys):

    """Create sorted representation of detections to improve search speed

    Sort by time, then xs, then ys.

    Return:
     np.array: ordered inds
     np.array: corresponding times (always in sorted order)
     np.array: xs and ys (sorted, subject to time).
    """
    time = fire_dets.acq_datetime.values.astype(np.uint64)

    sort_inds = np.lexsort((ys, xs, time))

    return (
        np.argsort(sort_inds),
        time[sort_inds],
        np.hstack([xs[sort_inds, None], ys[sort_inds, None]]),
    )
