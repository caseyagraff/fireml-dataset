import numpy as np
import time

LANDFIRE_CELL_SIZE = 30  # meters

# import fireml.processing.pooling as pool

# Two types: classes -> one hot, real-valued -> leave as is


def get_land_cover_data(
    center_point, obs_datetime, land_cover, land_cover_meta, window_size, cell_size
):
    raster_window_width = window_size // LANDFIRE_CELL_SIZE
    num_raster_cells = window_size // cell_size

    rasters = []
    for k, (ind, transform, metadata) in land_cover_meta.items():
        start = time.time()
        raster = extract_raster(
            center_point, land_cover[ind], transform, raster_window_width
        )
        # print("extract:", time.time() - start)

        start_b = time.time()
        if k == "landfire_evt":
            # raster[raster == -32768] = 3000
            # raster[raster == -9999] = 3000

            raster = convert_evt_classes(raster, metadata[0])
            raster = one_hot_encode(raster, metadata[1])
            # raster = pool.avg_pool_3d(raster, num_raster_cells, dtype=np.float32)
            raster = np.rollaxis(raster, 2, 0)
        else:
            raster = raster.astype(np.float32)
            raster[raster == -32768] = np.nan
            raster[raster == -9999] = np.nan
            # raster = pool.avg_pool_2d(raster, num_raster_cells, dtype=np.float32)
            raster = raster[None, :]
        # print("body:", k, time.time() - start_b)

        rasters.append(raster)

    start_c = time.time()
    stack = np.vstack(rasters)
    # print("stack", time.time() - start_c)

    return stack


def extract_raster(center_point, vals, transform, width):
    raster = np.full((width, width), fill_value=-32768, dtype=np.int16)

    col, row = ~transform * center_point
    col, row = int(col), int(row)

    offset = width // 2
    round_off = width % 2

    top = row - offset
    bottom = row + offset
    left = col - offset
    right = col + offset

    # Compute bounded indices
    top_b = max(top, 0)
    bottom_b = min(bottom, vals.shape[0] - 1)
    left_b = max(left, 0)
    right_b = min(right, vals.shape[1] - 1)

    # If bounded and unbounded indices deviate, only assign to subsection of extracted raster
    a = top_b - top
    b = width - (bottom - bottom_b)
    c = left_b - left
    d = width - (right - right_b)

    raster[a:b, c:d] = vals[top_b : bottom_b + round_off, left_b : right_b + round_off]

    return raster


def convert_evt_classes(arr, evt_to_class):
    return evt_to_class[arr - 3000]


def one_hot_encode(arr, num_classes):
    res = np.zeros((arr.size, num_classes), dtype=np.int16)

    res[np.arange(arr.size), arr.flatten()] = 1

    return res.reshape((arr.shape[0], arr.shape[1], num_classes))


def create_evt_to_lf_arr(evt_metadata):
    evt_to_lf = {
        k: evt_metadata[evt_metadata.VALUE == k].EVT_LF.values[0]
        for k in evt_metadata.VALUE.values
        if k != -9999
    }
    evt_to_lf[3000] = "Nodata - Land"
    classes = list(evt_metadata.EVT_LF.unique()) + ["Nodata - Land"]
    classes.remove("Nodata")
    lf_to_int = {k: i for i, k in enumerate(classes)}

    evt_to_lf_arr = np.zeros(max(evt_to_lf.keys()) - 3000 + 1, dtype=np.int16)

    for k, v in evt_to_lf.items():
        evt_to_lf_arr[k - 3000] = lf_to_int[v]

    return evt_to_lf_arr, len(np.unique(evt_to_lf_arr))
