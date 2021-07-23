import numpy as np

import fireml.processing.rasterization.rasterize as rast
from . import select_window as sel_wind

TARGET_STEPS = [1, 2]
BURN_WINDOW = [np.timedelta64(24 * 365 * 2, "h").astype("timedelta64[ns]")]


def get_detection_data(
    center_point,
    obs_datetime,
    search_table,
    rasterize_method,
    lags,
    time_window_lower,
    time_window_upper,
    time_offset,
    window_size,
    num_raster_cells,
):
    # Get observed rasters
    observed = []
    for lag in lags:
        lag_time_offset = time_offset * lag
        center_datetime = obs_datetime - lag_time_offset

        obs_raster = get_detections_raster(
            center_point,
            center_datetime,
            search_table,
            rasterize_method,
            time_window_lower,
            time_window_upper,
            window_size,
            num_raster_cells,
            obs_datetime,
        )

        observed.append(obs_raster)

    observed = np.stack(observed)

    # Get y values
    target = []
    for step in TARGET_STEPS:
        step_time_offset = time_offset * step
        center_datetime = obs_datetime + step_time_offset
        target_raster = get_detections_raster(
            center_point,
            center_datetime,
            search_table,
            rasterize_method,
            time_window_lower,
            time_window_upper,
            window_size,
            num_raster_cells,
            obs_datetime,
        )
        target.append(target_raster)

    target = np.stack(target)

    # Get burned values
    burned = []
    for window in BURN_WINDOW:
        center_datetime = obs_datetime - (window / 2) - time_offset + time_window_upper
        target_raster = get_detections_raster(
            center_point,
            center_datetime,
            search_table,
            "date",
            window / 2,
            window / 2,
            window_size,
            num_raster_cells,
            obs_datetime,
        )
        burned.append(target_raster)

    burned = np.stack(burned)

    return observed, target, burned


def select_rasterize_method(method):
    if method == "exact":
        rasterize = rast.rasterize_exact, np.uint8
    elif method == "date":
        rasterize = rast.rasterize_date, np.uint16
    elif method == "circle":
        rasterize = rast.rasterize_overlap, np.float32
    elif method == "fractional":
        rasterize = rast.rasterize_fractional, np.float32
    else:
        raise ValueError(f'Invalid method: "{method}"')

    return rasterize


def get_detections_raster(
    center_point,
    center_datetime,
    search_table,
    rasterize_method,
    time_window_lower,
    time_window_upper,
    window_size,
    cell_size,
    reference_datetime,
):

    num_raster_cells = window_size // cell_size

    rasterize_detections, det_dtype = select_rasterize_method(rasterize_method)

    inds = sel_wind.select_spatiotemporal_window(
        search_table[1],
        search_table[2],
        center_datetime,
        time_window_lower,
        time_window_upper,
        center_point[0],
        center_point[1],
        window_size,
    )

    raster = np.zeros((num_raster_cells, num_raster_cells), dtype=det_dtype)

    if inds is not None:
        points = search_table[2][inds, :]
        times = search_table[1][inds]

        rasterize_detections(
            raster,
            points,
            times,
            cell_size,
            window_size,
            center_point[0],
            center_point[1],
            reference_datetime,
        )
        raster = raster[::-1]  # Flip image to make North the top side

    return raster
