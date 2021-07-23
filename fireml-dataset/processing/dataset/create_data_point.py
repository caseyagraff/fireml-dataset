import numpy as np
import time
from affine import Affine

from . import fire_detections as fds, land_cover as lcov, meteorology as meteo


total_detection_time = 0
total_land_cover_time = 0
total_meteorology_time = 0

points_processed = 0


def create_data_point(
    point,
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
    method="exact",
    filter_data=None,
):
    """Get all data layers for a single data point (detections, land cover, and weather)."""
    global total_detection_time
    global total_land_cover_time
    global total_meteorology_time
    global points_processed

    # Get position and datetime for current point
    center_point, obs_datetime = get_position_and_datetime(
        point, search_table, filter_data
    )

    detection_time = time.time()
    observed, target, burned = fds.get_detection_data(
        center_point,
        obs_datetime,
        search_table,
        method,
        lags,
        time_window_lower,
        time_window_upper,
        time_offset,
        window_size,
        cell_size,
    )

    total_detection_time += time.time() - detection_time
    # print("Det: ", time.time() - detection_time)

    if land_cover is not None:
        land_cover_time = time.time()
        land_cover = lcov.get_land_cover_data(
            center_point,
            obs_datetime,
            land_cover,
            land_cover_meta,
            window_size,
            cell_size,
        )
        total_land_cover_time += time.time() - land_cover_time
        # print("Land: ", time.time() - land_cover_time)
    else:
        land_cover = np.zeros((1, 1, 1))

    if meteorology is not None:
        meteorology_time = time.time()
        transform = Affine.from_gdal(*meteorology.attrs["transform"])
        weather = meteo.get_weather_data(
            center_point,
            obs_datetime,
            meteorology,
            transform,
            time_offset,
            window_size,
            cell_size,
        )
        # print("Met: ", time.time() - meteorology_time)
        total_meteorology_time += time.time() - meteorology_time
    else:
        weather = np.zeros((1, 1, 1))

    points_processed += 1

    if points_processed % 1000 == 0:
        print(f"Processed: {points_processed}")

    return observed, target, burned, land_cover, weather


def get_position_and_datetime(point, search_table, filter_data):
    """Get the position and datetime for the current point."""
    conv_point = search_table[0][point]

    if filter_data is not None:
        p = filter_data.iloc[point]
        det_x, det_y = p.x, p.y
        det_acq_datetime = np.array([p.acq_datetime]).astype("datetime64[ns]")[0]
    else:
        det_x, det_y = search_table[2][conv_point]
        det_acq_datetime = search_table[1][conv_point].astype("datetime64[ns]")

    center_point = (det_x, det_y)

    return center_point, det_acq_datetime
