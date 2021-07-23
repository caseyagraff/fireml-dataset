import numpy as np
import pandas as pd
import time

#WEATHER_TIME_STEPS = [-1, 0, 1]
WEATHER_TIME_STEPS = [0]
WEATHER_MEASUREMENTS = [
    "2_metre_temperature",
    "2_metre_relative_humidity",
    "10_metre_u_wind_component",
    "10_metre_v_wind_component",
    "precipitation_rate",
]
WEATHER_MEASUREMENTS_LONG_TERM = [
    "2_metre_temperature",
    "2_metre_relative_humidity",
    "precipitation_rate",
]
WEATHER_RESOLVE_RESOLUTION = 375

LONG_TERM_1000 = np.timedelta64(1000, "h").astype("timedelta64[ns]")
LONG_TERM_100 = np.timedelta64(100, "h").astype("timedelta64[ns]")


def get_weather_data(
    center_point,
    obs_datetime,
    meteorology,
    transform,
    time_offset,
    window_size,
    cell_size,
):
    num_raster_cells = window_size // cell_size

    weather = []
    for step in WEATHER_TIME_STEPS:
        center_datetime = obs_datetime + time_offset * step

        weather_rast = get_weather_for_time(
            center_point,
            center_datetime,
            meteorology,
            transform,
            window_size,
            cell_size,
        )

        weather.append(weather_rast)

    # Near-term aggregate
    #weather.append(
    #    get_aggregate_weather(
    #        center_point,
    #        center_datetime,
    #        meteorology,
    #        transform,
    #        window_size,
    #        cell_size,
    #        time_offset,
    #    )
    #)

    # Long-term: 1000 hour, 100 hour
    # weather.append(
    #    get_aggregate_weather(
    #        center_point,
    #        center_datetime - LONG_TERM_100 / 2,
    #        meteorology,
    #        transform,
    #        window_size,
    #        cell_size,
    #        LONG_TERM_100 / 2,
    #        daytime=True,
    #    )
    #)
    #weather.append(
    #    get_aggregate_weather(
    #        center_point,
    #        center_datetime - LONG_TERM_1000 / 2,
    #        meteorology,
    #        transform,
    #        window_size,
    #        cell_size,
    #        LONG_TERM_1000 / 2,
    #        daytime=True,
    #    )
    #)

    return np.stack(weather)


def get_weather_for_time(
    center_point, center_datetime, meteorology, transform, window_size, cell_size
):
    # Get date ind
    num_raster_cells = window_size // cell_size
    # num_raster_cells = 3

    weather_raster = np.full(
        (len(WEATHER_MEASUREMENTS), num_raster_cells, num_raster_cells),
        fill_value=np.nan,
    )

    try:
        weather = meteorology.sel(
            time=center_datetime, method="nearest", tolerance="6H"
        )
    except:
        return weather_raster

    for i, measurement in enumerate(WEATHER_MEASUREMENTS):
        weather_raster[i] = get_around(
            weather[measurement].values,
            transform,
            transform.a,
            *center_point,
            width=window_size,
            resolution=WEATHER_RESOLVE_RESOLUTION,
            subwindow=True
        )

    return weather_raster


def get_aggregate_weather(
    center_point,
    center_datetime,
    meteorology,
    transform,
    window_size,
    cell_size,
    time_offset,
    measurements=WEATHER_MEASUREMENTS,
    daytime=False,
):
    # Get date ind
    num_raster_cells = window_size // cell_size
    # num_raster_cells = 3

    weather_raster = np.full(
        (len(measurements), num_raster_cells, num_raster_cells), fill_value=np.nan
    )

    try:
        weather = meteorology.sel(
            time=slice(center_datetime - time_offset, center_datetime + time_offset)
        )
    except:
        return weather_raster

    if daytime:
        times = pd.to_datetime(weather.time.values)
        daytime = (times.hour > 18) & (times.hour <= 24)

        weather = weather.sel(time=daytime)

    for i, measurement in enumerate(measurements):
        aggregate_method = np.nanmean if "wind" not in measurement else np.nanmax

        weather_raster[i] = get_around(
            weather[measurement].values,
            transform,
            transform.a,
            *center_point,
            width=window_size,
            resolution=WEATHER_RESOLVE_RESOLUTION,
            subwindow=True,
            aggregate=aggregate_method
        )

    return weather_raster


def index(transform, x, y, op=np.floor):
    res = ~transform * (x, y)

    if op is not None:
        res = op(res)

    return res


def xy(transform, i, j, position="center"):
    if position == "center":
        off = 0.5
    elif position == "upper":
        off = 0.0
    else:
        off = 1
    return transform * (i + off, j + off)


def upsample(arr, upsample_rate):
    return arr.repeat(upsample_rate, axis=0).repeat(upsample_rate, axis=1)


def downsample(arr, downsample_rate):
    target_size = arr.shape[0] // downsample_rate
    return arr.reshape(target_size, downsample_rate, target_size, downsample_rate).mean(
        axis=(1, 3)
    )


def get_around(
    data,
    transform,
    pixel_width,
    x,
    y,
    width,
    resolution=375,
    subwindow=True,
    aggregate=None,
):
    if subwindow:
        outer_window_ind = width / pixel_width
        outer_window_off = (
            1
            if int(outer_window_ind) % 2 == 1
            or int(np.round(outer_window_ind)) % 2 == 1
            else 0
        )
        outer_window_ind = int(np.ceil(outer_window_ind / 2))

        col_out, row_out = index(transform, x, y)
        row_out, col_out = int(row_out), int(col_out)

        # TODO: What to do when this goes outside of data array
        if aggregate:
            data = aggregate(data, axis=0)

        data = data[
            row_out - outer_window_ind : row_out + outer_window_ind + outer_window_off,
            col_out - outer_window_ind : col_out + outer_window_ind + outer_window_off,
        ]

        #data = data[
        #row_out : row_out + 1,
        #col_out : col_out + 1
        #]

        shift_row = row_out - outer_window_ind
        shift_col = col_out - outer_window_ind
    else:
        shift_row, shift_col = 0, 0

    # print(data.shape)
    # return data

    upsample_rate = pixel_width / resolution
    data = upsample(data, int(upsample_rate))

    width_ind = width / transform.a
    width_ind *= upsample_rate

    width_ind = int(width_ind)
    off = width_ind % 2
    width_ind = width_ind // 2

    col, row = index(transform, x, y, op=None)
    row -= shift_row
    col -= shift_col
    row *= int(upsample_rate)
    col *= int(upsample_rate)

    row, col = int(row), int(col)

    return data[
        (row - width_ind) : (row + width_ind + off),
        (col - width_ind) : (col + width_ind + off),
    ]
