import os
import h5py
import pytz
from fireml.helpers.file_io import load_pickle
from typing import Optional, cast
import datetime as dt
from fireml.helpers.timing import timer
import pandas as pd
import geopandas as gpd
import numpy as np

from fireml.processing.dataset.realtime.settings import (
    DEFAULT_CONFIG,
    DEFAULT_FILTERS,
    DEFAULT_PATHS,
    FireDetectionsConfig,
    Positions,
    PreprocessData,
    PreprocessDataArtifacts,
    PreprocessDataConfig,
    PreprocessFilters,
    PreprocessPaths,
)


def filter_geodataframe(gdf: gpd.GeoDataFrame, filter: pd.Series) -> gpd.GeoDataFrame:
    return cast(gpd.GeoDataFrame, gdf[filter])


def load_data_points(path: str) -> np.ndarray:
    with h5py.File(path) as hf:
        points = cast(h5py.Dataset, hf["points"])
        return cast(np.ndarray, points[:])


def datetime64_to_datetime(dt64: np.datetime64) -> dt.datetime:
    ts = (dt64 - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
    return dt.datetime.fromtimestamp(cast(float, ts), tz=pytz.UTC)


@timer("Loaded data: ")
def load_data(paths: PreprocessPaths) -> PreprocessData:
    fire_detections: gpd.GeoDataFrame = load_pickle(
        os.path.join(paths["data_dir"], paths["fire_detections"])
    )

    filter_points_fn = paths["filter_points"]
    filter_points: Optional[gpd.GeoDataFrame] = (
        load_pickle(os.path.join(paths["data_dir"], filter_points_fn))
        if filter_points_fn
        else None
    )

    data_points_path = paths["data_points"]
    data_points = (
        load_data_points(os.path.join(paths["data_dir"], data_points_path))
        if data_points_path
        else None
    )

    return {
        "fire_detections": fire_detections,
        "data_points": data_points,
        "filter_points": filter_points,
    }


@timer("Filtered fire detections: ")
def filter_fire_detections(
    fire_detections: gpd.GeoDataFrame,
    filters: PreprocessFilters,
    fire_detections_config: FireDetectionsConfig,
) -> gpd.GeoDataFrame:

    max_lag = max(
        fire_detections_config["lags"] + fire_detections_config["aggregate_lags"]
    )
    lower_datetime = (
        filters["start_datetime"]
        - max_lag
        - fire_detections_config["detection_window_lower"]
    )

    max_forecast_offset = max(fire_detections_config["forecast_offsets"])
    upper_datetime = (
        filters["end_datetime"]
        + max_forecast_offset
        + fire_detections_config["forecast_window_upper"]
    )

    acq_datetime = fire_detections["acq_datetime"]
    datetime_sel = (acq_datetime >= datetime64_to_datetime(lower_datetime)) & (acq_datetime < datetime64_to_datetime(upper_datetime))  # type: ignore

    return filter_geodataframe(fire_detections, datetime_sel)


@timer("Filtered data: ")
def filter_data(
    data: PreprocessData, filters: PreprocessFilters, config: PreprocessDataConfig
) -> PreprocessData:

    fire_detections_filtered = filter_fire_detections(
        data["fire_detections"], filters, config["fire_detections"]
    )

    return {
        "fire_detections": fire_detections_filtered,
        "data_points": data["data_points"],
        "filter_points": data["filter_points"],
    }


def extract_positions_from_geodataframe(gdf: gpd.GeoDataFrame) -> Positions:
    xs = gdf.geometry.apply(lambda g: g.x).values
    ys = gdf.geometry.apply(lambda g: g.y).values

    return {"xs": xs, "ys": ys}


def extract_points_from_geodataframe(gdf: gpd.GeoDataFrame) -> np.ndarray:
    time = gdf["acq_datetime"].values.astype(np.uint64)
    positions = extract_positions_from_geodataframe(gdf)

    return np.stack([time, positions["xs"], positions["ys"]], axis=1)


@timer("Created data points: ")
def create_data_points(
    fire_detections: gpd.GeoDataFrame,
    fire_detection_config: FireDetectionsConfig,
    filters: PreprocessFilters,
    data_points: np.ndarray = None,
    filter_points: gpd.GeoDataFrame = None,
) -> np.ndarray:
    if data_points:
        print(f"Using points from existing data points: {len(data_points)}")

        return data_points

    elif filter_points is not None:
        return extract_points_from_geodataframe(filter_points)

    else:
        print(f"Creating points from all detections: {len(fire_detections)}")

        filtered_detections = fire_detections

        # Within region
        within_region_sel = filtered_detections["within_region"]
        filtered_detections = filter_geodataframe(
            filtered_detections, within_region_sel
        )

        # Start datetime
        start_datetime_filter = filtered_detections[
            "acq_datetime"
        ] >= datetime64_to_datetime(
            filters["start_datetime"]
        )  # type: ignore
        filtered_detections = filter_geodataframe(
            filtered_detections, start_datetime_filter
        )

        # End datetime
        end_datetime_filter = filtered_detections[
            "acq_datetime"
        ] < datetime64_to_datetime(
            filters["end_datetime"]
        )  # type: ignore
        filtered_detections = filter_geodataframe(
            filtered_detections, end_datetime_filter
        )

        # Time of day lower
        if fire_detection_config["time_of_day_lower"]:
            tod_lower_sel = (
                filtered_detections["acq_datetime"].dt.time
                >= fire_detection_config["time_of_day_lower"]
            )
            filtered_detections = filter_geodataframe(
                filtered_detections, tod_lower_sel
            )

        # Time of day upper
        if fire_detection_config["time_of_day_upper"]:
            tod_upper_sel = (
                filtered_detections["acq_datetime"].dt.time
                < fire_detection_config["time_of_day_upper"]
            )
            filtered_detections = filter_geodataframe(
                filtered_detections, tod_upper_sel
            )

        print(f"Remaining points: {len(filtered_detections)}")

        return extract_points_from_geodataframe(filtered_detections)


@timer("Created fire detections search table: ")
def create_fire_detections_search_table(
    fire_detections: gpd.GeoDataFrame, config: FireDetectionsConfig
) -> np.ndarray:
    points = extract_points_from_geodataframe(fire_detections)

    sort_inds = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))

    return points[sort_inds]


@timer("Created data artifacts: ")
def create_data_artifacts(
    data: PreprocessData, config: PreprocessDataConfig, filters: PreprocessFilters
) -> PreprocessDataArtifacts:
    data_points = create_data_points(
        data["fire_detections"],
        config["fire_detections"],
        filters,
        data_points=data["data_points"],
        filter_points=data["filter_points"],
    )

    fire_detections_search_table = create_fire_detections_search_table(
        data["fire_detections"], config["fire_detections"]
    )

    print("Points:", data_points.shape)
    print("Fire Detections Search Table:", fire_detections_search_table.shape)

    if config["preprocessing"]["apply_filter_to_data_points"]:
        raise NotImplementedError()

    return {
        "data_points": data_points,
        "fire_detections_search_table": fire_detections_search_table,
    }


def create_save_path(paths: PreprocessPaths, filters: PreprocessFilters):
    start_datetime = np.datetime_as_string(filters["start_datetime"], unit="D")
    end_datetime = np.datetime_as_string(filters["end_datetime"], unit="D")
    current_datetime = dt.datetime.today().strftime("%Y-%m-%dT%H%M%S")

    save_fn = paths["save"]
    save_fn = (
        save_fn
        if save_fn is not None
        else f"{start_datetime}_{end_datetime}_{current_datetime}.hdf5"
    )

    return os.path.join(paths["save_dir"], save_fn)


@timer("Saved data artifacts: ")
def save_data_artifacts(data_artifacts: PreprocessDataArtifacts, save_path: str):
    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("points", data=data_artifacts["data_points"])
        hf.create_dataset(
            "fire_detections_search_table",
            data=data_artifacts["fire_detections_search_table"],
        )


def preprocess(
    paths: PreprocessPaths, filters: PreprocessFilters, config: PreprocessDataConfig
):
    """Generate necessary artifacts for online data generation

    Including:
        data points array
        fire detections search table
    """

    data = load_data(paths)

    filtered_data = filter_data(data, filters, config)

    data_artifacts = create_data_artifacts(filtered_data, config, filters)

    # Create save path

    save_path = create_save_path(paths, filters)
    save_data_artifacts(data_artifacts, save_path)


if __name__ == "__main__":
    preprocess(DEFAULT_PATHS, DEFAULT_FILTERS, DEFAULT_CONFIG)