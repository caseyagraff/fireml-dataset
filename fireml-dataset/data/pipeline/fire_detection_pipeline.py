import os
import pandas as pd
import datetime as dt
from dagma import create_node
import pickle

from fireml.helpers import file_io as fio, projections as proj
import fireml.data.fire_detections as fds

FILE_FMT_FILTERED_VIIRS_375M = "viirs_375m_v1_{}_t{}.pkl"
FILE_FMT_PROJECTED_VIIRS_375M = "viirs_375m_v1_{}_{}_t{}_{}_{}.pkl"

DIR_FILTERED = "filtered/"
DIR_REGIONS = "regions/"


# === File Path Functions ===
def get_fire_det_filtered_path(params):
    foreach_var = params["%foreach"]

    if foreach_var is None:
        return None

    year = foreach_var.iloc[0].acq_date[:4]

    detection_dir = (
        params["data_dir"]
        / fio.DIR_INTERIM
        / fds.DIR_FIRE_DETECTIONS
        / params["fire_detections.dataset"]
        / DIR_FILTERED
    )
    detection_dir.mkdir(parents=True, exist_ok=True)

    path = detection_dir / FILE_FMT_FILTERED_VIIRS_375M.format(
        year, "".join(map(str, params["fire_detections.det_types"])),
    )

    print("fds", path, params)

    return path


def get_fire_det_projected_path(params):
    detection_dir = (
        params["data_dir"]
        / fio.DIR_INTERIM
        / fds.DIR_FIRE_DETECTIONS
        / params["fire_detections.dataset"]
        / DIR_REGIONS
        / params["region"]
    )
    detection_dir.mkdir(parents=True, exist_ok=True)

    path = detection_dir / FILE_FMT_PROJECTED_VIIRS_375M.format(
        params["fire_detections.start_datetime"].strftime("%Y-%m-%d"),
        params["fire_detections.end_datetime"].strftime("%Y-%m-%d"),
        "".join(map(str, params["fire_detections.det_types"])),
        params["region"],
        params["projection"],
    )

    return path


# === Pipeline ===
@create_node
def get_years(start_datetime, end_datetime):
    """Get all years between start and end (inclusive) that have at least one full day in the range."""
    print("get years")
    return list(
        range(start_datetime.year, (end_datetime - dt.timedelta(hours=24)).year + 1)
    )


@create_node
def load_fire_detections_year(year, data_dir, dataset):
    print("load year")
    detection_dir = data_dir / fio.DIR_RAW / fds.DIR_FIRE_DETECTIONS
    detections = fds.load_fire_detections(detection_dir, dataset, year)
    return detections


@create_node(
    file_path=get_fire_det_filtered_path, load=fio.load_pickle, save=fio.save_pickle
)
def filter_detections(fire_detections, det_types, start_datetime, end_datetime):
    """Apply basic detections filtering (detection type and dates)."""
    print("filter dets")
    fire_detections = fds.filter_det_type(fire_detections, det_types)
    fire_detections = fds.add_datetimes(fire_detections)
    fire_detections = fds.filter_datetime(fire_detections, start_datetime, end_datetime)

    return fire_detections


@create_node
def aggregate_detections(fire_detection_dfs):
    """Combine multiple detection dataframes."""
    print("aggregate dets")
    return pd.concat(fire_detection_dfs)


@create_node
def filter_region(fire_detections, region):
    print("filter region")
    region, region_no_pad = region
    return fds.filter_shape(fire_detections, region, region_no_pad)


@create_node(
    file_path=get_fire_det_projected_path, load=fio.load_pickle, save=fio.save_pickle
)
def to_geodataframe_and_project_detections(fire_detections, projection):
    """Convert to GeoDataframe, project to common CRS, and filter by shape."""
    print("to geodf")
    fire_detections = fds.to_geodataframe(fire_detections)
    fire_detections = fire_detections.to_crs(proj.PROJECTION_DICT[projection])

    return fire_detections


def get_fire_detections(region):
    years = get_years("fire_detections.start_datetime", "fire_detections.end_datetime")

    fire_dets_years = load_fire_detections_year(
        years, "data_dir", "fire_detections.dataset", foreach=0
    )
    fire_dets_years = filter_detections(
        fire_dets_years,
        "fire_detections.det_types",
        "fire_detections.start_datetime",
        "fire_detections.end_datetime",
        foreach=0,
    )

    fire_dets = aggregate_detections(fire_dets_years)
    fire_dets = filter_region(fire_dets, region)
    fire_dets = to_geodataframe_and_project_detections(fire_dets, "projection")

    return fire_dets
