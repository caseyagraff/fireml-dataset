from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from typing_extensions import TypedDict
import geopandas as gpd
import datetime as dt


class Point(TypedDict):
    x: np.float64
    y: np.float64
    datetime: np.datetime64


class PreprocessPaths(TypedDict):
    data_dir: str
    save_dir: str
    fire_detections: str
    land_cover: str
    land_cover_filter: str
    evt_metadata: str
    meteorology: str
    data_points: Optional[str]
    filter_points: Optional[str]
    save: str


class PreprocessData(TypedDict):
    fire_detections: gpd.GeoDataFrame
    data_points: Optional[np.ndarray]
    filter_points: Optional[gpd.GeoDataFrame]


class PreprocessDataArtifacts(TypedDict):
    data_points: np.ndarray
    fire_detections_search_table: np.ndarray


class PreprocessFilters(TypedDict):
    start_datetime: np.datetime64
    end_datetime: np.datetime64


class FireDetectionsConfig(TypedDict):
    detection_window_lower: np.timedelta64
    detection_window_upper: np.timedelta64

    time_of_day_lower: Optional[dt.time]
    time_of_day_upper: Optional[dt.time]

    forecast_window_lower: np.timedelta64
    forecast_window_upper: np.timedelta64
    forecast_offsets: List[np.timedelta64]

    lags: List[np.timedelta64]
    aggregate_lags: List[np.timedelta64]


class LandCoverConfig(TypedDict):
    use_land_cover: bool
    land_cover_layers: List[str]


class MeteorologyConfig(TypedDict):
    use_meteorology: bool
    meteorology_layers: List[str]
    lags: List[np.timedelta64]


class PreprocessConfig(TypedDict):
    apply_filter_to_data_points: bool


class PreprocessDataConfig(TypedDict):
    fire_detections: FireDetectionsConfig
    land_cover: LandCoverConfig
    meteorology: MeteorologyConfig
    preprocessing: PreprocessConfig


class DetectionData(TypedDict):
    points: np.ndarray
    detections_search_table: np.ndarray


class LandCoverData(TypedDict):
    land_cover_layers: np.ndarray
    land_cover_layer_types: np.ndarray
    land_cover_transform: Dict[str, float]
    evt_metadata: pd.DataFrame
    land_cover_filter_layer: np.ndarray
    land_cover_filter_transform: Dict[str, float]


class MeteorlogyData(TypedDict):
    meteorology_times: np.ndarray
    meteorology_xys: np.ndarray
    meteorology_layers: np.ndarray
    meteorology_layer_types: np.ndarray
    meteorology_transform: Dict[str, float]


class AllData(TypedDict):
    detections: DetectionData
    land_cover: LandCoverData
    meteorology: MeteorlogyData


class SetupData(TypedDict):
    config: Dict[str, Any]
    points_time: np.ndarray
    points_xy: np.ndarray
    detections_search_table_times: np.ndarray
    detections_search_table_xys: np.ndarray
    evt_to_class: np.ndarray


class OutputBuffers(TypedDict):
    detections: np.ndarray
    land_cover: np.ndarray
    meteorology: np.ndarray
    filters: np.ndarray


SaveBuffers = OutputBuffers


Positions = TypedDict("Positions", {"xs": np.ndarray, "ys": np.ndarray})

DEFAULT_CELL_SIZE = 375
DEFAULT_LAND_COVER_CELL_SIZE = 375
DEFAULT_METEOROLOGY_CELL_SIZE = 375 * 30
DEFAULT_WINDOW_SIZE = 375 * 30 * 2

DEFAULT_SHOULD_FILTER_BY_DETECTIONS = True
DEFAULT_SHOULD_FILTER_BY_LAND_COVER = True


DEFAULT_FILTERS: PreprocessFilters = {
    "start_datetime": np.datetime64("2012-05-09T00:00:00", "ns"),
    "end_datetime": np.datetime64("2018-01-01T00:00:00", "ns"),
}

PREPROCESS_CA = "2012-05-09_2018-01-01_california_2021-04-30T0410.hdf5"
PREPROCESS_OR = "2012-05-09_2018-01-01_oregon_2021-04-30T0410.hdf5"
PREPROCESS_WA = "2012-05-09_2018-01-01_washington_2021-04-30T0410.hdf5"

CURRENT_REGION = "california"
PREPROCESS_DICT = {
    "california": PREPROCESS_CA,
    "oregon": PREPROCESS_OR,
    "washington": PREPROCESS_WA,
}

print(f"==== {CURRENT_REGION} ====")

DEFAULT_PATHS: PreprocessPaths = {
    "data_dir": "/extra/datalab_scratch0/graffc0/fireml/",
    "save_dir": "/extra/datalab_scratch0/graffc0/fireml/data/processed/datasets/",
    "fire_detections": f"data/interim/fire_detections/viirs_375m/regions/{CURRENT_REGION}/viirs_375m_v1_2012-01-01_2020-01-01_t0123_{CURRENT_REGION}_usa_aea.pkl",
    "land_cover": f"data/interim/land_cover/combined/US_ASP-CBD-CBH-CC-CH-DEM-EVT-SLP_{CURRENT_REGION}_usa_aea.hdf5",
    "land_cover_filter": f"data/interim/land_cover/vegetation/nlcd/projected/NLCD_2011_{CURRENT_REGION}_usa_aea.hdf5",
    "evt_metadata": "data/raw/land_cover/vegetation/landfire_evt/US_130EVT/CSV_Data/US_130EVT_02092015.csv",
    "meteorology": f"data/interim/meteorology/rapid_refresh_13km/aggregated/rap_130_2012-05-09_2018-01-01_{CURRENT_REGION}_usa_aea.nc",
    "data_points": None,
    "filter_points": None,  # "data/interim/fire_detections/viirs_375m/case_studies/case_studies.pkl",
    "save": f"data/processed/datasets/{PREPROCESS_DICT[CURRENT_REGION]}",
}
DEFAULT_CONFIG: PreprocessDataConfig = {
    "fire_detections": {
        "detection_window_lower": np.timedelta64(3, "h"),
        "detection_window_upper": np.timedelta64(3, "h"),
        "time_of_day_lower": dt.time(6, 30),
        "time_of_day_upper": dt.time(12, 30),
        "forecast_window_lower": np.timedelta64(15, "h"),
        "forecast_window_upper": np.timedelta64(3, "h"),
        "forecast_offsets": [
            np.timedelta64(24, "h"),
            np.timedelta64(48, "h"),
            np.timedelta64(72, "h"),
            np.timedelta64(96, "h"),
            np.timedelta64(120, "h"),
        ],
        "lags": [np.timedelta64(i, "h") for i in [12, 24, 36, 48, 60]],
        "aggregate_lags": [
            np.timedelta64(60, "h"),
            np.timedelta64(168, "h"),
            np.timedelta64(168, "h"),
            np.timedelta64(365, "D"),
        ],
    },
    "land_cover": {
        "use_land_cover": True,
        # "land_cover_layers": ["landfire_dem", "landfire_slp", "landfire_evt"],
        "land_cover_layers": ["landfire_dem"],
    },
    "meteorology": {
        "use_meteorology": True,
        "meteorology_layers": [
            "temperature",
            "2_metre_relative_humidity",
            "10_metre_u_wind_component",
            "10_metre_v_wind_component",
            "precipitation_rate",
        ],
        # Each lag is a pair of two times lower, upper
        "lags": [
            np.timedelta64(i, "h") for i in [0, 24, 24, 48, 48, 72, 72, 96, 96, 120]
        ],
    },
    "preprocessing": {"apply_filter_to_data_points": False},
}
