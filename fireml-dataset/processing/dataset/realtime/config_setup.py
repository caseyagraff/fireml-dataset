from fireml.processing.dataset.realtime.constants import DiscretizationMethods
from typing import Any, Dict, cast
from fireml.processing.dataset.realtime.util import (
    create_evt_to_lf_arr,
    timedelta64_to_uint,
)
import numpy as np
from fireml.processing.dataset.realtime.settings import (
    DEFAULT_CELL_SIZE,
    DEFAULT_LAND_COVER_CELL_SIZE,
    DEFAULT_METEOROLOGY_CELL_SIZE,
    DEFAULT_SHOULD_FILTER_BY_DETECTIONS,
    DEFAULT_SHOULD_FILTER_BY_LAND_COVER,
    DEFAULT_WINDOW_SIZE,
    PreprocessDataConfig,
    SetupData,
)


def setup(
    config: PreprocessDataConfig,
    points: np.ndarray,
    detections_search_table: np.ndarray,
    evt_metadata,
) -> SetupData:
    created_config: Dict[str, Any] = {
        k: timedelta64_to_uint(config["fire_detections"][k])
        for k in [
            "detection_window_lower",
            "detection_window_upper",
            "forecast_window_lower",
            "forecast_window_upper",
        ]
    }

    created_config["use_land_cover"] = config["land_cover"]["use_land_cover"]
    created_config["use_meteorology"] = config["meteorology"]["use_meteorology"]

    created_config["lags"] = timedelta64_to_uint(
        np.array(config["fire_detections"]["lags"])
    )
    created_config["aggregate_lags"] = timedelta64_to_uint(
        np.array(config["fire_detections"]["aggregate_lags"])
    )

    created_config["meteorology_lags"] = timedelta64_to_uint(
        np.array(config["meteorology"]["lags"])
    )

    created_config["forecast_offsets"] = timedelta64_to_uint(
        np.array(config["fire_detections"]["forecast_offsets"])
    )

    created_config["cell_size"] = DEFAULT_CELL_SIZE
    created_config["land_cover_cell_size"] = DEFAULT_LAND_COVER_CELL_SIZE
    created_config["meteorology_cell_size"] = DEFAULT_METEOROLOGY_CELL_SIZE
    created_config["window_size"] = DEFAULT_WINDOW_SIZE
    created_config["discretization_method"] = DiscretizationMethods.EXACT.value

    created_config["width"] = int(
        created_config["window_size"] // created_config["cell_size"]
    )

    created_config["land_cover_width"] = int(
        created_config["window_size"] // created_config["land_cover_cell_size"]
    )

    created_config["meteorology_width"] = int(
        created_config["window_size"] // created_config["meteorology_cell_size"]
    )

    num_layers = (
        1
        + len(created_config["lags"])
        + (len(created_config["aggregate_lags"]) // 2)
        + len(created_config["forecast_offsets"])
    )

    created_config["num_detection_layers"] = num_layers

    evt_to_class, num_classes = (
        create_evt_to_lf_arr(evt_metadata)
        if config["land_cover"]["use_land_cover"]
        else np.empty((1))
    )

    created_config["num_one_hot_classes"] = num_classes

    created_config["num_land_cover_layers"] = (
        len(config["land_cover"]["land_cover_layers"])
        + created_config["num_one_hot_classes"]
        - 1
    )

    # created_config["should_convert_evt_classes"] = True
    created_config["should_convert_evt_classes"] = False

    created_config["num_meteorology_layers"] = len(
        config["meteorology"]["meteorology_layers"]
    ) * len(config["meteorology"]["lags"])

    points_time = cast(np.ndarray, points[:, 0].astype("uint64"))
    points_xy = np.ascontiguousarray(points[:, 1:])

    detections_search_table_times = cast(
        np.ndarray,
        detections_search_table[:, 0].astype("datetime64[ns]").astype("uint64"),
    )
    detections_search_table_xys = np.ascontiguousarray(detections_search_table[:, 1:])

    # Filtering
    created_config["should_filter_by_detections"] = DEFAULT_SHOULD_FILTER_BY_DETECTIONS
    created_config["should_filter_by_land_cover"] = DEFAULT_SHOULD_FILTER_BY_LAND_COVER
    created_config["minimum_detections"] = 5
    created_config["minimum_land_cover_detections"] = 3
    created_config["minimum_land_cover_detections_percentage"] = 0.0
    created_config["maximum_lags_detections"] = 2
    created_config["maximum_lags_land_cover"] = 2

    return {
        "config": created_config,
        "points_time": points_time,
        "points_xy": points_xy,
        "detections_search_table_times": detections_search_table_times,
        "detections_search_table_xys": detections_search_table_xys,
        "evt_to_class": evt_to_class,
    }
