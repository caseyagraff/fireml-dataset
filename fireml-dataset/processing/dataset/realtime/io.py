import os
from fireml.processing.dataset.realtime.constants import (
    LandCoverLayerType,
    MeteorologyLayerType,
)
from affine import Affine
from fireml.processing.dataset.realtime.util import affine_to_transform_dict
from fireml.helpers.timing import timer
import h5py
import numpy as np
from fireml.processing.dataset.realtime.settings import (
    AllData,
    DetectionData,
    LandCoverConfig,
    LandCoverData,
    MeteorlogyData,
    MeteorologyConfig,
    OutputBuffers,
    PreprocessDataConfig,
    PreprocessFilters,
    PreprocessPaths,
    SaveBuffers,
    SetupData,
)
from typing import Any, cast
import xarray as xr
import fireml.helpers.file_io as fio

import pandas as pd
import pickle

from .preprocess import extract_points_from_geodataframe

lf_to_int = {
    "Sparse": 0,
    "Tree": 1,
    "Shrub": 2,
    "Herb": 3,
    "Water": 4,
    "Barren": 5,
    "Developed": 6,
    "Snow-Ice": 7,
    "Agriculture": 8,
    "Nodata - Land": 9,
}
nlcd_to_evt_classes = np.zeros((96), dtype=np.uint8)
nlcd_to_evt_classes[:] = lf_to_int["Nodata - Land"]

nlcd_to_evt_classes[0] = lf_to_int["Nodata - Land"]
nlcd_to_evt_classes[11] = lf_to_int["Water"]
nlcd_to_evt_classes[12] = lf_to_int["Snow-Ice"]
nlcd_to_evt_classes[21] = lf_to_int["Developed"]
nlcd_to_evt_classes[22] = lf_to_int["Developed"]
nlcd_to_evt_classes[23] = lf_to_int["Developed"]
nlcd_to_evt_classes[24] = lf_to_int["Developed"]
nlcd_to_evt_classes[31] = lf_to_int["Barren"]
nlcd_to_evt_classes[41] = lf_to_int["Tree"]
nlcd_to_evt_classes[42] = lf_to_int["Tree"]
nlcd_to_evt_classes[43] = lf_to_int["Tree"]
nlcd_to_evt_classes[52] = lf_to_int["Shrub"]
nlcd_to_evt_classes[71] = lf_to_int["Herb"]
nlcd_to_evt_classes[81] = lf_to_int["Agriculture"]
nlcd_to_evt_classes[82] = lf_to_int["Agriculture"]
nlcd_to_evt_classes[90] = lf_to_int["Sparse"]
nlcd_to_evt_classes[95] = lf_to_int["Sparse"]


nlcd_to_compact_classes = np.zeros((96), dtype=np.uint8)
nlcd_to_compact_classes[:] = lf_to_int["Nodata - Land"]

nlcd_to_compact_classes[0] = 0
nlcd_to_compact_classes[11] = 1
nlcd_to_compact_classes[12] = 2
nlcd_to_compact_classes[21] = 3
nlcd_to_compact_classes[22] = 4
nlcd_to_compact_classes[23] = 5
nlcd_to_compact_classes[24] = 6
nlcd_to_compact_classes[31] = 7
nlcd_to_compact_classes[41] = 8
nlcd_to_compact_classes[42] = 9
nlcd_to_compact_classes[43] = 10
nlcd_to_compact_classes[52] = 11
nlcd_to_compact_classes[71] = 12
nlcd_to_compact_classes[81] = 13
nlcd_to_compact_classes[82] = 14
nlcd_to_compact_classes[90] = 15
nlcd_to_compact_classes[95] = 16


def load_evt_metadata(path) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(path))
    df.drop("OID_", axis=1, inplace=True)

    return df


@timer("Loaded detection data: ")
def load_detection_data(path: str) -> DetectionData:
    with h5py.File(path, "r") as hf:
        points: np.ndarray = cast(np.ndarray, cast(h5py.Dataset, hf["points"])[:])
        detections_search_table: np.ndarray = cast(
            np.ndarray, cast(h5py.Dataset, hf["fire_detections_search_table"])[:]
        )

    return {"points": points, "detections_search_table": detections_search_table}


@timer("Loaded land cover: ")
def load_land_cover_data(
    path: str, metadata_path: str, filter_path: str, config: LandCoverConfig
) -> LandCoverData:
    with h5py.File(filter_path, "r") as hf_filter:
        land_cover_filter: np.ndarray = cast(
            np.ndarray, cast(h5py.Dataset, hf_filter["nlcd"])[:]
        )[0]

        filter_transform: Any = Affine.from_gdal(
            *cast(
                np.ndarray,
                cast(h5py.Dataset, hf_filter["nlcd"]).attrs["transform"],
            )
        )

    filter_transform_dict = affine_to_transform_dict(filter_transform)

    if not config["use_land_cover"]:
        return {
            "land_cover_layers": np.empty((1, 1, 1), dtype=np.int16),
            "land_cover_layer_types": np.empty((1), dtype=np.uint8),
            "land_cover_transform": {},
            "evt_metadata": pd.DataFrame(),
            "land_cover_filter_layer": land_cover_filter,
            "land_cover_filter_transform": filter_transform_dict,
        }

    with h5py.File(path, "r") as hf:
        land_cover = np.vstack(
            [
                cast(np.ndarray, cast(h5py.Dataset, hf[k])[:])
                for k in config["land_cover_layers"]
            ]
        )

        transform: Any = Affine.from_gdal(
            *cast(
                np.ndarray,
                cast(h5py.Dataset, hf[config["land_cover_layers"][0]]).attrs[
                    "transform"
                ],
            )
        )

    # Crop NLCD to same domain as LANDFIRE
    land_cover_shape = land_cover.shape
    start = ~filter_transform * np.array(transform * (0, 0))
    end = ~filter_transform * np.array(
        transform * (land_cover_shape[2], land_cover_shape[1])
    )

    start = int(round(start[0]))
    end = int(round(end[0]))

    nlcd_cropped = land_cover_filter[:, start:end]

    # Convert NLCD to EVT classes
    # nlcd_as_evt = nlcd_to_evt_classes[nlcd_cropped]
    nlcd_as_evt = nlcd_to_compact_classes[nlcd_cropped]

    print("n", np.unique(nlcd_as_evt))

    # Use NLCD instead of EVT
    land_cover = np.vstack([land_cover, nlcd_as_evt[None]])

    land_cover_layer_types = np.array(
        [
            LandCoverLayerType.EVT.value
            if n == "landfire_evt"
            else LandCoverLayerType.OTHER.value
            for n in config["land_cover_layers"]
        ],
        dtype=np.uint8,
    )

    # Label NLCD as EVT
    land_cover_layer_types = np.concatenate(
        [
            land_cover_layer_types,
            np.array([LandCoverLayerType.EVT.value], dtype=np.uint8),
        ]
    )

    evt_metadata = load_evt_metadata(metadata_path)

    transform_dict = affine_to_transform_dict(transform)

    return {
        "land_cover_layers": land_cover,
        "land_cover_layer_types": land_cover_layer_types,
        "land_cover_transform": transform_dict,
        "evt_metadata": evt_metadata,
        "land_cover_filter_layer": land_cover_filter,
        "land_cover_filter_transform": filter_transform_dict,
    }


@timer("Loaded meteorology: ")
def load_meteorology_data(
    path: str, filters: PreprocessFilters, config: MeteorologyConfig
) -> MeteorlogyData:
    if not config["use_meteorology"]:
        return {
            "meteorology_times": np.empty((1), dtype=np.uint64),
            "meteorology_xys": np.empty((1, 1, 1), dtype=np.float64),
            "meteorology_layers": np.empty((1, 1, 1, 1), dtype=np.float32),
            "meteorology_layer_types": np.empty((1), dtype=np.uint8),
            "meteorology_transform": {
                "a": 0,
                "b": 0,
                "c": 0,
                "d": 0,
                "e": 0,
                "f": 0,
                "a_i": 0,
                "b_i": 0,
                "c_i": 0,
                "d_i": 0,
                "e_i": 0,
                "f_i": 0,
            },
        }

    meteorology: xr.Dataset = fio.load_netcdf(path)
    meteorology["time"] = pd.to_datetime(meteorology.time.values)
    meteorology = meteorology.sel(
        time=slice(filters["start_datetime"], filters["end_datetime"])
    )
    meteorology = meteorology.isel(time=meteorology.missing == False)

    meteorology_times = cast(np.ndarray, meteorology.time.data).astype(np.uint64)
    meteorology_xys = np.stack(np.meshgrid(meteorology.x.data, meteorology.y.data))

    meteorology_layers = (
        np.stack([meteorology[l].data for l in config["meteorology_layers"]])
        if len(config["meteorology_layers"]) > 0
        else np.empty((1, 1, 1, 1), dtype=np.float32)
    )

    meteorology_layer_types = np.array(
        [MeteorologyLayerType.OTHER.value for n in config["meteorology_layers"]],
        dtype=np.uint8,
    )

    transform: Any = Affine.from_gdal(*cast(np.ndarray, meteorology.transform))
    meteorology_transform_dict = affine_to_transform_dict(transform)

    return {
        "meteorology_times": meteorology_times,
        "meteorology_xys": meteorology_xys,
        "meteorology_layers": meteorology_layers,
        "meteorology_layer_types": meteorology_layer_types,
        "meteorology_transform": meteorology_transform_dict,
    }


def load_all_data(
    paths: PreprocessPaths, filters: PreprocessFilters, config: PreprocessDataConfig
) -> AllData:
    detection_data = load_detection_data(paths["save"])

    if paths["filter_points"]:
        with open(paths["filter_points"], "rb") as fin:
            df = pickle.load(fin)

        filter_points = extract_points_from_geodataframe(df)

        detection_data["points"] = filter_points
    else:
        filter_points = None

    land_cover_data = load_land_cover_data(
        os.path.join(paths["data_dir"], paths["land_cover"]),
        os.path.join(paths["data_dir"], paths["evt_metadata"]),
        os.path.join(paths["data_dir"], paths["land_cover_filter"]),
        config["land_cover"],
    )

    meteorology_data = load_meteorology_data(
        os.path.join(paths["data_dir"], paths["meteorology"]),
        filters,
        config["meteorology"],
    )

    return {
        "detections": detection_data,
        "land_cover": land_cover_data,
        "meteorology": meteorology_data,
        "filter_points": filter_points,
    }


def create_output_buffers(setup_data: SetupData, batch_size: int) -> OutputBuffers:
    detection_data_out = np.zeros(
        (
            batch_size,
            setup_data["config"]["num_detection_layers"],
            setup_data["config"]["width"],
            setup_data["config"]["width"],
        ),
        dtype=np.uint8,
    )

    land_cover_data_out = np.zeros(
        (
            batch_size,
            setup_data["config"]["num_land_cover_layers"],
            setup_data["config"]["land_cover_width"],
            setup_data["config"]["land_cover_width"],
        ),
        dtype=np.float32,
    )

    meteorology_data_out = np.zeros(
        (
            batch_size,
            setup_data["config"]["num_meteorology_layers"],
            setup_data["config"]["meteorology_width"],
            setup_data["config"]["meteorology_width"],
        ),
        dtype=np.float32,
    )

    filtered_data_points_out = np.zeros(
        batch_size,
        dtype=np.uint8,
    )

    return {
        "detections": detection_data_out,
        "land_cover": land_cover_data_out,
        "meteorology": meteorology_data_out,
        "filters": filtered_data_points_out,
    }


def clear_output_buffers(buffers: OutputBuffers):
    buffers["detections"][:] = 0
    buffers["land_cover"][:] = 0
    buffers["meteorology"][:] = 0
    buffers["filters"][:] = 0


def create_save_buffers(setup_data: SetupData, total_points: int) -> SaveBuffers:
    all_filters = np.zeros(total_points, dtype=np.uint8)
    all_detections = np.empty(
        (
            total_points,
            setup_data["config"]["num_detection_layers"],
            setup_data["config"]["width"],
            setup_data["config"]["width"],
        ),
        dtype=np.uint8,
    )

    all_land_cover = np.empty(
        (
            total_points,
            setup_data["config"]["num_land_cover_layers"],
            setup_data["config"]["land_cover_width"],
            setup_data["config"]["land_cover_width"],
        )
    )

    all_meteorology = np.empty(
        (
            total_points,
            setup_data["config"]["num_meteorology_layers"],
            setup_data["config"]["meteorology_width"],
            setup_data["config"]["meteorology_width"],
        )
    )

    return {
        "detections": all_detections,
        "land_cover": all_land_cover,
        "meteorology": all_meteorology,
        "filters": all_filters,
    }


def save_output_buffers(
    output_buffers: OutputBuffers,
    save_buffers: SaveBuffers,
    start_ind: int,
    stop_ind: int,
):
    num_inds = stop_ind - start_ind
    save_buffers["filters"][start_ind:stop_ind] = output_buffers["filters"][:num_inds]
    save_buffers["detections"][start_ind:stop_ind] = output_buffers["detections"][
        :num_inds
    ]
    save_buffers["land_cover"][start_ind:stop_ind] = output_buffers["land_cover"][
        :num_inds
    ]
    save_buffers["meteorology"][start_ind:stop_ind] = output_buffers["meteorology"][
        :num_inds
    ]


def write_save_buffers_to_hdf5(
    path: str, setup_data: SetupData, save_buffers: SaveBuffers, total_points: int
):
    target_detections_ind = setup_data["config"]["num_detection_layers"] - len(
        setup_data["config"]["forecast_offsets"]
    )
    with h5py.File(
        path,
        "w",
    ) as hf:
        hf.create_dataset("filters", data=save_buffers["filters"])
        hf.create_dataset(
            "observed", data=save_buffers["detections"][:, :target_detections_ind]
        )
        hf.create_dataset(
            "target", data=save_buffers["detections"][:, target_detections_ind:]
        )
        hf.create_dataset("land_cover", data=save_buffers["land_cover"])
        hf.create_dataset("meteorology", data=save_buffers["meteorology"])
        hf.create_dataset("datetimes", data=setup_data["points_time"][:total_points])
