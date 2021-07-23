import datetime as dt

from dagma import create_node

import fireml.helpers.file_io as fio

from .region_pipeline import get_region
from .fire_detection_pipeline import get_fire_detections
from .fire_cluster_pipeline import get_fire_clusters
from .land_cover_pipeline import get_land_cover
from .meteorology_pipeline import get_meteorology

FILE_FMT_DATASET = "dataset_{}_{}_{}_{}.pkl"
DIR_DATASETS = "datasets/"


def flatten_and_resolve(d, prefix=None):
    flattened_dict = {}

    for k, v in d.items():
        resolved_key = ".".join([prefix, str(k)]) if prefix is not None else str(k)

        if isinstance(v, dict):
            flattened_dict = {
                **flattened_dict,
                **flatten_and_resolve(v, prefix=resolved_key),
            }
        else:
            flattened_dict[resolved_key] = v

    return flattened_dict


def make_dataset_path(params):
    """Create directory and build path for saving dataset."""

    dataset_dir = params["data_dir"] / fio.DIR_PROCESSED / DIR_DATASETS
    dataset_dir.mkdir(parents=True, exist_ok=True)

    return dataset_dir / FILE_FMT_DATASET.format(
        params["dataset.start_datetime"].strftime("%Y-%m-%d"),
        params["dataset.end_datetime"].strftime("%Y-%m-%d"),
        params["region"],
        params["dataset.dataset_name"],
    )


# @create_node(file_path=make_dataset_path, save=fio.save_pickle, load=fio.load_pickle)
# def make_dataset(fire_dets, land_cover, meteorology):
@create_node
def make_dataset(
    fire_detections, land_cover, meteorology, start_datetime, end_datetime, dataset_name
):
    """Combine all processed data sources into single dataset."""
    return meteorology


def get_dataset(data_params):
    """Define dataset pipeline."""
    region = get_region()

    fire_dets = (
        get_fire_detections(region)
        if "fire_detections" in data_params["components"]
        else 0
    )

    # clusters = get_fire_clusters(fire_dets)

    land_cover = (
        get_land_cover(region) if "land_cover" in data_params["components"] else 0
    )

    meteorology = (
        get_meteorology(region) if "meteorology" in data_params["components"] else 0
    )

    data_params = flatten_and_resolve(data_params)
    dataset = make_dataset(
        fire_dets,
        land_cover,
        meteorology,
        "dataset.start_datetime",
        "dataset.end_datetime",
        "dataset.dataset_name",
    )(data_params)

    return dataset
