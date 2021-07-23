import datetime as dt
import itertools
import numpy as np

from dagma import create_node

import fireml.helpers.file_io as fio
import fireml.processing.clustering.clustering as clust

DIR_CLUSTERS = "clusters/"

FILE_FMT_CLUSTER_IDS = "cluster_ids_{}_{}_{}_t{}_{}_{}km_{}hours.pkl"


def make_cluster_ids_path(params):
    cluster_dir = params["data_dir"] / fio.DIR_INTERIM / DIR_CLUSTERS
    cluster_dir.mkdir(parents=True, exist_ok=True)

    cluster_params = params["cluster"]

    return cluster_dir / FILE_FMT_CLUSTER_IDS.format(
        params["fire_detections.dataset"],
        params["fire_detections.start_datetime"].strftime("%Y-%m-%d"),
        params["fire_detections.end_datetime"].strftime("%Y-%m-%d"),
        "".join(map(str, params["det_types"])),
        params["region"],
        cluster_params["spatial_threshold_km"],
        cluster_params["temporal_threshold_hours"],
    )


@create_node(
    file_path=make_cluster_ids_path, load=fio.load_pickle, save=fio.save_pickle
)
def compute_cluster_ids(fire_dets, spatial_threshold_km, temporal_threshold_hours):
    cluster_ids = clust.cluster_spatial_temporal_forwards_bt(
        fire_dets, spatial_threshold_km, np.timedelta64(temporal_threshold_hours, "h"),
    )

    return cluster_ids


def get_fire_clusters(fire_dets):
    clusters = compute_cluster_ids(
        fire_dets, "clusters.spatial_threshold_km", "clusters.temporal_threshold_hours"
    )

    return clusters
