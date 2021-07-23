"""
Algorithms for clustering data.
"""
from typing import List, Dict, Any
import bisect
import numpy as np
import math

import tqdm

from sklearn.neighbors import BallTree

CLUST_TYPE_SPATIAL_TEMPORAL_FORWARDS = "spatial_temporal_forwards"

EARTH_RADIUS_KM = 6371.0
Point = List[Any]  # Tuple[float, float, dt.datetime, int, int]


def remove_self(inds, dist):
    new_inds = []
    new_dist = []
    for i in range(len(inds)):
        pos = np.where(inds[i] == i)
        new_inds.append(np.delete(inds[i], pos))
        new_dist.append(np.delete(dist[i], pos))

    return np.array(new_inds), np.array(new_dist)


def compute_all_spatial_distances(data, max_thresh_km):
    lats = data.latitude.values
    lons = data.longitude.values

    lats_rad = [math.radians(l) for l in lats]
    lons_rad = [math.radians(l) for l in lons]

    X = np.stack([lats_rad, lons_rad], axis=-1)

    bt = BallTree(X, metric="haversine", leaf_size=20)

    inds, dist = bt.query_radius(
        X, r=max_thresh_km / EARTH_RADIUS_KM, return_distance=True
    )

    inds, dist = remove_self(inds, dist)

    return inds, dist * EARTH_RADIUS_KM


def compute_temporal_distance(det_dt, point_dts, max_thresh_time):
    dists = np.abs(point_dts - det_dt)
    inds = (dists < max_thresh_time) & (point_dts <= det_dt)

    return inds, dists[inds]


def compute_all_temporal_distances(
    data, neighbor_inds, neighbor_spatial_dists, max_thresh_time
):
    new_neighbor_inds = []
    neighbor_temporal_dists = []
    new_neighbor_spatial_dists = []

    det_datetimes = data.acq_datetime.values

    for i, (det_dt, n_inds, spatial_dists) in enumerate(
        zip(det_datetimes, neighbor_inds, neighbor_spatial_dists)
    ):
        active_point_dts = det_datetimes[n_inds]
        inds, temporal_dists = compute_temporal_distance(
            det_dt, active_point_dts, max_thresh_time
        )

        new_neighbor_inds.append(n_inds[inds])
        new_neighbor_spatial_dists.append(spatial_dists[inds])
        neighbor_temporal_dists.append(temporal_dists)

    return new_neighbor_inds, new_neighbor_spatial_dists, neighbor_temporal_dists


def sort_neighbors(n_inds, s_dists, t_dists):
    new_n_inds = []
    new_s_dists = []
    new_t_dists = []

    # Sorting by most recent, then by spatial distance
    for i in range(len(n_inds)):
        new_order = np.lexsort([s_dists[i], t_dists[i]])
        new_n_inds.append(n_inds[i][new_order])
        new_s_dists.append(s_dists[i][new_order])
        new_t_dists.append(t_dists[i][new_order])

    return new_n_inds, new_s_dists, new_t_dists


NO_CLUSTER_VAL = -1


def cluster_spatial_temporal_forwards_bt(data, max_thresh_km, max_thresh_time):
    cluster_id_counter = 0

    num_points = len(data)
    point_to_cluster_id = np.full(num_points, fill_value=NO_CLUSTER_VAL, dtype=np.int64)

    last_datetime = None

    neighbor_inds, neighbor_spatial_dists = compute_all_spatial_distances(
        data, max_thresh_km
    )
    print("Done spatial", len(neighbor_inds))

    (
        neighbor_inds,
        neighbor_spatial_dists,
        neighbor_temporal_dists,
    ) = compute_all_temporal_distances(
        data, neighbor_inds, neighbor_spatial_dists, max_thresh_time
    )
    print("Done temporal", len(neighbor_inds))

    neighbor_inds, neighbor_spatial_dists, neighbor_temporal_dists = sort_neighbors(
        neighbor_inds, neighbor_spatial_dists, neighbor_temporal_dists
    )

    for i in tqdm.tqdm(range(num_points)):
        neighbors = neighbor_inds[i]

        if len(neighbors) == 0:
            # Add to new cluster
            point_to_cluster_id[i] = cluster_id_counter
            cluster_id_counter += 1
        else:
            neighbor_cluster_ids = point_to_cluster_id[neighbors]
            neighbor_cluster_ids = neighbor_cluster_ids[
                neighbor_cluster_ids != NO_CLUSTER_VAL
            ]

            if len(neighbor_cluster_ids) == 0:
                # Add to new cluster
                point_to_cluster_id[i] = cluster_id_counter
                cluster_id_counter += 1
            else:
                point_to_cluster_id[i] = neighbor_cluster_ids[0]

    return point_to_cluster_id


"""
def cluster_spatial_temporal_forwards(data, max_thresh_km, max_thresh_time, t_id):
    cluster_id_counter = 0

    active_points: List[Point] = []
    all_points = []

    # cluster_id: (earliest active point, [points,])
    pending_clusters: Dict[int, List[Point]] = {}

    # Iterate over each detection·
    last_datetime = None
    for row in tqdm.tqdm(list(data.itertuples())):

        p_new = [row.latitude, row.longitude, row.acq_datetime, None, row.Index]

        if last_datetime != p_new[2]:
            # Discard old points
            remove_datetime = p_new[2] - max_thresh_time
            remove_ind = bisect.bisect_left(
                [p[2] for p in active_points], remove_datetime
            )

            active_points = active_points[remove_ind:]

            last_datetime = p_new[2]

            # Close pending clusters
            pending_clusters = {}

        # Compute connection to all active points
        distances = (
            (
                spatial_temporal_dist(p_new[0], p_new[1], p_old[0], p_old[1], p_new[2], p_old[2], max_thresh_km, max_thresh_time),
                p_old[3],
            )
            for p_old in active_points
        )
        connections = [d for d in distances if d[0][0]]

        # If only one connect point
        if len(connections) == 1:
            p_new[3] = connections[0][1]

        # Multiple connected points
        if len(connections) > 1:
            # All connections belong to same cluster
            if all(c[1] == connections[0][1] for c in connections):
                p_new[3] = connections[0][1]

            # Multiple clusters are connected
            else:
                unique_cluster_ids = set(c[1] for c in connections)

                pending_cluster_ids = [
                    id for id in unique_cluster_ids if id in pending_clusters
                ]
                non_pending_cluster_ids = list(
                    unique_cluster_ids.difference(pending_cluster_ids)
                )

                # Check if any are pending
                if len(pending_cluster_ids) > 0:
                    main_pending_cluster_id = pending_cluster_ids[0]

                    # Merge all pending together
                    if len(pending_cluster_ids) > 1:
                        for cluster_id in pending_cluster_ids[1:]:
                            merge_clusters(
                                pending_clusters, main_pending_cluster_id, cluster_id,
                            )

                    # If any non-pending
                    if len(non_pending_cluster_ids) > 0:
                        # Merge pending with closest non-pending and remove from pending list
                        closest_non_pending_cluster_id = min(
                            (c[0][1], c[1])
                            for c in connections
                            if c[1] in non_pending_cluster_ids
                        )[1]

                        settle_pending_cluster(
                            pending_clusters,
                            main_pending_cluster_id,
                            closest_non_pending_cluster_id,
                        )

                        p_new[3] = closest_non_pending_cluster_id

                # No connected pending
                else:
                    p_new[3] = min((c[0][1], c[1]) for c in connections)[1]

        if p_new[3] in pending_clusters:
            pending_clusters[p_new[3]].append(p_new)

        # If no nearby points, create new cluster
        elif p_new[3] is None:
            p_new[3] = cluster_id_counter
            pending_clusters[cluster_id_counter] = [p_new]
            cluster_id_counter += 1

        active_points.append(p_new)
        all_points.append(p_new)

    fire_cluster_ids = [p[3] for p in all_points]
    # data_indices = [p[4] for p in prev_points]
    # data.loc[data_indices, 'cluster_id'] = fire_cluster_ids

    return fire_cluster_ids

"""
