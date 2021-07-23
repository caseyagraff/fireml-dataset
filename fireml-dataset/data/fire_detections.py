"""
Loading and manipulating fire detection data.
"""
import os
import pytz
import datetime as dt

import pandas as pd
import geopandas
import shapely

DIR_FIRE_DETECTIONS = "fire_detections/"

DATASET_VIIRS_375M = "viirs_375m"
DIR_VIIRS_375M = "viirs_375m/"
FILE_FMT_VIIRS_375M = "viirs_375m_v1_{}.csv"


def load_fire_detections(dir_path, dataset, year):
    if dataset == DATASET_VIIRS_375M:
        file_path = dir_path / DIR_VIIRS_375M / FILE_FMT_VIIRS_375M.format(year)
    else:
        raise ValueError(f'Invalid value "{dataset}" for dataset.')

    return pd.read_csv(file_path)


def filter_det_type(det_df, det_types):
    return det_df[det_df.type.isin(det_types)]


def add_datetimes(det_df):
    datetime_strings = det_df.acq_date + " " + det_df.acq_time.astype('int32').apply("{0:0>4}".format)
    acq_datetimes = pd.to_datetime(datetime_strings, utc=True)

    return det_df.assign(acq_datetime=acq_datetimes)


def filter_datetime(det_df, start_datetime, end_datetime):
    """Inclusive lower, exclusive upper."""
    return det_df[
        (start_datetime <= det_df.acq_datetime) & (det_df.acq_datetime < end_datetime)
    ]


def filter_latlon(det_df, bounding_box):
    min_lon, min_lat, max_lon, max_lat = bounding_box
    print("bb", min_lon, min_lat, max_lon, max_lat)

    is_in_bounding_box = [
        min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
        for lon, lat in zip(det_df.longitude.astype('float32'), det_df.latitude.astype('float32'))
    ]

    return det_df[is_in_bounding_box]


def filter_shape(det_df, shapefile, shapefile_no_pad):
    # Filter first using bounding box
    det_df = filter_latlon(det_df, shapefile.bounds)

    # Filter remaining points using shape (more expensive check)
    preped_shape = shapely.prepared.prep(shapefile)

    is_contained = [
        preped_shape.contains(shapely.geometry.Point(lon, lat))
        for lon, lat in zip(det_df.longitude, det_df.latitude)
    ]

    preped_shape = shapely.prepared.prep(shapefile_no_pad)

    is_contained_no_pad = [
        preped_shape.contains(shapely.geometry.Point(lon, lat))
        for lon, lat in zip(det_df.longitude, det_df.latitude)
    ]

    det_df["within_region"] = is_contained_no_pad

    return det_df[is_contained]


def to_geodataframe(det_df):
    points = [
        shapely.geometry.Point(lon, lat)
        for lon, lat in zip(det_df.longitude, det_df.latitude)
    ]

    geo_df = geopandas.GeoDataFrame(det_df, geometry=points)
    geo_df.crs = {"init": "epsg:4326"}  # WGS84 is CRS used by VIIRS

    return geo_df
