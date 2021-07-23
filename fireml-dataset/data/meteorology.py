"""
Loading and manipulating meteorology data.
"""

from pathlib import Path
import json
import numpy as np

import rioxarray
import xarray as xr
import pygrib
import fiona
import pandas as pd
import rasterio

from fireml.helpers import grib, raster as rast

DIR_METEOROLOGY = "meteorology/"

DATASET_RAP_REFRESH_13KM_REANAL = "rap13_reanal"
DIR_RAP_REFRESH_13KM_REANAL = "rapid_refresh_13km/"
FILE_FMT_RAP_REFRESH_13KM_REANAL = "rap_130_{}_{}_{}.grb2"

RAP_REFRESH_INSTANT_LAYERS = [
    {"typeOfLevel": "surface", "name": "Wind speed (gust)"},
    {"typeOfLevel": "surface", "name": "Temperature"},
    {"typeOfLevel": "surface", "name": "Lightning"},
    {"typeOfLevel": "surface", "name": "Precipitation rate"},
    {"typeOfLevel": "surface", "name": "Surface pressure"},
    {"typeOfLevel": "surface", "name": "Snow depth"},
    {"typeOfLevel": "heightAboveGround", "name": "2 metre temperature"},
    {"typeOfLevel": "heightAboveGround", "name": "2 metre relative humidity"},
    {"typeOfLevel": "heightAboveGround", "name": "10 metre U wind component"},
    {"typeOfLevel": "heightAboveGround", "name": "10 metre V wind component"},
    {"typeOfLevel": "isobaricInhPa", "name": "U component of wind", "level": 900},
    {"typeOfLevel": "isobaricInhPa", "name": "V component of wind", "level": 900},
    {"typeOfLevel": "unknown", "name": "Low cloud cover"},
    {"typeOfLevel": "unknown", "name": "Medium cloud cover"},
    {"typeOfLevel": "unknown", "name": "High cloud cover"},
    {"typeOfLevel": "atmosphere", "name": "Total Cloud Cover"},
    {"name": "Moisture availability"},
]

RAP_REFRESH_INSTANT_LAYERS_SAMPLE = [
    {"typeOfLevel": "surface", "name": "Wind speed (gust)"},
    {"typeOfLevel": "surface", "name": "Temperature"},
    {"typeOfLevel": "surface", "name": "Precipitation rate"},
    {"typeOfLevel": "surface", "name": "Surface pressure"},
    {"typeOfLevel": "heightAboveGround", "name": "2 metre temperature"},
    {"typeOfLevel": "heightAboveGround", "name": "2 metre relative humidity"},
    {"typeOfLevel": "heightAboveGround", "name": "10 metre U wind component"},
    {"typeOfLevel": "heightAboveGround", "name": "10 metre V wind component"},
    {"typeOfLevel": "isobaricInhPa", "name": "U component of wind", "level": 900},
    {"typeOfLevel": "isobaricInhPa", "name": "V component of wind", "level": 900},
]


RAP_REFRESH_ACCUM_LAYERS = [
    {"typeOfLevel": "surface", "name": "Total Precipitation"},
    {"typeOfLevel": "surface", "name": "Total snowfall"},
    {"name": "Convective precipitation (water)"},
    {"name": "Large scale precipitation (non-convective)"},
]

RAP_REFRESH_DIMENSIONS = (337, 451)


class MultiDateRaster:
    def __init__(self, data, metadata, dates, missing, layer_to_units):
        self.data = data
        self.metadata = metadata
        self.dates = dates
        self.missing = missing
        self.layer_to_units = layer_to_units


def get_dimensions(dataset):
    return RAP_REFRESH_DIMENSIONS


def make_meteorology_file_path(dataset, datetime, offset):
    year = str(datetime.year)
    month = str(datetime.month).zfill(2)
    day = str(datetime.day).zfill(2)

    year_month = "".join([year, month])
    year_month_day = "".join([year, month, day])

    time = str(datetime.hour).zfill(2) + "00"
    offset = str(offset).zfill(3)

    return (
        Path(DIR_METEOROLOGY)
        / DIR_RAP_REFRESH_13KM_REANAL
        / year
        / year_month
        / year_month_day
        / FILE_FMT_RAP_REFRESH_13KM_REANAL.format(year_month_day, time, offset)
    )


def load_meteorology(dir_path, dataset, layers, datetime, offset):
    path = dir_path / make_meteorology_file_path(dataset, datetime, offset)

    try:
        with pygrib.open(str(path)) as grbs:
            layers, _ = grib.find_layers(grbs, layers)
            return layers
    except Exception as e:
        return None


def get_metadata(path, layers):
    with rasterio.Env(), rasterio.open(path) as raster:
        metadata = raster.meta.copy()

    with pygrib.open(str(path)) as grbs:
        layers, unfound_layers = grib.find_layers(grbs, layers)

        layers_to_units = {l.name: l.units for (_, l) in layers}
        layers_to_no_units = {l["name"]: "none" for (_, l) in unfound_layers}

    return metadata, {**layers_to_units, **layers_to_no_units}


def make_name(name):
    return name.lower().replace("(", "").replace(")", "").replace(" ", "_")


def make_layer(data, layer, layer_ind, num_layers, layer_to_units, dtype):
    layer_data = data[layer_ind::num_layers].astype(dtype)
    return (["time", "y", "x",], layer_data, {"units": layer_to_units[layer["name"]]})


def build_projected_xys(transform, height, width):
    xs = np.empty(width, dtype=np.float64)
    ys = np.empty(height, dtype=np.float64)

    for i in range(height):
        _, ys[i] = transform * (0.5, i + 0.5)  # +.5 to get value for center of pixel

    for i in range(width):
        xs[i], _ = transform * (i + 0.5, 0.5)  # +.5 to get value for center of pixel

    return xs, ys


def to_netcdf(md_raster, layers, dtype=np.float32):
    # Create spatial xs and ys
    xs, ys = build_projected_xys(
        md_raster.metadata["transform"],
        md_raster.metadata["height"],
        md_raster.metadata["width"],
    )

    # Create dictionary of each layer
    num_layers = len(layers)
    layers = {
        make_name(l["name"]): make_layer(
            md_raster.data, l, i, num_layers, md_raster.layer_to_units, dtype
        )
        for i, l in enumerate(layers)
    }

    layers["missing"] = (["time"], md_raster.missing)

    ds = xr.Dataset(
        layers,
        coords={"x": xs, "y": ys, "time": pd.to_datetime(md_raster.dates),},
        attrs={"transform": md_raster.metadata["transform"].to_gdal(),},
    )

    ds = ds.rio.write_crs(md_raster.metadata["crs"])

    return ds
