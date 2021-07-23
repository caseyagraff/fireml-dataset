import xarray as xr
import pandas as pd
import geopandas as gpd
import pickle
import h5py
import rasterio
from pathlib import Path

DIR_RAW = "raw/"
DIR_INTERIM = "interim/"
DIR_PROCESSED = "processed/"


def load_pickle(path):
    with open(path, "rb") as f_in:
        return pickle.load(f_in)  # nosec


def save_pickle(path, data):
    with open(path, "wb") as f_out:
        pickle.dump(data, f_out)


def load_dataframe_csv(path):
    return pd.read_csv(path)


def save_dataframe_csv(path, df):
    df.to_csv(path)


def load_raster(path):
    with rasterio.Env():
        return rasterio.open(path)


def load_raster_all(path):
    with rasterio.Env(), rasterio.open(path) as raster:
        return raster.read(), raster.meta


def save_raster(path, value):
    data, meta = value
    with rasterio.open(path, "w", **meta) as f_out:
        f_out.write(data)


def load_hdf5(path):
    with h5py.File(path, "r") as hf:
        return {k: (hf[k][:], hf[k].attrs, hf[k].dtype) for k in hf.keys()}


def save_hdf5(path, datasets):
    with h5py.File(path, "w") as f_out:
        for name, (val, attrs, dtype) in datasets.items():
            ds = f_out.create_dataset(name, data=val, dtype=dtype)

            for k, v in attrs.items():
                ds.attrs.create(k, v)


def load_netcdf(path):
    return xr.open_dataset(path)


def save_netcdf(path, xds):
    xds.to_netcdf(path, format="NETCDF4", engine="netcdf4")
