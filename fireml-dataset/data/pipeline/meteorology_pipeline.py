import rasterio
import fiona
import tempfile
import datetime as dt
import numpy as np
import tqdm
import xarray as xr
from dagma import create_node
import pickle

from fireml.helpers import (
    raster as rast,
    file_io as fio,
    dates as du,
    projections as proj,
)
import fireml.data.meteorology as meteo
import fireml.data.regions as reg

FILE_FMT_RAP_REFRESH_13KM_REANAL_AGGREGATED = "rap_130_{}_{}_{}.nc"
FILE_FMT_RAP_REFRESH_13KM_REANAL_AGGREGATED_ALL = "rap_130_{}_{}_{}_{}.nc"
DIR_AGGREGATED = "aggregated/"


def get_netcdf_path(params):
    foreach_var = params["%foreach"]

    if foreach_var is None:
        return None

    # year = str(foreach_var.dates[0].year)
    # month = str(foreach_var.dates[0].month).zfill(2)
    year = str(foreach_var[0].year)
    month = str(foreach_var[0].month).zfill(2)

    year_month = "".join([year, month])

    dir_path = (
        params["data_dir"]
        / fio.DIR_INTERIM
        / meteo.DIR_METEOROLOGY
        / meteo.DIR_RAP_REFRESH_13KM_REANAL
        / year
        / year_month
    )
    dir_path.mkdir(parents=True, exist_ok=True)

    path = dir_path / FILE_FMT_RAP_REFRESH_13KM_REANAL_AGGREGATED.format(
        year_month, params["region"], params["projection"]
    )

    return path


def make_combined_all_path(params):
    dir_path = (
        params["data_dir"]
        / fio.DIR_INTERIM
        / meteo.DIR_METEOROLOGY
        / meteo.DIR_RAP_REFRESH_13KM_REANAL
        / DIR_AGGREGATED
    )
    dir_path.mkdir(parents=True, exist_ok=True)

    path = dir_path / FILE_FMT_RAP_REFRESH_13KM_REANAL_AGGREGATED_ALL.format(
        params["meteorology.start_datetime"].strftime("%Y-%m-%d"),
        params["meteorology.end_datetime"].strftime("%Y-%m-%d"),
        params["region"],
        params["projection"],
    )

    return path


@create_node
def get_month_ranges(start_datetime, end_datetime):
    """Get all months between start and end (inclusive) that have at least one full day in the range."""
    print("range", start_datetime, end_datetime)
    months = du.month_range(start_datetime.date(), end_datetime.date())

    months = [
        dt.datetime.combine(m, dt.datetime.min.time(), tzinfo=start_datetime.tzinfo)
        for m in months
    ]

    start_months = [start_datetime] + months[1:]
    end_months = months[1:] + [end_datetime]

    month_ranges = [(start, end) for (start, end) in zip(start_months, end_months)]

    return month_ranges


# TODO: Load and combine CSV metadata if available (esp. for EVT)
# @create_node
def combine_meteorology(data_dir, dataset, layers, month_range):
    print("combine", month_range)
    data_dir = data_dir / fio.DIR_RAW

    start_datetime, end_datetime = month_range
    datetimes = list(
        du.date_range(start_datetime, end_datetime, increment=dt.timedelta(hours=1))
    )

    missing_datetimes = np.zeros(len(datetimes), dtype=bool)
    num_layers = len(layers)

    # Choose first rap date to get rap metadata from (could choose any valid rap file)
    path = data_dir / meteo.make_meteorology_file_path(
        dataset, dt.datetime(2012, 5, 9, 0), offset=0
    )
    metadata, layer_to_units = meteo.get_metadata(path, layers)
    metadata["count"] = len(datetimes) * num_layers
    metadata["dtype"] = "float32"

    data = np.full(
        (metadata["count"], metadata["height"], metadata["width"]),
        fill_value=np.nan,
        dtype=metadata["dtype"],
    )

    # Load all meteorology into data array
    for i, datetime in enumerate(tqdm.tqdm(datetimes)):

        layer_vals = meteo.load_meteorology(
            data_dir, dataset, layers, datetime, offset=0
        )

        if layer_vals is not None:
            for j, layer in layer_vals:
                data[i * num_layers + j] = layer.values[::-1]
        else:
            missing_datetimes[i] = True

    # temp_file = tempfile.NamedTemporaryFile(dir='/srv/disk00/graffc/tmp/')
    # with rasterio.Env() :
    #    with rasterio.open(temp_file.name, 'w', **metadata) as raster:
    #        raster.write(data)

    # TODO: Currently using temp_file to conserve memory usage, once foreach supports correct execution order
    # (the full chain for each value is run consecutively), then we can switch to the non-tempfile approach
    return meteo.MultiDateRaster(
        data, metadata, datetimes, missing_datetimes, layer_to_units
    )
    # return temp_file, datetimes, missing_datetimes, layer_to_units


# @create_node
def filter_meteorology_by_shape(md_raster, region):
    region, _ = region  # Unpack region, region_no_pad

    combined_memfile = rast.write_memfile(md_raster.data, md_raster.metadata)
    # temp_file, dates, missing, layer_to_units = md_raster

    # with rasterio.Env(), rasterio.open(temp_file.name) as raster:
    with rasterio.Env(), combined_memfile.open() as raster:
        data, metadata = rast.filter_by_region(
            raster,
            region,
            fiona.crs.from_epsg(reg.REGION_BASE_EPSG_CODE),
            new_dtype="float32",
            fill_val=np.nan,
        )

    md_raster = meteo.MultiDateRaster(
        data, metadata, md_raster.dates, md_raster.missing, md_raster.layer_to_units
    )
    # md_raster = meteo.MultiDateRaster(data, metadata, dates, missing, layer_to_units)

    return md_raster


# @create_node
def project_meteorology(md_raster, projection):
    print("=== project")
    md_raster.metadata["nodata"] = np.nan
    data, metadata = rast.project(
        md_raster.data, md_raster.metadata, proj.PROJECTION_DICT[projection]
    )

    md_raster = meteo.MultiDateRaster(
        data, metadata, md_raster.dates, md_raster.missing, md_raster.layer_to_units
    )

    return md_raster


# @create_node(file_path=get_netcdf_path, save=fio.save_netcdf, load=fio.load_netcdf)
def to_netcdf(md_raster, layers):
    return meteo.to_netcdf(md_raster, layers)


@create_node(
    file_path=get_netcdf_path,
    save=fio.save_netcdf,
    load=fio.load_netcdf,
    mem_cache=False,
    # volatile_vars={"meteorology.start_datetime", "meteorology.end_datetime"},
)
def build_meteorology(data_dir, dataset, layers, month_range, region, projection):
    md_raster = combine_meteorology(data_dir, dataset, layers, month_range)
    md_raster = filter_meteorology_by_shape(md_raster, region)
    md_raster = project_meteorology(md_raster, projection)

    md_raster = to_netcdf(md_raster, layers)

    return md_raster


@create_node(
    file_path=make_combined_all_path,
    save=fio.save_netcdf,
    load=fio.load_netcdf,
    hash_alg=None,
)
def combine_all_rasters(multidate_rasters):
    new_dataset = xr.concat(multidate_rasters, dim="time")

    zero_mask = new_dataset.temperature[0] == 0

    for k in list(new_dataset):
        if k == "missing":
            continue
        new_dataset[k].values[:, zero_mask] = np.nan

    return new_dataset


def get_meteorology(region):
    month_ranges = get_month_ranges(
        "meteorology.start_datetime", "meteorology.end_datetime"
    )

    # multidate_raster = combine_meteorology("data_dir", "meteorology.dataset", "meteorology.layers", month_ranges, foreach=3)
    # multidate_raster = filter_meteorology_by_shape(multidate_raster, region, foreach=0)
    # multidate_raster = project_meteorology(multidate_raster, "projection", foreach=0)

    # rap_data = to_netcdf(multidate_raster, "meteorology.layers", foreach=0)

    rap_data = build_meteorology(
        "data_dir",
        "meteorology.dataset",
        "meteorology.layers",
        month_ranges,
        region,
        "projection",
        foreach=3,
    )

    combined_rap = combine_all_rasters(rap_data)

    return combined_rap
