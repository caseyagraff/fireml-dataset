import os
import json
import numpy as np

import rasterio
import fiona
from dagma import create_node
import pickle

from fireml.helpers import file_io as fio, raster as rast, projections as proj
from fireml.data import land_cover as lcov, regions as reg

DIR_PROJECTED = "projected/"
DIR_COMBINED = "combined/"

FILE_FMT_LAND_COVER_PROJECTED = "US_{}{}_{}_{}.pkl"
FILE_FMT_LAND_COVER_COMBINED = "US_{}_{}_{}.hdf5"


class Raster:
    def __init__(self, name, data, metadata):
        self.name = name
        self.data = data
        self.metadata = metadata


def make_projected_path(params):
    foreach_var = params["%foreach"]

    if foreach_var is None:
        return None

    dataset = foreach_var.name

    dir_path = (
        params["data_dir"]
        / fio.DIR_INTERIM
        / lcov.make_land_cover_dir_path(dataset)
        / DIR_PROJECTED
    )
    dir_path.mkdir(parents=True, exist_ok=True)
    print("make")

    version = lcov.DATASET_TO_VERSION[dataset]
    abreviation = lcov.DATASET_TO_ABREV[dataset]

    path = dir_path / FILE_FMT_LAND_COVER_PROJECTED.format(
        version, abreviation, params["region"], params["projection"],
    )

    return path


def make_combined_path(params):
    dir_path = params["data_dir"] / fio.DIR_INTERIM / lcov.DIR_LAND_COVER / DIR_COMBINED
    dir_path.mkdir(parents=True, exist_ok=True)

    datasets = params["land_cover.datasets"]
    abreviations = "-".join(sorted([lcov.DATASET_TO_ABREV[d] for d in datasets]))

    path = dir_path / FILE_FMT_LAND_COVER_COMBINED.format(
        abreviations, params["region"], params["projection"],
    )

    print(path, params)
    with open("test.pkl", "wb") as fout:
        pickle.dump((params, None, None), fout)

    return path


@create_node
def filter_land_cover_by_region(data_dir, dataset, region):
    print("--- filter", dataset)
    region, _ = region

    dir_path = data_dir / fio.DIR_INTERIM
    with rasterio.Env(), lcov.load_land_cover(dir_path, dataset) as raster:
        data, metadata = rast.filter_by_region(
            raster, region, fiona.crs.from_epsg(reg.REGION_BASE_EPSG_CODE)
        )

    # Pad landcover
    data, metadata = rast.pad_raster(data, metadata)

    # Compute water mask
    memfile = rast.write_memfile(data, metadata)

    ocean_region = lcov.load_ocean_shape(data_dir)

    with rasterio.Env(), memfile.open() as raster:
        ocean_mask, ocean_meta = rast.build_raster_mask(
            raster, metadata, ocean_region, fiona.crs.from_epsg("4326")
        )

    memfile.close()

    ocean_mask = ((data == -9999) | (data == -32768)) & (~ocean_mask)

    # Modify data using ocean_mask (dependent on type of land cover)
    data = lcov.apply_ocean_mask(data, ocean_mask, dataset)

    return Raster(dataset, data, metadata)


# TODO: Add saving/loading once dagma correctly supports loading/check foreach nodes
@create_node
def project_land_cover(land_cover_raster, region_name, projection):
    print("--- project", land_cover_raster.name)

    data, metadata = rast.project(
        land_cover_raster.data,
        land_cover_raster.metadata,
        proj.PROJECTION_DICT[projection],
    )

    ret = Raster(land_cover_raster.name, data, metadata)

    dataset_name = land_cover_raster.name
    category = lcov.DATASET_TO_CATEGORY[dataset_name]
    version = lcov.DATASET_TO_VERSION[dataset_name]
    abrev = lcov.DATASET_TO_ABREV[dataset_name]

    path = f"data/interim/land_cover/{category}/{dataset_name}/projected/US_{version}{abrev}_{region_name}_{projection}.hdf5"
    metadata["transform"] = metadata["transform"].to_gdal()
    fio.save_hdf5(path, {land_cover_raster.name: (data, metadata, data.dtype)})

    return ret


# TODO : Add load/save
@create_node(file_path=make_combined_path, load=fio.load_hdf5, hash_alg=None)
def combine_land_cover(land_cover_rasters, data_dir, datasets, region, projection):
    print("--- combine", len(land_cover_rasters))

    combined = {r.name: (r.data, r.metadata, r.data.dtype) for r in land_cover_rasters}

    params = {
        "data_dir": data_dir,
        "land_cover.datasets": datasets,
        "region": region,
        "projection": projection,
    }
    combined_path = make_combined_path(params)
    fio.save_hdf5(combined_path, combined)

    return combined


def get_land_cover(region):
    land_cover_raster = filter_land_cover_by_region(
        "data_dir", "land_cover.datasets", region, foreach="land_cover.datasets"
    )
    land_cover_raster = project_land_cover(
        land_cover_raster, "region", "projection", foreach=0
    )

    land_cover_rasters = combine_land_cover(
        land_cover_raster, "data_dir", "land_cover.datasets", "region", "projection"
    )

    return land_cover_rasters
