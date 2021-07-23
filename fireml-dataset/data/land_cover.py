from pathlib import Path
import geopandas as gpd

import fireml.helpers.file_io as fio

DIR_LAND_COVER = "land_cover/"
DIR_VEGETATION = "vegetation/"
DIR_FUEL = "fuel/"
DIR_TOPOGRAPHY = "topography/"

DIR_TO_TIFF = "to_tiff/"

DATASET_LANDFIRE_EVT = "landfire_evt"

DATASET_LANDFIRE_ASP = "landfire_asp"
DATASET_LANDFIRE_DEM = "landfire_dem"
DATASET_LANDFIRE_SLP = "landfire_slp"

DATASET_LANDFIRE_CBD = "landfire_cbd"
DATASET_LANDFIRE_CBH = "landfire_cbh"
DATASET_LANDFIRE_CC = "landfire_cc"
DATASET_LANDFIRE_CH = "landfire_ch"

DATASET_TO_CATEGORY = {
    DATASET_LANDFIRE_EVT: DIR_VEGETATION,
    DATASET_LANDFIRE_ASP: DIR_TOPOGRAPHY,
    DATASET_LANDFIRE_DEM: DIR_TOPOGRAPHY,
    DATASET_LANDFIRE_SLP: DIR_TOPOGRAPHY,
    DATASET_LANDFIRE_CBD: DIR_FUEL,
    DATASET_LANDFIRE_CBH: DIR_FUEL,
    DATASET_LANDFIRE_CC: DIR_FUEL,
    DATASET_LANDFIRE_CH: DIR_FUEL,
}

DATASET_TO_ABREV = {
    DATASET_LANDFIRE_EVT: "EVT",
    DATASET_LANDFIRE_ASP: "ASP",
    DATASET_LANDFIRE_DEM: "DEM",
    DATASET_LANDFIRE_SLP: "SLP",
    DATASET_LANDFIRE_CBD: "CBD",
    DATASET_LANDFIRE_CBH: "CBH",
    DATASET_LANDFIRE_CC: "CC",
    DATASET_LANDFIRE_CH: "CH",
}

DATASET_TO_VERSION = {
    DATASET_LANDFIRE_EVT: "130",
    DATASET_LANDFIRE_ASP: "120",
    DATASET_LANDFIRE_DEM: "120",
    DATASET_LANDFIRE_SLP: "120",
    DATASET_LANDFIRE_CBD: "130",
    DATASET_LANDFIRE_CBH: "130",
    DATASET_LANDFIRE_CC: "130",
    DATASET_LANDFIRE_CH: "130",
}

LANDFIRE_VEG_LAYERS = [DATASET_LANDFIRE_EVT]
LANDFIRE_TOPO_LAYERS = [
    DATASET_LANDFIRE_ASP,
    DATASET_LANDFIRE_DEM,
    DATASET_LANDFIRE_SLP,
]
LANDFIRE_FUEL_LAYERS = [
    DATASET_LANDFIRE_CBD,
    DATASET_LANDFIRE_CBH,
    DATASET_LANDFIRE_CC,
    DATASET_LANDFIRE_CH,
]

LANDFIRE_INVALID = (-32768, -9999)

FILE_FMT_LAND_COVER_DIR = "US_{}{}/"
FILE_FMT_LAND_COVER_FILENAME = "US_{}{}.tiff"


def make_land_cover_dir_path(dataset):
    category = DATASET_TO_CATEGORY[dataset]

    return Path(DIR_LAND_COVER) / category / dataset


def load_land_cover(dir_path, dataset):
    version = DATASET_TO_VERSION[dataset]
    abreviation = DATASET_TO_ABREV[dataset]

    file_name = FILE_FMT_LAND_COVER_FILENAME.format(version, abreviation)
    path = dir_path / make_land_cover_dir_path(dataset) / DIR_TO_TIFF / file_name

    return fio.load_raster(path)


def load_ocean_shape(data_dir):
    ocean_fn = data_dir / fio.DIR_RAW / "boundaries/ne_10m_ocean/ne_10m_ocean.shp"
    oceans = gpd.read_file(ocean_fn).to_crs("epsg:4326").geometry[0][0]

    return oceans


def apply_ocean_mask(data, ocean_mask, dataset):
    land_cover_category = DATASET_TO_CATEGORY[dataset]

    if land_cover_category == DIR_VEGETATION:
        # Anything in the ocean (that isn't already classified) is marked as ocean (3292)
        data[ocean_mask] = 3292  # Open water class

        # Anything  that is not in the ocean (and isn't already classified) is marked as land (3000)
        data[data == -9999] = 3000
        data[data == -32768] = 3000

    elif land_cover_category == DIR_FUEL or land_cover_category == DIR_TOPOGRAPHY:
        # There are no fuels in the ocean AND the elevation/slope/aspect are all zero
        data[ocean_mask] = 0

        # Mark non-ocean unclassified locations as unkown (to be handled by the training code) -- unify -9999 & -32768
        data[data == -9999] = -32768

    else:
        raise ValueError(f"Invalid dataset category {land_cover_category}.")

    return data
