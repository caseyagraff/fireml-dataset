import geopandas as gpd
import fireml.helpers.file_io as fio
from shapely.ops import unary_union


DIR_BOUNDARIES = "boundaries/"

REGION_BASE_EPSG_CODE = 4326
REGION_BASE_PROJECTION = f"epsg:{REGION_BASE_EPSG_CODE}"

REGION_NAME_CA = "california"
REGION_SHAPEFILE_PATH_CA = "CA_State/CA_State_TIGER2016.shp"
REGION_IND_CONTIGUOUS_CALIFORNIA = 6

REGION_NAME_OR = "oregon"
REGION_SHAPEFILE_PATH_OR = "tl_2016_us_state/tl_2016_us_state.shp"
REGION_SELECT_OR = ("NAME", "Oregon")

REGION_NAME_WA = "washington"
REGION_SHAPEFILE_PATH_WA = "tl_2016_us_state/tl_2016_us_state.shp"
REGION_SELECT_WA = ("NAME", "Washington")

states = [
    "Alabama",
    "AL",
    "Arizona",
    "AZ",
    "Arkansas",
    "AR",
    "California",
    "CA",
    "Colorado",
    "CO",
    "Connecticut",
    "CT",
    "Delaware",
    "DE",
    "District of Columbia",
    "DC",
    "Florida",
    "FL",
    "Georgia",
    "GA",
    "Idaho",
    "ID",
    "Illinois",
    "IL",
    "Indiana",
    "IN",
    "Iowa",
    "IA",
    "Kansas",
    "KS",
    "Kentucky",
    "KY",
    "Louisiana",
    "LA",
    "Maine",
    "ME",
    "Montana",
    "MT",
    "Nebraska",
    "NE",
    "Nevada",
    "NV",
    "New Hampshire",
    "NH",
    "New Jersey",
    "NJ",
    "New Mexico",
    "NM",
    "New York",
    "NY",
    "North Carolina",
    "NC",
    "North Dakota",
    "ND",
    "Ohio",
    "OH",
    "Oklahoma",
    "OK",
    "Oregon",
    "OR",
    "Maryland",
    "MD",
    "Massachusetts",
    "MA",
    "Michigan",
    "MI",
    "Minnesota",
    "MN",
    "Mississippi",
    "MS",
    "Missouri",
    "MO",
    "Pennsylvania",
    "PA",
    "Rhode Island",
    "RI",
    "South Carolina",
    "SC",
    "South Dakota",
    "SD",
    "Tennessee",
    "TN",
    "Texas",
    "TX",
    "Utah",
    "UT",
    "Vermont",
    "VT",
    "Virginia",
    "VA",
    "Washington",
    "WA",
    "West Virginia",
    "WV",
    "Wisconsin",
    "WI",
    "Wyoming",
    "WY",
]

REGION_NAME_US_CONT = "us_contiguous"
REGION_SHAPEFILE_PATH_US_CONT = "tl_2016_us_state/tl_2016_us_state.shp"
REGION_SELECT_US_CONT = ("NAME", states[::2])


REGIONS_PATH_LOOKUP = {
    REGION_NAME_CA: (REGION_SHAPEFILE_PATH_CA, REGION_IND_CONTIGUOUS_CALIFORNIA, None),
    REGION_NAME_OR: (REGION_SHAPEFILE_PATH_OR, None, REGION_SELECT_OR),
    REGION_NAME_WA: (REGION_SHAPEFILE_PATH_WA, None, REGION_SELECT_WA),
    REGION_NAME_US_CONT: (REGION_SHAPEFILE_PATH_US_CONT, None, REGION_SELECT_US_CONT),
}


def load_region_shapefile(file_path):
    gdf = gpd.read_file(file_path)

    return gdf.to_crs(REGION_BASE_PROJECTION)


def load_region(data_dir, region_name, region_padding):
    region_path, region_ind, region_select = REGIONS_PATH_LOOKUP[region_name]
    path = data_dir / fio.DIR_RAW / DIR_BOUNDARIES / region_path

    region = load_region_shapefile(path)
    region_geometry = None

    if region_select is not None:
        column, value = region_select

        if isinstance(value, str):
            region = region[region[column] == value]
            region_geometry = np.array(region.geometry)[0]
        else:
            region = region[region[column].isin(value)]
            region_geometry = unary_union(region.geometry)

    region = region_geometry[region_ind] if region_ind is not None else region_geometry

    if region_padding is not None:
        region_padded = region.buffer(region_padding)
    else:
        region_padded = None

    return region_padded, region
