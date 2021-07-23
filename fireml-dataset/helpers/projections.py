# USA Albers Equal Area
PROJECTION_NAME_USA_AEA = "usa_aea"
PROJECTION_CRS_USA_AEA = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs=True"

# USA Albers Equal Area USGS
PROJECTION_NAME_USA_AEA_USGS = "usa_aea_usgs"
PROJECTION_CRS_USA_AEA_USGS = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs=True"

PROJECTION_DICT = {
    PROJECTION_NAME_USA_AEA: PROJECTION_CRS_USA_AEA,
    PROJECTION_NAME_USA_AEA_USGS: PROJECTION_CRS_USA_AEA_USGS,
}
