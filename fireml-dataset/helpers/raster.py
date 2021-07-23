import json

import numpy as np
import rasterio
import rasterio.io as rio
import rasterio.coords as rcoords
import geopandas as gpd

import rasterio.warp as rwarp
import rasterio.mask as rmask


def write_memfile(data, metadata):
    memfile = rio.MemoryFile()
    with memfile.open(**metadata) as dest:
        dest.write(data)

    return memfile


def get_bounds(metadata):
    ul = metadata["transform"] * (0, 0)
    br = metadata["transform"] * (metadata["width"], metadata["height"])

    return rcoords.BoundingBox(left=ul[0], top=ul[1], right=br[0], bottom=br[1])


def get_shape_features(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def filter_by_region(
    raster, region, region_crs, new_dtype=None, fill_val=None, pad_width=None
):
    """Filter raster using region boundary."""

    gdf = gpd.GeoDataFrame({"geometry": region}, index=[0], crs=region_crs)
    gdf = gdf.to_crs(crs=raster.crs.data)

    shapes = get_shape_features(gdf)

    out_img, out_transform = rmask.mask(
        dataset=raster,
        shapes=shapes,
        all_touched=True,
        nodata=fill_val,
        crop=True,
        pad=pad_width is not None,
        pad_width=pad_width,
    )
    out_meta = raster.meta.copy()

    if new_dtype is not None:
        out_meta["dtype"] = new_dtype

    out_img = out_img.astype(out_meta["dtype"])

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "crs": raster.crs,
        }
    )

    return out_img, out_meta


def project(data, metadata, projection):
    """Project raster to new crs."""

    transform, width, height = rwarp.calculate_default_transform(
        metadata["crs"],
        projection,
        metadata["width"],
        metadata["height"],
        *get_bounds(metadata)
    )

    projected_data = np.full(
        (metadata["count"], height, width),
        fill_value=metadata["nodata"],
        dtype=metadata["dtype"],
    )

    projected_metadata = metadata.copy()
    projected_metadata.update(
        {
            "crs": projection,
            "transform": transform,
            "width": width,
            "height": height,
        }
    )

    rwarp.reproject(
        source=data,
        destination=projected_data,
        src_transform=metadata["transform"],
        src_crs=metadata["crs"],
        dst_transform=projected_metadata["transform"],
        dst_crs=projected_metadata["crs"],
        resampling=rwarp.Resampling.nearest,
    )

    return projected_data, projected_metadata


def project_raster_to_raster(raster, dest_fn, projection):
    """Project raster to new crs."""

    transform, width, height = rwarp.calculate_default_transform(
        metadata["crs"],
        projection,
        metadata["width"],
        metadata["height"],
        *get_bounds(metadata)
    )

    projected_data = np.full((metadata["count"], height, width), np.nan)

    projected_metadata = metadata.copy()
    projected_metadata.update(
        {
            "crs": projection,
            "transform": transform,
            "width": width,
            "height": height,
        }
    )

    rwarp.reproject(
        source=data,
        destination=projected_data,
        src_transform=metadata["transform"],
        src_crs=metadata["crs"],
        dst_transform=projected_metadata["transform"],
        dst_crs=projected_metadata["crs"],
        resampling=rwarp.Resampling.nearest,
    )


def pad_raster(data, meta, pad_width=1000, mode="constant", constant_values=-32768):
    out_img, pad_transform = rasterio.pad(
        data[0],
        meta["transform"],
        pad_width=pad_width,
        mode=mode,
        constant_values=constant_values,
    )

    out_meta = meta.copy()
    out_meta["transform"] = pad_transform
    out_meta["width"] = out_img.shape[1]
    out_meta["height"] = out_img.shape[0]

    return out_img[None], out_meta


def build_raster_mask(raster, meta, region, region_crs):
    gdf = gpd.GeoDataFrame({"geometry": region}, index=[0], crs=region_crs)
    gdf = gdf.to_crs(crs=raster.crs.data)

    shapes = get_shape_features(gdf)

    masked, transform, window = rmask.raster_geometry_mask(
        dataset=raster,
        shapes=shapes,
        all_touched=True,
        crop=False,
    )

    new_meta = meta.copy()
    new_meta["transform"] = transform

    return masked, new_meta
