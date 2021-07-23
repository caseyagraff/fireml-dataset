from .timing import timer
import numpy as np
from fireml.helpers import convex

import rasterio.features


def get_points(vals, thresh):
    return np.array(list(zip(*np.where(vals > thresh))))


@timer("Base perimeter loss: ")
def perimeter_loss(y_list, y_hat_list, alpha=0.1, thresh=0.05, is_weighted=False):
    total = 0
    examples = 0
    errors = []
    weights = []

    for ind, (y, y_hat) in enumerate(zip(y_list, y_hat_list)):
        points_y = get_points(y, 0.5)
        points_y_hat = get_points(y_hat, thresh)

        height = y.shape[0]
        width = y.shape[1]

        if len(points_y) < 3:
            continue

        if len(points_y_hat) < 3:
            examples += 1
            total += 1
            continue

        try:
            y_union, _ = convex.alpha_shape(points_y, alpha)
        except:
            y_union = None

        try:
            y_hat_union, _ = convex.alpha_shape(points_y_hat, alpha)
        except:
            y_hat_union = None

        if y_union is None or y_union.is_empty:
            y_rast = y > 0.5
        else:
            y_rast = rasterio.features.rasterize([y_union], out_shape=(height, width))

        if y_hat_union is None or y_hat_union.is_empty:
            y_hat_rast = y_hat > thresh
        else:
            y_hat_rast = rasterio.features.rasterize(
                [y_hat_union], out_shape=(height, width)
            )

        intersection = np.sum((y_rast == 1) & (y_hat_rast == 1))
        union = np.sum((y_rast == 1) | (y_hat_rast == 1))

        if union == 0:
            continue

        total += 1 - (intersection / union)
        examples += 1
        if is_weighted:
            errors.append(1 - (intersection / union))
            weights.append(np.sum(y_rast))

    return total, examples, errors, weights


@timer("New perimeter loss: ")
def new_perimeter_loss(
    x_hist_list, y_list, y_hat_list, alpha=0.1, thresh=0.05, is_weighted=False
):
    total = 0
    examples = 0
    errors = []
    weights = []

    for ind, (x_hist, y, y_hat) in enumerate(zip(x_hist_list, y_list, y_hat_list)):
        height = y.shape[0]
        width = y.shape[1]

        # x = np.sum(x_hist, axis=0) > 0
        # xy = np.sum([x, y], axis=0) > 0

        x = x_hist
        xy = np.sum([x, y], axis=0)

        points_x = get_points(x, 0.5)
        points_y = get_points(y, 0.5)
        points_xy = get_points(xy, 0.5)
        points_y_hat = get_points(y_hat, thresh)

        if len(points_y) < 3:
            continue

        if len(points_y_hat) < 3:
            total += 1
            examples += 1
            continue

        try:
            x_union, _ = convex.alpha_shape(points_x, alpha)
        except:
            x_union = None

        if x_union is None or x_union.is_empty:
            x_rast = x > 0.5
        else:
            x_rast = rasterio.features.rasterize([x_union], out_shape=(height, width))

        try:
            y_union, _ = convex.alpha_shape(points_y, alpha)
        except:
            y_union = None

        if y_union is None or y_union.is_empty:
            y_rast = y > 0.5
        else:
            y_rast = rasterio.features.rasterize([y_union], out_shape=(height, width))

        try:
            xy_union, _ = convex.alpha_shape(points_xy, alpha)
        except:
            xy_union = None

        if xy_union is None or xy_union.is_empty:
            xy_rast = xy > 0.5
        else:
            xy_rast = rasterio.features.rasterize([xy_union], out_shape=(height, width))

        xy_rast = xy_rast | x_rast | y_rast

        try:
            y_hat_union, _ = convex.alpha_shape(points_y_hat, alpha)
        except:
            y_hat_union = None

        if y_hat_union is None or y_hat_union.is_empty:
            rast_pred = y_hat > thresh
        else:
            rast_pred = rasterio.features.rasterize(
                [y_hat_union], out_shape=(height, width)
            )

        rast_delta = x_rast != xy_rast

        intersection = np.sum((rast_delta == 1) & (rast_pred == 1))
        union = np.sum((rast_delta == 1) | ((rast_pred == 1) & (x_rast != 1)))

        if union == 0:
            continue

        total += 1 - (intersection / union)
        examples += 1
        if is_weighted:
            errors.append(1 - (intersection / union))
            weights.append(np.sum(rast_delta))

    return total, examples, errors, weights
