from typing import cast
import numpy as np
import h5py
import datetime as dt
import time


def compute_vpd(temp, humid):
    vpsat = 0.611 * 10 ** ((7.5 * temp - 2048.6) / (temp - 35.85))
    vp = humid / 100 * vpsat
    vpd = vpsat - vp

    # Ensure it is non-negative
    vpd[vpd < 0] = 0

    return vpd


def load_data(
    parameters,
    file_path,
    ind_cut=np.iinfo(np.int32).max,
    use_filters=True,
):
    use_land_cover = parameters["data"]["use_land_cover"]
    use_topography = parameters["data"].get("use_topography", False)
    use_weather = parameters["data"]["use_weather"]

    learned_normalization_coefficients = parameters["data"].get(
        "normalization_coefficients", None
    )

    print(parameters["data"]["data_dir"])
    print(parameters["data"]["file_name"])

    if learned_normalization_coefficients is not None:
        print("Normalization", learned_normalization_coefficients)

    normalization_coefficients = (
        {}
        if learned_normalization_coefficients is None
        else learned_normalization_coefficients
    )

    with h5py.File(file_path, "r") as hf:

        # data_x = hf["observed"][:].astype(np.float32)
        # data_y = hf["target"][:, 0].astype(np.long)

        if use_filters:
            filters = cast(np.ndarray, cast(h5py.Dataset, hf["filters"])[:])
            filter_sel = np.nonzero(filters == 0)
            inds = np.arange(len(filters))[filter_sel][:ind_cut]
        else:
            inds = np.arange(hf["filters"].shape[0])

        forecast_horizon = parameters["data"]["forecast_horizon"]
        print("Forecast Horizon", forecast_horizon)

        start_time = time.time()
        data_x = hf["observed"][inds].astype(np.float32)

        print("Det", data_x.shape)

        data_y = hf["target"][inds][:, :forecast_horizon].astype(np.long)
        print("Data Y Shape", data_y.shape)
        data_y = np.any(data_y, axis=1)
        print("Data Y Shape", data_y.shape)

        datetimes = hf["datetimes"][inds].astype("datetime64[ns]")
        print("Load took", time.time() - start_time)

        if use_topography:
            data_dem = hf["land_cover"][inds, 0:1].astype(np.float32)
            data_dem[np.isnan(data_dem)] = 0

            if learned_normalization_coefficients is None:
                mean, std = (
                    np.mean(data_dem, axis=(1, 2, 3))[:, None, None, None],
                    np.std(data_dem),
                )

                print("Topo", mean.shape, std.shape)
                normalization_coefficients["topography"] = (mean, std)
            else:
                mean = np.mean(data_dem, axis=(1, 2, 3))[:, None, None, None]
                _, std = learned_normalization_coefficients["topography"]

            data_dem -= mean
            data_dem /= std

        if use_land_cover:
            data_lc = hf["land_cover"][inds, 1:].astype(np.float32)
            print(data_lc.shape)
            data_lc[np.isnan(data_lc)] = 0
            if learned_normalization_coefficients is None:
                mean, std = (
                    np.mean(data_lc),
                    np.std(data_lc),
                )

                print("LC", mean, std)
                normalization_coefficients["land_cover"] = (mean, std)
            else:
                mean, std = learned_normalization_coefficients["land_cover"]

            # std[std == 0] = 1
            data_lc = (data_lc - mean) / std
            data_lc[np.isnan(data_lc)] = 0

        if use_weather:
            # data_weather: np.ndarray = hf["meteorology"][
            #     inds, : forecast_horizon * 10
            # ].astype(np.float32)

            data_weather: np.ndarray = hf["meteorology"][inds, :5].astype(np.float32)

            # vpd = compute_vpd(temp / 7, humid / 7)
            # print(vpd.shape)

            data_weather = np.stack(
                [
                    data_weather[:, 0],
                    data_weather[:, 1],
                    data_weather[:, 2],
                    data_weather[:, 3],
                ],
                axis=1,
            )

            print(data_weather.shape)

            if learned_normalization_coefficients is None:
                mean, std = (
                    np.mean(data_weather, axis=(0, 2, 3))[:, None, None],
                    np.std(data_weather, axis=(0, 2, 3))[:, None, None],
                )

                print("Met", mean, std)
                normalization_coefficients["meteorology"] = (mean, std)
            else:
                mean, std = learned_normalization_coefficients["meteorology"]

            std[std == 0] = 1
            data_weather = (data_weather - mean) / std
            data_weather[np.isnan(data_weather)] = 0

    num_lags = parameters["data"]["num_lags"]
    data_x = data_x[:, 0 : num_lags + 1]

    if use_topography:
        print("Dem", data_dem.shape)
        data_x = np.concatenate([data_x, data_dem], axis=1)

    if use_land_cover:
        print("LC", data_lc.shape)
        data_x = np.concatenate([data_x, data_lc], axis=1)

    if use_weather:
        # reshape: combine time step & weather variables
        data_weather = data_weather.repeat(30, axis=2).repeat(30, axis=3)
        s0, s1, s2, s3 = data_weather.shape
        print(data_weather.shape)
        parameters["model"]["weather_time_steps"] = s1
        parameters["model"]["weather_variables"] = s2
        # data_weather = data_weather.reshape(s0, s1 * s2, s3, s4)

        print("weather", data_weather.shape)

        data_x = np.concatenate([data_x, data_weather], axis=1)

    print("shape", np.shape(data_x))
    print("shape", np.shape(data_y))
    print("shape", np.shape(datetimes))

    parameters["model"]["num_classes"] = len(np.unique(data_y))

    return data_x, data_y, datetimes, normalization_coefficients
