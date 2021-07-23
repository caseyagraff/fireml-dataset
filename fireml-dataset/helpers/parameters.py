from typing import Dict, Any
import os
import datetime as dt
from pathlib import Path
import uuid
import numpy as np

import yaml
import pytz

from fireml.models.neural.neural_helpers import OPTIMIZER_ADAM

from fireml.helpers.projections import PROJECTION_NAME_USA_AEA
from fireml.data.regions import REGION_NAME_CA
from fireml.data.fire_detections import DATASET_VIIRS_375M
from fireml.data.meteorology import (
    DATASET_RAP_REFRESH_13KM_REANAL,
    RAP_REFRESH_INSTANT_LAYERS_SAMPLE,
)
from fireml.data.land_cover import (
    LANDFIRE_VEG_LAYERS,
    LANDFIRE_TOPO_LAYERS,
    LANDFIRE_FUEL_LAYERS,
)

from .saving import PickleSaveLoadMixin

DEFAULT_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "general": {"seed": 123},
    "data": {
        "data_dir": Path("data"),
        "region": REGION_NAME_CA,
        "region_padding": 0.5,
        "projection": PROJECTION_NAME_USA_AEA,
        "fire_detections": {
            "dataset": DATASET_VIIRS_375M,
            "start_datetime": dt.datetime(2012, 1, 1, 0, 0, tzinfo=pytz.utc),
            "end_datetime": dt.datetime(2021, 1, 1, 0, 0, tzinfo=pytz.utc),
            "det_types": [0, 1, 2, 3],
        },
        "clusters": {
            "spatial_threshold_km": 10,
            "temporal_threshold_hours": 48,
        },
        "land_cover": {
            "datasets": LANDFIRE_TOPO_LAYERS
            + LANDFIRE_FUEL_LAYERS
            + LANDFIRE_VEG_LAYERS
        },
        "meteorology": {
            "dataset": DATASET_RAP_REFRESH_13KM_REANAL,
            "start_datetime": dt.datetime(2012, 5, 9, 0, 0, tzinfo=pytz.utc),
            "end_datetime": dt.datetime(2021, 1, 1, 0, 0, tzinfo=pytz.utc),
            "layers": RAP_REFRESH_INSTANT_LAYERS_SAMPLE,
        },
        "dataset": {
            "dataset_name": str(uuid.uuid4()),
            "start_datetime": dt.datetime(2012, 5, 9, 0, 0, tzinfo=pytz.utc),
            "end_datetime": dt.datetime(2021, 1, 1, 0, 0, tzinfo=pytz.utc),
        },
        # "components": ["fire_detections", "land_cover", "meteorology"],
        "components": ["meteorology"],
        "forecast_horizon": 1,
    },
    "preprocessing": {
        "standardize": False,
        "detrend": False,
    },
    "model": {
        "cuda_device": 0,
        "l1_weight": 0.0,
        "l2_weight": 0.0,
        "learning_rate": 1e-2,
        "max_num_epochs": 10,
        "optimizer": OPTIMIZER_ADAM,
        "momentum": 0.9,
        "dropout_rate": 0.0,
        "batch_size": 5,
        "shuffle": True,
        "criterion": "cross_entropy",
        "activation_function": "relu",
        "early_stopping": {"use": False, "patience": 0},
        "logging_interval": 1,
        "isotropic_kernel": False,
        "rotation_type": "radial",
    },
    "results": {
        "save_dir": "results/",
        "run_name": "run",
        "save": True,
        "save_model": True,
        "save_predictions": False,
    },
}

REQUIRED_PARAMETERS = {"model": ["model_name"]}

SAVE_DATETIME_FMT = "%Y%m%dT%H%M%S.%f"


class Parameters(PickleSaveLoadMixin):
    def __init__(self, parameters):
        Parameters.check_required_params(parameters)
        self.parameters = Parameters.add_default_params(parameters)

    @staticmethod
    def check_required_params(parameters):
        for param_type, required in REQUIRED_PARAMETERS.items():
            try:
                _ = [parameters[param_type][r] for r in required]
            except KeyError as e:
                raise ValueError(
                    f'Missing required parameter: "{param_type}: {e.args[0]}'
                )

    @staticmethod
    def add_default_params(parameters):
        # Add constant defaults
        updated_parameters = DEFAULT_PARAMETERS.copy()

        for param_type in parameters:
            if param_type in updated_parameters:
                if parameters[param_type] is None:
                    continue

                updated_parameters[param_type].update(parameters[param_type])
            else:
                if parameters[param_type] is None:
                    raise ValueError(
                        f'Param type "{param_type}" must be non-null if defaults aren\'t set.'
                    )

                updated_parameters[param_type] = parameters[param_type]

        # Add dynamic defaults
        if "experiment_name" not in updated_parameters["results"]:
            updated_parameters["results"]["experiment_name"] = updated_parameters[
                "model"
            ]["model_name"]

        if "seed" not in updated_parameters["data"]:
            updated_parameters["data"]["seed"] = updated_parameters["general"]["seed"]

        if "seed" not in updated_parameters["model"]:
            updated_parameters["model"]["seed"] = updated_parameters["general"]["seed"]

        updated_parameters["results"][
            "experiment_started"
        ] = dt.datetime.now().strftime(SAVE_DATETIME_FMT)

        return updated_parameters

    def __getitem__(self, item):
        return self.parameters[item]

    def make_save_path(self, suffix=""):
        file_name = "_".join(
            [
                self.parameters["results"]["run_name"],
                self.parameters["results"]["experiment_started"] + suffix,
            ]
        )

        return os.path.join(
            self.parameters["results"]["save_dir"],
            self.parameters["results"]["experiment_name"],
            file_name,
        )

    @staticmethod
    def handle_imports(parameters, path):
        parameters = parameters.copy()

        for k, v in parameters.items():
            if isinstance(v, str) and v.startswith("import("):
                import_path = Path(v.split("import(")[1][:-1])

                if import_path.is_absolute():
                    parameters[k] = Parameters.__parse(import_path)[k]
                else:
                    parameters[k] = Parameters.__parse(path.parent / import_path)[k]

        return parameters

    @staticmethod
    def __parse(file_name):
        with open(file_name, "rb") as f_in:
            params = yaml.safe_load(f_in)
            params = Parameters.handle_imports(params, file_name)

        return params

    @staticmethod
    def parse(file_name):
        return Parameters(Parameters.__parse(file_name))

    def save_name(self):
        return "parameters.pkl"

    def save_data(self):
        return {"parameters": self.parameters}

    def __str__(self):
        return str(self.parameters)


class ParameterPrinter:
    def __init__(self, indent_size=3):
        self.indent_size = indent_size

    def pprint(self, params, indent=1):
        val = ""

        if isinstance(params, dict):
            val += self.print_dict(params, indent)

        elif isinstance(params, list):
            val += self.print_list(params, indent)

        else:
            val += self.print_other(params)

        return val

    def print_dict(self, params, indent):
        val = ""

        prefix = " " * (indent * self.indent_size)
        val += "{\n"

        for k in params.keys():
            val += prefix + f'"{k}": '
            val += self.pprint(params[k], indent=indent + 1)

        prefix = " " * ((indent - 1) * self.indent_size)
        val += prefix + "}\n"

        return val

    def print_list(self, params, indent):
        val = ""

        prefix = " " * (indent * self.indent_size)
        val += "[\n"

        for v in params:
            val += prefix
            val += self.pprint(v, indent=indent + 1)

        prefix = " " * ((indent - 1) * self.indent_size)
        val += prefix + "]\n"

        return val

    def print_other(self, params):
        return str(params) + "\n"
