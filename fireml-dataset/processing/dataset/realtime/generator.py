from typing import TypedDict
import numpy as np
from fireml.processing.dataset.realtime.settings import (
    PreprocessDataConfig,
    PreprocessFilters,
    PreprocessPaths,
)
from fireml.processing.dataset.realtime.main import (
    OutputBuffers,
    clear_output_buffers,
    create_batch_data_points,
    create_output_buffers,
    load_all_data,
    setup,
)


class DataArrays(TypedDict):
    observed: np.ndarray
    targets: np.ndarray
    land_cover: np.ndarray
    meteorology: np.ndarray
    filters: np.ndarray


class OnlineGenerator:
    def __init__(
        self,
        paths: PreprocessPaths,
        filters: PreprocessFilters,
        config: PreprocessDataConfig,
        max_batch_size: int,
        num_workers: int = 1,
    ):
        self.max_batch_size = max_batch_size
        self.all_data = load_all_data(paths, filters, config)
        self.num_workers = num_workers

        self.setup_data = setup(
            config,
            self.all_data["detections"]["points"],
            self.all_data["detections"]["detections_search_table"],
            self.all_data["land_cover"]["evt_metadata"],
        )

        self.output_buffers = create_output_buffers(
            self.setup_data, self.max_batch_size
        )

        self.total_points = len(self.all_data["detections"]["points"])

        self.target_detections_ind = self.setup_data["config"][
            "num_detection_layers"
        ] - len(self.setup_data["config"]["forecast_offsets"])

    def __len__(self):
        return self.total_points

    def get_consecutive_batch(self, start_ind: int, batch_size: int) -> DataArrays:
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size ({batch_size}) must not exceed max batch size ({self.max_batch_size})."
            )

        clear_output_buffers(self.output_buffers)

        stop_ind = min(start_ind + batch_size, self.total_points)

        inds = np.arange(start_ind, stop_ind)
        true_batch_size = stop_ind - start_ind

        create_batch_data_points(
            self.output_buffers,
            inds,
            self.setup_data,
            self.all_data,
            self.num_workers,
        )

        return {
            "observed": self.output_buffers["detections"][
                :true_batch_size, : self.target_detections_ind
            ],
            "targets": self.output_buffers["detections"][
                :true_batch_size, self.target_detections_ind :
            ],
            "land_cover": self.output_buffers["land_cover"][:true_batch_size],
            "meteorology": self.output_buffers["meteorology"][:true_batch_size],
            "filters": self.output_buffers["filters"][:true_batch_size],
        }