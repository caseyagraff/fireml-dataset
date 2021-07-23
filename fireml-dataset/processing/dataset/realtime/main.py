from fireml.processing.dataset.realtime.config_setup import setup
from fireml.processing.dataset.realtime.io import (
    clear_output_buffers,
    create_output_buffers,
    create_save_buffers,
    load_all_data,
    save_output_buffers,
    write_save_buffers_to_hdf5,
)

import tqdm
import math
import numpy as np
import time
from fireml.processing.dataset.realtime.preprocess import (
    DEFAULT_PATHS,
    DEFAULT_CONFIG,
)
from fireml.processing.dataset.realtime.settings import (
    CURRENT_REGION,
    AllData,
    DEFAULT_FILTERS,
    OutputBuffers,
    SetupData,
)

import fireml.processing.dataset.realtime.create as create


def create_batch_data_points(
    output_buffers: OutputBuffers,
    inds: np.ndarray,
    setup_data: SetupData,
    all_data: AllData,
    num_workers: int,
):

    create.create_batch_data_points(
        setup_data["points_time"][inds],
        setup_data["points_xy"][inds],
        output_buffers["detections"],
        output_buffers["land_cover"],
        output_buffers["meteorology"],
        output_buffers["filters"],
        setup_data["detections_search_table_times"],
        setup_data["detections_search_table_xys"],
        all_data["meteorology"]["meteorology_times"],
        all_data["meteorology"]["meteorology_xys"],
        all_data["land_cover"]["land_cover_layers"],
        all_data["land_cover"]["land_cover_filter_layer"],
        all_data["meteorology"]["meteorology_layers"],
        setup_data["config"],
        setup_data["config"]["lags"],
        setup_data["config"]["aggregate_lags"],
        setup_data["config"]["forecast_offsets"],
        setup_data["config"]["meteorology_lags"],
        all_data["land_cover"]["land_cover_layer_types"],
        all_data["meteorology"]["meteorology_layer_types"],
        setup_data["evt_to_class"],
        all_data["land_cover"]["land_cover_transform"],
        all_data["land_cover"]["land_cover_filter_transform"],
        all_data["meteorology"]["meteorology_transform"],
        num_workers,
    )


def main(batch_size: int = 32 * 10, save_data: bool = True):
    all_data = load_all_data(DEFAULT_PATHS, DEFAULT_FILTERS, DEFAULT_CONFIG)

    setup_data = setup(
        DEFAULT_CONFIG,
        all_data["detections"]["points"],
        all_data["detections"]["detections_search_table"],
        all_data["land_cover"]["evt_metadata"],
    )

    output_buffers = create_output_buffers(setup_data, batch_size)

    start_time = time.time()

    points_processed = 0
    total_points = len(setup_data["points_time"])

    if save_data:
        save_buffers = create_save_buffers(setup_data, total_points)
    else:
        save_buffers = None

    num_batches = math.ceil(total_points / batch_size)

    print(f"Starting Processing for {total_points} points")

    total_filtered_points = 0

    for i in tqdm.tqdm(range(num_batches)):
        start_ind = i * batch_size
        stop_ind = min((i + 1) * batch_size, total_points)

        inds = np.arange(start_ind, stop_ind)
        create_batch_data_points(
            output_buffers,
            inds,
            setup_data,
            all_data,
            32,
        )

        points_processed += len(inds)
        total_filtered_points += np.sum(output_buffers["filters"] != 0)

        if save_buffers:
            save_output_buffers(output_buffers, save_buffers, start_ind, stop_ind)

        clear_output_buffers(output_buffers)

    print(f"Processed: {points_processed} / {total_points}")
    print(
        f"Filtered: {total_filtered_points} ({total_filtered_points / total_points * 100}%)"
    )

    if save_buffers:
        write_save_buffers_to_hdf5(
            f"/extra/datalab_scratch0/graffc0/fireml/data/processed/datasets/dataset-{CURRENT_REGION}_5days_met_nlcd_true.hdf5",
            setup_data,
            save_buffers,
            total_points,
        )

    # cProfile.runctx(
    #     "create_batch_data_points(detection_data_out, land_cover_data_out, np.arange(batch_size), points_time, points_xy, detections_search_table_times, detections_search_table_xys, land_cover_layers, land_cover_transform, config)",
    #     globals(),
    #     locals(),
    #     "Profile.prof",
    # )
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats(100)

    end_time = time.time()

    print("Elapsed:", end_time - start_time)
    print("Per 1:", (end_time - start_time) / points_processed)


if __name__ == "__main__":
    main()
