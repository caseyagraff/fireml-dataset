# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
from fireml.processing.dataset.realtime.transform cimport apply_inverse_transform
cimport fireml.processing.dataset.select_window as sel_window
from libc.stdio cimport printf

cdef void create_meteorology_data(
    Point* point,
    unsigned long [::1] meteorology_times,
    double [:,:,::1] meteorology_xys,
    float [:,:,::1] meteorology_data_out,
    float [:,:,:,::1] meteorology_layers,
    unsigned char [:] meteorology_layer_types,
    DataConfig* data_config,
    unsigned long[:] aggregate_lags,
    Transform t
) nogil:
    cdef size_t num_aggregate_lags = len(aggregate_lags)
    cdef size_t num_layers = meteorology_layers.shape[0]
    cdef size_t current_layer = 0
    cdef size_t i
    cdef unsigned long time_window_lower, time_window_upper


    for i in range(num_aggregate_lags):
        # time_window_lower = 0
        # time_window_upper = aggregate_lags[i]

        # Lags come in pairs of two defining upper and lower times
        time_window_lower = aggregate_lags[i * 2 + 0]
        time_window_upper = aggregate_lags[i * 2 + 1]

        create_meteorology_raster(
                point,
                time_window_lower,
                time_window_upper,
                meteorology_times,
                meteorology_xys,
                meteorology_data_out[current_layer:current_layer+num_layers],
                meteorology_layers,
                t
            )


cdef void create_meteorology_raster(
    Point* point,
    unsigned long time_window_lower,
    unsigned long time_window_upper,
    unsigned long [::1] meteorology_times,
    double [:,:,::1] meteorology_xys,
    float [:,:,::1] meteorology_data_out,
    float [:,:,:,::1] meteorology_layers,
    Transform t
) nogil:
    cdef size_t num_layers = meteorology_layers.shape[0]
    cdef size_t output_width = meteorology_data_out.shape[1]
    cdef size_t time_step, l, i, j
    cdef unsigned long current_time = point.datetime

    cdef unsigned long start_ind, end_ind

    # Get temporal window
    start_ind, end_ind = sel_window.select_temporal_window(meteorology_times, point.datetime, time_window_lower, time_window_upper)

    cdef size_t num_timesteps = end_ind - start_ind

    # Get spatial "center"
    cdef size_t col = <size_t> apply_inverse_transform(t, point.x, point.y)[0]
    cdef size_t row = <size_t> apply_inverse_transform(t, point.x, point.y)[1]

    # Find upper-left corner of spatial window
    col -= (output_width - 1) // 2
    row -= (output_width - 1) // 2

    # For each layer
    for l in range(num_layers):
        # Walk forward through time (average through timesteps)
        for time_step in range(start_ind, end_ind):
            for i in range(output_width):
                for j in range(output_width):
                    meteorology_data_out[l, i, j] += (meteorology_layers[l, time_step, row + i, col + j] / num_timesteps)