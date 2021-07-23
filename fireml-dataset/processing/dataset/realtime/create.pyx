# cython: language_level=3, boundscheck=False, wraparound=False

from cython.parallel import prange

from fireml.processing.dataset.realtime.types cimport DataConfig, Point, Transform
from fireml.processing.dataset.realtime.fire_detections cimport create_fire_detection_data
from fireml.processing.dataset.realtime.land_cover cimport create_land_cover_data
from fireml.processing.dataset.realtime.meteorology cimport create_meteorology_data



cdef unsigned char create_one_data_point(
    double x,
    double y,
    unsigned long datetime,
    unsigned char[:,:,::1] fire_detection_data_out,
    float[:,:,::1] land_cover_data_out,
    float[:,:,::1] meteorology_data_out,
    unsigned long [::1] detection_search_table_times,
    double [:,::1] detection_search_table_xys,
    unsigned long [::1] meteorology_times,
    double [:,:,::1] meteorology_xys,
    short[:,:,::1] land_cover_layers,
    unsigned char[:,::1] land_cover_filter_layer,
    float[:,:,:,::1] meteorology_layers,
    DataConfig data_config,
    unsigned long[:] lags,
    unsigned long[:] aggregate_lags,
    unsigned long[:] forecast_offsets,
    unsigned long[:] meteorology_lags,
    unsigned char [:] land_cover_layer_types,
    unsigned char [:] meteorology_layer_types,
    short [:] evt_to_class,
    Transform t,
    Transform t_filter,
    Transform t_meteorology
) nogil :
    cdef unsigned char is_filtered

    cdef Point point;
    point.x = x
    point.y = y
    point.datetime = datetime

    is_filtered = create_fire_detection_data(
        &point, 
        detection_search_table_times, 
        detection_search_table_xys, 
        fire_detection_data_out, 
        &data_config, 
        lags, 
        aggregate_lags,
        forecast_offsets,
        land_cover_filter_layer,
        t_filter
    )

    if data_config.use_land_cover:
        create_land_cover_data(
            &point, 
            land_cover_data_out, 
            land_cover_layers,
            land_cover_layer_types,
            evt_to_class,
            &data_config, 
            t
        )


    if data_config.use_meteorology:
        create_meteorology_data(
            &point, 
            meteorology_times,
            meteorology_xys,
            meteorology_data_out, 
            meteorology_layers, 
            meteorology_layer_types,
            &data_config, 
            meteorology_lags, 
            t_meteorology
        )

    return is_filtered


def create_batch_data_points(
    unsigned long [:] points_time,
    double [:,:] points_xy,
    unsigned char[:,:,:,::1] fire_detection_data_out,
    float[:,:,:,::1] land_cover_data_out,
    float[:,:,:,::1] meteorology_data_out,
    unsigned char [:] filtered_data_points_out,
    unsigned long [::1] detection_search_table_times,
    double [:,::1] detection_search_table_xys,
    unsigned long [::1] meteorology_times,
    double [:,:,::1] meteorology_xys,
    short[:,:,::1] land_cover_layers,
    unsigned char[:,::1] land_cover_filter_layer,
    float[:,:,:,::1] meteorology_layers,
    DataConfig data_config,
    unsigned long[:] lags,
    unsigned long[:] aggregate_lags,
    unsigned long[:] forecast_offsets,
    unsigned long[:] meteorology_lags,
    unsigned char [:] land_cover_layer_types,
    unsigned char [:] meteorology_layer_types,
    short [:] evt_to_class,
    Transform t,
    Transform t_filter,
    Transform t_meteorology,
    int num_threads
):
    cdef int num_points = len(points_time)

    cdef int i

    for i in prange(num_points, nogil=True, num_threads=num_threads):
    # for i in range(num_points):
        filtered_data_points_out[i] = create_one_data_point(
            points_xy[i, 0],
            points_xy[i, 1],
            points_time[i],
            fire_detection_data_out[i],
            land_cover_data_out[i],
            meteorology_data_out[i],
            detection_search_table_times,
            detection_search_table_xys,
            meteorology_times,
            meteorology_xys,
            land_cover_layers,
            land_cover_filter_layer,
            meteorology_layers,
            data_config,
            lags,
            aggregate_lags,
            forecast_offsets,
            meteorology_lags,
            land_cover_layer_types,
            meteorology_layer_types,
            evt_to_class,
            t,
            t_filter,
            t_meteorology
        )