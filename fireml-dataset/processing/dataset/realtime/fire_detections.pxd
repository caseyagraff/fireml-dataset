# cython: language_level=3

from fireml.processing.dataset.realtime.types cimport Point, DataConfig, FilterReason, Transform

cdef FilterReason create_fire_detection_data(
    Point* point,
    unsigned long [::1] detection_search_table_times,
    double [:,::1] detection_search_table_xys,
    unsigned char[:,:,::1] fire_detection_data,
    DataConfig* data_config,
    unsigned long[:] lags,
    unsigned long[:] aggregate_lags,
    unsigned long[:] forecast_offsets,
    unsigned char [:,::1] land_cover_filter_layer,
    Transform t_filter
) nogil