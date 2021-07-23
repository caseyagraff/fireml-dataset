# cython: language_level=3

from fireml.processing.dataset.realtime.types cimport DataConfig, Point, Transform

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
) nogil