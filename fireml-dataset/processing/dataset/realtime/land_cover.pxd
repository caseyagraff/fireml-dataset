# cython: language_level=3

from fireml.processing.dataset.realtime.types cimport DataConfig, Point, Transform

cdef void create_land_cover_data(
    Point* point,
    float [:,:,::1] land_cover_data_out,
    short [:,:,::1] land_cover_layers,
    unsigned char [:] land_cover_layer_types,
    short [:] evt_to_class,
    DataConfig* data_config,
    Transform t
) nogil