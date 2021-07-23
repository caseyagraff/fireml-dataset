# cython: language_level=3

cdef enum DetectionDiscretizationMethod:
    exact = 1
    square = 2

cdef enum LandCoverLayerType:
    evt = 1
    filter = 2
    other = 3

cdef enum FilterReason:
    nominal = 0
    detections = 1
    land_cover = 2

cdef struct DataConfig:
    unsigned long detection_window_lower
    unsigned long detection_window_upper
    unsigned long forecast_window_lower
    unsigned long forecast_window_upper
    unsigned long cell_size
    unsigned long window_size
    DetectionDiscretizationMethod discretization_method

    size_t land_cover_cell_size
    unsigned char should_convert_evt_classes
    unsigned int num_one_hot_classes

    unsigned char should_filter_by_detections
    unsigned char should_filter_by_land_cover

    size_t minimum_detections
    size_t minimum_land_cover_detections
    float minimum_land_cover_detections_percentage
    size_t maximum_lags_detections
    size_t maximum_lags_land_cover

    unsigned char use_land_cover
    unsigned char use_meteorology


cdef struct Point:
    double x
    double y
    unsigned long datetime

cdef struct Transform:
    float a
    float b
    float c
    float d
    float e
    float f

    float a_i
    float b_i
    float c_i
    float d_i
    float e_i
    float f_i