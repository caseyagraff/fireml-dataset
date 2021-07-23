# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

from fireml.processing.dataset.realtime.types cimport Point, DataConfig, FilterReason, LandCoverLayerType, Transform
cimport fireml.processing.rasterization.rasterize as rast
cimport fireml.processing.dataset.select_window as sel_window
from libc.stdlib cimport malloc, free

from fireml.processing.dataset.vector cimport Vector, create_vector, free_vector, get_length, get_item
from fireml.processing.dataset.realtime.transform cimport apply_inverse_transform

cdef int URB_LO = 22 # Developed, low intensity
cdef int URB_MD = 23 # Developed, medium intensity
cdef int URB_HI = 24 # Developed, high intensity
cdef int AGR_PASTURE = 81 # Pasture/hay
cdef int AGR_CULTIVATED = 82 # Cultivated crops
cdef int NO_DATA = 0

cdef int* LANDCOVER_FILTER_CLASSES = [URB_LO, URB_MD, URB_HI, AGR_PASTURE, AGR_CULTIVATED, NO_DATA]
cdef size_t NUM_LANDCOVER_FILTER_CLASSES = 6

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
) nogil:
    cdef size_t num_layers = fire_detection_data.shape[0];

    cdef size_t num_lags = len(lags)
    cdef size_t num_aggregate_lags = len(aggregate_lags)
    cdef size_t num_forecast_offsets = len(forecast_offsets)

    cdef size_t width = fire_detection_data.shape[1];

    cdef size_t i;
    cdef size_t current_layer = 0
    cdef unsigned long start_offset, end_offset, aggregate_lag_half_duration, aggregate_lag_midpoint;
    cdef Point modified_point;

    cdef FilterReason filter_data_point, aggregate_filter_data_point = FilterReason.nominal
    cdef unsigned char any_filter_data_point_nominal = 0
    cdef unsigned char should_filter_lag_by_detections, should_filter_lag_by_land_cover

    modified_point.x = point.x
    modified_point.y = point.y

    # Get current observation
    filter_data_point = create_detections_raster(
        point, 
        detection_search_table_times, 
        detection_search_table_xys, 
        fire_detection_data[current_layer], 
        data_config.detection_window_lower,
        data_config.detection_window_upper,
        data_config.cell_size,
        data_config.window_size,

        land_cover_filter_layer,
        t_filter,
        data_config.should_filter_by_detections,
        data_config.should_filter_by_land_cover,
        data_config.minimum_detections,
        data_config.minimum_land_cover_detections,
        data_config.minimum_land_cover_detections_percentage,
        data_config.land_cover_cell_size
    )

    any_filter_data_point_nominal = 1 if filter_data_point == FilterReason.nominal else any_filter_data_point_nominal
    aggregate_filter_data_point = filter_data_point if filter_data_point > aggregate_filter_data_point else aggregate_filter_data_point

    current_layer += 1

    # Get lags
    for i in range(num_lags):
        modified_point.datetime = point.datetime - lags[i]

        should_filter_lag_by_detections = data_config.should_filter_by_detections and i < data_config.maximum_lags_detections
        should_filter_lag_by_land_cover = data_config.should_filter_by_land_cover and i < data_config.maximum_lags_land_cover

        filter_data_point = create_detections_raster(
            &modified_point, 
            detection_search_table_times, 
            detection_search_table_xys, 
            fire_detection_data[current_layer], 
            data_config.detection_window_lower,
            data_config.detection_window_upper,
            data_config.cell_size,
            data_config.window_size,

            land_cover_filter_layer,
            t_filter,
            should_filter_lag_by_detections,
            should_filter_lag_by_land_cover,
            data_config.minimum_detections,
            data_config.minimum_land_cover_detections,
            data_config.minimum_land_cover_detections_percentage,
            data_config.land_cover_cell_size
        )

        if should_filter_lag_by_detections or should_filter_lag_by_land_cover:
            any_filter_data_point_nominal = 1 if filter_data_point == FilterReason.nominal else any_filter_data_point_nominal
            aggregate_filter_data_point = filter_data_point if filter_data_point > aggregate_filter_data_point else aggregate_filter_data_point

        current_layer += 1


    # Get aggregate lags (must come in pairs of 2)
    for i in range(num_aggregate_lags // 2):
        start_offset = aggregate_lags[i * 2 + 0]
        end_offset = aggregate_lags[i * 2 + 1]

        aggregate_lag_half_duration = (end_offset - start_offset) // 2
        aggregate_lag_midpoint = (point.datetime - start_offset) - aggregate_lag_half_duration

        modified_point.datetime = aggregate_lag_midpoint

        create_detections_raster(
            &modified_point, 
            detection_search_table_times, 
            detection_search_table_xys, 
            fire_detection_data[current_layer], 
            aggregate_lag_half_duration,
            aggregate_lag_half_duration,
            data_config.cell_size,
            data_config.window_size,

            land_cover_filter_layer,
            t_filter,
            False,
            False,
            0,
            0,
            0,
            0
        )
        current_layer += 1


    # Get targets
    for i in range(num_forecast_offsets):
        modified_point.datetime = point.datetime + forecast_offsets[i]

        create_detections_raster(
            &modified_point, 
            detection_search_table_times, 
            detection_search_table_xys, 
            fire_detection_data[current_layer], 
            data_config.forecast_window_lower,
            data_config.forecast_window_upper,
            data_config.cell_size,
            data_config.window_size,

            land_cover_filter_layer,
            t_filter,
            False,
            False,
            0,
            0,
            0,
            0
        )

        current_layer += 1

    return aggregate_filter_data_point if not any_filter_data_point_nominal else FilterReason.nominal

cdef FilterReason create_detections_raster(
    Point* point,
    unsigned long [::1] detection_search_table_times,
    double [:,::1] detection_search_table_xys,
    unsigned char[:,::1] fire_detection_data,
    unsigned long time_window_lower,
    unsigned long time_window_upper,
    unsigned long cell_size,
    unsigned long window_size,
    unsigned char [:,::1] land_cover_filter_layer,
    Transform t_filter,
    unsigned char should_filter_by_detections,
    unsigned char should_filter_by_land_cover,
    size_t minimum_detections,
    size_t minimum_land_cover_detections,
    float minimum_land_cover_detections_percentage,
    unsigned long land_cover_cell_size,
) nogil:
    cdef size_t num_inds, i, j, ind, width;

    cdef Vector *inds = create_vector(50)

    cdef FilterReason filter_data_point = FilterReason.nominal

    sel_window.select_spatiotemporal_window(
       detection_search_table_times,
       detection_search_table_xys,
       point.datetime,
       time_window_lower,
       time_window_upper,
       point.x,
       point.y,
       window_size,
       inds
    )

    num_inds = get_length(inds)

    cdef FilterReason filter_reason = FilterReason.nominal

    if num_inds == 0:
        free_vector(inds)
        return FilterReason.detections if should_filter_by_detections and minimum_detections > 0 else FilterReason.nominal;


    cdef double *xys = <double *> malloc(num_inds * 2 * sizeof(double))

    for i in range(num_inds):
        ind = get_item(inds, i)

        xys[i * 2] = detection_search_table_xys[ind,0]
        xys[i * 2  + 1] = detection_search_table_xys[ind,1]


    free_vector(inds)

    if should_filter_by_detections and num_inds < minimum_detections:
        filter_reason = FilterReason.detections

    if should_filter_by_land_cover and filter_by_land_cover(point, xys, num_inds, land_cover_filter_layer, t_filter, land_cover_cell_size, minimum_land_cover_detections, minimum_land_cover_detections_percentage) != FilterReason.nominal:
        filter_reason = FilterReason.land_cover

    rast.rasterize_exact(
        fire_detection_data,
        xys,
        num_inds,
        cell_size,
        window_size,
        point.x,
        point.y,
        point.datetime
    )

    free(xys)

    return filter_reason

cdef FilterReason filter_by_land_cover(
    Point *point,
    double *xys,
    size_t num_points,
    unsigned char [:,::1] land_cover_filter_layer,
    Transform t_filter,
    size_t land_cover_cell_size,
    size_t minimum_land_cover_detections,
    float minimum_land_cover_detections_percentage
) nogil:
    cdef size_t i, j

    cdef size_t row, col, valid_detections = 0
    cdef unsigned char land_cover_class, is_valid_detection

    for i in range(num_points):
        col = <size_t> apply_inverse_transform(t_filter, xys[i * 2], xys[i * 2 + 1])[0]
        row = <size_t> apply_inverse_transform(t_filter, xys[i * 2], xys[i * 2 + 1])[1]

        land_cover_class = land_cover_filter_layer[row, col]

        is_valid_detection = 1
        for j in range(NUM_LANDCOVER_FILTER_CLASSES):
            if land_cover_class == LANDCOVER_FILTER_CLASSES[j]:
                is_valid_detection = 0
                break

        valid_detections += is_valid_detection

    if valid_detections >= minimum_land_cover_detections and (valid_detections / num_points) >= minimum_land_cover_detections_percentage:
        return FilterReason.nominal

    # There must be at least one non-valid detection to filter this due to land cover (o.w. the reason should come from FilterReason.detections)
    return FilterReason.land_cover if valid_detections < num_points else FilterReason.nominal