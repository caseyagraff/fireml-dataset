# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

from libc.stdlib cimport malloc, free
from libc.math cimport NAN

from fireml.processing.dataset.realtime.types cimport DataConfig, Point, Transform, LandCoverLayerType
from fireml.processing.dataset.realtime.transform cimport apply_inverse_transform

from fireml.processing.dataset.realtime.pooling cimport avg_pool_2d, avg_pool_3d


cdef size_t LANDFIRE_CELL_SIZE = 30
cdef short MISSING_VALUE = -32768
cdef short UNKNOWN_VALUE = -9999

cdef void copy_raster_to_memview_2d(
    float *raster,
    float [:, ::1] memview
) nogil:
    cdef size_t width = memview.shape[0]
    cdef size_t i, j

    for i in range(width):
        for j in range(width):
            memview[i,j] = raster[i * width + j]

"""
Expected input shape: (# layers, width, width)
"""
cdef void copy_raster_to_memview_3d(
    float *raster,
    float [:, :, ::1] memview
) nogil:
    cdef size_t num_layers = memview.shape[0]
    cdef size_t width = memview.shape[1]
    cdef size_t i, j, k

    for i in range(num_layers):
        for j in range(width):
            for k in range(width):
                memview[i,j,k] = raster[i * width * width + j * width + k]

cdef void copy_memview_to_raster(
    short [:, ::1] memview,
    float *raster
) nogil:
    cdef size_t width = memview.shape[0]
    cdef size_t i, j

    for i in range(width):
        for j in range(width):
            raster[i * width + j] = memview[i,j]

cdef void extract_raster(
    Point* point,
    float *raster,
    short [:,::1] land_cover_layer,
    size_t width,
    Transform t
) nogil:
    cdef size_t col = <size_t> apply_inverse_transform(t, point.x, point.y)[0]
    cdef size_t row = <size_t> apply_inverse_transform(t, point.x, point.y)[1]

    cdef size_t layer_height = land_cover_layer.shape[0];
    cdef size_t layer_width = land_cover_layer.shape[1];

    cdef size_t offset = width // 2
    cdef size_t round_off = width % 2

    cdef size_t top = row - offset
    cdef size_t bottom = row + offset
    cdef size_t left = col - offset
    cdef size_t right = col + offset

    # Compute bounded indices
    cdef size_t top_b = top if top > 0 else 0
    cdef size_t bottom_b = bottom if bottom < layer_height - 1 else layer_height - 1
    cdef size_t left_b = left if left > 0 else 0
    cdef size_t right_b = right if right < layer_width - 1 else layer_width - 1

    # If bounded and unbounded indices deviate, only assign to subsection of extracted raster
    cdef size_t a = top_b - top
    cdef size_t b = width - (bottom - bottom_b)
    cdef size_t c = left_b - left
    cdef size_t d = width - (right - right_b)

    # raster[a:b, c:d] = land_cover_layer[top_b : bottom_b + round_off, left_b : right_b + round_off]
    copy_memview_to_raster(land_cover_layer[top_b : bottom_b + round_off, left_b : right_b + round_off], raster)

cdef void process_raster(
    float *raster,
    float [:,::1] land_cover_data_out,
    size_t width,
    size_t land_cover_cell_size,
) nogil:
    cdef size_t i, j

    for i in range(width):
        for j in range(width):
            if raster[i * width + j] == MISSING_VALUE or raster[i * width + j] == UNKNOWN_VALUE:
                raster[i * width + j] = NAN

    # Pooling
    if land_cover_cell_size > LANDFIRE_CELL_SIZE:
        avg_pool_2d(raster, width, land_cover_data_out)
    else:
        copy_raster_to_memview_2d(raster, land_cover_data_out)


cdef void convert_evt_classes(float *arr, size_t width, short [:] evt_to_class) nogil:
    cdef size_t i, j
    for i in range(width):
        for j in range(width):
            arr[i * width + j] = evt_to_class[(<size_t> arr[i * width + j]) - 3000]

"""
Expected input shape: (width, width)
Expected output shape: (# one_hot_classes, width, width)
"""
cdef void one_hot_encode(float *arr, float *out, size_t width, unsigned int num_one_hot_classes) nogil:
    cdef size_t i, j, k
    cdef size_t class_ind;

    for i in range(width):
        for j in range(width):
            class_ind = <size_t> arr[i * width + j]

            for k in range(num_one_hot_classes):
                if class_ind == k:
                    out[k * width * width + i * width + j] = 1
                else:
                    out[k * width * width + i * width + j] = 0

cdef void process_raster_evt(
    float *raster,
    float *evt_raster,
    float [:,:,::1] land_cover_data_out,
    size_t width,
    size_t land_cover_cell_size,
    unsigned char should_convert_evt_classes,
    unsigned int num_one_hot_classes,
    short [:] evt_to_class
) nogil:
    cdef float *raster_ptr = raster

    # Convert to classes
    if should_convert_evt_classes:
        convert_evt_classes(raster_ptr, width, evt_to_class)

    # One hot encode
    if num_one_hot_classes > 1:
        one_hot_encode(raster, evt_raster, width, num_one_hot_classes)

        raster_ptr = evt_raster
    
    # Spatial pooling; behaves identically to 2d pooling if first dimension size is 1
    if land_cover_cell_size > LANDFIRE_CELL_SIZE:
       avg_pool_3d(raster_ptr, width, land_cover_data_out)

    else:
       copy_raster_to_memview_3d(raster_ptr, land_cover_data_out)

cdef void create_land_cover_data(
    Point* point,
    float [:,:,::1] land_cover_data_out,
    short [:,:,::1] land_cover_layers,
    unsigned char [:] land_cover_layer_types,
    short [:] evt_to_class,
    DataConfig* data_config,
    Transform t
) nogil:
    cdef size_t width = data_config.window_size // LANDFIRE_CELL_SIZE
    cdef size_t num_land_cover_layers = len(land_cover_layers)
    cdef size_t i

    # Need intermediate raster before shrinking with pooling
    cdef float *raster = <float *> malloc(sizeof(float) * width * width)
    cdef float *evt_raster = <float *> 0;
    
    if data_config.num_one_hot_classes > 1:
        evt_raster = <float *> malloc(sizeof(float) * width * width * data_config.num_one_hot_classes)

    for i in range(num_land_cover_layers):
        extract_raster(
            point, 
            raster,
            land_cover_layers[i],
            width,
            t,
        )
        if land_cover_layer_types[i] == LandCoverLayerType.filter:
            continue;

        elif land_cover_layer_types[i] == LandCoverLayerType.evt:
            process_raster_evt(
                raster,
                evt_raster,
                land_cover_data_out[i:i+data_config.num_one_hot_classes],
                width,
                data_config.land_cover_cell_size,
                data_config.should_convert_evt_classes,
                data_config.num_one_hot_classes,
                evt_to_class
            )
 
        else:
            process_raster(
                raster,
                land_cover_data_out[i],
                width,
                data_config.land_cover_cell_size ,
            )    
            

    free(raster)

    if data_config.num_one_hot_classes > 1:
        free(evt_raster)

