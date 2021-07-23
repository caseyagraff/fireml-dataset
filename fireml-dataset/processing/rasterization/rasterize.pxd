# cython: language_level=3

cdef unsigned long ONE_HOUR_NS = 3600000000000

cdef void offset_and_scale_points(
    double *points,
    const size_t num_points,
    const unsigned int raster_size,
    const unsigned int cell_size,
    const double center_point_x,
    const double center_point_y) nogil

cdef void rasterize_exact(
    unsigned char [:,::1] raster,
    double *points,
    const size_t num_points,
    const int cell_size,
    const int raster_size,
    const double center_point_x,
    const double center_point_y,
    const unsigned long end_datetime) nogil

cdef double offset_and_scale_point(
    double p,
    const unsigned int raster_size,
    const unsigned int cell_size,
    const double center_point
) nogil

"""
cdef void rasterize_exact(
    unsigned char [:,::1] raster,
    const double [:,:] points,
    const int cell_size,
    const int raster_size,
    const double center_point_x,
    const double center_point_y
    )

cdef void rasterize_overlap(
        unsigned char [:,::1] raster,
        const double [:,:] points,
        const int cell_size,
        const int raster_size,
        const double center_point_x,
        const double center_point_y
        )

cdef void rasterize_fractional(
        float [:,::1] raster,
        const double [:,:] points,
        const int cell_size,
        const int raster_size,
        const double center_point_x,
        const double center_point_y
        )
"""
