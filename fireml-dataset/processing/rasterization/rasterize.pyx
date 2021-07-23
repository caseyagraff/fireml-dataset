# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

# from .generate_mask cimport generate_circle_mask


cdef inline double offset_and_scale_point(
    double p,
    const unsigned int raster_size,
    const unsigned int cell_size,
    const double center_point,
) nogil:
    cdef double offset = center_point - (raster_size / 2.0)

    return (p - offset) / cell_size

cdef void offset_and_scale_points(
    double *points,
    const size_t num_points,
    const unsigned int raster_size,
    const unsigned int cell_size,
    const double center_point_x,
    const double center_point_y) nogil:

    # cdef unsigned int midpoint = raster_size # 2

    # cdef double offset_x = center_point_x - midpoint
    # cdef double offset_y = center_point_y - midpoint

    cdef size_t i

    for i in range(num_points):
        points[i * 2] = offset_and_scale_point(points[i * 2], raster_size, cell_size, center_point_x)
        points[i * 2 + 1] = offset_and_scale_point(points[i * 2 + 1], raster_size, cell_size, center_point_y)
        # points[i * 2] = (points[i * 2] - offset_x) / cell_size
        # points[i * 2 + 1] = (points[i * 2 + 1] - offset_y) / cell_size

    # cdef double [:,:] points_offset = np.subtract(points, (offset_x, offset_y))
    # cdef double [:,:] points_scaled = np.divide(points_offset, cell_size)

cdef void rasterize_exact(
    unsigned char [:,::1] raster,
    double *points,
    const size_t num_points,
    const int cell_size,
    const int raster_size,
    const double center_point_x,
    const double center_point_y,
    const unsigned long end_datetime) nogil:

    cdef double point_x, point_y
    cdef size_t i

    if num_points == 0:
        return

    offset_and_scale_points(
           points, num_points, raster_size, cell_size, center_point_x, center_point_y
    )

    cdef size_t raster_height = raster.shape[0]

    for i in range(num_points):
        point_x = points[i * 2]
        point_y = raster_height - points[i * 2 + 1]

        raster[<unsigned int> point_y, <unsigned int> point_x] = 1

# def rasterize_date(
#     unsigned short [:,::1] raster,
#     double [:,:] points,
#     const unsigned long [::1] times,
#     const int cell_size,
#     const int raster_size,
#     const double center_point_x,
#     const double center_point_y,
#     const unsigned long end_datetime):

#     cdef unsigned int num_points = points.shape[0]
#     cdef const double [:] point

#     if num_points == 0:
#         return

#     offset_and_scale_points(
#             points, raster_size, cell_size, center_point_x, center_point_y
#     )

#     for i in range(num_points):
#         point = points[i]
#         raster[<unsigned int> point[1], <unsigned int> point[0]] = (end_datetime - times[i]) / ONE_HOUR_NS


# def rasterize_overlap(
#         unsigned char [:,::1] raster,
#         double [:,:] points,
#         const unsigned long [::1] times,
#         const int cell_size,
#         const int raster_size,
#         const double center_point_x,
#         const double center_point_y,
#         const unsigned long end_datetime):
#     cdef unsigned int num_points = points.shape[0]
#     cdef const double [:] point

#     if num_points == 0:
#         return

#     offset_and_scale_points(
#             points, raster_size, cell_size, center_point_x, center_point_y
#     )

#     for i in range(num_points):
#         point = points[i]
#         generate_circle_mask(raster, 375, cell_size, point[0], point[1])

# cdef float [:,::1] downsample(const unsigned char [:,::1] raster, const unsigned int new_size):
#     cdef unsigned int raster_size = raster.shape[0]

#     cdef unsigned int ratio = raster_size // new_size

#     return np.reshape(raster, (new_size, ratio, new_size, ratio)).mean(-1).mean(1).astype(np.float32)

# def rasterize_fractional(
#         float [:,::1] raster,
#         const double [:,:] points,
#         const unsigned long [::1] times,
#         const int cell_size,
#         const int raster_size,
#         const double center_point_x,
#         const double center_point_y,
#         const unsigned long end_datetime):

#     cdef unsigned int aproximation_cell_size = 15
#     cdef unsigned int ratio = cell_size // aproximation_cell_size
#     cdef unsigned int num_raster_cells = raster.shape[0]
#     cdef unsigned int temp_raster_size = num_raster_cells * ratio

#     cdef unsigned char [:,::1] temp_raster = np.zeros((temp_raster_size, temp_raster_size), dtype=np.uint8)

#     rasterize_overlap(temp_raster, points, aproximation_cell_size, raster_size, center_point_x, center_point_y)

#     raster[:,:] = downsample(temp_raster, num_raster_cells)
