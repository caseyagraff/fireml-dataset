# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport sqrt, ceil

cdef double magnitude(double x, double y):
    return sqrt(x**2 + y**2)

cdef void generate_mask_quadrant(
        unsigned char[:,:] mask,
        char x_dir,
        char y_dir,
        double radius_grid_units,
        int radius_grid_units_ceil,
        double center_x,
        double center_y):

    cdef double x_adj, y_adj, distance
    cdef int mask_width = mask.shape[0]
    cdef int mask_height = mask.shape[1]

    cdef int y_max = (radius_grid_units_ceil + 1)
    cdef int x_max = (radius_grid_units_ceil + 1)


    cdef int y, x

    for y in range(y_max):
        for x in range(x_max):
            x_adj = x * x_dir + center_x
            y_adj = y * y_dir + center_y

            if x_adj < 0 or y_adj < 0 or x_adj >= mask_width or y_adj >= mask_height:
                continue

            distance = magnitude(x,y)

            if distance <= radius_grid_units:
                mask[<unsigned int>y_adj, <unsigned int>x_adj] = 1

cdef void generate_circle_mask(
        unsigned char [:,:] mask,
        unsigned short radius,
        unsigned short cell_size,
        double center_x,
        double center_y):

    cdef double radius_grid_units = radius / cell_size
    cdef int radius_grid_units_ceil = <int> ceil(radius_grid_units)

    generate_mask_quadrant(mask, 1, 1, radius_grid_units, radius_grid_units_ceil, center_x, center_y)
    generate_mask_quadrant(mask, 1, -1, radius_grid_units, radius_grid_units_ceil, center_x, center_y)
    generate_mask_quadrant(mask, -1, 1, radius_grid_units, radius_grid_units_ceil, center_x, center_y)
    generate_mask_quadrant(mask, -1, -1, radius_grid_units, radius_grid_units_ceil, center_x, center_y)

    mask[<unsigned int> center_y, <unsigned int> center_x] = 1
