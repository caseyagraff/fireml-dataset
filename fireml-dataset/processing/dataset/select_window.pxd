# cython: language_level=3

from .vector cimport Vector

cdef (unsigned long, unsigned long) select_temporal_window(
    const unsigned long [::1] times,
    const unsigned long datetime,
    const unsigned long time_window_lower,
    const unsigned long time_window_upper) nogil

cdef void select_spatial_window(
    const double [:,::1] xys,
    const unsigned long start_ind,
    const unsigned long end_ind,
    const double center_x,
    const double center_y,
    const unsigned int window_size,
    Vector *inds
    ) nogil

cdef void select_spatiotemporal_window(
    const unsigned long [::1] times,
    const double [:,::1] xys,
    const unsigned long datetime,
    const unsigned long time_window_lower,
    const unsigned long time_window_upper,
    const double center_x,
    const double center_y,
    const unsigned int window_size,
    Vector *inds
    ) nogil
