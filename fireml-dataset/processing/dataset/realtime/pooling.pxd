# cython: language_level=3

cdef void avg_pool_2d(float *arr, const size_t current_width, float [:,::1] result) nogil
cdef void avg_pool_3d(float *arr, const size_t current_width, float [:,:,::1] result) nogil