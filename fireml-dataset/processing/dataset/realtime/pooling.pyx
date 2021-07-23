# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

from libc.math cimport ceil, floor

cdef double mean_2d(
    float *sel,  
    size_t width,
    size_t start_i, 
    size_t end_i, 
    size_t start_j, 
    size_t end_j
  ) nogil:
    cdef size_t i, j
    cdef double acc = 0

    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            acc += sel[i * width + j]

    return acc / ((end_i - start_i) + (end_j - start_j))

cdef void avg_pool_2d(float *arr, const size_t current_width, float [:,::1] result) nogil:
    cdef size_t target_width = result.shape[0]

    cdef double width_ratio = current_width / target_width

    cdef size_t start_i, end_i, start_j, end_j, i, j

    for i in range(target_width):
        for j in range(target_width):
            start_i = <size_t> floor(i * width_ratio)
            end_i = <size_t> ceil((i+1) * width_ratio)
            start_j = <size_t> floor(j * width_ratio)
            end_j = <size_t> ceil((j+1) * width_ratio)

            result[i,j] = mean_2d(arr, current_width, start_i, end_i, start_j, end_j)


"""
Expected input shape: (# layers, current_width, current_width)
Expected output shape: (# layers, new_width, new_width)
"""
cdef void avg_pool_3d(float *arr, const size_t current_width, float [:,:,::1] result) nogil:
    cdef size_t num_layers = result.shape[0]
    cdef size_t i

    for i in range(num_layers):
        avg_pool_2d(arr + (i * current_width * current_width), current_width, result[i])