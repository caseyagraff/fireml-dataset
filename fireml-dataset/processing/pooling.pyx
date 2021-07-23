# cython: language_level=3, boundscheck=False, wraparound=False
from libc.math cimport ceil, floor

import numpy as np

cdef float [:,:] avg_pool_2d_(const short [:,:] arr, const int target_size):
    cdef double width = arr.shape[0] / target_size

    cdef float [:,:] result = np.empty((target_size, target_size), dtype=np.float32)

    cdef int start_i
    cdef int end_i
    cdef int start_j
    cdef int end_j
    cdef const short [:,:] sel

    cdef int i,j

    for i in range(target_size):
        for j in range(target_size):
            start_i = <int> floor(i * width)
            end_i = <int> ceil((i+1) * width)
            start_j = <int> floor(j * width)
            end_j = <int> ceil((j+1) * width)

            sel = arr[start_i:end_i,start_j:end_j]
            result[i,j] = np.mean(sel)

    return result

def avg_pool_2d(arr, target_size, dtype=None):

    rem = arr.shape[0] % target_size

    if rem != 0:
        upsample_rate = target_size // rem
        arr = arr.repeat(upsample_rate, axis=0).repeat(upsample_rate, axis=1)

    ratio = arr.shape[0] // target_size

    return np.nanmean(arr.reshape(target_size, ratio, target_size, ratio), axis=(1,3), dtype=dtype)

def avg_pool_3d_(arr, target_size):
    num_channels = arr.shape[2]
    result = np.empty((target_size, target_size, num_channels), dtype=np.float32)

    for i in range(num_channels):
        result[:,:,i] = avg_pool_2d(arr[:,:,i], target_size)

    return result

def avg_pool_3d(arr, target_size, dtype=None):
    num_channels = arr.shape[2]

    rem = arr.shape[0] % target_size

    if rem != 0:
        upsample_rate = target_size // rem
        arr = arr.repeat(upsample_rate, axis=0).repeat(upsample_rate, axis=1)

    ratio = arr.shape[0] // target_size

    return np.nanmean(arr.reshape(target_size, ratio, target_size, ratio, num_channels), axis=(1,3), dtype=dtype)
