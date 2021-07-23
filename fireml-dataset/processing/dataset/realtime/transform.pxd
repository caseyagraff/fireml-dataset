# cython: language_level=3

from fireml.processing.dataset.realtime.types cimport Transform 

cdef (double, double) apply_transform(Transform t, double x, double y) nogil

cdef (double, double) apply_inverse_transform(Transform t, double x, double y) nogil