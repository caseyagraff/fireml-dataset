from fireml.processing.dataset.realtime.types cimport Transform 

cdef (double, double) apply_transform(Transform t, double x, double y) nogil:
    cdef double x_ = t.a * x + t.b * y + t.c
    cdef double y_ = t.d * x + t.e * y + t.f

    return x_, y_

cdef (double, double) apply_inverse_transform(Transform t, double x, double y) nogil:

    cdef double x_ = t.a_i * x + t.b_i * y + t.c_i
    cdef double y_ = t.d_i * x + t.e_i * y + t.f_i

    return x_, y_