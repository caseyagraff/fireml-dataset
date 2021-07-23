# cython: language_level=3

cdef struct Vector:
    size_t max_size
    size_t length
    size_t* data

cdef Vector *create_vector(size_t intial_count) nogil

cdef size_t *free_vector(Vector *p) nogil

cdef void change_max_size(Vector *p, size_t new_max_size) nogil

cdef void push_item(Vector *p, size_t item) nogil

cdef size_t pop_item(Vector *p) nogil

cdef size_t get_item(Vector *p, size_t ind) nogil

cdef void clear_vector(Vector *p, size_t new_size) nogil

cdef size_t get_length(Vector *p) nogil