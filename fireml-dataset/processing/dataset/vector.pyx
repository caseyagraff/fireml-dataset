# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

from libc.stdlib cimport malloc, realloc, free

from .vector cimport Vector

cdef Vector *create_vector(size_t intial_count) nogil:
    cdef Vector *vector = <Vector *> malloc(sizeof(Vector))

    vector.data = <size_t *> malloc(sizeof(size_t) * intial_count)
    vector.max_size = intial_count
    vector.length = 0

    return vector

cdef size_t *free_vector(Vector *p) nogil:
    free(p.data)
    free(p)


cdef void change_max_size(Vector *p, size_t new_max_size) nogil:
    # increase size
    p.data = <size_t *> realloc(p.data, new_max_size * sizeof(size_t))

    p.max_size = new_max_size

cdef void push_item(Vector *p, size_t item) nogil:
    if (p.length + 1) > (p.max_size // 2):
        change_max_size(p, p.max_size + 50)

    (p.data + p.length)[0] = item

    p.length += 1


cdef size_t pop_item(Vector *p) nogil:
    if p.length == 0:
        return 0;

    cdef size_t item = (p.data + (p.length - 1))[0]
    p.length -= 1

    if p.length < (p.max_size // 2):
        if p.max_size > 200:
            change_max_size(p, p.max_size - 50)
        elif p.max_size > 50:
            change_max_size(p, p.max_size - 10)

    return item; 

cdef size_t get_item(Vector *p, size_t ind) nogil:
    return (p.data + ind)[0]

cdef void clear_vector(Vector *p, size_t new_size) nogil:
    p.length = 0;
    if new_size == 0:
        change_max_size(p, 5)
    else:
        change_max_size(p, new_size)

cdef size_t get_length(Vector *p) nogil:
    return p.length