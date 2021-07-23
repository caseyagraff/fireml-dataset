# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

from .vector cimport Vector, push_item

cdef int compare_times(const void *a, const void *b) nogil:
    cdef unsigned long a_v = (<unsigned long*>a)[0]
    cdef unsigned long b_v = (<unsigned long*>b)[0]

    if a_v < b_v:
        return -1;
    elif a_v > b_v:
        return 1;
    else:
        return 0;

# Insertion side = left
cdef size_t search(const void *key, const void *array, size_t count, size_t size, int (*compare)(const void *, const void *) nogil) nogil:
    if compare(key, array) <= 0:
        return 0;
    
    if compare(key, array + size * (count - 1)) >= 0:
        return count - 1;

    cdef size_t lo = 0;
    cdef size_t hi = count; 
    cdef size_t mid = count; 
    cdef int res;

    while lo < hi:
        mid = lo + (hi - lo) // 2;
        res = compare(key, array + size * mid) 

        if res < 0:
            hi = mid - 1
        elif res > 0:
            lo = mid + 1
        else:
            # Key is equal to mid, move left until this changes
            while mid > 0: 
                if compare(key, array + size * (mid - 1)) > 0 or mid == 0:
                    break

                mid -= 1

            return mid;

    if hi < 0:
        return 0

    # Move right if we overshot
    while compare(key, array + size * hi) > 0 and hi < (count - 1): 
        hi += 1 

    # Move left if we overshot
    while compare(key, array + size * (hi - 1)) <= 0 and hi > 0: 
        hi -= 1

    return hi;



cdef (unsigned long, unsigned long) select_temporal_window(
    const unsigned long [::1] times,
    const unsigned long datetime,
    const unsigned long time_window_lower, 
    const unsigned long time_window_upper) nogil:

    cdef unsigned long arr_length = times.shape[0]

    cdef unsigned long start_time = datetime - time_window_lower
    cdef unsigned long end_time = datetime + time_window_upper

    cdef size_t start_ind = search(&start_time, &(times[0]), arr_length, sizeof(unsigned long), &compare_times)

    cdef unsigned long end_ind = start_ind

    while end_ind < (arr_length - 1):
        if times[end_ind + 1] >= end_time:
            break;

        end_ind += 1

    return start_ind, end_ind

cdef void select_spatial_window(
    const double [:,::1] xys,
    const unsigned long start_ind,
    const unsigned long end_ind,
    const double center_x,
    const double center_y,
    const unsigned int window_size,
    Vector *inds,
    ) nogil:

    cdef unsigned long max_len = end_ind - start_ind

    cdef double start_x = center_x - window_size / 2
    cdef double end_x = center_x + window_size / 2

    cdef double start_y = center_y - window_size / 2
    cdef double end_y = center_y + window_size / 2

    cdef double x, y
    cdef size_t i

    # inds: Vector* = []

    for i in range(start_ind, end_ind):
        x = xys[i, 0]
        y = xys[i, 1]

        if (start_x <= x <= end_x) and (start_y <= y <= end_y):
            # ind_list.append(i)
            push_item(inds, i)


cdef void select_spatiotemporal_window(
    const unsigned long [::1] times,
    const double [:,::1] xys,
    const unsigned long datetime,
    const unsigned long time_window_lower,
    const unsigned long time_window_upper,
    const double center_x,
    const double center_y,
    const unsigned int window_size,
    Vector* inds) nogil:

    cdef unsigned long start_ind, end_ind

    start_ind, end_ind = select_temporal_window(times, datetime, time_window_lower, time_window_upper)

    if start_ind != end_ind:
        select_spatial_window(xys, start_ind, end_ind, center_x, center_y, window_size, inds)

