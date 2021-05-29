from __future__ import print_function
cimport numpy as np
import numpy as np

cpdef np.ndarray[np.int, ndim=1] fast_filter(node, ratings):
    cdef int len_node = len(node)
    cdef int len_rate = len(ratings)
    cdef int i = 0
    cdef int j = 0
    cdef np.ndarray[np.uint8_t, ndim=1] mask = np.zeros((len_rate), dtype=np.uint8)
    while i != len_rate and j != len_node:
        if ratings[i] == node[j]:
            mask[i] = 1
            i += 1
        elif ratings[i] > node[j]:
            j += 1
        else:
            i += 1
    return mask


def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b

    print()