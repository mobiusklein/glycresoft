cimport cython

from libc.math cimport pow, sqrt

cimport numpy as np

np.import_array()


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef (np.float32_t, np.float32_t) delta_distribution_parameters(np.float32_t[::1] apex_times, np.float32_t[::1] weights):
    cdef:
        Py_ssize_t i, n, size
        np.float32_t time, weight, acc, weight_acc, mean_time, time_stddev

    n = len(apex_times)
    with nogil:
        acc = 0
        weight_acc = 0
        size = 0
        for i in range(n):
            time = apex_times[i]
            weight = weights[i]
            acc += time * weight
            weight_acc += weight
        mean_time = acc / weight_acc
        acc = 0
        for i in range(n):
            time = apex_times[i]
            acc += pow(time - mean_time, 2) * weight
        time_stddev = sqrt(acc / (weight_acc - 1))
    return mean_time, time_stddev


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef (np.float32_t, np.float32_t, Py_ssize_t) filtered_delta_distribution_parameters(
                                                            np.float32_t[::1] apex_times,
                                                            np.float32_t[::1] weights,
                                                            cython.floating upper,
                                                            cython.floating lower):
    cdef:
        Py_ssize_t i, n, size
        np.float32_t time, weight, acc, weight_acc, mean_time, time_stddev

    n = len(apex_times)
    with nogil:
        acc = 0
        weight_acc = 0
        size = 0
        for i in range(n):
            time = apex_times[i]
            if time > lower and time < upper:
                weight = weights[i]
                acc += time * weight
                weight_acc += weight
                size += 1
        if size == 0:
            return 0, 1, 0
        mean_time = acc / weight_acc
        acc = 0
        for i in range(n):
            time = apex_times[i]
            if time > lower and time < upper:
                acc += pow(time - mean_time, 2) * weight
        time_stddev = sqrt(acc / (weight_acc - 1))
    return mean_time, time_stddev, size

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef (double, double, double) posprocess_linear_fit(np.ndarray[np.float64_t, ndim=1, mode='c'] y,
                                                     np.float64_t[::1] residuals_,
                                                     np.float64_t[::1] w,
                                                     np.float64_t[::1] diag_H_):
    cdef:
        Py_ssize_t i, n, size
        np.float64_t[::1] y_
        double press
        double rss
        double tss
        double ybar
        double leave_one_out_err

    y_ = y
    ybar = y.mean()

    with nogil:
        press = 0.0
        tss = 0.0
        rss = 0.0
        n = y_.shape[0]

        for i in range(n):
            leave_one_out_err = (residuals_[i] / (1 - diag_H_[i]))
            press += w[i] * pow(leave_one_out_err, 2)
            rss += w[i] * pow(residuals_[i], 2)
            tss += w[i] * pow(y_[i] - ybar, 2)
    return (press, rss, tss)
