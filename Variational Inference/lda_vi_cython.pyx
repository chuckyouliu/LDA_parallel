#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from libc.math cimport exp, log
from libc.stdlib cimport malloc
from cython.parallel import parallel, prange


cpdef double [:] parallel_test(double[:] temp, int num_threads):
    cdef unsigned int i, k, N = temp.shape[0]
    cdef double a

    with nogil, parallel(num_threads=num_threads):
        for i in prange(N):
            a = 0
            for k in range(1000):
                a = a + i
            temp[i] = a

    return temp


cdef double digamma(double x) nogil:
    '''
    Approximate Psi function
    '''
    cdef double result = 0, xx, xx2, xx4

    while x < 7:
        result -= 1/x
        x += 1
    x -= 1.0/2.0
    xx = 1.0/x
    xx2 = xx*xx
    xx4 = xx2*xx2
    result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4
    return result


cdef double dot(double[:, ::1] v1, int* v2, unsigned int N, unsigned int j) nogil:
    cdef double result = 0
    cdef unsigned int i = 0
    cdef double el1 = 0
    cdef double el2 = 0
    # We assume N < min(len(v1, v2))
    for i in range(N):
        el1 = v1[j, i]
        el2 = v2[i]
        result += el1*el2
    return result


cdef void exp_digamma_vec(double[:, ::1] data, double[::1] dig, unsigned int d) nogil:
    cdef:
        unsigned int i
        unsigned int N = data.shape[1]
        double s = 0
        double val

    for i in range(N):
        val = data[d, i]
        dig[i] = digamma(val)
        s += val

    s = digamma(s)

    for i in range(N):
        dig[i] = exp(dig[i] - s)


cdef void exp_digamma_arr(double[:, ::1] data, double[:, ::1] dig_arr) nogil:
    cdef:
        unsigned int i, j
        unsigned int N = data.shape[0]
        unsigned int M = data.shape[1]
        float s, d

    for i in range(N):
        s = 0
        for j in range(M):
            d = data[i, j]
            dig_arr[i, j] = digamma(d)
            s += d

        s = digamma(s)

        for j in range(M):
            dig_arr[i, j] = exp(dig_arr[i, j] - s)


cpdef void lda_batch(long[:, ::1] dtm, unsigned int ntopic, unsigned int batch_size,
                unsigned int num_threads, float tau, float kappa, double[:, ::1] lambda_,
                double[:, ::1] gamma,  double[:, ::1] lambda_int,
                double[:, ::1] phi, double[::1] ExpLogTethad,
                double[:, ::1] ExpELogBeta) nogil:
    '''
    Online variational inference lda with mini-batch
    '''
    cdef:
        unsigned int ndoc, nvoc, it_batch, k, d, inner_it, numbatch, j_v, j_t, j_v_r, nvoc_loc, batch_pos
        float eta, alpha, a_dot, err, diff, el1, el2, s, rt

    ndoc = dtm.shape[0]
    nvoc = dtm.shape[1]
    eta = 1./float(ntopic)
    alpha = eta

    # Allocate memory for current counts and ids on the document
    cdef int* counts = <int*> malloc(nvoc*sizeof(int))
    cdef int* ids = <int*> malloc(nvoc*sizeof(int))
    cdef int* indices = <int*> malloc(nvoc*sizeof(int))

    numbatch = ndoc / batch_size

    for it_batch in range(numbatch):
        # Current batch number
        batch_pos = it_batch*batch_size
        lambda_int[:, :] = 0.

        exp_digamma_arr(lambda_, ExpELogBeta)

        # Initializing the indices of the words in the whole batch
        for j_v in range(nvoc):
            indices[j_v] = 0

        # TODO: parallelize
        # Going through the documents in the batch
        for k in prange(batch_size, nogil=True, schedule='guided', num_threads=num_threads):
            d = batch_pos + k

            # Number of words in the current doc
            nvoc_loc = 0

            # ids : index of the words in the current doc
            # counts : counts of the words
            for j_v in range(nvoc):
                # word j_v in document
                if dtm[d, j_v] > 0:
                    ids[nvoc_loc] = j_v
                    counts[nvoc_loc] = dtm[d, j_v]
                    indices[j_v] = 1
                    # Reduction variable
                    nvoc_loc = nvoc_loc + 1

            for inner_it in range(1000):

                exp_digamma_vec(gamma, ExpLogTethad, d)

                # Computing phi
                for j_v in range(nvoc_loc):
                    s = 0
                    for j_t in range(ntopic):
                        el1 = ExpELogBeta[j_t, ids[j_v]]
                        el2 = ExpLogTethad[j_t]
                        phi[j_t, j_v] = el1 * el2
                        s += el1 * el2
                    # Normalizing phi
                    for j_t in range(ntopic):
                        phi[j_t, j_v] /= (s + 0.00000001)

                # Final criterion
                err = 0
                # Updating gamma
                for j_t in range(ntopic):
                    a_dot = alpha + dot(phi, counts, nvoc_loc, j_t)
                    # Computing the squared diff for the errors
                    diff = a_dot - gamma[d, j_t]
                    err += diff * diff
                    gamma[d, j_t] = a_dot

                # Reduction variable
                err = err / (1. * ntopic)

                if err < 0.0000001:
                    break

            # print inner_it

            for j_v in range(nvoc_loc):
                j_v_r = ids[j_v]
                for j_t in range(ntopic):
                    el1 = counts[j_v]
                    el2 = phi[j_t, j_v]
                    lambda_int[j_t, j_v_r] += el1 * el2

        rt = (tau + it_batch)**(- kappa)

        # Updating the topics
        for j_v in range(nvoc):
            if indices[j_v]:
                for j_t in range(ntopic):
                    lambda_[j_t, j_v] *= (1 - rt)
                    el1 = lambda_int[j_t, j_v]
                    lambda_[j_t, j_v] += rt * ndoc * (eta + el1) / batch_size