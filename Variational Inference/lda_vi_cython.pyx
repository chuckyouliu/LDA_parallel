#cython: boundscheck=False, wraparound=False, nonecheck=False
from scipy.special._ufuncs import psi
from libc.math cimport exp
from libc.stdlib cimport malloc
cimport numpy as np
from cython cimport view


# cdef digamma_vec(np.ndarray[np.float64_t, ndim=1] data):
#     return psi(data) - psi(data.sum())

cdef double dot(double[::1] v1, int[::1] v2, int N):
    cdef double result = 0
    cdef int i = 0
    cdef double el1 = 0
    cdef double el2 = 0
    # We assume N < min(len(v1, v2))
    for i in range(N):
        el1 = v1[i]
        el2 = v2[i]
        result += el1*el2
    return result


cdef double[::1] exp_digamma_vec_new(double[::1] data):
    cdef:
        unsigned int i
        unsigned int N = data.shape[0]
        double[::1] dig = view.array(shape=(N,), itemsize=sizeof(double), format="d")
        double s = 0
        double d

    for i in range(N):
        d = data[i]
        dig[i] = psi(d)
        s += d

    s = psi(s)

    for i in range(N):
        dig[i] = exp(dig[i] - s)

    return dig


cdef double[:, ::1] exp_digamma_arr_new(double[:, ::1] data):
    cdef:
        unsigned int i, j
        unsigned int N = data.shape[0]
        unsigned int M = data.shape[1]
        double[:, ::1] dig_arr = view.array(shape=(N,M), itemsize=sizeof(double), format="d")
        float s, d

    for i in range(N):
        s = 0
        for j in range(M):
            d = data[i, j]
            dig_arr[i, j] = psi(d)
            s += d

        s = psi(s)

        for j in range(M):
            dig_arr[i, j] = exp(dig_arr[i, j] - s)

    return dig_arr



# cdef digamma_arr(np.ndarray[np.float64_t, ndim=2] data):
#     return psi(data) - psi(np.sum(data, axis=1))[:, np.newaxis]


cpdef lda_batch(double[:, ::1] lambda_, long[:, ::1] dtm, unsigned int ntopic, unsigned int batch_size, float tau, float kappa):
    '''
    Online variational inference lda with mini-batch
    '''
    cdef:
        unsigned int ndoc, nvoc, it_batch, k, d, inner_it, numbatch, j_v, j_t, j_v_r, nvoc_loc, batch_pos
        double[:, ::1] ExpELogBeta
        double[::1] ExpLogTethad
        float eta, alpha, a_dot, err, diff, el1, el2, s, rt

    ndoc = dtm.shape[0]
    nvoc = dtm.shape[1]
    eta = 1./float(ntopic)
    alpha = eta

    # # Allocate memory for topic distribution on the current document
    # cdef double* gammad = <double*> malloc(ntopic*sizeof(double))
    # # Allocate memory for current counts and ids on the document
    # cdef int* counts = <int*> malloc(nvoc*sizeof(int))
    # cdef int* ids = <int*> malloc(nvoc*sizeof(int))
    
    cdef int[::1] counts = view.array(shape=(nvoc,), itemsize=sizeof(int), format="i")
    cdef int[::1] ids = view.array(shape=(nvoc,), itemsize=sizeof(int), format="i")
    cdef int[::1] indices = view.array(shape=(nvoc,), itemsize=sizeof(int), format="i")
    cdef double[:, ::1] gamma = view.array(shape=(ndoc, ntopic), itemsize=sizeof(double), format="d")
    cdef double[:, ::1] lambda_int = view.array(shape=(ntopic, nvoc), itemsize=sizeof(double), format="d")
    cdef double[:, ::1] ExpELogBetad = view.array(shape=(ntopic, nvoc), itemsize=sizeof(double), format="d")
    cdef double[:, ::1] phi = view.array(shape=(ntopic, nvoc), itemsize=sizeof(double), format="d")

    # Initialization
    # lambda_ = np.random.gamma(100., 1./100., (ntopic, nvoc))
    gamma[:, :] = 1

    print lambda_[0, 0]
    numbatch = ndoc / batch_size
    docs = range(ndoc)

    for it_batch in range(numbatch):
        # Current batch number
        batch_pos = it_batch*batch_size

        ExpELogBeta = exp_digamma_arr_new(lambda_)

        lambda_int[:, :] = 0.

        # Initializing the indices of the words in the batch
        for j_v in range(nvoc):
            indices[j_v] = 0

        # Going over the batch documents
        for k in range(batch_size):
            d = docs[batch_pos + k]

            # Number of words in the current doc
            nvoc_loc = 0

            # ids : index of the present words
            # counts : counts of the words
            for j_v in range(nvoc):
                # word j_v in document
                if dtm[d, j_v] > 0:
                    ids[nvoc_loc] = j_v
                    counts[nvoc_loc] = dtm[d, j_v]
                    indices[j_v] = 1
                    # Isolating the expElogBetad on the present words
                    for j_t in range(ntopic):
                        ExpELogBetad[j_t, nvoc_loc] = ExpELogBeta[j_t, j_v]
                    nvoc_loc += 1

            for inner_it in range(1000):

                ExpLogTethad = exp_digamma_vec_new(gamma[d])

                # Computing phi
                for j_v in range(nvoc_loc):
                    s = 0
                    for j_t in range(ntopic):
                        el1 = ExpELogBetad[j_t, j_v]
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
                    a_dot = alpha + dot(phi[j_t], counts, nvoc_loc)
                    # Computing the squared diff for the errors
                    diff = a_dot - gamma[d, j_t]
                    err += diff * diff
                    gamma[d, j_t] = a_dot

                err /= (1. * ntopic)
                #print avg_err
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

    return lambda_, gamma
