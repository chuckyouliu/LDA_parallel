cimport numpy as np
import numpy as np
cimport cython
from cython.parallel import prange, parallel
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc, free

'''
Collapsed Gibbs Sampling
- Dynamic sampling (http://jmlr.csail.mit.edu/proceedings/papers/v13/xiao10a/xiao10a.pdf)
'''
@cython.boundscheck(False)
cpdef void CGS(int[:, ::1] documents, double[:, ::1] K_V, double[:, ::1] D_K, int[::1] sum_K, int[::1] curr_K, double alpha, double beta, unsigned int n) nogil:
    cdef size_t i, j, k, it, p
    # number of documents
    cdef int D = documents.shape[0]
    # number of types
    cdef int V = documents.shape[1]
    # number of topics
    cdef int K = K_V.shape[0]
    # malloc K integers to store probability distribution
    cdef double* p_K = <double*> malloc(K*sizeof(double))
    cdef double cumulative_p
    cdef double randnum
        
    #initialize random topics to start
    cdef int count = 0
    cdef int topic = 0
    for i in range(D):
        for j in range(V):
            for k in range(documents[i,j]):
                topic = count % K
                curr_K[count] = topic
                inc(sum_K[topic])
                inc(K_V[topic, j])
                inc(D_K[i, topic])
                inc(count)
    
    #proceed to Gibbs sampling
    for it in range(n):
        count = 0
        for i in range(D):
            for j in range(V):
                for k in range(documents[i,j]):
                    #retrieve current topic and decrement
                    topic = curr_K[count]
                    dec(sum_K[topic])
                    dec(K_V[topic,j])
                    dec(D_K[i,topic])
                    
                    cumulative_p = 0
                    for p in range(K):
                        cumulative_p += (D_K[i, p] + alpha)*(K_V[p,j] + beta)/(sum_K[p] + V*beta)
                        p_K[p] = cumulative_p
                    randnum = rand()/float(RAND_MAX)*cumulative_p
                    topic = get_index(p_K, randnum, K)
                    
                    #assign new topic
                    curr_K[count] = topic
                    inc(sum_K[topic])
                    inc(K_V[topic,j])
                    inc(D_K[i, topic])
                    inc(count)
    free(p_K)
    
    #normalize K_V and D_K
    for i in range(K):
        cumulative_p = 0
        for j in range(V):
            K_V[i,j] += beta
            cumulative_p += K_V[i,j]
        for j in range(V):
            K_V[i,j] /= cumulative_p
    for i in range(D):
        cumulative_p = 0
        for j in range(K):
            D_K[i,j] += alpha
            cumulative_p += D_K[i,j]
        for j in range(K):
            D_K[i,j] /= cumulative_p
        
    
@cython.boundscheck(False)
cdef int get_index(double* p_K, double randnum, int K) nogil:
    cdef int mink = 0
    cdef int maxk = K-1
    cdef int midk
    while mink < maxk:
        midk = mink + (maxk-mink)/2
        if p_K[midk] < randnum:
            mink = midk + 1
        else:
            maxk = midk
    return maxk