cimport numpy as np
import numpy as np
cimport cython
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc, free

'''
Initial topic sampling for Gibbs sampling
'''
@cython.boundscheck(False)
cpdef int init_topics(int[:, ::1] documents, double[:, ::1] K_V, double[:, ::1] D_K, int[::1] sum_K, int[::1] curr_K, int thread_num, int num_threads, int d_interval, int doc_num, int w_interval, int word_num, int count) nogil:
    cdef size_t j,k
    # number of topics
    cdef int K = K_V.shape[0]
    cdef int topic = 0    
    cdef int w_start = ((thread_num+word_num)%num_threads)*w_interval
    cdef int w_end = min(documents.shape[1], w_start + w_interval)
    cdef int document = thread_num*d_interval + doc_num
    
    if document < documents.shape[0]:
        for j in xrange(w_start, w_end):
            for k in xrange(documents[document,j]):
                topic = count % K
                curr_K[count] = topic
                inc(sum_K[topic])
                inc(K_V[topic, j])
                inc(D_K[document, topic])
                inc(count)
    return count

'''
Normalization with increment mu
'''
@cython.boundscheck(False)
cpdef void normalize(double[:, ::1] arr, double mu) nogil:
    cdef double cumulative_p
    cdef size_t i, j
    for i in xrange(arr.shape[0]):
        cumulative_p = 0
        for j in xrange(arr.shape[1]):
            arr[i,j] += mu
            cumulative_p += arr[i,j]
        for j in xrange(arr.shape[1]):
            arr[i,j] /= cumulative_p
            
'''
Array methods in cython
'''
@cython.boundscheck(False)
cpdef void copy(int[::1] sum_K, int[::1] t_sum_K) nogil:
    cdef size_t i
    for i in xrange(sum_K.shape[0]):
        t_sum_K[i] = sum_K[i]
        
@cython.boundscheck(False)
cpdef void subtract(int[::1] t_sum_K, int[::1] sum_K) nogil:
    cdef size_t i
    for i in xrange(sum_K.shape[0]):
        t_sum_K[i] -= sum_K[i]
        
@cython.boundscheck(False)
cpdef void add(int[::1] sum_K, int[::1] t_sum_K) nogil:
    cdef size_t i
    for i in xrange(sum_K.shape[0]):
        sum_K[i] += t_sum_K[i]

'''
Single iteration of CGS with dynamic sampling with specified document and bounds on words
'''
@cython.boundscheck(False)
cpdef int CGS_iter(int[:, ::1] documents, double[:, ::1] K_V, double[:, ::1] D_K, int[::1] sum_K, int[::1] curr_K, double alpha, double beta, int[:, :, ::1] sampling, double[::1] p_K, int[::1] uniq_K, int thread_num, int num_threads, int d_interval, int doc_num, int w_interval, int word_num, int count) nogil:
    cdef size_t j,k,p
    # number of types
    cdef int V = documents.shape[1]
    # number of topics
    cdef int K = K_V.shape[0]
    cdef double cumulative_p, randnum
    cdef int num_samples, topic
    cdef int w_start = ((thread_num+word_num)%num_threads)*w_interval
    cdef int w_end = min(documents.shape[1], w_start + w_interval)
    cdef int document = thread_num*d_interval + doc_num
    
    if document < documents.shape[0]:
        for j in xrange(w_start, w_end):
            #check if document has type j, if so go to sampling procedure
            if documents[document,j] > 0:
                num_samples = documents[document,j]
                if num_samples > 2:
                    #more than 2 occurrences, so do dynamic sampling procedure
                    cumulative_p = 0
                    for p in xrange(num_samples):
                        cumulative_p += sampling[document,j,p]
                    #choose random number between 0 and sum of samples
                    #if cumulative_p is 0, must initialize the sampling vector
                    if cumulative_p == 0:
                        sampling[document,j,num_samples-1] = 1
                        cumulative_p = 1
                    randnum = rand()/float(RAND_MAX)*cumulative_p
                    num_samples = -1
                    cumulative_p = 0
                    while cumulative_p < randnum and num_samples < documents[document,j]:
                        cumulative_p += sampling[document,j,inc(num_samples)]
                #only looping through num_samples times, reset uniq_k
                for k in xrange(K):
                    uniq_K[k] = 0
                for k in xrange(num_samples):
                    #retrieve current topic and decrement
                    topic = curr_K[count]
                    dec(sum_K[topic])
                    dec(K_V[topic,j])
                    dec(D_K[document,topic])
                    
                    cumulative_p = 0
                    for p in xrange(K):
                        cumulative_p += (D_K[document, p] + alpha)*(K_V[p,j] + beta)/(sum_K[p] + V*beta)
                        p_K[p] = cumulative_p
                    randnum = rand()/float(RAND_MAX)*cumulative_p
                    topic = get_index(p_K, randnum, K)
                    
                    #assign new topic
                    curr_K[count] = topic
                    inc(sum_K[topic])
                    inc(K_V[topic,j])
                    inc(D_K[document, topic])
                    inc(count)
                    
                    #increment index in uniq_k
                    inc(uniq_K[topic])
                #make sure to keep count consistent with documents table
                count += documents[document,j] - num_samples
                #find number of unique topics used and increment index
                num_samples = 0
                for k in xrange(K):
                    if uniq_K[k] != 0:
                        inc(num_samples)
                inc(sampling[document,j,num_samples])
    return count

'''
Initial Serial Collapsed Gibbs Sampling
- Dynamic sampling (http://jmlr.csail.mit.edu/proceedings/papers/v13/xiao10a/xiao10a.pdf)
'''
@cython.boundscheck(False)
cpdef void CGS(int[:, ::1] documents, double[:, ::1] K_V, double[:, ::1] D_K, int[::1] sum_K, int[::1] curr_K, double alpha, double beta, unsigned int n, int[:, :, ::1] sampling, double[::1] p_K, int[::1] uniq_K) nogil:
    cdef size_t i, j, k, it, p
    # number of documents
    cdef int D = documents.shape[0]
    # number of types
    cdef int V = documents.shape[1]
    # number of topics
    cdef int K = K_V.shape[0]
    cdef double cumulative_p
    cdef double randnum
    cdef int num_samples, count, topic    
        
    #initialize random topics to start
    count = 0
    for i in xrange(documents.shape[0]):
        for j in xrange(documents.shape[1]):
            for k in xrange(documents[i,j]):
                topic = count % K
                curr_K[count] = topic
                inc(sum_K[topic])
                inc(K_V[topic, j])
                inc(D_K[i, topic])
                inc(count) 
    
    #proceed to Gibbs sampling
    for it in range(n):
        count = 0
        for i in xrange(D):
            for j in xrange(V):
                #check if document has type j, if so go to sampling procedure
                if documents[i,j] > 0:
                    num_samples = documents[i,j]
                    if num_samples > 2:
                        #more than 2 occurrences, so do dynamic sampling procedure
                        cumulative_p = 0
                        for p in xrange(num_samples):
                            cumulative_p += sampling[i,j,p]
                        #choose random number between 0 and sum of samples
                        #if cumulative_p is 0, must initialize the sampling vector
                        if cumulative_p == 0:
                            sampling[i,j,num_samples-1] = 1
                            cumulative_p = 1
                        randnum = rand()/float(RAND_MAX)*cumulative_p
                        num_samples = -1
                        cumulative_p = 0
                        while cumulative_p < randnum and num_samples < documents[i,j]:
                            cumulative_p += sampling[i,j,inc(num_samples)]
                    #only looping through num_samples times, reset uniq_k
                    for k in xrange(K):
                        uniq_K[k] = 0
                    for k in xrange(num_samples):
                        #retrieve current topic and decrement
                        topic = curr_K[count]
                        dec(sum_K[topic])
                        dec(K_V[topic,j])
                        dec(D_K[i,topic])
                        
                        cumulative_p = 0
                        for p in xrange(K):
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
                        
                        #increment index in uniq_k
                        inc(uniq_K[topic])
                    #make sure to keep count consistent with documents table
                    count += documents[i,j] - num_samples
                    #find number of unique topics used and increment index
                    num_samples = 0
                    for k in xrange(K):
                        if uniq_K[k] != 0:
                            inc(num_samples)
                    inc(sampling[i,j,num_samples])                    
    
    #normalize K_V and D_K
    for i in xrange(K):
        cumulative_p = 0
        for j in xrange(V):
            K_V[i,j] += beta
            cumulative_p += K_V[i,j]
        for j in xrange(V):
            K_V[i,j] /= cumulative_p
    for i in xrange(D):
        cumulative_p = 0
        for j in xrange(K):
            D_K[i,j] += alpha
            cumulative_p += D_K[i,j]
        for j in xrange(K):
            D_K[i,j] /= cumulative_p  
        
    
@cython.boundscheck(False)
cdef int get_index(double[::1] p_K, double randnum, int K) nogil:
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
    