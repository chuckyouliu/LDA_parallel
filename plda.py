import cyplda
import numpy as np
import threading
import time

class LDA:
    def __init__(self, num_topics, iterations = 500, damping = 1, sync_interval = 1):
        self.num_topics = num_topics
        self.iterations = iterations
        self.damping = damping
        self.sync_interval = sync_interval
        
    def set_topics(self, n):
        self.num_topics = n
        
    def set_iterations(self, i):
        self.iterations = i
        
    def set_damping(self, d):
        self.damping = d
        
    def set_sync_interval(self, s):
        self.sync_interval = s
        
    #baseline serial cython CGS
    def sCGS(self, documents, alpha=None, beta=None):
        if alpha is None or alpha <= 0:
            alpha = 50./self.num_topics
        if beta is None or beta <= 0:
            beta = 0.1
        
        # topic -> words distribution
        K_V = np.zeros((self.num_topics,documents.shape[1]), dtype=np.float)
        # documents -> topic distribution
        D_K = np.zeros((documents.shape[0], self.num_topics), dtype=np.float)
        # sum of types per topic
        sum_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        # current topic for ith word in corpus
        curr_K = np.zeros((np.sum(documents)), dtype=np.dtype("i"))
        # sampling distributions
        sampling = np.zeros((documents.shape[0], documents.shape[1], np.max(documents)), dtype=np.dtype("i"))
        # vectors for probability and topic counts
        p_K = np.zeros((self.num_topics), dtype=np.float)
        uniq_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        cyplda.CGS(documents, K_V, D_K, sum_K, curr_K, alpha, beta, self.iterations, sampling, p_K, uniq_K)
        self.K_V = K_V
        self.D_K = D_K
        
    def pCGS(self, documents, num_threads = 4, alpha=None, beta=None):
        if alpha is None or alpha <= 0:
            alpha = 50./self.num_topics
        if beta is None or beta <= 0:
            beta = 0.1
            
        # topic -> words distribution
        K_V = np.zeros((self.num_topics,documents.shape[1]), dtype=np.float)
        # documents -> topic distribution
        D_K = np.zeros((documents.shape[0], self.num_topics), dtype=np.float)
        # sum of types per topic
        sum_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        # current topic for ith word in corpus
        curr_K = np.zeros((np.sum(documents)), dtype=np.dtype("i"))
        # sampling distributions
        sampling = np.zeros((documents.shape[0], documents.shape[1], np.max(documents)), dtype=np.dtype("i"))
        
        start = time.time()
        cyplda.init_topics(documents, K_V, D_K, sum_K, curr_K)
        init_end = time.time()
        print "Init:{}".format(init_end - start)
        #vector of threads
        tList = [None]*num_threads
        #update iterations on threads
        updateCount = [0]
        #copy iterations on threads
        copyCount = [0]
        #condition objects to synchronize iList
        updateCondition = threading.Condition()
        copyCondition = threading.Condition()
        for i in range(num_threads):
            tList[i] = threading.Thread(target=self.workerCGS, args=(updateCondition, copyCondition, updateCount, copyCount, i, num_threads, documents, K_V, D_K, sum_K, curr_K, alpha, beta, sampling))
            tList[i].start()
        for i in range(num_threads):
            tList[i].join()
            
        cgs_end = time.time()
        print "CGS:{}".format(cgs_end-init_end)
        
        cyplda.normalize(K_V, beta)
        cyplda.normalize(D_K, alpha)           
        
        norm_end = time.time()
        print "Norm:{}".format(norm_end - cgs_end)
            
        self.K_V = K_V
        self.D_K = D_K
        assert np.sum(sum_K) == np.sum(documents), "Not syncing correctly"
        
    def workerCGS(self, updateCondition, copyCondition, updateCount, copyCount, thread_num, num_threads, documents, K_V, D_K, sum_K, curr_K, alpha, beta, sampling):
        # vectors for probability and topic counts
        p_K = np.zeros((self.num_topics), dtype=np.float)
        uniq_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        
        #create a copy of sum_K to work over
        t_sum_K = sum_K.copy()
        
        for i in xrange(self.iterations):
            cyplda.CGS_iter(documents, K_V, D_K, t_sum_K, curr_K, alpha, beta, sampling, p_K, uniq_K, thread_num, num_threads)
            
            if i%self.sync_interval == 0:
                #must synchronize sum_K
                cyplda.subtract(t_sum_K, sum_K)
                #notify threads waiting on condition that you've completed this sync iteration
                #threads will continue in wait loop until all threads have completed their sync iteration
                with updateCondition:
                    updateCount[0] += 1
                    if updateCount[0]%num_threads == 0:
                        updateCondition.notifyAll()
                    
                #wait for all threads to reach this point, then one at a time update sum_K
                with updateCondition:
                    while updateCount[0]%num_threads != 0:
                        updateCondition.wait()
                    cyplda.add(sum_K, t_sum_K)
                    with copyCondition:
                        copyCount[0] += 1
                        if copyCount[0]%num_threads == 0:
                            copyCondition.notifyAll()
                    
                #at this point need to wait for all threads to update sum_K with their changes
                #once all threads reach this point it's safe to copy sum_K to t_sum_K
                with copyCondition:
                    while copyCount[0]%num_threads != 0:
                        copyCondition.wait()
                    cyplda.copy(sum_K, t_sum_K)
            