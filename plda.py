import cyplda
import numpy as np
import threading

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
        
        
    '''
        Implement a parallel version of CGS similar to:
            https://www.cs.purdue.edu/homes/alanqi/papers/Parallel-Inf-LDA-GPU-NIPS.pdf
        with the variation of instead of having num_threads separate iterations
        we utilize locking mechanisms to do the calculations all at once
    '''
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
        # sampling distributions
        sampling = np.zeros((documents.shape[0], documents.shape[1], np.max(documents)), dtype=np.dtype("i"))
        
        #vector of threads
        tList = [None]*num_threads
        #count arrays to synchronize
        updateCount = [0]
        copyCount = [0]
        #condition objects to synchronize, both required as one thread may be waiting in a copy region while another notifies in an update region
        updateCondition = threading.Condition()
        copyCondition = threading.Condition()
        for i in range(num_threads):
            tList[i] = threading.Thread(target=self.workerCGS, args=(updateCondition, copyCondition, updateCount, copyCount, i, num_threads, documents, K_V, D_K, sum_K, alpha, beta, sampling))
            tList[i].start()
        for i in range(num_threads):
            tList[i].join()
            
        cyplda.normalize(K_V, beta)
        cyplda.normalize(D_K, alpha)           
        
        self.K_V = K_V
        self.D_K = D_K
        assert np.sum(sum_K) == np.sum(documents), "Not syncing correctly: {}, {}".format(np.sum(sum_K), np.sum(documents))
        
    def workerCGS(self, updateCondition, copyCondition, updateCount, copyCount, thread_num, num_threads, documents, K_V, D_K, sum_K, alpha, beta, sampling):
        # vectors for probability and topic counts
        p_K = np.zeros((self.num_topics), dtype=np.float)
        uniq_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        
        #create a copy of sum_K to work over
        t_sum_K = np.zeros(sum_K.shape, dtype=np.dtype("i"))
        
        #specify the boundaries of documents to work over
        d_interval = (documents.shape[0] + (documents.shape[0]%num_threads))/num_threads
        d_start = thread_num*d_interval
        d_end = min(documents.shape[0], (thread_num+1)*d_interval)
        w_interval = (documents.shape[1] + (documents.shape[1]%num_threads))/num_threads
        
        #create a custom curr_K that maps to the document boundaries
        curr_K = np.zeros((np.sum(documents[d_start:d_end])), dtype=np.dtype("i"))
        
        #initialize topics, we separate a window of words to do at a time for each
        #thread doc and we loop until we do all words. Each thread must interact with a different
        #window of words simultaneously so there are no conflicts with K_V        
        count = 0
        for doc_num in xrange(d_interval):
            for word_num in xrange(num_threads):
                count = cyplda.init_topics(documents, K_V, D_K, t_sum_K, curr_K, thread_num, num_threads, d_interval, doc_num, w_interval, word_num, count)
                self.updateConditionCheck(updateCount, num_threads, updateCondition)
        
        #have sum_K be the sum of all thread-specific t_sum_K's
        with copyCondition:
            cyplda.add(sum_K, t_sum_K)
            self.copyConditionCheck(copyCount, num_threads, copyCondition) 
        
        #have t_sum_K be a copy of the summed sum_K
        cyplda.copy(sum_K, t_sum_K)
        
        #start the gibb sampling iterations
        for i in xrange(self.iterations):
            count = 0
            for doc_num in xrange(d_interval):
                for word_num in xrange(num_threads):
                    count = cyplda.CGS_iter(documents, K_V, D_K, t_sum_K, curr_K, alpha, beta, sampling, p_K, uniq_K, thread_num, num_threads, d_interval, doc_num, w_interval, word_num, count)
                    self.updateConditionCheck(updateCount, num_threads, updateCondition)
            
            #wait for all threads to finish one run before continuing
            with copyCondition:
                self.copyConditionCheck(copyCount, num_threads, copyCondition)
                
            if i%self.sync_interval == 0:
                #must synchronize sum_K                    
                #this subtraction can be done in parallel as sum_K unmodified and then wait for every thread to do that
                cyplda.subtract(t_sum_K, sum_K)                
                with copyCondition:
                    self.copyConditionCheck(copyCount, num_threads, copyCondition)
                    
                #one at a time update sum_K
                with copyCondition:
                    cyplda.add(sum_K, t_sum_K)
                    self.copyConditionCheck(copyCount, num_threads, copyCondition)
                    
                #at this point need to wait for all threads to update sum_K with their changes
                #once all threads reach this point it's safe to copy sum_K to t_sum_K
                cyplda.copy(sum_K, t_sum_K)
        
    def copyConditionCheck(self, copyCount, num_threads, copyCondition):
        copyCount[0] +=1
        if copyCount[0] == num_threads:
            copyCount[0] = 0
            copyCondition.notifyAll()
        else:
            copyCondition.wait()
            
    def updateConditionCheck(self, updateCount, num_threads, updateCondition):
        with updateCondition:
            updateCount[0] += 1
            if updateCount[0] == num_threads:
                updateCount[0] = 0
                updateCondition.notifyAll()
            else:
                updateCondition.wait()
            
    
        