import cyplda
import numpy as np
import threading

class LDA:
    def __init__(self, num_topics, iterations = 500, damping = 1, sync_interval = 1):
        assert sync_interval < iterations, "Cannot have sync_interval greater than iterations"
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
        assert s < self.iterations, "Cannot have sync_interval greater than iterations"
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
        Implement a parallel version of CGS over documents
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
        #count array to synchronize
        copyCount = [0]
        #condition object to synchronize
        copyCondition = threading.Condition()
        for i in range(num_threads):
            tList[i] = threading.Thread(target=self.workerCGS, args=(copyCondition, copyCount, i, num_threads, documents, K_V, D_K, sum_K, alpha, beta, sampling))
            tList[i].start()
        for i in range(num_threads):
            tList[i].join()
            
        assert np.sum(sum_K) == np.sum(documents), "Sum_K not synced: {}, {}".format(np.sum(sum_K), np.sum(documents))
        assert np.sum(K_V) == np.sum(documents), "K_V not synced: {}, {}".format(np.sum(K_V), np.sum(documents))
        cyplda.normalize(K_V, beta)
        cyplda.normalize(D_K, alpha)           
        
        self.K_V = K_V
        self.D_K = D_K
        
    def workerCGS(self, copyCondition, copyCount, thread_num, num_threads, documents, K_V, D_K, sum_K, alpha, beta, sampling):
        # vectors for probability and topic counts
        p_K = np.zeros((self.num_topics), dtype=np.float)
        uniq_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        
        #create a copy of sum_K and K_V to work over
        t_sum_K = np.zeros(sum_K.shape, dtype=np.dtype("i"))
        t_K_V = np.zeros(K_V.shape, dtype=np.float)
        
        #specify the boundaries of documents to work over
        d_interval = (documents.shape[0] - (documents.shape[0]%num_threads))/num_threads + 1 if documents.shape[0]%num_threads != 0 else documents.shape[0]/num_threads
        d_start = thread_num*d_interval
        d_end = min(documents.shape[0], (thread_num+1)*d_interval)
        #create a custom curr_K and that maps to the document boundaries
        curr_K = np.zeros((np.sum(documents[d_start:d_end])), dtype=np.dtype("i"))        
        #initialize topics for each thread   
        cyplda.init_topics(documents, t_K_V, D_K, t_sum_K, curr_K, d_start, d_end)
                
        #have sum_K and K_V be the sum of all thread-specific t_sum_K's/t_K_V's
        with copyCondition:
            cyplda.add1d(sum_K, t_sum_K)
            cyplda.add2d(K_V, t_K_V)
            self.copyConditionCheck(copyCount, num_threads, copyCondition) 
        
        #have t_sum_K/t_K_V be a copy of the summed sum_K/K_V
        cyplda.copy1d(sum_K, t_sum_K)
        cyplda.copy2d(K_V, t_K_V)
        #start the gibb sampling iterations
        for i in xrange(self.iterations/self.sync_interval):
            cyplda.CGS_iter(documents, t_K_V, D_K, t_sum_K, curr_K, alpha, beta, sampling, p_K, uniq_K, d_start, d_end, self.sync_interval)
            #must synchronize sum_K and K_V              
            #this subtraction can be done in parallel as originals unmodified and then wait for every thread to do that
            
            cyplda.subtract1d(t_sum_K, sum_K)
            cyplda.subtract2d(t_K_V, K_V)
          
            with copyCondition:
                self.copyConditionCheck(copyCount, num_threads, copyCondition)
                
            #one at a time update sum_K
            with copyCondition:
                cyplda.add1d(sum_K, t_sum_K)
                cyplda.add2d(K_V, t_K_V)
                self.copyConditionCheck(copyCount, num_threads, copyCondition)
                
            #at this point need to wait for all threads to update sum_K with their changes
            #once all threads reach this point it's safe to copy sum_K to t_sum_K
            cyplda.copy1d(sum_K, t_sum_K)
            cyplda.copy2d(K_V, t_K_V)
         
    def copyConditionCheck(self, copyCount, num_threads, copyCondition):
        copyCount[0] +=1
        if copyCount[0] == num_threads:
            copyCount[0] = 0
            copyCondition.notifyAll()
        else:
            copyCondition.wait()
            
    
        