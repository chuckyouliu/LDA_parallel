import cyplda
import numpy as np

class LDA:
    def __init__(self, num_topics, iterations = 500, damping = 1):
        self.num_topics = num_topics
        self.iterations = iterations
        self.damping = damping
        
    def set_topics(self, n):
        self.num_topics = n
        
    def set_iterations(self, i):
        self.iterations = i
        
    def CGS(self, documents, alpha=None, beta=None):
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
        cyplda.CGS(documents, K_V, D_K, sum_K, curr_K, alpha, beta, self.iterations, sampling)
        self.K_V = K_V
        self.D_K = D_K
        self.sum_K = sum_K
        self.curr_K = curr_K