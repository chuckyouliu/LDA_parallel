import lda_vi_cython
import numpy as np
import threading


class LDA_vi:
    def __init__(self, num_topics, num_threads=1):
        self.num_topics = num_topics
        self.num_threads = num_threads

    def set_topics(self, n):
        self.num_topics = n

    def set_threads(self, t):
        self.num_threads = t

    def fit_s(self, dtm, S, tau=512, kappa=0.7):
        '''
        Serial version of the lda
        '''
        # Initialisation
        num_docs, num_words = dtm.shape
        np.random.seed(0)
        topics = np.random.gamma(100., 1./100., (self.num_topics, num_words))
        gamma = np.ones((num_docs, self.num_topics))
        topics_int = np.zeros((self.num_topics, num_words))
        phi = np.zeros((self.num_topics, num_words))
        ExpLogTethad = np.zeros(self.num_topics)
        ExpELogBeta = np.zeros((self.num_topics, num_words))

        # Lda
        lda_vi_cython.lda_online(dtm, self.num_topics, S, self.num_threads,
                                 tau, kappa, topics, gamma,
                                 topics_int, phi, ExpLogTethad, ExpELogBeta)
        # Attributes update
        self.topics = topics
        self.gamma = gamma

    def fit_batch(self, dtm, S, tau=1, kappa=0):
        '''
        Serial version of the lda
        '''
        # Initialisation
        num_docs, num_words = dtm.shape
        self.topics = np.random.gamma(100., 1./100., (self.num_topics, num_words))
        self.gamma = np.ones((num_docs, self.num_topics))
        topics_int = np.zeros((self.num_topics, num_words))
        phi = np.zeros((self.num_topics, num_words))
        ExpLogTethad = np.zeros(self.num_topics)
        ExpELogBeta = np.zeros((self.num_topics, num_words))

        # Lda
        # Loop for lda batch
        for it in range(50):
            lda_vi_cython.lda_online(dtm, self.num_topics, S, self.num_threads,
                                     tau, kappa, self.topics, self.gamma,
                                     topics_int, phi, ExpLogTethad, ExpELogBeta)
            # Attributes update
            self.topics = self.topics
            self.gamma = self.gamma

    def fit_p(self, dtm, S, tau=512, kappa=0.7):
        '''
        Parallel version of the lda: the temporary topics are computed in
        parallel for each document inside a mini-batch

        '''
        # Initialisation
        num_docs, num_words = dtm.shape
        topics = np.random.gamma(100., 1./100., (self.num_topics, num_words))
        gamma = np.ones((num_docs, self.num_topics))
        topics_int = np.zeros((self.num_topics, num_words))
        phi = np.zeros((self.num_topics, num_words))
        ExpLogTethad = np.zeros(self.num_topics)
        ExpELogBeta = np.zeros((self.num_topics, num_words))



