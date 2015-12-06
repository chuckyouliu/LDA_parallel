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

    def fit_s(self, dtm, batch_size, tau=512, kappa=0.7):
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
        lda_vi_cython.lda_online(dtm, self.num_topics, batch_size, self.num_threads,
                                 tau, kappa, topics, gamma,
                                 topics_int, phi, ExpLogTethad, ExpELogBeta)
        # Attributes update
        self.topics = topics
        self.gamma = gamma

    def fit_batch(self, dtm, batch_size, tau=1, kappa=0):
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
            lda_vi_cython.lda_online(dtm, self.num_topics, batch_size, self.num_threads,
                                     tau, kappa, self.topics, self.gamma,
                                     topics_int, phi, ExpLogTethad, ExpELogBeta)
            # Attributes update
            self.topics = self.topics
            self.gamma = self.gamma

    def fit_p(self, dtm, batch_size, tau=512, kappa=0.7, locks=False):
        '''
        Parallel version of the lda: the temporary topics are computed in
        parallel for each document inside a mini-batch

        '''
        # Initialisation
        num_docs, num_words = dtm.shape
        np.random.seed(0)
        topics = np.random.gamma(100., 1./100., (self.num_topics, num_words))
        gamma = np.ones((num_docs, self.num_topics))
        ExpELogBeta = np.zeros((self.num_topics, num_words))
        topics_int = np.zeros((self.num_threads, self.num_topics, num_words))

        num_batch = num_docs / batch_size
        batches = np.array_split(np.arange(num_docs, dtype=np.int32), num_batch)

        for it_batch in range(num_batch):
            lda_vi_cython.exp_digamma_arr(topics, ExpELogBeta)
            # Indices present in the batch
            indices = np.zeros(num_words, dtype=np.int32)

            # Splitting the current batch among threads
            # batch = np.arange(it_batch*num_batch, (it_batch + 1)*num_batch,
            #                   dtype=np.int32)
            # docs_thread = np.array_split(batch, self.num_threads)
            docs_thread = np.array_split(batches[it_batch], self.num_threads)

            #vector of threads
            threads = [None]*self.num_threads

            wlocks = None

            if locks:
                wlocks = [None]*self.num_threads
                for i in range(self.num_threads):
                    wlocks[i] = threading.Lock()

            for tid in range(self.num_threads):
                threads[tid] = threading.Thread(target=self.worker_estep,
                                                args=(tid,docs_thread[tid], dtm,
                                                      topics_int[tid, :, :],
                                                      gamma, ExpELogBeta, indices, wlocks))
                threads[tid].start()

            for thread in threads:
                thread.join()

            # Synchronizing the topics_int
            topics_int_tot = np.sum(topics_int, axis=0)
            # Initialize the list of topics int for the next batch
            topics_int[:, :, :] = 0
            # M-step
            indices = (np.sum(dtm[batches[it_batch], :], axis=0) > 0).astype(np.int32)
            lda_vi_cython.m_step(topics, topics_int_tot, indices, num_docs,
                                 batch_size, tau, kappa, it_batch)

        self.topics = topics
        self.gamma = gamma

    def worker_estep(self, tid, docs, dtm, topics_int_t, gamma, ExpELogBeta, indices, wlocks):
        # Local initialization
        num_words = dtm.shape[1]

        ExpLogTethad = np.zeros(self.num_topics)
        phi = np.zeros((self.num_topics, num_words))

        # Lambda_int is shared among the threads
        if wlocks is None:
            lda_vi_cython.e_step(docs, dtm, gamma, ExpELogBeta, ExpLogTethad, topics_int_t, phi,
               indices, self.num_topics, 0, num_words, num_words)
        else:
            w_interval = (num_words - (num_words%self.num_threads))/self.num_threads + 1 if num_words%self.num_threads != 0 else num_words/self.num_threads
            for i in xrange(self.num_threads):
                word_group = (i+tid)%self.num_threads
                with wLocks[word_group]:
                    w_start = (word_group)*w_interval
                    w_end = min(num_words, w_start + w_interval)
                    lda_vi_cython.e_step(docs, dtm, gamma, ExpELogBeta, ExpLogTethad, topics_int_t, phi,
                        indices, self.num_topics, w_start, w_end, w_interval)
         



