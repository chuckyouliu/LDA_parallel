import lda_vi_cython
import numpy as np
import threading


class LDA:

    def __init__(self, num_topics, num_threads=1):
        self.num_topics = num_topics
        self.num_threads = num_threads
        self.topics = None
        self.gamma = None

    def set_topics(self, n):
        self.num_topics = n

    def set_threads(self, t):
        self.num_threads = t

    def fit(self, dtm, batch_size, tau=512, kappa=0.7):
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
        batches = np.array_split(
            np.arange(num_docs, dtype=np.int32), num_batch)

        for it_batch in range(num_batch):
            lda_vi_cython.exp_digamma_arr(topics, ExpELogBeta)

            docs_thread = np.array_split(batches[it_batch], self.num_threads)

            # vector of threads
            threads = [None]*self.num_threads

            for tid in range(self.num_threads):
                threads[tid] = threading.Thread(target=self.worker_estep,
                                                args=(docs_thread[tid], dtm,
                                                      topics_int[tid, :, :],
                                                      gamma, ExpELogBeta))
                threads[tid].start()

            for thread in threads:
                thread.join()

            # Synchronizing the topics_int
            topics_int_tot = np.sum(topics_int, axis=0)
            # Initialize the list of topics int for the next batch
            topics_int[:, :, :] = 0
            # M-step
            indices = (np.sum(dtm[batches[it_batch], :], axis=0) > 0).astype(
                np.int32)
            lda_vi_cython.m_step(topics, topics_int_tot, indices, num_docs,
                                 batch_size, tau, kappa, it_batch)

        self.topics = topics
        self.gamma = gamma

    def transform(self, dtm, batch_size, tau=512, kappa=0.7):
        '''
        Transform dtm into gamma according to the previously trained model.

        '''
        if self.topics is None:
            raise NameError('The model has not been trained yet')
        # Initialisation
        num_docs, num_words = dtm.shape
        np.random.seed(0)
        gamma = np.ones((num_docs, self.num_topics))
        ExpELogBeta = np.zeros((self.num_topics, num_words))
        topics_int = np.zeros((self.num_threads, self.num_topics, num_words))

        num_batch = num_docs / batch_size
        batches = np.array_split(
            np.arange(num_docs, dtype=np.int32), num_batch)

        for it_batch in range(num_batch):
            lda_vi_cython.exp_digamma_arr(self.topics, ExpELogBeta)

            docs_thread = np.array_split(batches[it_batch], self.num_threads)

            # vector of threads
            threads = [None]*self.num_threads

            for tid in range(self.num_threads):
                threads[tid] = threading.Thread(target=self.worker_estep,
                                                args=(docs_thread[tid], dtm,
                                                      topics_int[tid, :, :],
                                                      gamma, ExpELogBeta))
                threads[tid].start()

            for thread in threads:
                thread.join()

        return gamma

    def perplexity(self, dtm_test, batch_size, tau=512, kappa=0.7):
        '''
        Compute the log-likelihood of the documents in dtm_test based on the
        topic distribution already learned by the model
        '''
        gamma = self.transform(dtm_test, tau, kappa)
        # Normalizing the topics and gamma
        topics = self.topics/self.topics.sum(axis=0)
        gamma = gamma/gamma.sum(axis=1)[:, np.newaxis]

        if len(gamma.shape) == 1:
            doc_idx = np.nonzero(dtm_test)[0]
            doc_cts = dtm_test[doc_idx]
            return np.sum(np.log(np.dot(gamma[i, :],
                          topics[:, doc_idx]))*doc_cts)
        else:
            # Initialization
            num = 0
            denom = 0
            for i in range(gamma.shape[0]):
                doc_idx = np.nonzero(dtm_test[i, :])[0]
                doc_cts = dtm_test[i, doc_idx]
                num += np.sum(np.log(np.dot(gamma[i, :],
                              topics[:, doc_idx]))*doc_cts)
                denom += np.sum(doc_cts)
            return num/denom

    def print_topic(self, vocabulary, num_top_words=10):
        if self.topics is None:
            raise NameError('The model has not been trained yet')
        topic_word = self.topics
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(
                vocabulary)[np.argsort(topic_dist)][:-(num_top_words+1):-1]
            print(u'Topic {}: {}'.format(i, ' '.join(topic_words)))

    def worker_estep(self, docs, dtm, topics_int_t, gamma, ExpELogBeta):
        # Local initialization
        num_words = dtm.shape[1]
        ExpLogTethad = np.zeros(self.num_topics)
        phi = np.zeros((self.num_topics, num_words))

        # Lambda_int is shared among the threads
        lda_vi_cython.e_step(docs, dtm, gamma, ExpELogBeta, ExpLogTethad, topics_int_t, phi,
                             self.num_topics)
