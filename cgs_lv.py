import sys
sys.path.append('util/')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import plda
from timer import Timer
import logging
import numpy as np

logger = logging.getLogger('lda')
logger.propagate = False

X = np.load('Lasvegas/lv_dtm.npy').astype(np.int32)
iterations = 1000
vocab = np.load('Lasvegas/lv_vocab10.npy')
test = plda.LDA(40, iterations)
n_top_words = 10

times = {}
print str(iterations) + " Iterations"
for num_threads in [4, 8, 16]:
    for sync in [100]:
        test.set_sync_interval(sync)
        with Timer() as t:
            test.fit(X, num_threads, 0.1, 0.01)
            topic_word = test.K_V 
            print "Threads:{}, Locking:{}".format(num_threads, False)
            for i, topic_dist in enumerate(topic_word):
                topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
                print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        times[(num_threads, False)] = t.interval
        with Timer() as t:
            test.fit(X, num_threads, 0.1, 0.01, True)
            topic_word = test.K_V 
            print "Threads:{}, Locking:{}".format(num_threads, True)
            for i, topic_dist in enumerate(topic_word):
                topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
                print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        times[(num_threads, True)] = t.interval

with Timer() as t:
    test.sCGS(X, 0.1, 0.01)
    topic_word = test.K_V 
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
print "Serial: {}".format(t.interval)
print "Parallel"
for time in times:
    num_threads, lock_regions = time
    print "Threads-{}-Lock-{}:{}".format(num_threads, lock_regions, times[time])