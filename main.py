import lda
import sys
sys.path.append('util/')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import plda
from timer import Timer
import logging

logger = logging.getLogger('lda')
logger.propagate = False

X = lda.datasets.load_reuters()
iterations = 1000
vocab = lda.datasets.load_reuters_vocab()
test = plda.LDA(20, iterations)
n_top_words = 8

print str(iterations) + " Iterations"
for num_threads in [1,2,4,8]:
    for sync in [1,50,100]:
        test.set_sync_interval(sync)
        with Timer() as t:
            test.pCGS(X, num_threads, 0.1, 0.01)
            #topic_word = test.K_V 
            #for i, topic_dist in enumerate(topic_word):
            #    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            #    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        print "{} threads, {} sync_step:".format(num_threads, sync) + str(t.interval)


#with Timer() as t:
#    test.sCGS(X, 0.1, 0.01)
    #topic_word = test.K_V 
    #for i, topic_dist in enumerate(topic_word):
    #    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    #    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
#print "Serial:" + str(t.interval)