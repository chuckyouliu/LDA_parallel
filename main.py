import lda
import sys
sys.path.append('util')

import set_compiler
set_compiler.install()

import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import plda
from timer import Timer
import logging

logger = logging.getLogger('lda')
logger.propagate = False

X = lda.datasets.load_reuters()
iterations = 25
vocab = lda.datasets.load_reuters_vocab()
test = plda.LDA(20, iterations)
#base = lda.LDA(20, iterations)
n_top_words = 8

with Timer() as t:
    test.pCGS(X, 8, 0.1, 0.01)
    topic_word = test.K_V 
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
print "Parallel:" + str(t.interval)


with Timer() as t:
    test.sCGS(X, 0.1, 0.01)
    topic_word = test.K_V 
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
print "Serial:" + str(t.interval)
'''
with Timer() as t:
    base.fit(X)
    topic_word = base.topic_word_ 
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
print "LDA Package:" + str(t.interval)
'''