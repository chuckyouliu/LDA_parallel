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
test = plda.LDA(10, 500)
base = lda.LDA(10, 500)

with Timer() as t:
    test.CGS(X, 0.1, 0.01)
print "Dynamic:" + str(t.interval)
with Timer() as t:
    base.fit(X)
print "LDA Package:" + str(t.interval)