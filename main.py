import lda
import sys
sys.path.append('util')

import set_compiler
set_compiler.install()

import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import plda

X = lda.datasets.load_reuters()
test = plda.LDA(10)
test.CGS(X)
print test.K_V
print test.D_K