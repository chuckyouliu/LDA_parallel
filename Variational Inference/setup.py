from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension('lda_vi_cython',
                               ['lda_vi_cython.pyx'],
                               etra_compile_args=['-fopenmp'],
                               extra_link_args=['-fopenmp'])]
      )