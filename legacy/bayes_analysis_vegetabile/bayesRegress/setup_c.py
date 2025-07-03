import numpy
from distutils.core import setup
from distutils.extension import Extension 
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext = Extension('truncated_norm_C', sources=['truncated_norm_C.pyx', 'truncated_normal.c'],
    include_dirs=[numpy.get_include()])

setup(ext_modules=[ext], cmdclass={'build_ext':build_ext})