import numpy
from distutils.core import setup
from distutils.extension import Extension 
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext = Extension('bayesRegress', sources=['bayesRegress.pyx'],
    include_dirs=[numpy.get_include()])

setup(ext_modules=[ext], cmdclass={'build_ext':build_ext})