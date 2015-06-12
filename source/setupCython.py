

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np




print "run with 'python setupCython.py build_ext --inplace'"
#opt = {'build_ext': {'compiler': 'mingw32'}}
opt = {}
modules = cythonize("_hough_transform.pyx")
setup(name='test',
      ext_modules=modules,
      include_dirs=np.get_include(),
      options=opt)
