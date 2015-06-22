from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

print "run with 'python setupCython.py build_ext --inplace'"

numpy_dirs=numpy.get_include()

ext_modules = [Extension("_hough_transform", ["source/_hough_transform.pyx"], include_dirs=[numpy_dirs], extra_compile_args=[])]
setup(
      name = 'ctqa_cp',
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(ext_modules, annotate=False)
    )
