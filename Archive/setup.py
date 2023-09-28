from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("person_tracker.pyx"),
    include_dirs=[numpy.get_include()],
    compiler_directives={'language_level' : "3"}
)
