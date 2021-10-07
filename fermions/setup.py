#import sys
#import importlib
#clive = sys.path
#sys.path.insert(0,'/home/atrant01/venvs/HCQLIB3/lib/python3.6/site-packages/')
#print(setuptools.__version__)
#importlib.reload(setuptools)
#print(setuptools.__version__)
#sys.path = clive
#print(setuptools.__version__)
import setuptools
from Cython.Build import cythonize
setuptools.setup(
    name='yaferp',
    version='0.451',
    packages=setuptools.find_namespace_packages(),
    url='',
    license='',
    author='atranter',
    author_email='',
    description='',
    ext_modules = cythonize(["yaferp/general/oneXOnState/oneXOnState.pyx","yaferp/general/oneZOnState/oneZOnState.pyx"],language_level=3)
)
