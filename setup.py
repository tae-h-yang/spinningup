from os.path import join
from setuptools import setup
import sys

assert sys.version_info >= (3, 8), "Python 3.8 or higher is required."

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    py_modules=['spinup'],
    version=__version__,
    python_requires='>=3.8, <3.11',
    install_requires=[
        'cloudpickle>=2.0',
        'gym[classic_control]==0.26.2',
        'box2d-py',
        'ipython',
        'joblib',
        'matplotlib>=3.5',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn>=0.11',
        'tensorflow-macos',
        'tensorflow-metal',
        'torch>=1.12',
        'tqdm'
    ],
    description="Teaching tools for introducing people to deep RL.",
    author="Joshua Achiam",
)
