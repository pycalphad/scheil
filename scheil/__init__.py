"""
scheil
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import warnings
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', message='invalid value encountered in greater')
warnings.filterwarnings('ignore', message='divide by zero encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')
warnings.filterwarnings('ignore', message='divide by zero encountered')

from scheil.solidification_result import SolidificationResult
from scheil.simulate import simulate_scheil_solidification, simulate_equilibrium_solidification
