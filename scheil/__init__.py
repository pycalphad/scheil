"""
scheil
"""

import warnings
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', message='invalid value encountered in greater')
warnings.filterwarnings('ignore', message='divide by zero encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')
warnings.filterwarnings('ignore', message='divide by zero encountered')

from scheil.solidification_result import SolidificationResult
from scheil.simulate import simulate_scheil_solidification, simulate_equilibrium_solidification

# Set the version of scheil
try:
    from ._dev import get_version
    # We have a local (editable) installation and can get the version based on the
    # source control management system at the project root.
    __version__ = get_version(root='..', relative_to=__file__)
    del get_version
except ImportError:
    # Fall back on the metadata of the installed package
    try:
        from importlib.metadata import version
    except ImportError:
        # backport for Python<3.8
        from importlib_metadata import version
    __version__ = version("scheil")
    del version