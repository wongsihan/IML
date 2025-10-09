"""
SPINN: Physics-Informed Sparse Neural Network

Official implementation of the paper:
"Physics-Informed Sparse Neural Network for Permanent Magnet Eddy Current Device Modeling and Analysis"

Authors: D. Wang, S. Wang, D. Kong, J. Wang, W. Li and M. Pecht
Journal: IEEE Magnetics Letters, vol. 14, pp. 1-5, 2023
DOI: 10.1109/LMAG.2023.3288388
"""

__version__ = "1.0.0"
__author__ = "D. Wang, S. Wang, D. Kong, J. Wang, W. Li and M. Pecht"
__email__ = ""
__license__ = "MIT"

from .core import *
from .pde import *
from .neural_network import *
from .cavity import *

__all__ = [
    'SPINN2D', 'CavityPDE', 'Plotter2D', 'App2D',
    'PDE', 'RegularPDE', 'Optimizer', 'App'
]
