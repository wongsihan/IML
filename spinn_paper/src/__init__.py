"""
SPINN: Physics-Informed Sparse Neural Network for Permanent Magnet Eddy Current Device Modeling and Analysis

This package implements the SPINN method for solving PDEs using sparse neural networks.
"""

__version__ = "1.0.0"
__author__ = "D. Wang, S. Wang, D. Kong, J. Wang, W. Li and M. Pecht"

from .core import *
from .pde import *
from .neural_network import *
from .optimizer import *
