"""
Example script for running the cavity PDE problem.
This reproduces the main results from the paper.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from cavity_pde import CavityPDE
from neural_network import SPINN2D, Plotter2D, App2D
from core import App


def main():
    """Run the cavity PDE example."""
    print("SPINN: Physics-Informed Sparse Neural Network")
    print("Permanent Magnet Eddy Current Device Modeling")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the application
    app = App(CavityPDE, SPINN2D, Plotter2D)
    
    # Default arguments for the paper results
    args = [
        '--nodes', '25',           # Number of nodes
        '--samples', '225',        # Number of sample points
        '--b-nodes', '25',         # Boundary nodes
        '--b-samples', '25',       # Boundary samples
        '--sample-frac', '0.1',    # Sampling fraction
        '--n-train', '2500',       # Training iterations
        '--n-skip', '500',         # Print interval
        '--lr', '0.001',           # Learning rate
        '--optimizer', 'Adam',     # Optimizer
        '--layers', '3',           # Network layers
        '--neurons', '50',         # Neurons per layer
        '--directory', './results' # Output directory
    ]
    
    # Run the application
    app.run(args)


if __name__ == "__main__":
    main()
