"""
Simple example for running SPINN.
This is a minimal example that demonstrates the basic usage.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from cavity import CavityPDE
from neural_network import SPINN2D, Plotter2D, gaussian
from core import App


def simple_cavity_example():
    """Simple cavity example."""
    print("SPINN Simple Example")
    print("=" * 30)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create PDE
    pde = CavityPDE(n_nodes=10, ns=25, nb=10, nbs=10, sample_frac=0.5)
    
    # Create neural network
    nn = SPINN2D(pde, gaussian, fixed_h=False, use_pu=False)
    
    # Create plotter
    plotter = Plotter2D(pde, nn)
    
    # Test forward pass
    x, y = pde.plot_points()
    x_flat = torch.tensor(x.ravel(), requires_grad=True)
    y_flat = torch.tensor(y.ravel(), requires_grad=True)
    
    output = nn(x_flat, y_flat)
    print(f"Network output shape: {output.shape}")
    
    # Test loss computation
    loss = pde.loss(nn)
    print(f"Initial loss: {loss.item():.6f}")
    
    # Test gradient computation
    loss.backward()
    print(f"Gradient computation successful")
    
    print("✓ Simple example completed successfully!")


if __name__ == "__main__":
    simple_cavity_example()
