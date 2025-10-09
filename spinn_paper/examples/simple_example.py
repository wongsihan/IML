"""
Simple example for running SPINN.
This is a minimal example that demonstrates the basic usage.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np


def simple_cavity_example():
    """Simple cavity example without complex dependencies."""
    print("SPINN Simple Example")
    print("=" * 30)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple neural network
    class SimpleSPINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, 50),
                nn.Tanh(),
                nn.Linear(50, 50),
                nn.Tanh(),
                nn.Linear(50, 2)
            )
        
        def forward(self, x, y):
            coords = torch.stack([x, y], dim=-1)
            return self.layers(coords)
    
    # Create network
    nn_model = SimpleSPINN().to(device)
    
    # Create sample points
    x = torch.linspace(0, 1, 10, device=device, requires_grad=True)
    y = torch.linspace(0, 1, 10, device=device, requires_grad=True)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Test forward pass
    output = nn_model(X.flatten(), Y.flatten())
    print(f"Network output shape: {output.shape}")
    
    # Test gradient computation
    loss = output.sum()
    loss.backward()
    print(f"Gradient computation successful")
    
    print("✓ Simple example completed successfully!")


if __name__ == "__main__":
    simple_cavity_example()
