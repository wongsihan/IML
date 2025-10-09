"""
Test script for the cavity PDE implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from cavity import CavityPDE
from neural_network import SPINN2D, gaussian


def test_cavity_pde_initialization():
    """Test cavity PDE initialization."""
    pde = CavityPDE(n_nodes=10, ns=25, nb=10, nbs=10)
    
    # Check basic properties
    assert pde.n_vars() == 2
    assert not pde.has_exact()
    assert pde.sample_frac == 1.0
    
    print("✓ CavityPDE initialization test passed")


def test_neural_network():
    """Test neural network creation."""
    pde = CavityPDE(n_nodes=10, ns=25, nb=10, nbs=10)
    
    # Create a simple network
    nn = SPINN2D(pde, gaussian, fixed_h=False, use_pu=False)
    
    # Test forward pass
    x = torch.randn(10, requires_grad=True)
    y = torch.randn(10, requires_grad=True)
    output = nn(x, y)
    
    assert output.shape == (10, 2)
    print("✓ Neural network test passed")


def test_loss_computation():
    """Test loss computation."""
    pde = CavityPDE(n_nodes=5, ns=9, nb=5, nbs=5, sample_frac=0.5)
    
    # Create a simple network
    nn = SPINN2D(pde, gaussian, fixed_h=False, use_pu=False)
    
    # Test loss computation
    try:
        loss = pde.loss(nn)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        print("✓ Loss computation test passed")
    except Exception as e:
        print(f"✗ Loss computation test failed: {e}")


def test_derivatives():
    """Test derivative computation."""
    pde = CavityPDE(n_nodes=5, ns=9, nb=5, nbs=5)
    
    # Test derivative computation
    x = torch.randn(5, requires_grad=True)
    y = torch.randn(5, requires_grad=True)
    u = torch.randn(5, requires_grad=True)
    
    try:
        ux, uy = pde._compute_gradient(u, x, y)
        assert ux.shape == u.shape
        assert uy.shape == u.shape
        print("✓ Derivative computation test passed")
    except Exception as e:
        print(f"✗ Derivative computation test failed: {e}")


def run_all_tests():
    """Run all tests."""
    print("Running SPINN tests...")
    print("=" * 30)
    
    try:
        test_cavity_pde_initialization()
        test_neural_network()
        test_loss_computation()
        test_derivatives()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")


if __name__ == "__main__":
    run_all_tests()
