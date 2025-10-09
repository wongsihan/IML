"""
PDE implementations for SPINN.
"""

import numpy as np
import torch
import torch.autograd as ag
from .core import PDE, tensor


class RegularPDE(PDE):
    """Base class for regular PDE problems in 2D."""
    
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples,
                   args.b_nodes, args.b_samples,
                   args.sample_frac)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--nodes', '-n', dest='nodes',
            default=kw.get('nodes', 25), type=int,
            help='Number of nodes to use.'
        )
        p.add_argument(
            '--samples', '-s', dest='samples',
            default=kw.get('samples', 100), type=int,
            help='Number of sample points to use.'
        )
        p.add_argument(
            '--b-nodes', dest='b_nodes',
            default=kw.get('b_nodes', None), type=int,
            help='Number of boundary nodes to use per edge'
        )
        p.add_argument(
            '--b-samples', dest='b_samples',
            default=kw.get('b_samples', None), type=int,
            help='Number of boundary samples to use per edge.'
        )
        p.add_argument(
            '--sample-frac', '-f', dest='sample_frac',
            default=kw.get('sample_frac', 1.0), type=float,
            help='Fraction of interior nodes used for sampling.'
        )

    def _compute_derivatives(self, u, xs, ys):
        """Compute derivatives using automatic differentiation."""
        du = ag.grad(
            outputs=u, inputs=(xs, ys), grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )
        ux, uy = du[0], du[1]
        
        uxx = ag.grad(
            outputs=ux, inputs=(xs,), grad_outputs=torch.ones_like(ux),
            retain_graph=True, create_graph=True
        )[0]
        
        uyy = ag.grad(
            outputs=uy, inputs=(ys,), grad_outputs=torch.ones_like(uy),
            retain_graph=True, create_graph=True
        )[0]
        
        return u, ux, uy, uxx, uyy

    def _compute_gradient(self, u, xs, ys):
        """Compute gradient using automatic differentiation."""
        du = ag.grad(
            outputs=u, inputs=(xs, ys), grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )
        return du[0], du[1]

    def interior(self):
        """Get interior sample points."""
        if not hasattr(self, '_interior_cache'):
            self._interior_cache = self._get_interior_samples()
        return self._interior_cache

    def _get_interior_samples(self):
        """Get interior sample points for training."""
        # This should be implemented by subclasses
        raise NotImplementedError()

    def boundary(self):
        """Get boundary sample points."""
        if not hasattr(self, '_boundary_cache'):
            self._boundary_cache = self._get_boundary_samples()
        return self._boundary_cache

    def _get_boundary_samples(self):
        """Get boundary sample points for training."""
        # This should be implemented by subclasses
        raise NotImplementedError()

    def plot_points(self):
        """Get points for plotting."""
        if not hasattr(self, '_plot_cache'):
            self._plot_cache = self._get_plot_points()
        return self._plot_cache

    def _get_plot_points(self):
        """Get points for plotting results."""
        # This should be implemented by subclasses
        raise NotImplementedError()
