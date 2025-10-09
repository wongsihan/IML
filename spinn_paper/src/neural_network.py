"""
Neural network implementations for SPINN.
"""

import numpy as np
import torch
import torch.nn as nn
from .core import tensor


class SPINN2D(nn.Module):
    """Sparse Physics-Informed Neural Network for 2D problems."""
    
    @classmethod
    def from_args(cls, pde, activation, args):
        return cls(
            pde.n_vars(), args.layers, args.neurons,
            activation, pde.nodes(), pde.fixed_nodes()
        )

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--layers', dest='layers',
            default=kw.get('layers', 3), type=int,
            help='Number of hidden layers.'
        )
        p.add_argument(
            '--neurons', dest='neurons',
            default=kw.get('neurons', 50), type=int,
            help='Number of neurons per layer.'
        )

    def __init__(self, n_vars, n_layers, n_neurons, activation, nodes, fixed_nodes):
        super().__init__()
        self.n_vars = n_vars
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        
        # Node coordinates
        self.nodes = nodes
        self.fixed_nodes = fixed_nodes
        
        # Network layers
        self.layers = nn.ModuleList()
        
        # Input layer (2D coordinates)
        self.layers.append(nn.Linear(2, n_neurons))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        
        # Output layer
        self.layers.append(nn.Linear(n_neurons, n_vars))
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, y):
        """Forward pass of the network."""
        # Convert to tensors if needed
        if not isinstance(x, torch.Tensor):
            x = tensor(x)
        if not isinstance(y, torch.Tensor):
            y = tensor(y)
        
        # Combine coordinates
        coords = torch.stack([x, y], dim=-1)
        
        # Forward through layers
        for i, layer in enumerate(self.layers[:-1]):
            coords = layer(coords)
            coords = self.activation(coords)
        
        # Output layer (no activation)
        output = self.layers[-1](coords)
        
        return output

    def get_weights(self):
        """Get network weights for analysis."""
        weights = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.data.cpu().numpy())
        return weights

    def set_weights(self, weights):
        """Set network weights."""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear) and i < len(weights):
                layer.weight.data = torch.tensor(weights[i], dtype=torch.float32)


class Plotter2D:
    """2D plotter for SPINN results."""
    
    @classmethod
    def from_args(cls, pde, nn, args):
        return cls(pde, nn, args.no_show_exact)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--no-show-exact', dest='no_show_exact',
            action='store_true', default=False,
            help='Do not show exact solution even if available.'
        )

    def __init__(self, pde, nn, no_show_exact=False):
        self.pde = pde
        self.nn = nn
        self.show_exact = not no_show_exact

    def get_error(self, xn=None, yn=None, pn=None):
        """Compute error metrics."""
        if not self.pde.has_exact():
            return 0.0, 0.0, 0.0

        if xn is None and pn is None:
            xn, yn, pn = self.get_plot_data()

        un = self.pde.exact(xn, yn)
        diff = un - pn
        err_L1 = np.mean(np.abs(diff))
        err_L2 = np.sqrt(np.mean(diff**2))
        err_Linf = np.max(np.abs(diff))
        return err_L1, err_L2, err_Linf

    def save(self, dirname):
        """Save the model and results."""
        import os
        modelfname = os.path.join(dirname, 'model.pt')
        torch.save(self.nn.state_dict(), modelfname)
        
        rfile = os.path.join(dirname, 'results.npz')
        x, y, u = self.get_plot_data()
        if self.pde.has_exact():
            u_exact = self.pde.exact(x, y)
            np.savez(rfile, x=x, y=y, u=u, u_exact=u_exact)
        else:
            np.savez(rfile, x=x, y=y, u=u)

    def get_plot_data(self):
        """Get data for plotting."""
        x, y = self.pde.plot_points()
        xt, yt = tensor(x.ravel()), tensor(y.ravel())
        pn = self.nn(xt, yt).detach().cpu().numpy()
        pn.shape = x.shape
        return x, y, pn

    def plot_solution(self):
        """Plot the solution."""
        import matplotlib.pyplot as plt
        
        x, y, u = self.get_plot_data()
        
        plt.figure(figsize=(12, 5))
        
        # Plot predicted solution
        plt.subplot(1, 2, 1)
        plt.contourf(x, y, u, levels=20, cmap='viridis')
        plt.colorbar()
        plt.title('SPINN Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plot exact solution if available
        if self.pde.has_exact() and self.show_exact:
            u_exact = self.pde.exact(x, y)
            plt.subplot(1, 2, 2)
            plt.contourf(x, y, u_exact, levels=20, cmap='viridis')
            plt.colorbar()
            plt.title('Exact Solution')
            plt.xlabel('x')
            plt.ylabel('y')
        
        plt.tight_layout()
        plt.show()

    def plot_weights(self):
        """Plot network weights."""
        weights = self.nn.get_weights()
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(weights), figsize=(4*len(weights), 4))
        
        if len(weights) == 1:
            axes = [axes]
        
        for i, w in enumerate(weights):
            im = axes[i].imshow(w, cmap='RdBu', aspect='auto')
            axes[i].set_title(f'Layer {i+1} Weights')
            axes[i].set_xlabel('Input')
            axes[i].set_ylabel('Output')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()

    def plot(self):
        """Main plotting function."""
        self.plot_solution()
        self.plot_weights()


class App2D:
    """2D application for SPINN."""
    
    def __init__(self, pde_cls, nn_cls, plotter_cls, optimizer=None):
        from .core import Optimizer
        self.pde_cls = pde_cls
        self.nn_cls = nn_cls
        self.plotter_cls = plotter_cls
        self.optimizer = optimizer or Optimizer

    def run(self, args=None, **kw):
        """Run the 2D SPINN application."""
        from .core import App
        app = App(self.pde_cls, self.nn_cls, self.plotter_cls, self.optimizer)
        return app.run(args, **kw)
