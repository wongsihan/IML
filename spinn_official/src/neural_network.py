"""
Neural network implementations for SPINN.
"""

import numpy as np
import torch
import torch.nn as nn
from .core import tensor


class Shift2D(nn.Module):
    """2D shift layer for SPINN."""
    
    def __init__(self, points, fixed_points, fixed_h=False):
        super().__init__()
        n_free = len(points[0])
        n_fixed = len(fixed_points[0])
        self.n = n = n_free + n_fixed
        
        if n_fixed:
            dsize = np.ptp(np.hstack((points[0], fixed_points[0])))
        else:
            dsize = np.ptp(points[0])
        dx = dsize / np.sqrt(n)
        self.dx = dx
        
        self.x = nn.Parameter(tensor(points[0]))
        self.y = nn.Parameter(tensor(points[1]))
        fp = fixed_points
        self.xf = tensor(fp[0])
        self.yf = tensor(fp[1])

        self.fixed_h = fixed_h
        if fixed_h:
            self.h = nn.Parameter(tensor(dx))
        else:
            self.h = nn.Parameter(dx * torch.ones(n))

    def centers(self):
        return torch.cat((self.x, self.xf)), torch.cat((self.y, self.yf))

    def widths(self):
        if self.fixed_h:
            return self.h * torch.ones(self.n)
        else:
            return self.h

    def forward(self, x, y):
        xc, yc = self.centers()
        fac = 1.0 / self.h
        xh = (x - xc) * fac
        yh = (y - yc) * fac
        return xh, yh


class SPINN2D(nn.Module):
    """Sparse Physics-Informed Neural Network for 2D problems."""
    
    @classmethod
    def from_args(cls, pde, activation, args):
        return cls(pde, activation,
                   fixed_h=args.fixed_h, use_pu=args.pu)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--fixed-h', dest='fixed_h', action='store_true', default=False,
            help='Use fixed width nodes.'
        )
        p.add_argument(
            '--pu', dest='pu', action='store_true', default=False,
            help='Use a partition of unity.'
        )

    def __init__(self, pde, activation, fixed_h=False, use_pu=False):
        super().__init__()
        self.activation = activation
        self.use_pu = use_pu
        self.layer1 = Shift2D(pde.nodes(), pde.fixed_nodes(), fixed_h=fixed_h)
        n = self.layer1.n
        self.layer2 = nn.Linear(n, pde.n_vars(), bias=not use_pu)
        self.layer2.weight.data.fill_(0.0)
        if not self.use_pu:
            self.layer2.bias.data.fill_(0.0)

    def forward(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        xh, yh = self.layer1(x, y)
        z = self.activation(xh, yh)
        if self.use_pu:
            zsum = z.sum(axis=1).unsqueeze(1)
        else:
            zsum = 1.0
        z = self.layer2(z / zsum)
        return z.squeeze()

    def centers(self):
        return self.layer1.centers()

    def widths(self):
        return self.layer1.widths()

    def weights(self):
        return self.layer2.weight


def gaussian(x, y):
    """Gaussian activation function."""
    return torch.exp(-0.5 * (x * x + y * y))


class SoftPlus:
    """SoftPlus activation function."""
    
    def __init__(self):
        self._sp = nn.Softplus()
        self.k = tensor([1.0 + 4.0 * np.log(2.0)])
        self.fac = self._sp(tensor([1.0]))

    def __call__(self, x, y):
        sp = self._sp
        return sp(
            self.k - sp(2.0 * x) - sp(-2.0 * x) - sp(2.0 * y) - sp(-2.0 * y)
        ) / self.fac


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
        weights = self.nn.weights().detach().cpu().numpy()
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(weights, cmap='RdBu', aspect='auto')
        plt.colorbar()
        plt.title('Network Weights')
        plt.xlabel('Input')
        plt.ylabel('Output')
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

    def _setup_activation_options(self, p, **kw):
        p.add_argument(
            '--activation', '-a', dest='activation',
            default=kw.get('activation', 'gaussian'),
            choices=['gaussian', 'softplus'],
            help='Select the activation function for particles.'
        )

    def _get_activation(self, args):
        activations = {
            'gaussian': lambda x: gaussian,
            'softplus': lambda x: SoftPlus()
        }
        return activations[args.activation](None)

    def run(self, args=None, **kw):
        """Run the 2D SPINN application."""
        from .core import App
        app = App(self.pde_cls, self.nn_cls, self.plotter_cls, self.optimizer)
        app._setup_activation_options = self._setup_activation_options
        app._get_activation = self._get_activation
        return app.run(args, **kw)
