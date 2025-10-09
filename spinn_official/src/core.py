"""
Core utilities for SPINN implementation.
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


# Global device management
_device = torch.device("cpu")


def device(dev=None):
    """Set/Get the device."""
    global _device
    if dev is None:
        return _device
    else:
        _device = torch.device(dev)


def tensor(x, **kw):
    """Returns a suitable device specific tensor."""
    return torch.tensor(x, dtype=torch.float32, device=device(), **kw)


class PDE:
    """Base class for PDE problems."""
    
    @classmethod
    def from_args(cls, args):
        pass

    @classmethod
    def setup_argparse(cls, parser, **kw):
        pass

    def nodes(self):
        pass

    def fixed_nodes(self):
        pass

    def interior(self):
        pass

    def boundary(self):
        pass

    def plot_points(self):
        pass

    def n_vars(self):
        """Return the number of variables per point in your PDE."""
        return 1

    def pde(self, *args):
        raise NotImplementedError()

    def has_exact(self):
        return True

    def exact(self, *args):
        pass

    def interior_loss(self, nn):
        raise NotImplementedError()

    def boundary_loss(self, nn):
        raise NotImplementedError()

    def loss(self, nn):
        """Total loss is computed by default as (interior loss + boundary loss)"""
        loss_int = self.interior_loss(nn)
        loss_bdy = self.boundary_loss(nn)
        loss = loss_int + loss_bdy
        return loss

    def _compute_derivatives(self, u, x):
        raise NotImplementedError()


class Plotter:
    """Base class for plotting results."""
    
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
        self.plt1 = None
        self.plt2 = None
        self.show_exact = not no_show_exact

    def get_plot_data(self):
        pass

    def get_error(self, **kw):
        pass

    def plot_solution(self):
        pass

    def plot(self):
        pass

    def plot_weights(self):
        """Implement this method to plot any weights."""
        pass

    def save(self, dirname):
        """Save the model and results."""
        pass

    def show(self):
        pass


class Optimizer:
    """Optimizer for training SPINN models."""
    
    @classmethod
    def from_args(cls, pde, nn, plotter, args):
        optimizers = {'Adam': optim.Adam, 'LBFGS': optim.LBFGS}
        o = optimizers[args.optimizer]
        return cls(
            pde, nn, plotter, n_train=args.n_train,
            n_skip=args.n_skip, tol=args.tol, lr=args.lr,
            plot=args.plot, out_dir=args.directory, opt_class=o
        )

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--n-train', '-t', dest='n_train',
            default=kw.get('n_train', 2500), type=int,
            help='Number of training iterations.'
        )
        p.add_argument(
            '--n-skip', dest='n_skip',
            default=kw.get('n_skip', 1000), type=int,
            help='Number of iterations after which we print/update plots.'
        )
        p.add_argument(
            '--tol', '-e', dest='tol',
            default=kw.get('tol', 1e-6), type=float,
            help='Tolerance for loss computation.'
        )
        p.add_argument(
            '--plot', dest='plot', action='store_true', default=False,
            help='Show a live plot of the results.'
        )
        p.add_argument(
            '--lr', dest='lr', default=kw.get('lr', 1e-2), type=float,
            help='Learning rate.'
        )
        p.add_argument(
            '--optimizer', dest='optimizer',
            default=kw.get('optimizer', 'Adam'),
            choices=['Adam', 'LBFGS'], help='Optimizer to use.'
        )
        p.add_argument(
            '-d', '--directory', dest='directory',
            default=kw.get('directory', './results'),
            help='Output directory (output files are dumped here).'
        )

    def __init__(self, pde, nn, plotter, n_train, n_skip=100, tol=1e-6,
                 lr=1e-2, plot=True, out_dir=None, opt_class=optim.Adam):
        self.pde = pde
        self.nn = nn
        self.plotter = plotter
        self.n_train = n_train
        self.n_skip = n_skip
        self.tol = tol
        self.lr = lr
        self.plot = plot
        self.out_dir = out_dir
        self.opt_class = opt_class
        self.time_taken = 0
        self.loss_history = []
        self.error_history = []

    def solve(self):
        """Solve the PDE using the neural network."""
        start_time = time.time()
        
        if self.opt_class == optim.LBFGS:
            self._solve_lbfgs()
        else:
            self._solve_adam()
            
        self.time_taken = time.time() - start_time
        print(f"Training completed in {self.time_taken:.2f} seconds")

    def _solve_adam(self):
        """Solve using Adam optimizer."""
        optimizer = self.opt_class(self.nn.parameters(), lr=self.lr)
        
        for i in range(self.n_train):
            optimizer.zero_grad()
            loss = self.pde.loss(self.nn)
            loss.backward()
            optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if i % self.n_skip == 0:
                print(f"Iteration {i}, Loss: {loss.item():.6f}")
                if self.plot:
                    self.plotter.plot()

    def _solve_lbfgs(self):
        """Solve using LBFGS optimizer."""
        optimizer = self.opt_class(self.nn.parameters(), lr=self.lr)
        
        def closure():
            optimizer.zero_grad()
            loss = self.pde.loss(self.nn)
            loss.backward()
            return loss
            
        for i in range(self.n_train):
            loss = optimizer.step(closure)
            self.loss_history.append(loss.item())
            
            if i % self.n_skip == 0:
                print(f"Iteration {i}, Loss: {loss.item():.6f}")
                if self.plot:
                    self.plotter.plot()

    def save(self):
        """Save the results."""
        if self.out_dir is None:
            return
            
        os.makedirs(self.out_dir, exist_ok=True)
        dirname = os.path.join(
            self.out_dir,
            f"model_{int(time.time())}"
        )
        os.makedirs(dirname, exist_ok=True)
        
        # Save model
        torch.save(self.nn.state_dict(), os.path.join(dirname, 'model.pt'))
        
        # Save training history
        np.savez(
            os.path.join(dirname, 'training_history.npz'),
            loss=self.loss_history,
            error=self.error_history,
            time_taken=self.time_taken
        )
        
        # Save results
        self.plotter.save(dirname)


class App:
    """Main application class for running SPINN."""
    
    def __init__(self, pde_cls, nn_cls, plotter_cls, optimizer=Optimizer):
        self.pde_cls = pde_cls
        self.nn_cls = nn_cls
        self.plotter_cls = plotter_cls
        self.optimizer = optimizer

    def _setup_activation_options(self, p, **kw):
        pass

    def _get_activation(self, args):
        pass

    def setup_argparse(self, **kw):
        """Setup the argument parser."""
        p = ArgumentParser(
            description="SPINN: Physics-Informed Sparse Neural Network",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        p.add_argument(
            '--gpu', dest='gpu', action='store_true',
            default=kw.get('gpu', False),
            help='Run code on the GPU.'
        )
        classes = (
            self.pde_cls, self.nn_cls, self.plotter_cls, self.optimizer
        )
        for c in classes:
            c.setup_argparse(p, **kw)

        self._setup_activation_options(p)
        return p

    def run(self, args=None, **kw):
        """Run the SPINN application."""
        parser = self.setup_argparse(**kw)
        args = parser.parse_args(args)

        if args.gpu:
            device("cuda")
        else:
            device("cpu")

        activation = self._get_activation(args)

        pde = self.pde_cls.from_args(args)
        self.pde = pde
        dev = device()

        nn = self.nn_cls.from_args(pde, activation, args).to(dev)
        self.nn = nn
        plotter = self.plotter_cls.from_args(pde, nn, args)
        self.plotter = plotter

        solver = self.optimizer.from_args(pde, nn, plotter, args)
        self.solver = solver

        solver.solve()
        if args.directory is not None:
            solver.save()
