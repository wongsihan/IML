import os
import time

import numpy as np
import torch
import torch.nn as nn
torch.cuda.empty_cache()

import torch.optim as optim

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


_device = torch.device("cpu")


def device(dev=None):
    '''Set/Get the device.
    '''
    global _device
    if dev is None:
        return _device
    else:
        _device = torch.device(dev)


def tensor(x, **kw):
    '''Returns a suitable device specific tensor.

    '''
    return torch.tensor(x, dtype=torch.float32, device=device(),  **kw)


class PDE:
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

    # #### Override these for different differential equations ############
    def n_vars(self):
        '''Return the number of variables per point in your PDE.

        For example, for scalar equations, return 1 (which is the default). For
        a system of 3 equations return 3.

        '''
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

    ###################################################################

    def loss(self, nn):
        '''Total loss is computed by default as
           (interior loss + boundary loss)
        '''
        #tasks_relative_loss = torch.zeros(11)
        #loss_int, tasks_relative_loss = self.interior_loss(nn)#20mb 采样点*frac
        loss_int, loss_one, loss_two = self.interior_loss(nn)
        loss_bdy, loss_three,loss_four, loss_five,loss_six,loss_seven,loss_eight,loss_nine,loss_ten,loss_eleven = self.boundary_loss(nn)#14mb
        loss = loss_int + loss_bdy
        return loss, loss_one, loss_two, loss_three,loss_four, loss_five,loss_six,loss_seven,loss_eight,loss_nine,loss_ten,loss_eleven

    def _compute_derivatives(self, u, x):
        raise NotImplementedError()


class Plotter:
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
        '''Initializer

        Parameters
        -----------

        pde: PDE: object managing the pde.
        nn: Neural network for the solution
        eq: DiffEq: Differential equation to evaluate.
        no_show_exact: bool: Show exact solution or not.
        '''
        self.pde = pde
        self.nn = nn
        self.plt1 = None
        self.plt2 = None  # For weights
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
        '''Implement this method to plot any weights.
        Note this is always called *after* plot_solution.
        '''
        pass

    def save(self, dirname):
        '''Save the model and results.

        '''
        pass

    def show(self):
        pass


class Optimizer:
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
            # default=kw.get('directory', '/root/data/SPINN/code/save'),
            default=kw.get('directory', 'D:\Github\SPINN2\SPINN2\code\save'),
            help='Output directory (output files are dumped here).'
        )

    def __init__(self, pde, nn, plotter, n_train, n_skip=100, tol=1e-6,
                 lr=1e-2, plot=True, out_dir=None, opt_class=optim.Adam):
        '''Initializer

        Parameters
        -----------

        pde: Solver: The problem being solved
        nn: SPINN1D/SPINN2D/...: Neural Network class.
        plotter: Plotter: Handles all the plotting
        n_train: int: Training steps
        n_skip: int: Print loss every so often.
        lr: float: Learming rate
        plot: bool: Plot live solution.
        out_dir: str: Output directory.
        opt_class: Optimizer to use.
        '''

        self.pde = pde
        self.nn = nn
        self.plotter = plotter

        self.opt_class = opt_class
        self.errors_L1 = []
        self.errors_L2 = []
        self.errors_Linf = []
        self.loss = []
        self.time_taken = 0.0
        self.n_train = n_train
        self.n_skip = n_skip
        self.tol = tol
        self.lr = lr
        self.plot = plot
        self.out_dir = out_dir
        self.opt = None
        self.loss_for_tolerance = False
        #self.loss_scale = torch.nn.Parameter(torch.tensor([-0.5] * 11))

    def use_loss_for_tolerance(self):
        self.loss_for_tolerance = True

    def closure(self):#19个点的
        #with torch.autograd.set_detect_anomaly(True):
        opt = self.opt
        opt.zero_grad()
        if torch.cuda.is_available():
            # 设置默认设备为CUDA设备
            device = torch.device("cuda")
        else:
            # 如果CUDA不可用，则使用CPU设备
            device = torch.device("cpu")
        tmp = torch.tensor([-0.5] * 11).to(device)
        loss, loss_one, loss_two,loss_three,loss_four, loss_five,loss_six,loss_seven,loss_eight,loss_nine,loss_ten,loss_eleven = self.pde.loss(self.nn)
        loss_re = [loss_one, loss_two,loss_three,loss_four, loss_five,loss_six,loss_seven,loss_eight,loss_nine,loss_ten,loss_eleven]
        loss_re_3 = 0
        for i, task in enumerate(loss_re):
            loss_re_2 = (task / (2 * self.nn.loss_scale[i].exp()) + self.nn.loss_scale[i] / 2)
            loss_re_3 = loss_re_3 + loss_re_2

        # print("opt.param_groups[0]['params']]")
        # print([x.grad for x in opt.param_groups[0]['params']])
        # print("--------------------------------")
        loss_re_3.backward(retain_graph=True)
        # print("after opt.param_groups[0]['params']]")
        print([x.grad for x in opt.param_groups[0]['params']])
        # print("--------------------------------")
        self.loss.append(loss_re_3.item())
        return loss_re_3

        # loss.backward(retain_graph=True)
        # self.loss.append(loss.item())
        # return loss

    def solve(self):
        plotter = self.plotter
        n_train = self.n_train
        n_skip = self.n_skip
        if self.opt is None:
            opt = self.opt_class(self.nn.parameters(), lr=self.lr)
            self.opt = opt
        else:
            opt = self.opt
        if self.plot:
            (err_L1, err_L2, err_Linf), torque=plotter.plot()#用25个点画

        iterations_done = False
        start = time.perf_counter()
        err = 1.0
        #torque = 0
        for i in range(1, n_train+1):
            loss = opt.step(self.closure)
            if err < self.tol:
                iterations_done = True
            if i % n_skip == 0 or i == n_train or iterations_done:
                err_L1 = 0.0
                err_L2 = 0.0
                err_Linf = 0.0
                if self.plot:
                   (err_L1, err_L2, err_Linf), torque = plotter.plot()#这是25个点的
                else:
                    err_L1, err_L2, err_Linf = plotter.get_error()
                self.errors_L1.append(err_L1)
                self.errors_L2.append(err_L2)
                self.errors_Linf.append(err_Linf)
                if self.pde.has_exact():
                    e_str = f", Linf error={err_Linf:.3e}"
                else:
                    e_str = ''
                print(
                    f"Iteration ({i}/{n_train}): Loss={loss.item():.3e}" +
                    e_str
                )
                print("after 1000 Torque is :", torque)
                if abs(err_Linf) < 1e-8 or self.loss_for_tolerance:
                    err = loss.item()
                else:
                    err = err_Linf
            if iterations_done:
                break

        time_taken = time.perf_counter() - start
        self.time_taken = time_taken
        print(f"Done. Took {time_taken:.3f} seconds.")
        if self.plot:
            plotter.show()

    def save(self):
        timestr = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        dirname = self.out_dir
        if self.out_dir is None:
            print("No output directory set.  Skipping.")
            return
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print("Saving output to", dirname)
        fname = os.path.join(dirname,'solver.npz'+ timestr)
        np.savez(
            fname, loss=self.loss, error_L1=self.errors_L1,
            error_L2=self.errors_L2, error_Linf=self.errors_Linf,
            time_taken=self.time_taken
        )
        self.plotter.save(dirname)


class App:
    def __init__(self, pde_cls, nn_cls, plotter_cls,
                 optimizer=Optimizer):
        self.pde_cls = pde_cls
        self.nn_cls = nn_cls
        self.plotter_cls = plotter_cls
        self.optimizer = optimizer

    def _setup_activation_options(self, p, **kw):
        pass

    def _get_activation(self, args):
        pass

    def setup_argparse(self, **kw):
        '''Setup the argument parser.

        Any keyword arguments are used as the default values for the
        parameters.
        '''
        p = ArgumentParser(
            description="Configurable options.",
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
        # if os.path.exists('D:\Github\SPINN2\SPINN2\code\save'):
        #     nn = torch.load('D:\Github\SPINN2\SPINN2\code\save\model.pt2023-04-04-15-11-51')
        self.nn = nn
        plotter = self.plotter_cls.from_args(pde, nn, args)
        self.plotter = plotter

        solver = self.optimizer.from_args(pde, nn, plotter, args)
        self.solver = solver

        solver.solve()
        if args.directory is not None:
            solver.save()
