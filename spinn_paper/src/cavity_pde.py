"""
Cavity PDE implementation for permanent magnet eddy current device modeling.
This is the main implementation from the paper.
"""

import numpy as np
import torch
import torch.autograd as ag
from scipy import integrate
from .pde import RegularPDE
from .core import tensor


class CavityPDE(RegularPDE):
    """
    Cavity PDE for permanent magnet eddy current device modeling.
    
    This implements the physics-informed sparse neural network for solving
    the electromagnetic field equations in a permanent magnet eddy current device.
    """
    
    def __init__(self, n_nodes, ns, nb=None, nbs=None, sample_frac=1.0):
        """
        Initialize the cavity PDE.
        
        Parameters:
        -----------
        n_nodes : int
            Number of nodes for the neural network
        ns : int
            Number of sample points
        nb : int, optional
            Number of boundary nodes
        nbs : int, optional
            Number of boundary samples
        sample_frac : float
            Fraction of interior nodes used for sampling
        """
        self.sample_frac = sample_frac
        self.omega = 150 * 2 * np.pi / 60  # Angular velocity
        
        # Physical parameters
        Pi = np.pi
        alpha = 1
        rm = 0.045  # Magnet radius
        p = 5       # Number of pole pairs
        self.lengthX = alpha * rm * Pi / p
        
        # Layer thicknesses
        self.thickness_mag = 0.010    # Magnet thickness
        self.thickness_air = 0.003    # Air gap thickness
        self.thickness_copper = 0.005 # Copper thickness
        self.fitness = 0.0           # Boundary fitness
        
        # Generate interior nodes (three regions)
        self._generate_interior_nodes(n_nodes)
        
        # Generate fixed nodes (boundary conditions)
        self._generate_fixed_nodes(nb or n_nodes)
        
        # Generate sample points for training
        self._generate_sample_points(ns, nbs or ns)
        
        # Cache for interior and boundary samples
        self._interior_cache = None
        self._boundary_cache = None
        self._plot_cache = None

    def _generate_interior_nodes(self, n_nodes):
        """Generate interior nodes for the three regions."""
        n = n_nodes
        
        # Magnet region
        xl, xr = self.fitness, self.lengthX - self.fitness
        yb, yt = self.fitness, self.thickness_mag - self.fitness
        slx = slice(xl, xr, n * 1j)
        sly = slice(yb, yt, n * 1j)
        x1, y1 = np.mgrid[slx, sly]
        
        # Air gap region
        yb, yt = (self.thickness_mag + self.fitness, 
                  self.thickness_mag + self.thickness_air - self.fitness)
        sly = slice(yb, yt, n * 1j * 0.3)
        x2, y2 = np.mgrid[slx, sly]
        
        # Copper region
        yb, yt = (self.thickness_mag + self.thickness_air + self.fitness,
                  self.thickness_mag + self.thickness_air + self.thickness_copper - self.fitness)
        sly = slice(yb, yt, n * 1j * 0.5)
        x3, y3 = np.mgrid[slx, sly]
        
        # Combine all regions
        x = np.append(x1, x2)
        x = np.append(x, x3)
        y = np.append(y1, y2)
        y = np.append(y, y3)
        
        self.i_nodes = (x.ravel(), y.ravel())

    def _generate_fixed_nodes(self, nb):
        """Generate fixed nodes for boundary conditions."""
        if nb == 0:
            self.f_nodes = ([], [])
            return
            
        self.nb = nb
        lengthY4 = self.thickness_mag + self.thickness_air + self.thickness_copper
        
        _x = np.linspace(self.fitness, self.lengthX - self.fitness, nb)
        _o = np.ones_like(_x)
        _l = np.linspace(self.fitness, lengthY4 - self.fitness, nb)
        
        # Boundary points: top, air-copper interface, magnet-air interface, bottom, left, right
        x = np.hstack((_x, _x, _x, _x, _x*0.0, _o*self.lengthX))
        y = np.hstack((_o*0.0, _o*self.thickness_mag, 
                      _o*(self.thickness_mag+self.thickness_air), 
                      _o*(self.thickness_mag+self.thickness_air+self.thickness_copper), 
                      _l, _l))
        
        self.f_nodes = (x, y)

    def _generate_sample_points(self, ns, nbs):
        """Generate sample points for training."""
        self.ns = ns = round(np.sqrt(ns) + 0.49)
        self.nbs = nbs
        
        # Generate interior samples for each region
        self._generate_interior_samples(ns)
        
        # Generate boundary samples
        self._generate_boundary_samples(nbs)
        
        # Set up sampling
        self.n_interior = len(self.p_samples[0])
        self.rng_interior = np.arange(self.n_interior)
        self.sample_size = int(self.sample_frac * self.n_interior)

    def _generate_interior_samples(self, ns):
        """Generate interior sample points for each region."""
        # Magnet region samples
        xl, xr = self.fitness, self.lengthX - self.fitness
        yb, yt = self.fitness, self.thickness_mag - self.fitness
        slx = slice(xl, xr, ns * 1j)
        sly = slice(yb, yt, ns * 1j)
        x1, y1 = np.mgrid[slx, sly]
        
        self.label = len(x1[0]) * len(x1[:, 0])
        self.p_samples_x1 = x1
        self.p_samples_y1 = y1
        
        # Air gap region samples
        yb2, yt2 = (self.thickness_mag + self.fitness,
                    self.thickness_mag + self.thickness_air - self.fitness)
        sly = slice(yb2, yt2, ns * 1j * 0.3)
        x2, y2 = np.mgrid[slx, sly]
        
        self.label2 = len(x2[0]) * len(y2[:, 0])
        self.p_samples_x2 = x2
        self.p_samples_y2 = y2
        
        # Copper region samples
        yb3, yt3 = (self.thickness_mag + self.thickness_air + self.fitness,
                    self.thickness_mag + self.thickness_air + self.thickness_copper - self.fitness)
        sly = slice(yb3, yt3, ns * 1j)
        x3, y3 = np.mgrid[slx, sly]
        
        self.label3 = len(x3[0]) * len(y3[:, 0])
        self.p_samples_x3 = x3
        self.p_samples_y3 = y3
        
        # Combine all samples
        x = np.append(x1, x2)
        x = np.append(x, x3)
        y = np.append(y1, y2)
        y = np.append(y, y3)
        
        self.p_samples = tensor((x.ravel(), y.ravel()), requires_grad=True)

    def _generate_boundary_samples(self, nbs):
        """Generate boundary sample points."""
        lengthY4 = self.thickness_mag + self.thickness_air + self.thickness_copper
        
        _x = np.linspace(self.fitness, self.lengthX - self.fitness, nbs)
        _o = np.ones_like(_x)
        _l = np.linspace(self.fitness, lengthY4 - self.fitness, nbs)
        
        def tg(x):
            return tensor(x, requires_grad=True)
        
        # Boundary segments
        self.one = one = (tg(_x), tg(_o * lengthY4))
        self.two = two = (tg(_x), tg(_o * (self.thickness_mag + self.thickness_air)))
        self.three = three = (tg(_x), tg(_o * self.thickness_mag))
        self.four = four = (tg(_x), tg(_o * 0.00))
        self.five = five = (tg(_o * 0.00), tg(_l))
        self.six = six = (tg(_o * self.lengthX), tg(_l))
        
        self.b_samples = (
            torch.cat([x[0] for x in (one, two, three, four, five, six)]),
            torch.cat([x[1] for x in (one, two, three, four, five, six)])
        )

    def _get_interior_samples(self):
        """Get interior sample points for training."""
        if self.sample_size < self.n_interior:
            indices = np.random.choice(self.n_interior, self.sample_size, replace=False)
            xs = self.p_samples[0][indices]
            ys = self.p_samples[1][indices]
        else:
            xs, ys = self.p_samples[0], self.p_samples[1]
        
        return (xs, ys), self.p_samples_x1, self.p_samples_x2, self.p_samples_x3, \
               self.p_samples_y1, self.p_samples_y2, self.p_samples_y3

    def _get_boundary_samples(self):
        """Get boundary sample points for training."""
        return self.b_samples

    def _get_plot_points(self):
        """Get points for plotting results."""
        # Create a fine grid for plotting
        nx, ny = 100, 100
        x = np.linspace(self.fitness, self.lengthX - self.fitness, nx)
        y = np.linspace(self.fitness, 
                       self.thickness_mag + self.thickness_air + self.thickness_copper - self.fitness, 
                       ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def n_vars(self):
        """Return number of variables (2 for A_dy and A_dz)."""
        return 2

    def has_exact(self):
        """No exact solution available."""
        return False

    def interior_loss(self, nn):
        """Compute interior loss (Maxwell equations)."""
        weight = 10
        (xs, ys), x1, x2, x3, y1, y2, y3 = self.interior()
        
        U = nn(xs, ys)
        Ady = U[:, 0]  # A_dy component
        Adz = U[:, 1]  # A_dz component
        
        # Split by regions
        A1dy = Ady[:self.label]
        A1dz = Adz[:self.label]
        A2dy = Ady[self.label:self.label + self.label2]
        A2dz = Adz[self.label:self.label + self.label2]
        A3dy = Ady[self.label + self.label2:]
        A3dz = Adz[self.label + self.label2:]
        
        # Compute derivatives
        A1dy2, A1dydz = self._compute_gradient(A1dy, xs[:self.label], ys[:self.label])
        A1dzdy, A1dz2 = self._compute_gradient(A1dz, xs[:self.label], ys[:self.label])
        A2dy2, A2dydz = self._compute_gradient(A2dy, xs[self.label:self.label + self.label2], 
                                               ys[self.label:self.label + self.label2])
        A2dzdy, A2dz2 = self._compute_gradient(A2dz, xs[self.label:self.label + self.label2], 
                                               ys[self.label:self.label + self.label2])
        A3dy2, A3dydz = self._compute_gradient(A3dy, xs[self.label + self.label2:], 
                                               ys[self.label + self.label2:])
        A3dzdy, A3dz2 = self._compute_gradient(A3dz, xs[self.label + self.label2:], 
                                               ys[self.label + self.label2:])
        
        # Physical constants
        u0 = tensor(4 * np.pi * 1e-7)
        Pi = tensor(np.pi)
        Br = tensor(1.25)
        alpha = tensor(1)
        rm = tensor(0.045)
        beta = tensor(5 / rm)
        sigma = tensor(57000000)
        omega = tensor(self.omega)
        
        # Maxwell equations for each region
        # Magnet region (region 1)
        p = 5  # Number of pole pairs
        K1 = ((4 * Br * rm) / (Pi * p)) * torch.sin(alpha * Pi / 2)
        maxwell1_dy = A1dy2 + A1dz2 - beta**2 * A1dy + K1 * beta * torch.cos(beta * xs[:self.label])
        maxwell1_dz = A1dz2 + A1dy2 - beta**2 * A1dz - K1 * beta * torch.sin(beta * xs[:self.label])
        
        # Air gap region (region 2)
        maxwell2_dy = A2dy2 + A2dz2 - beta**2 * A2dy
        maxwell2_dz = A2dz2 + A2dy2 - beta**2 * A2dz
        
        # Copper region (region 3)
        maxwell3_dy = A3dy2 + A3dz2 - beta**2 * A3dy + u0 * sigma * omega * A3dy
        maxwell3_dz = A3dz2 + A3dy2 - beta**2 * A3dz + u0 * sigma * omega * A3dz
        
        # Total loss
        loss1 = (maxwell1_dy**2).sum() + (maxwell1_dz**2).sum()
        loss2 = (maxwell2_dy**2).sum() + (maxwell2_dz**2).sum()
        loss3 = (maxwell3_dy**2).sum() + (maxwell3_dz**2).sum()
        
        return weight * (loss1 + loss2 + loss3)

    def boundary_loss(self, nn):
        """Compute boundary loss."""
        bc_weight = self.ns * 20
        xs, ys = self.boundary()
        
        U = nn(xs, ys)
        Ady = U[:, 0]
        Adz = U[:, 1]
        
        # Boundary conditions (simplified for this example)
        # In practice, these would be more complex based on the specific problem
        bc_loss = (Ady**2).sum() + (Adz**2).sum()
        
        return bc_weight * bc_loss

    def cal_torque(self, Ady, x, y):
        """Calculate torque from the solution."""
        alpha = 1
        Pi = np.pi
        br = 1.25
        rm = 0.045
        p = 5
        sigma = 57000000
        omega = self.omega
        L = 0.03
        
        K = ((4 * br * rm) / (Pi * p)) * (torch.sin(tensor(alpha * Pi / 2)))
        beta = p / rm
        
        exact_eddy_current = -sigma * omega * rm * Ady
        J2 = exact_eddy_current * exact_eddy_current
        dy = y
        dx = x
        
        torque = integrate.simps(J2.T.cpu().detach().numpy(), dy.T.cpu().detach().numpy(), axis=0)
        torque = (integrate.simps(torque, dx[:, 0].cpu().detach().numpy()) * L * 10) / (sigma * omega)
        
        return torque
