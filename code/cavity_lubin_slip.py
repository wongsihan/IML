import os

# from mayavi import mlab
import numpy as np
import torch

torch.cuda.empty_cache()

import torch.autograd as ag
import time
from scipy import integrate
from common_nomayavi import tensor
from spinn2d_nomayavi import Plotter2D, App2D, SPINN2D
from pde2d_base_nomayavi import RegularPDE
import matplotlib.pyplot as plt

class CavityPDE(RegularPDE):
    def __init__(self, n_nodes, ns, nb=None, nbs=None, sample_frac=1.0):#生成100+40个固定点和2500个采样点
        self.sample_frac = sample_frac
        self.omega = 150 * 2 * np.pi / 60
        # Interior nodes 100个内部固定点的坐标
        #new method 一共三个区域,run函数定义数量
        #永磁体区域 10000
        n = n_nodes
        Pi = np.pi
        alpha = 1
        rm = 0.045
        p = 5
        lengthX = alpha * rm * Pi / p

        # 永磁体
        self.thickness_mag = 0.010
        dxb2 = 1.00 / (1000*n)
        self.fitness = 0
        xl, xr = self.fitness, lengthX - self.fitness
        yb, yt = self.fitness, self.thickness_mag - self.fitness
        slx = slice(xl, xr, n * 1j)
        sly = slice(yb, yt, n * 1j)
        x1, y1 = np.mgrid[slx, sly]

        # 气隙
        self.thickness_air = 0.003
        yb, yt = self.thickness_mag + self.fitness, self.thickness_mag + self.thickness_air - self.fitness
        sly = slice(yb, yt, n * 1j * 0.3)
        x2, y2 = np.mgrid[slx, sly]

        # 铜
        self.thickness_copper = 0.005
        yb, yt = self.thickness_mag + self.thickness_air + self.fitness, self.thickness_mag + self.thickness_air + self.thickness_copper - self.fitness
        sly = slice(yb, yt, n * 1j * 0.5)
        x3, y3 = np.mgrid[slx, sly]

        x = np.append(x1,x2)
        x = np.append(x,x3)
        y = np.append(y1,y2)
        y = np.append(y,y3)
        self.i_nodes = (x.ravel(), y.ravel())
        # Fixed nodes 40个都在边界上
        nb = n if nb is None else nb
        self.nb = nb#10
        if nb == 0:
            self.f_nodes = ([], [])
        else:
            dxb2 = 1.0/(nb*1000)#0.1
            lengthX = alpha * rm * Pi / p
            lengthY4 = self.thickness_mag + self.thickness_air +self.thickness_copper
            _x = np.linspace(self.fitness, lengthX - self.fitness, nb)
            _o = np.ones_like(_x)
            _l = np.linspace(self.fitness, lengthY4 - self.fitness, nb)
            x = np.hstack((_x, _x, _x, _x, _x*0.0, _o*lengthX))
            y = np.hstack((_o*0.0, _o*self.thickness_mag, _o*(self.thickness_mag+self.thickness_air), _o*(self.thickness_mag+self.thickness_air+self.thickness_copper), _l, _l))
            self.f_nodes = (x, y)#一个40个都在边界上

        # Interior samples 2500个内部的0.02-0.98的点
        # 永磁体区域
        self.ns = ns = round(np.sqrt(ns) + 0.49)  # 15
        dxb2 = 1.00 / (1000*ns)#0.00000667
        xl, xr = self.fitness, lengthX - self.fitness
        yb, yt = self.fitness, self.thickness_mag - self.fitness
        slx = slice(xl, xr, ns * 1j)
        sly = slice(yb, yt, ns * 1j)
        x1, y1 = np.mgrid[slx, sly]

        self.label = len(x1[0]) * len(x1[:,0])
        self.p_samples_x1 = x1
        self.p_samples_y1 = y1


        yb2, yt2 = self.thickness_mag + self.fitness, self.thickness_mag + self.thickness_air - self.fitness
        sly = slice(yb2, yt2, ns * 1j * 0.3)
        x2, y2 = np.mgrid[slx, sly]
        self.label2 = len(x2[0]) * len(y2[:,0])
        self.p_samples_x2 = x2
        self.p_samples_y2 = y2

        yb3, yt3 = self.thickness_mag + self.thickness_air + self.fitness, self.thickness_mag + self.thickness_air + self.thickness_copper - self.fitness
        sly = slice(yb3, yt3, ns * 1j)
        x3, y3 = np.mgrid[slx, sly]
        self.label3 = len(x3[0]) * len(y3[:,0])
        self.p_samples_x3 = x3
        self.p_samples_y3 = y3

        # slx_fake = slice(0, lengthX, ns * 1j)
        # sly_fake = slice(thickness_mag + thickness_air, thickness_mag + thickness_air + thickness_copper, ns * 1j)
        # x3_fake, y3_fake = np.mgrid[slx_fake, sly_fake]
        x = np.append(x1, x2)
        x = np.append(x, x3)
        y = np.append(y1, y2)
        y = np.append(y, y3)
        self.p_samples = tensor((x.ravel(), y.ravel()), requires_grad=True)
        # self.p_samples_fake_x3 = x3_fake
        # self.p_samples_fake_y3 = y3_fake

        # Boundary samples 200个，正方形每条边50个
        nbs = ns if nbs is None else nbs
        self.nbs = nbs#50

        dxb2 = 1.0 / (nbs*1000)  # 0.1
        _x = np.linspace(self.fitness, lengthX - self.fitness, nbs)
        _o = np.ones_like(_x)
        _l = np.linspace(self.fitness, lengthY4 - self.fitness, nbs)

        def tg(x):
            return tensor(x, requires_grad=True)

        self.one = one = (tg(_x), tg(_o * lengthY4))
        self.two = two = (tg(_x), tg(_o * (self.thickness_mag + self.thickness_air)))
        self.three = three = (tg(_x), tg(_o * self.thickness_mag))
        self.four = four = (tg(_x), tg(_o * 0.00))
        self.five = five = (tg(_o*0.00), tg(_l))
        self.six = six = (tg(_o*lengthX), tg(_l))
        self.b_samples = (
            torch.cat([x[0] for x in (one, two, three, four, five, six)]),
            torch.cat([x[1] for x in (one, two, three, four, five, six)])
        )#200个，正方形每条边50个

        self.n_interior = len(self.p_samples[0])#2500
        self.rng_interior = np.arange(self.n_interior)#2500 应该是索引
        self.sample_size = int(self.sample_frac*self.n_interior)#250

    def cal_torque(self, Ady, x, y):
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
        b = self.thickness_mag
        c = self.thickness_air
        d = self.thickness_copper
        exact_eddy_current = -sigma * omega * rm * Ady  # [46,46]
        J2 = exact_eddy_current * exact_eddy_current
        dy = y
        dx = x
        torque = integrate.simps(J2.T.cpu().detach().numpy(), dy.T.cpu().detach().numpy(), axis=0)
        torque = (integrate.simps(torque, dx[:, 0].cpu().detach().numpy()) * L * 10) / (sigma * omega)

        return torque

    def _compute_gradient(self, u, xs, ys):
        du = ag.grad(
            outputs=u, inputs=(xs, ys), grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )
        return du[0], du[1]

    def n_vars(self):#输出几个变量
        return 2

    def has_exact(self):#是否有解析解
        return False

    def interior_loss(self, nn):
        weight = 10
        (xs, ys),x1,x2,x3,y1,y2,y3 = self.interior()  # 250 250
        U = nn(xs, ys)  # (250,3) U就是A1 和 A2
        Ady = U[:, 0]  # A1
        Adz = U[:, 1]  # A2
        A1dy = Ady[:self.label]
        A1dz = Adz[:self.label]
        A2dy = Ady[self.label:]
        A2dz = Adz[self.label:]
        Ady_cor_copper = xs[self.label + self.label2:]
        Adz_cor_copper = ys[self.label + self.label2:]
        Ady_copper = Ady[self.label + self.label2:]
        A1dy2, A1dydz = self._compute_gradient(A1dy, xs, ys)
        A1dzdy, A1dz2 = self._compute_gradient(A1dz, xs, ys)
        A2dy2, A2dydz = self._compute_gradient(A2dy, xs, ys)
        A2dzdy, A2dz2 = self._compute_gradient(A2dz, xs, ys)

        # The maxwell equations.
        u0 = tensor(4 * np.pi * 1e-7, requires_grad=True)
        Pi = tensor(np.pi, requires_grad=True)
        Br = tensor(1.25, requires_grad=True)
        alpha = tensor(1, requires_grad=True)
        rm = tensor(0.045, requires_grad=True)
        beta = tensor(5 / rm, requires_grad=True)
        xsA1 = xs[:self.label]
        mzy = (4 * Br / Pi * u0) * (torch.sin(alpha * Pi / 2)) * beta * (torch.cos(beta * xsA1))

        A1dy2 = A1dy2[:self.label]
        A1dz2 = A1dz2[:self.label]
        A2dy2 = A2dy2[self.label:]
        A2dz2 = A2dz2[self.label:]
        loss_one =(((A1dy2 + A1dz2 + (u0 * mzy)) ** 2)*weight).sum()
        loss_two = (((A2dy2 + A2dz2) ** 2)*weight).sum()

        # alpha = 1
        # Pi = np.pi
        # br = 1.25
        # rm = 0.045
        # p = 5
        # sigma = 57000000
        # omega = self.omega
        # L = 0.03
        # K = ((4 * br * rm) / (Pi * p)) * (torch.sin(tensor(alpha * Pi / 2)))
        # beta = p / rm
        # b = self.thickness_mag
        # c = self.thickness_air
        # d = self.thickness_copper
        # x1 = tensor(x1, requires_grad=True)
        # y1 = tensor(y1, requires_grad=True)
        # x3 = tensor(x3, requires_grad=True)
        # y3 = tensor(y3, requires_grad=True)
        # # exact_A2 = K * ((torch.sinh(tensor(beta * 0.1, requires_grad=True)) * torch.cosh(beta * (yn_copper - 0.18)))/(torch.sinh(tensor(beta*0.18, requires_grad=True)))) *(torch.cos(beta * xn_copper))
        # exact_A2 = K * (torch.sinh(tensor(beta * b)) * torch.cosh(beta * (y3 - (b + c + d))) * torch.cos(
        #     beta * x3))
        # exact_A2 = exact_A2 / torch.sinh(tensor(beta * (b + c + d)))
        # exact_A1 = 1-((torch.sinh(tensor(beta * (c+d))) * torch.cosh(beta*y1))/torch.sinh(tensor(beta*(b+c+d))))
        # exact_A1 = K * exact_A1 * torch.cos(beta*x1)
        # # zhuanju1
        # exact_A1dy, exact_A1dz = self._compute_gradient(exact_A1, x1, y1)
        # exact_A2dy, exact_A2dz = self._compute_gradient(exact_A2, x3, y3)
##################
        #torque = self.cal_torque(A2dy,x3,y3)
        # exact_eddy_current = -sigma * omega * rm * A2dy  # [46,46]
        # J2 = exact_eddy_current * exact_eddy_current
        # dy = y3
        # dx = x3
        # torque = integrate.simps(J2.T.cpu().detach().numpy(), dy.T.cpu().detach().numpy(), axis=0)
        # torque = (integrate.simps(torque, dx[:,0].cpu().detach().numpy()) * L * 10) / (sigma * omega)
        #print("loss里的转矩是：",torque)
 ########################

        # d1 = exact_A2dy.shape[0]
        # d2 = exact_A2dy.shape[1]
        # exact_A2dy = exact_A2dy.reshape(d1*d2)
        #
        # d3 = exact_A1dy.shape[0]
        # d4 = exact_A1dy.shape[1]
        # exact_A1dy = exact_A1dy.reshape(d1 * d2)
        # loss_three =(((exact_A2dy - Ady_copper)**2)*500).sum()#copper
        # loss_four = (((exact_A1dy - A1dy)**2)*500).sum()#magnet

        return loss_one + loss_two # [1]



    def conform_exact(self,Ady_pre,Adz_pre,y,z,pattern,boundary):
        weight = 1000
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
        b = 0.01
        c = 0.003
        d = 0.005
        n_one = len(self.one[0])
        n_two = len(self.two[0]) + n_one
        n_three = len(self.three[0]) + n_two
        n_four = len(self.four[0]) + n_three
        n_five = len(self.five[0]) + n_four
        n_six = len(self.six[0]) + n_five
        y = tensor(y, requires_grad=True)
        z = tensor(z, requires_grad=True)
        if pattern == 1:
            exact_A1 = 1 - ((torch.sinh(tensor(beta * (c + d))) * torch.cosh(beta * z)) / torch.sinh(
                tensor(beta * (b + c + d))))
            exact_A1 = K * exact_A1 * torch.cos(beta * y)
            exact_Ady, exact_Adz = self._compute_gradient(exact_A1, y, z)

        elif pattern == 2:
            exact_A2 = K * (torch.sinh(tensor(beta * b)) * torch.cosh(beta * (z - (b + c + d))) * torch.cos(
                beta * y))
            exact_A2 = exact_A2 / torch.sinh(tensor(beta * (b + c + d)))
            exact_Ady, exact_Adz = self._compute_gradient(exact_A2, y, z)

        # d1 = exact_A2dy.shape[0]
        # d2 = exact_A2dy.shape[1]
        # exact_A2dy = exact_A2dy.reshape(d1 * d2)
        # d3 = exact_A1dy.shape[0]
        # d4 = exact_A1dy.shape[1]
        # exact_A1dy = exact_A1dy.reshape(d1 * d2)
        loss_one = loss_two = 0
        if boundary == 1:
            exact_Ady = exact_Ady[:n_one]
            exact_Adz = exact_Adz[:n_one]

            loss_one = (((exact_Ady - Ady_pre) ** 2) * weight).sum()  # copper
            loss_two = (((exact_Adz - Adz_pre) ** 2) * weight).sum()  # magnet
        elif boundary == 2:
            exact_Ady = exact_Ady[n_one:n_two]
            exact_Adz = exact_Adz[n_one:n_two]

            loss_one = (((exact_Ady - Ady_pre) ** 2) * weight).sum()  # copper
            loss_two = (((exact_Adz - Adz_pre) ** 2) * weight).sum()
        elif boundary == 3:
            exact_Ady = exact_Ady[n_two:n_three]
            exact_Adz = exact_Adz[n_two:n_three]

            loss_one = (((exact_Ady - Ady_pre) ** 2) * weight).sum()  # copper
            loss_two = (((exact_Adz - Adz_pre) ** 2) * weight).sum()
        elif boundary == 4:
            exact_Ady = exact_Ady[n_three:n_four]
            exact_Adz = exact_Adz[n_three:n_four]

            loss_one = (((exact_Ady - Ady_pre) ** 2) * weight).sum()  # copper
            loss_two = (((exact_Adz - Adz_pre) ** 2) * weight).sum()
        return loss_one + loss_two

    def conform_exact_boundary(self, Ady_pre, Adz_pre, y, z, boundary):
        weight = 1000
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
        b = 0.01
        c = 0.003
        d = 0.005
        n_one = len(self.one[0])
        n_two = len(self.two[0]) + n_one
        n_three = len(self.three[0]) + n_two
        n_four = len(self.four[0]) + n_three
        n_five = len(self.five[0]) + n_four
        n_six = len(self.six[0]) + n_five
        y = tensor(y, requires_grad=True)
        z = tensor(z, requires_grad=True)

        exact_A1 = 1 - ((torch.sinh(tensor(beta * (c + d))) * torch.cosh(beta * z)) / torch.sinh(
            tensor(beta * (b + c + d))))
        exact_A1 = K * exact_A1 * torch.cos(beta * y)
        exact_A1dy, exact_A1dz = self._compute_gradient(exact_A1, y, z)


        exact_A2 = K * (torch.sinh(tensor(beta * b)) * torch.cosh(beta * (z - (b + c + d))) * torch.cos(
            beta * y))
        exact_A2 = exact_A2 / torch.sinh(tensor(beta * (b + c + d)))
        exact_A2dy, exact_A2dz = self._compute_gradient(exact_A2, y, z)

        loss_one = loss_two = 0
        if boundary == 5:
            exact_A1dy = exact_A1dy[n_four:n_five]
            exact_A2dy = exact_A2dy[n_four:n_five]
            exact_A1dz = exact_A1dz[n_four:n_five]
            exact_A2dz = exact_A2dz[n_four:n_five]

            loss_one = (((exact_A1dy[:6] - Ady_pre[:6]) ** 2) * weight).sum()  # copper
            loss_two = (((exact_A1dz[:6] - Adz_pre[:6]) ** 2) * weight).sum()  # magnet
            #loss_one += (((exact_A2dy[6:] - Ady_pre[6:]) ** 2) * weight).sum()
            #loss_two += (((exact_A2dz[6:] - Adz_pre[6:]) ** 2) * weight).sum()
        elif boundary == 6:
            exact_A1dy = exact_A1dy[n_five:n_six]
            exact_A2dy = exact_A2dy[n_five:n_six]
            exact_A1dz = exact_A1dz[n_five:n_six]
            exact_A2dz = exact_A2dz[n_five:n_six]
            loss_one = (((exact_A1dy[:6] - Ady_pre[:6]) ** 2) * weight).sum()  # copper
            loss_two = (((exact_A1dz[:6] - Adz_pre[:6]) ** 2) * weight).sum()  # magnet
            #loss_one += (((exact_A2dy[6:] - Ady_pre[6:]) ** 2) * weight).sum()
            #loss_two += (((exact_A2dz[6:] - Adz_pre[6:]) ** 2) * weight).sum()

        return loss_one + loss_two

    def boundary_loss(self, nn):
        bc_weight = self.ns * 20  # 1000
        xb, yb = self.boundary()  # 200 200
        Ub = nn(xb, yb)  # (200，3)
        Ady = Ub[:, 0]  # A1
        Adz = Ub[:, 1]  # A2
        u0 = tensor(4 * np.pi * 1e-7, requires_grad=True)
        Pi = tensor(np.pi, requires_grad=True)
        Br = tensor(1.25, requires_grad=True)
        alpha = tensor(1, requires_grad=True)
        rm = tensor(0.045, requires_grad=True)
        beta = tensor(5 / rm, requires_grad=True)
        ur = 1

        n_one = len(self.one[0])
        n_two = len(self.two[0]) + n_one
        n_three = len(self.three[0]) + n_two
        n_four = len(self.four[0]) + n_three
        n_five = len(self.five[0]) + n_four
        n_six = len(self.six[0]) + n_five

        Ady_one = Ady[:n_one]
        Adz_one = Adz[:n_one]
        Ady_two = Ady[n_one:n_two]
        Adz_two = Adz[n_one:n_two]
        Ady_three = Ady[n_two:n_three]
        Adz_three = Adz[n_two:n_three]
        Ady_four = Ady[n_three:n_four]
        Adz_four = Adz[n_three:n_four]
        Ady_five = Ady[n_four:n_five]
        Adz_five = Adz[n_four:n_five]
        Ady_six = Ady[n_five:n_six]
        Adz_six = Adz[n_five:n_six]

        # 1 2边遵循A2
        A1dy2, A1dydz = self._compute_gradient(Ady_one, xb, yb)
        A1dzdy, A1dz2 = self._compute_gradient(Adz_one, xb, yb)
        A2dy2, A2dydz = self._compute_gradient(Ady_two, xb, yb)
        A2dzdy, A2dz2 = self._compute_gradient(Adz_two, xb, yb)
        A1dy2 = A1dy2[:n_one]
        A1dz2 = A1dz2[:n_one]
        A2dy2 = A2dy2[n_one:n_two]
        A2dz2 = A2dz2[n_one:n_two]
        # bc_loss = self.conform_exact(Ady_one, Adz_one, xb, yb, 2, 1)
        # bc_loss += self.conform_exact(Ady_two, Adz_two, xb, yb, 2, 2)
        bc_loss = ((A1dy2 + A1dz2) ** 2).sum()
        bc_loss += ((A2dy2 + A2dz2) ** 2).sum()

        # 3边遵循 A1 A2
        A3dy2, A3dydz = self._compute_gradient(Ady_three, xb, yb)
        A3dzdy, A3dz2 = self._compute_gradient(Adz_three, xb, yb)
        A3dy2 = A3dy2[n_two:n_three]
        A3dz2 = A3dz2[n_two:n_three]

        xbA3 = xb[n_two:n_three]
        mzy = (4 * Br / Pi * u0) * (torch.sin(alpha * Pi / 2)) * beta * (torch.cos(beta * xbA3))
        bc_loss += self.conform_exact(Ady_three, Adz_three, xb, yb, 1, 3)
        bc_loss += self.conform_exact(Ady_three, Adz_three, xb, yb, 2, 3)
        bc_loss += (((A3dy2 + A3dz2 + u0 * mzy) ** 2)*bc_weight).sum()
        bc_loss += ((A3dy2 + A3dz2) ** 2).sum()

        # 4边遵循 A1
        A4dy2, A4dydz = self._compute_gradient(Ady_four, xb, yb)
        A4dzdy, A4dz2 = self._compute_gradient(Adz_four, xb, yb)
        A4dz2 = A4dz2[n_three:n_four]
        A4dy2 = A4dy2[n_three:n_four]
        xbA4 = xb[n_three:n_four]
        mzy = (4 * Br / Pi * u0) * (torch.sin(alpha * Pi / 2)) * beta * (torch.cos(beta * xbA4))
        bc_loss += self.conform_exact(Ady_four, Adz_four, xb, yb, 1, 4)
        bc_loss += ((A4dy2 + A4dz2 + u0 * mzy) ** 2).sum()

        # 5 6边赋常数
        # Ady_five = Ady_five * 0
        # Adz_five = Adz_five * 0
        # Ady_six = Ady_six * 0
        # Adz_six = Adz_six * 0
        bc_loss += self.conform_exact_boundary(Ady_five, Adz_five, xb, yb, 5)
        bc_loss += self.conform_exact_boundary(Ady_six, Adz_six, xb, yb, 6)

        return bc_loss

    def plot_points(self):
        n = min(self.ns * 2, 200)#30
        ns = round(np.sqrt(self.ns) + 0.49)#5
        #dxb2 = 1.00 / (1000 * ns)#0.0002
        Pi = np.pi
        alpha = 1
        rm = 0.045
        p = 5
        length = alpha * rm * Pi / p# ns 50 n 100
        thickness_sum = self.thickness_air + self.thickness_mag + self.thickness_copper
        thickness_mag_air = self.thickness_air + self.thickness_mag
        x, y = np.mgrid[self.fitness:length-self.fitness:n * 1j, self.fitness:thickness_sum-self.fitness:n * 1j]
        x = x
        y = y
        x_copper, y_copper = np.mgrid[0:length:n * 1j, thickness_mag_air:thickness_sum:n * 1j]
        #x_copper, y_copper = np.mgrid[self.fitness:length-self.fitness:n * 1j, thickness_mag_air:thickness_sum:n * 1j]
        x_copper = x_copper
        y_copper= y_copper
        return x, y, x_copper, y_copper,self.omega,self.thickness_mag,self.thickness_air,self.thickness_copper#(100,100) (100,100)在0-1,0-1这个区间，生成了10000个坐标，间隔0.01


class CavityPlotter(Plotter2D):
    def _compute_gradient(self, u, xs, ys):
        du = ag.grad(
            outputs=u, inputs=(xs, ys), grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )
        return du[0], du[1]

    def cal_torque(self, Ady, x, y):
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
        b = self.thickness_mag
        c = self.thickness_air
        d = self.thickness_copper
        exact_eddy_current = -sigma * omega * rm * Ady  # [46,46]
        J2 = exact_eddy_current * exact_eddy_current
        dy = y
        dx = x
        torque = integrate.simps(J2.T.cpu().detach().numpy(), dy.T.cpu().detach().numpy(), axis=0)
        torque = (integrate.simps(torque, dx[:, 0].cpu().detach().numpy()) * L * 10) / (sigma * omega)

        return torque
    def get_plot_data(self):
        x, y, x_copper, y_copper,self.omega,self.thickness_mag,self.thickness_air,self.thickness_copper = self.pde.plot_points()
        xt, yt = tensor(x.ravel()), tensor(y.ravel())  # 10000 10000
        x_copper_t, y_copper_t = tensor(x_copper.ravel()), tensor(y_copper.ravel())
        pn = self.nn(xt, yt).detach().cpu().numpy()  # 10000,3
        pn_copper = self.nn(x_copper_t, y_copper_t).detach().cpu().numpy()
        pn.shape = x.shape + (2,)  # 100，100，3
        pn_copper.shape = x_copper.shape + (2,)  # 100，100，3
        return x, y, pn, pn_copper, x_copper, y_copper

    def plot_weights(self):
        '''Implement this method to plot any weights.
        Note this is always called *after* plot_solution.
        '''
        nn = self.nn
        x, y = nn.centers()
        h = nn.widths().detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        # if not self.plt2:
        #     self.plt2 = mlab.points3d(
        #         x, y, np.zeros_like(x), h, mode='2dcircle',
        #         scale_factor=1.0, color=(1, 0, 0), opacity=0.4
        #     )
        #     self.plt2.glyph.glyph_source.glyph_source.resolution = 20
        #     mlab.pipeline.glyph(
        #         self.plt2, color=(1, 0, 0), scale_factor=0.025, opacity=0.4,
        #         mode='2dcross', scale_mode='none'
        #     )
        # else:
        #     self.plt2.mlab_source.trait_set(x=x, y=y, scalars=h)

    def plot_solution(self):
        xn, yn, pn, pn_copper, xn_copper, yn_copper = self.get_plot_data()  # 前两者合并就是所有坐标   第三项就是每个点对应的uvp三个值(100,100) (100,100) (100,100,3)
        Ady, Adz = pn[..., 0], pn[..., 1]
        Ady_copper, Adz_copper = pn_copper[..., 0], pn_copper[..., 1]
        xn_copper = tensor(xn_copper, requires_grad=True)
        yn_copper = tensor(yn_copper, requires_grad=True)
        alpha = 1
        Pi = np.pi
        br = 1.25
        rm = 0.045
        p = 5
        sigma = 57000000
        omega = self.omega
        L = 0.03
        K = ((4*br*rm)/(Pi*p)) * (torch.sin(tensor(alpha * Pi/2)))
        beta = p / rm
        b = self.thickness_mag
        c = self.thickness_air
        d = self.thickness_copper
        # exact_A2 = K * ((torch.sinh(tensor(beta * 0.1, requires_grad=True)) * torch.cosh(beta * (yn_copper - 0.18)))/(torch.sinh(tensor(beta*0.18, requires_grad=True)))) *(torch.cos(beta * xn_copper))
        exact_A2 = K * (torch.sinh(tensor(beta * b)) * torch.cosh(beta * (yn_copper - (b+c+d))) * torch.cos(beta * xn_copper))
        exact_A2 = exact_A2 / torch.sinh(tensor(beta * (b+c+d)))
        #exact_A2 = tensor(exact_A2, requires_grad=True)
        #旧的torque计算
        # eddy_current = 57000000 * 5 * np.pi * 0.45 * Ady_copper  # [46,46]
        # torque = integrate.simps(eddy_current * eddy_current, yn_copper.T, axis=0)
        # torque = (integrate.simps(torque, xn_copper[:, 0]) * 0.3 * 10) / (57000000 * 5 * np.pi)
       #新的解析解torque计算
        A2dy, A2dz = self._compute_gradient(exact_A2, xn_copper, yn_copper)
        # excat_eddy_current = -sigma * omega * rm * A2dy
        # exact_J2 = excat_eddy_current * excat_eddy_current
        # dy = yn_copper.T
        # dx = xn_copper[:, 0]
        # exact_torque = integrate.simps(exact_J2.T.cpu().detach().numpy(), dy.cpu().detach().numpy(), axis=0)
        # exact_torque = (integrate.simps(exact_torque, dx.cpu().detach().numpy()) * L * 10) / (sigma * omega)

        eddy_current = -sigma * omega * rm * Ady_copper
        # [46,46]
        J2 = eddy_current * eddy_current
        dy = yn_copper.T
        dx = xn_copper[:, 0]
        torque = integrate.simps(J2.T, dy.cpu().detach().numpy(), axis=0)
        torque = (integrate.simps(torque, dx.cpu().detach().numpy()) * L * 10) / (sigma * omega)
        pde = self.pde
        #画图
        plt.figure(figsize=(20,4))
        eddy_current = tensor(eddy_current, requires_grad=True)
        plt.subplot(1,2,1)
        plt.contour(xn_copper.cpu().detach().numpy(), yn_copper.cpu().detach().numpy(), eddy_current[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
       # plt.title("eddy current")

        Bz = -Ady_copper
        Bx = -Adz_copper
        plt.subplot(1, 2, 2)
        M = np.hypot(Bx,Bz)
        plt.quiver(-xn_copper.cpu().detach().numpy(), yn_copper.cpu().detach().numpy(),Bx,Bz,M,width=0.005,
               scale=10,cmap='jet')
        #
        # plt.contour(xn_copper.cpu().detach().numpy(), yn_copper.cpu().detach().numpy(),
        #             eddy_current[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
       # plt.title("vector B")
        plt.show()
        #exact_torque = self.cal_torque(A2dy, xn_copper ,yn_copper)
        #print("Plot里的转矩是 :", exact_torque)
        return self.get_error(xn, yn, pn),torque#用于与解析解计算误差

    def save(self, dirname):
        timestr = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        modelfname = os.path.join(dirname, 'model.pt'+ timestr)
        # torch.save(self.nn.state_dict(), modelfname)
        torch.save(self.nn, modelfname)
        rfile = os.path.join(dirname, 'results.npz' +timestr)
        #x, y, u = self.get_plot_data()
        x, y, u, pn_copper, x_copper, y_copper = self.get_plot_data()

        np.savez(rfile, x=x, y=y, u=u)


if __name__ == '__main__':
    app = App2D(
        pde_cls=CavityPDE, nn_cls=SPINN2D,
        plotter_cls=CavityPlotter,
    )

    app.run(nodes=50, samples=100, sample_frac=1, lr=5e-5, n_train=50000)#100个内部样本点，40个边界样本，2500总抽样
