import os
#from mayavi import mlab
import numpy as np
import torch
import torch.autograd as ag
import time
from scipy import integrate
from common_nomayavi import tensor
from spinn2d_nomayavi import Plotter2D, App2D, SPINN2D
from pde2d_base_nomayavi import RegularPDE


class CavityPDE(RegularPDE):
    def __init__(self, n_nodes, ns, nb=None, nbs=None, sample_frac=1.0):#生成100+40个固定点和2500个采样点
        self.sample_frac = sample_frac

        # Interior nodes 100个内部固定点的坐标
        # n = round(np.sqrt(n_nodes) + 0.49)
        # self.n = n
        # dxb2 = 1.0/(n + 1)
        # xl, xr = dxb2, 1.0 - dxb2
        # sl = slice(xl, xr, n*1j)
        # x, y = np.mgrid[sl, sl]
        # self.i_nodes = (x.ravel(), y.ravel())#100个固定点的坐标

        #new method 一共三个区域,run函数定义数量
        #永磁体区域 10000
        n = n_nodes

        lengthX = 56.54
        lengthY = 10.00
        dxb2 = 1.00 / (n + 1)
        xl, xr = dxb2, lengthX - dxb2
        yb, yt = dxb2, lengthY - dxb2
        slx = slice(xl, xr, n * 1j)
        sly = slice(yb, yt, n * 1j)
        x1, y1 = np.mgrid[slx, sly]

        # 气隙 900
        lengthY = 3
        yb, yt = 10 + dxb2, 10 + lengthY - dxb2
        sly = slice(yb, yt, n * 1j * 0.3)
        x2, y2 = np.mgrid[slx, sly]

        # 铜 2500
        lengthY = 5
        yb, yt = 13 + dxb2, 13 + lengthY - dxb2
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
            dxb2 = 1.0/(nb)#0.1
            lengthX = 56.54
            lengthY1 = 0
            lengthY2 = 10
            lengthY3 = 13
            lengthY4 = 18

            _x = np.linspace(dxb2, lengthX - dxb2, nb)
            _o = np.ones_like(_x)
            _l = np.linspace(dxb2, lengthY4 - dxb2, nb)
            # x = np.hstack((_x, _o, _x, 0.0*_o))#堆叠参数的个数决定多少个
            # y = np.hstack((_o*0.0, _x, _o, _x))
            x = np.hstack((_x, _x, _x, _x, _x*0.0, _o*56.54))
            y = np.hstack((_o*0.0, _o*10.0, _o*13.0, _o*18.0, _l, _l))

            self.f_nodes = (x, y)#一个40个都在边界上

        # Interior samples 2500个内部的0.02-0.98的点
        # self.ns = ns = round(np.sqrt(ns) + 0.49)#50
        # dxb2 = 1.0/(ns)#0.02
        # xl, xr = dxb2, 1.0 - dxb2
        # sl = slice(xl, xr, ns*1j)
        # x, y = np.mgrid[sl, sl]#(50,50) (50,50)
        # xs, ys = (tensor(t.ravel(), requires_grad=True)
        #           for t in (x, y))
        # self.p_samples = (xs, ys)#2500个内部的0.02-0.98的点

        # 永磁体区域 10000
        self.ns = ns = round(np.sqrt(ns) + 0.49)  # 50
        lengthX = 56.54
        lengthY = 10.00
        dxb2 = 1.00 / (ns + 1)
        xl, xr = dxb2, lengthX - dxb2
        yb, yt = dxb2, lengthY - dxb2
        slx = slice(xl, xr, ns * 1j)
        sly = slice(yb, yt, ns * 1j)
        x1, y1 = np.mgrid[slx, sly]
        self.label = len(x1) * len(y1)
        # 气隙 900
        lengthY = 3
        yb, yt = 10 + dxb2, 10 + lengthY - dxb2
        sly = slice(yb, yt, ns * 1j * 0.3)
        x2, y2 = np.mgrid[slx, sly]

        # 铜 2500
        lengthY = 5
        yb, yt = 13 + dxb2, 13 + lengthY - dxb2
        sly = slice(yb, yt, ns * 1j * 0.5)
        x3, y3 = np.mgrid[slx, sly]

        x = np.append(x1, x2)
        x = np.append(x, x3)
        y = np.append(y1, y2)
        y = np.append(y, y3)
        self.p_samples = tensor((x.ravel(), y.ravel()), requires_grad=True)

        # Boundary samples 200个，正方形每条边50个
        nbs = ns if nbs is None else nbs
        self.nbs = nbs#50
        # sl = slice(0, 1.0, nbs*1j)
        # _x = np.linspace(dxb2, 1.0 - dxb2, nbs)#50个
        # o = np.ones_like(_x)#50个

        dxb2 = 1.0 / (nbs)  # 0.1
        lengthX = 56.54
        lengthY1 = 0
        lengthY2 = 10
        lengthY3 = 13
        lengthY4 = 18

        _x = np.linspace(dxb2, lengthX - dxb2, nbs)
        _o = np.ones_like(_x)
        _l = np.linspace(dxb2, lengthY4 - dxb2, nbs)
        # x = np.hstack((_x, _o, _x, 0.0*_o))#堆叠参数的个数决定多少个
        # y = np.hstack((_o*0.0, _x, _o, _x))
        # x = np.hstack((_x, _x, _x, _x, _x * 0.0, _o * 56.54))
        # y = np.hstack((_o * 0.0, _o * 10.0, _o * 13.0, _o * 18.0, _l, _l))

        #self.f_nodes = (x, y)  # 一个40个都在边界上

        def tg(x):
            return tensor(x, requires_grad=True)

        # self.left = left = (tg(0.0*o), tg(_x))#50个
        # self.right = right = (tg(o), tg(_x))#50
        # self.bottom = bottom = (tg(_x), tg(o)*0.0)#50
        # self.top = top = (tg(_x), tg(o))#50个
        self.one = one = (tg(_x), tg(_o*18.00))
        self.two = two = (tg(_x), tg(_o * 13.00))
        self.three = three = (tg(_x), tg(_o * 10.00))
        self.four = four = (tg(_x), tg(_o * 0.00))
        self.five = five = (tg(_o*0.00), tg(_l))
        self.six = six = (tg(_o*56.54), tg(_l))
        self.b_samples = (
            torch.cat([x[0] for x in (one, two, three, four, five, six)]),
            torch.cat([x[1] for x in (one, two, three, four, five, six)])
        )#200个，正方形每条边50个

        self.n_interior = len(self.p_samples[0])#2500
        self.rng_interior = np.arange(self.n_interior)#2500 应该是索引
        self.sample_size = int(self.sample_frac*self.n_interior)#250

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
        xs, ys = self.interior()  # 250 250
        U = nn(xs, ys)  # (250,3) U就是A1 和 A2
        Ady = U[:, 0]  # A1
        Adz = U[:, 1]  # A2
        A1dy = Ady[:self.label]
        A1dz = Adz[:self.label]
        A2dy = Ady[self.label:]
        A2dz = Adz[self.label:]
        A1dy2, A1dydz = self._compute_gradient(A1dy, xs, ys)
        A1dzdy, A1dz2 = self._compute_gradient(A1dz, xs, ys)
        A2dy2, A2dydz = self._compute_gradient(A2dy, xs, ys)
        A2dzdy, A2dz2 = self._compute_gradient(A2dz, xs, ys)

        # The NS equations.
        u0 = tensor(4 * np.pi * 1e-7, requires_grad=True)
        Pi = tensor(np.pi, requires_grad=True)
        Br = tensor(1.25, requires_grad=True)
        alpha = tensor(0.9, requires_grad=True)
        rm = tensor(45, requires_grad=True)
        beta = tensor(5 / rm, requires_grad=True)
        xsA1 = xs[:self.label]
        mzy = (4 * Br / Pi * u0) * (torch.sin(alpha * Pi / 2)) * beta * (torch.cos(beta * xsA1))

        A1dy2 = A1dy2[:self.label]
        A1dz2 = A1dz2[:self.label]
        A2dy2 = A2dy2[self.label:]
        A2dz2 = A2dz2[self.label:]
        loss_one = (((A1dy2 + A1dz2 + (u0 * mzy)) ** 2)*1000).sum()
        loss_two = (((A2dy2 + A2dz2) ** 2)*1000).sum()
        return loss_one + loss_two  # [1]

    def boundary_loss(self, nn):
        bc_weight = self.ns * 20  # 1000
        xb, yb = self.boundary()  # 200 200
        Ub = nn(xb, yb)  # (200，3)
        Ady = Ub[:, 0]  # A1
        Adz = Ub[:, 1]  # A2
        u_magnet_back_iron = 1
        u_magent = 1
        u_copper = 1
        u_copper_back_iron = 1
        u0 = tensor(4 * np.pi * 1e-7, requires_grad=True)
        Pi = tensor(np.pi, requires_grad=True)
        Br = tensor(1.25, requires_grad=True)
        alpha = tensor(0.9, requires_grad=True)
        rm = tensor(45, requires_grad=True)
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
        Adz_four = Adz[n_two:n_four]
        Ady_five = Ady[n_two:n_five]
        Adz_five = Adz[n_two:n_five]
        Ady_six = Ady[n_two:n_six]
        Adz_six = Adz[n_two:n_six]

        # 1 2边遵循A2
        A1dy2, A1dydz = self._compute_gradient(Ady_one, xb, yb)
        A1dzdy, A1dz2 = self._compute_gradient(Adz_one, xb, yb)
        A2dy2, A2dydz = self._compute_gradient(Ady_two, xb, yb)
        A2dzdy, A2dz2 = self._compute_gradient(Adz_two, xb, yb)
        A1dy2 = A1dy2[:n_one]
        A1dz2 = A1dz2[:n_one]
        A2dy2 = A2dy2[n_one:n_two]
        A2dz2 = A2dz2[n_one:n_two]
        bc_loss = ((A1dy2 + A1dz2) ** 2).sum()
        bc_loss += ((A2dy2 + A2dz2) ** 2).sum()

        # 3边遵循 A1 A2
        A3dy2, A3dydz = self._compute_gradient(Ady_three, xb, yb)
        A3dzdy, A3dz2 = self._compute_gradient(Adz_three, xb, yb)
        A3dy2 = A3dy2[n_two:n_three]
        A3dz2 = A3dz2[n_two:n_three]
        bc_loss += (((A3dy2 + A3dz2) ** 2)*bc_weight).sum()
        xbA3 = xb[n_two:n_three]
        mzy = (4 * Br / Pi * u0) * (torch.sin(alpha * Pi / 2)) * beta * (torch.cos(beta * xbA3))
        bc_loss += (((A3dy2 + A3dz2 + u0 * mzy) ** 2)*bc_weight).sum()

        # 4边遵循 A1
        A4dy2, A4dydz = self._compute_gradient(Ady_four, xb, yb)
        A4dzdy, A4dz2 = self._compute_gradient(Adz_four, xb, yb)
        A4dz2 = A4dz2[n_three:n_four]
        A4dy2 = A4dy2[n_three:n_four]
        xbA4 = xb[n_three:n_four]
        mzy = (4 * Br / Pi * u0) * (torch.sin(alpha * Pi / 2)) * beta * (torch.cos(beta * xbA4))
        bc_loss += ((A4dy2 + A4dz2 + u0 * mzy) ** 2).sum()

        # 5 6边赋常数
        # Ady_five = Ady_five * 0
        # Adz_five = Adz_five * 0
        # Ady_six = Ady_six * 0
        # Adz_six = Adz_six * 0

        return bc_loss

    def plot_points(self):
        n = min(self.ns * 2, 200)  # ns 50 n 100
        x, y = np.mgrid[0:56.54:n * 1j, 0:18:n * 1j]
        x_copper, y_copper = np.mgrid[0:56.54:n * 1j, 13:18:n * 1j]
        # self.label2 =
        return x, y, x_copper, y_copper#(100,100) (100,100)在0-1,0-1这个区间，生成了10000个坐标，间隔0.01


class CavityPlotter(Plotter2D):

    def get_plot_data(self):
        x, y, x_copper, y_copper = self.pde.plot_points()
        xt, yt = tensor(x.ravel()), tensor(y.ravel())  # 10000 10000
        x_copper_t, y_copper_t = tensor(x_copper.ravel()), tensor(y_copper.ravel())
        pn = self.nn(xt, yt).detach().cpu().numpy()  # 10000,3
        pn_copper = self.nn(x_copper_t, y_copper_t).detach().cpu().numpy()
        pn.shape = x_copper.shape + (2,)  # 100，100，3
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
        # print("CATIVTY!!!!!!")
        # for arr in (xn, yn, u, v):
        #     arr.shape = arr.shape + (1, )
        # vmag = np.sqrt(u*u + v*v)#(100,100,1)u和v就是在x和y方向上的速度分量,所以这里是求总和，也就是每个点上的总速度
        #vmag = np.sqrt(Ady * Ady + Adz * Adz)
        vmag = np.sqrt(Ady * Ady + Adz * Adz)
        eddy_current = 57000000 * 5*np.pi * 0.45 * Ady_copper  # [46,46]
        torque = integrate.simps(eddy_current * eddy_current, yn_copper.T,axis=0)
        torque = (integrate.simps(torque, xn_copper[:, 0]) * 0.3 * 10) / (57000000 * 5*np.pi)
        pde = self.pde
        # if self.plt1 is None:
        #     mlab.figure(
        #         size=(700, 700), fgcolor=(0,0,0), bgcolor=(1, 1, 1)#字体的颜色
        #     )
        #     if self.show_exact and pde.has_exact():
        #         un = pde.exact(xn, yn)
        #         mlab.surf(xn, yn, un, colormap='viridis',
        #                   representation='wireframe')
        #     src = mlab.pipeline.vector_field(
        #         xn, yn, np.zeros_like(xn), u, v, np.zeros_like(u),#这里面zerolike项是之对应的z轴坐标和值都是0
        #         scalars=vmag,
        #         name='vectors'
        #     )
        #
        #     self.plt1 = mlab.pipeline.vectors(
        #         src, scale_factor=0.2, mask_points=3, colormap='viridis'
        #     )
        #     self.plt1.scene.z_plus_view()   #让z面面对人
        #     cgp = mlab.pipeline.contour_grid_plane(
        #         src, opacity=0.8,
        #         colormap='viridis',
        #     )
        #     cgp.enable_contours = False
        #     mlab.colorbar(self.plt1) #下标栏
        #     mlab.axes(self.plt1)
        # else:
        #     self.plt1.mlab_source.trait_set(u=u, v=v, scalars=vmag)
        print("Torque is :", torque)
        return self.get_error(xn, yn, pn),torque#用于与解析解计算误差

    def save(self, dirname):
        timestr = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        modelfname = os.path.join(dirname, 'model.pt'+ timestr)
        torch.save(self.nn.state_dict(), modelfname)
        rfile = os.path.join(dirname, 'results.npz' +timestr)
        #x, y, u = self.get_plot_data()
        x, y, u, pn_copper, x_copper, y_copper = self.get_plot_data()
        lc = tensor(np.linspace(0, 1, 100))
        midc = tensor(0.5*np.ones(100))
        xc = torch.cat((lc, midc))
        yc = torch.cat((midc, lc))
        data = self.nn(xc, yc).detach().cpu().numpy()
        vc = data[:, 1][:100]
        uc = data[:, 0][100:]

        np.savez(rfile, x=x, y=y, u=u, xc=lc, uc=uc, vc=vc)


if __name__ == '__main__':
    app = App2D(
        pde_cls=CavityPDE, nn_cls=SPINN2D,
        plotter_cls=CavityPlotter,
    )
    app.run(nodes=100, samples=200, sample_frac=1, lr=5e-4, n_train=10000)#100个内部样本点，40个边界样本，2500总抽样
