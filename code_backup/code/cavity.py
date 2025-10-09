import os
from mayavi import mlab
import numpy as np
import torch
import torch.autograd as ag
import time

from common import tensor
from spinn2d import Plotter2D, App2D, SPINN2D
from pde2d_base import RegularPDE


class CavityPDE(RegularPDE):
    def __init__(self, n_nodes, ns, nb=None, nbs=None, sample_frac=1.0):#生成100+40个固定点和2500个采样点
        self.sample_frac = sample_frac

        # Interior nodes 100个内部固定点的坐标,内部的基函数
        n = round(np.sqrt(n_nodes) + 0.49)
        self.n = n
        dxb2 = 1.0/(n + 1)
        xl, xr = dxb2, 1.0 - dxb2
        sl = slice(xl, xr, n*1j)
        x, y = np.mgrid[sl, sl]
        self.i_nodes = (x.ravel(), y.ravel())#100个固定点的坐标

        # Fixed nodes 40个都在边界上，边界上的基函数
        nb = n if nb is None else nb
        self.nb = nb#10
        if nb == 0:
            self.f_nodes = ([], [])
        else:
            dxb2 = 1.0/(nb)#0.1
            _x = np.linspace(dxb2, 1.0 - dxb2, nb)
            _o = np.ones_like(_x)
            x = np.hstack((_x, _o, _x, 0.0*_o))#堆叠参数的个数决定多少个
            y = np.hstack((_o*0.0, _x, _o, _x))
            self.f_nodes = (x, y)#一个40个都在边界上

        # Interior samples 2500个内部的0.02-0.98的点
        self.ns = ns = round(np.sqrt(ns) + 0.49)#50
        dxb2 = 1.0/(ns)#0.02
        xl, xr = dxb2, 1.0 - dxb2
        sl = slice(xl, xr, ns*1j)
        x, y = np.mgrid[sl, sl]#(50,50) (50,50)
        xs, ys = (tensor(t.ravel(), requires_grad=True)
                  for t in (x, y))
        self.p_samples = (xs, ys)#2500个内部的0.02-0.98的点

        # Boundary samples 200个，正方形每条边50个
        nbs = ns if nbs is None else nbs
        self.nbs = nbs#50
        sl = slice(0, 1.0, nbs*1j)
        _x = np.linspace(dxb2, 1.0 - dxb2, nbs)#50个
        o = np.ones_like(_x)#50个

        def tg(x):
            return tensor(x, requires_grad=True)

        self.left = left = (tg(0.0*o), tg(_x))#50个
        self.right = right = (tg(o), tg(_x))#50
        self.bottom = bottom = (tg(_x), tg(o)*0.0)#50
        self.top = top = (tg(_x), tg(o))#50个
        self.b_samples = (
            torch.cat([x[0] for x in (top, bottom, left, right)]),
            torch.cat([x[1] for x in (top, bottom, left, right)])
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
        return 3

    def has_exact(self):#是否有解析解
        return False

    def interior_loss(self, nn):
        xs, ys = self.interior()#250 250
        U = nn(xs, ys)#(250,3)
        u = U[:, 0]
        v = U[:, 1]
        p = U[:, 2]
        u, ux, uy, uxx, uyy = self._compute_derivatives(u, xs, ys)
        v, vx, vy, vxx, vyy = self._compute_derivatives(v, xs, ys)
        px, py = self._compute_gradient(p, xs, ys)

        # The NS equations.
        nu = 0.01# kinematic viscosity
        ce = ((ux + vy)**2).sum()#[1]
        mex = ((u*ux + v*uy + px - nu*(uxx + uyy))**2).sum()#[1]
        mey = ((u*vx + v*vy + py - nu*(vxx + vyy))**2).sum()#[1]

        return ce + (mex + mey)#[1]

    def boundary_loss(self, nn):
        bc_weight = self.ns*20#1000 对采样点开平方得到ns
        xb, yb = self.boundary()#200 200
        Ub = nn(xb, yb)#(200，3)
        ub = Ub[:, 0]#200
        vb = Ub[:, 1]
        pb = Ub[:, 2]
        pbx, pby = self._compute_gradient(pb, xb, yb)#200 200
        #top, bottom, left, right
        n_top = len(self.top[0])#50
        ub_top = ub[:n_top]#50
        bc_loss = ((ub_top - 1.0)**2).sum()#顶边的loss
        bc_loss += (ub[n_top:]**2).sum()*bc_weight#*bottom*bc_weight是放大1000倍
        bc_loss += ((vb)**2).sum()*bc_weight#原文中有一句v在所有边界上都是0
        bc_loss += (pby[:2*n_top]**2).sum()#原文中第五个公式，法向量为0
        bc_loss += (pbx[2*n_top:]**2).sum()#原文中第五个公式，法向量为0
        return bc_loss

    def plot_points(self):
        n = min(self.ns*2, 200)#ns 50 n 100
        x, y = np.mgrid[0:1:n*1j, 0:1:n*1j]
        return x, y#(100,100) (100,100)在0-1,0-1这个区间，生成了10000个坐标，间隔0.01


class CavityPlotter(Plotter2D):

    def get_plot_data(self):
        x, y = self.pde.plot_points()
        xt, yt = tensor(x.ravel()), tensor(y.ravel())#10000 10000
        pn = self.nn(xt, yt).detach().cpu().numpy()#10000,3
        pn.shape = x.shape + (3,)#100，100，3
        return x, y, pn

    def plot_weights(self):
        '''Implement this method to plot any weights.
        Note this is always called *after* plot_solution.
        '''
        nn = self.nn
        x, y = nn.centers()
        h = nn.widths().detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        if not self.plt2:
            self.plt2 = mlab.points3d(
                x, y, np.zeros_like(x), h, mode='2dcircle',
                scale_factor=1.0, color=(1, 0, 0), opacity=0.4
            )
            self.plt2.glyph.glyph_source.glyph_source.resolution = 20
            mlab.pipeline.glyph(
                self.plt2, color=(1, 0, 0), scale_factor=0.025, opacity=0.4,
                mode='2dcross', scale_mode='none'
            )
        else:
            self.plt2.mlab_source.trait_set(x=x, y=y, scalars=h)

    def plot_solution(self):
        xn, yn, pn = self.get_plot_data()#前两者合并就是所有坐标   第三项就是每个点对应的uvp三个值(100,100) (100,100) (100,100,3)
        u, v, p = pn[..., 0], pn[..., 1], pn[..., 2]
        for arr in (xn, yn, u, v, p):
            arr.shape = arr.shape + (1, )
        vmag = np.sqrt(u*u + v*v)#(100,100,1)u和v就是在x和y方向上的速度分量,所以这里是求总和，也就是每个点上的总速度
        pde = self.pde
        if self.plt1 is None:
            mlab.figure(
                size=(700, 700), fgcolor=(0,0,0), bgcolor=(1, 1, 1)#字体的颜色
            )
            if self.show_exact and pde.has_exact():
                un = pde.exact(xn, yn)
                mlab.surf(xn, yn, un, colormap='viridis',
                          representation='wireframe')
            # src = mlab.pipeline.vector_field(
            #     xn, yn, np.zeros_like(xn), u, v, np.zeros_like(u),#这里面zerolike项是之对应的z轴坐标和值都是0
            #     scalars=vmag,
            #     name='vectors'
            # )
            src = mlab.pipeline.vector_field(
                xn, yn, np.zeros_like(xn),   # 这里面zerolike项是之对应的z轴坐标和值都是0
                scalars=vmag,
                name='vectors'
            )
            self.plt1 = mlab.pipeline.vectors(
                src, scale_factor=0.2, mask_points=3, colormap='jet'
            )

            self.plt1.scene.z_plus_view()   #让z面面对人

            cgp = mlab.pipeline.contour_grid_plane(
                src, opacity=0.8,
                colormap='jet',
            )
            cgp.enable_contours = False

            mlab.colorbar(self.plt1)     #下标栏
            mlab.axes(self.plt1)

        else:
            self.plt1.mlab_source.trait_set(u=u, v=v, scalars=vmag)
        return self.get_error(xn, yn, pn)#用于与解析解计算误差

    def save(self, dirname):
        modelfname = os.path.join(dirname, 'model.pt')
        torch.save(self.nn.state_dict(), modelfname)
        rfile = os.path.join(dirname, 'results.npz')
        x, y, u = self.get_plot_data()
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
    app.run(nodes=100, samples=2500, sample_frac=0.1, lr=5e-4, n_train=10000)#100个内部样本点，40个边界样本，2500总抽样
