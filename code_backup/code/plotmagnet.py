import os
#from mayavi import mlab
import numpy as np
import torch
torch.cuda.empty_cache()

import torch.autograd as ag
import time
from scipy import integrate
from common_nomayavi import tensor
from spinn2d_nomayavi import Plotter2D, App2D, SPINN2D
from pde2d_base_nomayavi import RegularPDE
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'B.csv',header=None)
tdata = data.values
y = tdata[0:200:10,1]#400
z = tdata[0:40000:2000,2]
# z = z.reshape(200,200)[0:200:10,0]#20
#z=z.reshape(20,20)----------------------
by = np.array([])
bz =np.array([])
for i in range(20):
    by = np.append(by,tdata[(i*2000):200+(2000*i):10, 3])
    bz = np.append(bz,tdata[(i * 2000):200 + (2000 * i):10, 4])
# by = tdata[0:40000:10,3]
# bz = tdata[0:40000:100,4]
def dele_tensor(x):
    res = np.delete(x, [0], axis=0)
    res = np.delete(res, [0, 19], axis=1)
    return res

def dele_ver(x):
    res = np.delete(x,[0,19],axis=1)
    return res

[Y,Z] = np.meshgrid(y,z)
Y = dele_tensor(Y)
Z = dele_tensor(Z)
by2 = by.reshape(20,20)
bz2 = bz.reshape(20,20)
by2 = dele_tensor(by2)
bz2 = dele_tensor(bz2)
By = by2  # 20,20
Bz = -bz2  # 20,20
M = np.hypot(By, Bz)
plt.quiver(Y, Z, By, Bz, M, width=0.005,
           scale=10, cmap='jet')
#
# plt.contour(xn_copper.cpu().detach().numpy(), yn_copper.cpu().detach().numpy(),
#             eddy_current[:, :].cpu().detach().numpy(), 10000, cmap='jet', zorder=1)
plt.colorbar()
plt.xticks([])
plt.yticks([])
# plt.title("vector B")
plt.show()
