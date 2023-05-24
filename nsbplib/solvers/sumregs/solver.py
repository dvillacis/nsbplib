import numpy as np
from pyproximal.optimization.primaldual import PrimalDual

from ...operators.Patch import Patch
from ...operators.SDL2 import SDL2
from ...operators.SDL21 import SDL21

class SumRegsSolver_2D(object):
    def __init__(self,data,K):
        self.data = data
        self.K = K
        self.L = 8.0
        
    def solve(self,data_par:Patch,reg_par:Patch):
        reg_par.data = np.where(reg_par.data < 0, 1e-9, reg_par.data)
        data_par = data_par.map_to_img(self.data)
        reg_par = reg_par.map_to_img(self.data[:-1,:-1])
        l2 = SDL2(b=self.data.ravel(),sigma=data_par.ravel())
        l21 = SDL21(ndim=2,sigma=reg_par.ravel())
        tau = 1.0 / np.sqrt(self.L)
        mu = 1.0 / (tau*self.L)
        rec = PrimalDual(l2,l21,self.K,tau=tau,mu=mu,theta=1.0,x0=np.zeros_like(self.data.ravel()),niter=1000)
        return rec.reshape(self.data.shape)