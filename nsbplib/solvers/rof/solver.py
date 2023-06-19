
"""
Rudin Osher and Fatemi (ROF) Image Denoising Model Solver

Solver for the ROF Model based on the Primal-Dual Hybrid Gradient approach described in

[1] ...

Each class will have a smooth_gradient method that encapsulates the gradient of the l2-squared portion, 
and a prox_dual method that describes the proximal operator for the Fenchel dual function of the TV part.

"""

import numpy as np
from pyproximal import L2,L21
from pyproximal.optimization.primaldual import PrimalDual
from nsbplib.operators.FirstDerivative import FirstDerivative

from nsbplib.operators.Patch import Patch
from nsbplib.operators.SDL2 import SDL2
from nsbplib.operators.SDL21 import SDL21


class ROFSolver_1D(object):
    def __init__(self,data):
        self.data = data
        self.K = FirstDerivative(len(data))
        self.L = 8.0
        
    def solve(self,data_par,reg_par):
        l2 = L2(b=self.data,sigma=data_par)
        l21 = L21(1,sigma=reg_par)
        tau = 1.0 / np.sqrt(self.L)
        mu = 1.0 / (tau*self.L)
        rec = PrimalDual(l2,l21,self.K,tau=tau,mu=mu,theta=1.0,x0=np.zeros_like(self.data))
        return rec
    
class ROFSolver_2D(object):
    # Documentation for the ROFSolver_2D class
    # This class is used to solve the ROF problem 
    # The class is initialized with the data and the K operator
    # The solve method takes the data and regularization parameters and returns the reconstructed image using the Primal-Dual Hybrid Gradient approach
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
    
class TVDeblurring(object):
    
    def __init__(self,data,K,R,maxiter=1000) -> None:
        self.data = data
        self.K = K
        self.R = R # Define the Blurring Operator
        self.L = 8.0
        self.maxiter = maxiter
        
    def solve(self,data_par:Patch,reg_par:Patch):
        # print(f'Inpainting: {data_par.data.shape} {reg_par.data.shape}')
        data_par = data_par.map_to_img(self.data)
        reg_par = reg_par.map_to_img(self.data[:-1,:-1])
        l2 = SDL2(b=self.data.ravel(),sigma=data_par.ravel(),A=self.R)
        l21 = SDL21(ndim=2,sigma=reg_par.ravel())
        tau = 1.0 / np.sqrt(self.L)
        mu = 1.0 / (tau*self.L)
        rec = PrimalDual(l2,l21,self.K,tau=tau,mu=mu,theta=1.0,x0=np.zeros_like(self.data.ravel()),niter=self.maxiter,show=False)
        rec = rec.reshape(self.data.shape)
        return rec