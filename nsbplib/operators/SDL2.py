import numpy as np
from scipy.sparse.linalg import cg
from pylops import Identity, LinearOperator, Diagonal
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator

class SDL2(ProxOperator):
    def __init__(self, b, sigma:np.array, A:LinearOperator=None):
        super().__init__(None,True)
        self.b = b
        self.sigma = sigma
        self.A = A
        if A == None:
            self.A = Identity(N=b.shape[0])
        self.ATb = self.A.T * self.b
        self.ATA = self.A.T * self.A

    def __call__(self, x):
        return 0.5 * np.dot(self.sigma,(self.A*x-self.b)**2)

    @_check_tau
    def prox(self,x,tau):
        rhs = x + tau * Diagonal(self.sigma) * self.ATb
        lhs = Identity(N=x.shape[0]) + tau * Diagonal(self.sigma) * self.ATA
        # print(f'{type(rhs)=} {type(lhs)=}')
        # print(f'{rhs.shape=} {lhs.shape=}')
        x,exit_code = cg(lhs,rhs)
        if exit_code != 0:
            raise ValueError(f'Conjugate gradient for calculating proximal did not converge: {exit_code}')
        return x

    def grad(self,x):
        g = self.sigma * self.A.T*(self.A*x - self.b)
        return g