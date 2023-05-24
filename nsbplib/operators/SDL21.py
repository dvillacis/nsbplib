import numpy as np
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator

class SDL21(ProxOperator):
    def __init__(self,ndim,sigma:np.array):
        super().__init__(None,False)
        self.ndim = ndim
        self.sigma = sigma

    def __call__(self, x):
        x = x.reshape(self.ndim, len(x) // self.ndim)
        f = np.dot(self.sigma,np.sqrt(np.sum(x ** 2, axis=0)))
        return f

    @_check_tau
    def prox(self, x, tau):
        x = x.reshape(self.ndim, len(x) // self.ndim)
        aux = np.sqrt(np.sum(x ** 2, axis = 0))
        aux = np.vstack([aux] * self.ndim).ravel()
        s = np.vstack([self.sigma] * self.ndim).ravel()
        x = (1 - (tau * s) / np.maximum(aux, tau * s)) * x.ravel()
        return x
    
    @_check_tau
    def proxdual(self, x, tau):
        x = x.reshape(self.ndim, len(x) // self.ndim)
        aux = np.sqrt(np.sum(x ** 2, axis = 0))
        aux = np.vstack([aux] * self.ndim).ravel()
        s = np.vstack([self.sigma] * self.ndim).ravel()
        x = s * x.ravel() / np.maximum(aux, s)
        return x