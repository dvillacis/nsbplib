import numpy as np
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator

class SDL2(ProxOperator):
    def __init__(self, b, sigma:np.array):
        super().__init__(None,True)
        self.b = b
        self.sigma = sigma

    def __call__(self, x):
        return 0.5 * np.dot(self.sigma,(x-self.b)**2)

    @_check_tau
    def prox(self,x,tau):
        num = x + tau * self.sigma * self.b
        x = num / (1. + tau * self.sigma)
        return x

    def grad(self,x):
        g = self.sigma * (x - self.b)