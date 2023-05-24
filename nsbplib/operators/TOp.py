import numpy as np
from pylops import LinearOperator
from .FirstDerivative import FirstDerivative

def phi(nu,tol):
    return np.where(nu<tol,1.,1./nu)

def psi(nu,tol):
    return np.where(nu<tol,0,-1./nu**3)

class TOp(LinearOperator):
    def __init__(self, Kx, Ky, u, tol=1e-2, dtype='float64'):
        self.Kx = Kx
        self.Ky = Ky
        self.Kxu = Kx*u.ravel()
        self.Kyu = Ky*u.ravel()
        self.Kxu2 = self.Kxu ** 2
        self.Kyu2 = self.Kyu ** 2
        self.Kxyu = self.Kxu * self.Kyu
        self.nKu = np.linalg.norm(np.vstack((self.Kxu,self.Kyu)).T,axis=1)
        self.phi_u = phi(self.nKu,tol)
        self.psi_u = psi(self.nKu,tol)
        knx,kny = Kx.shape
        self.shape = (2*knx,kny)
        self.dtype = dtype
        self.explicit = False
        self.matvec_count = 0
        self.rmatvec_count = 0

    def _matvec(self,x):
        Kxx = self.Kx * x
        Kyx = self.Ky * x
        a = self.psi_u * (self.Kxu2 * Kxx + self.Kxyu * Kyx) + self.phi_u * Kxx
        b = self.psi_u * (self.Kyu2 * Kyx + self.Kxyu * Kxx) + self.phi_u * Kyx
        return np.concatenate((a,b))
    
    def _rmatvec(self,x):
        print(self.Kx.shape,x.shape)
        m = len(x)//2
        q1 = x[:m]
        q2 = x[m:]
        Kxx = self.Kx.transpose() * q1
        Kyx = self.Ky.transpose() * q2
        # a = self.psi_u * (self.Kxu2 * Kxx + self.Kxyu * Kyx) + self.phi_u * Kxx
        # b = self.psi_u * (self.Kyu2 * Kyx + self.Kxyu * Kxx) + self.phi_u * Kyx
        a = self.phi_u * Kxx
        b = self.phi_u * Kyx
        return a