import numpy as np
from pylops import LinearOperator
from .FirstDerivative import FirstDerivative

def phi(nu,gamma):
    return np.where(gamma*nu-1<=-1/(2*gamma),gamma,1./nu)

def psi(nu,gamma):
    return np.where(gamma*nu-1<-1/(2*gamma),0,-1./nu**3)

class Tgamma(LinearOperator):
    def __init__(self,Kx,Ky, u, gamma=1000, dtype='float64'):
        self.Kx = Kx
        self.Ky = Ky
        self.Kxu = self.Kx*u.ravel()
        self.Kyu = self.Ky*u.ravel()
        self.Kxu2 = self.Kxu**2
        self.Kyu2 = self.Kyu**2
        self.Kxyu = self.Kxu * self.Kyu
        self.nKu = np.linalg.norm(np.vstack((self.Kxu,self.Kyu)).T,axis=1)
        #print(np.min(self.nKu))
        #print(f'grad:\n{self.nKu}')
        self.phi_u = phi(self.nKu,gamma)
        self.psi_u = psi(self.nKu,gamma)
        knx,kny = self.Kx.shape
        self.shape = (2*knx,kny)
        self.dtype = dtype
        self.matvec_count = 0
        self.rmatvec_count = 0

    def _matvec(self,x):
        Kxx = self.Kx * x
        Kyx = self.Ky * x
        #print(self.phi_u * Kxx)
        #print(self.psi_u * (self.Kxu2 * Kxx + self.Kxyu * Kyx))
        a = self.psi_u * (self.Kxu2 * Kxx + self.Kxyu * Kyx) + self.phi_u * Kxx
        b = self.psi_u * (self.Kyu2 * Kyx + self.Kxyu * Kxx) + self.phi_u * Kyx
        return np.concatenate((a,b))
    
    def _rmatvec(self,y):
        n = y.shape
        y = np.reshape(y,(n[0]//2,2),order='F')
        y1 = y[:,0]
        y2 = y[:,1]
        a = self.psi_u * (self.Kxu2 * y1 + self.Kxyu * y2) + self.phi_u * y1
        b = self.psi_u * (self.Kyu2 * y2 + self.Kxyu * y1) + self.phi_u * y1
        return a+b