import numpy as np
from pylops import LinearOperator
from pylops.utils.backend import get_array_module

def pointwise_euclidean_norm(u):
    n = u.shape
    if len(n)>1:
        raise ValueError('Input must be a 1D vector...')
    u = np.reshape(u,(n[0]//2,2),order='F')
    nu = np.linalg.norm(u,axis=1)
    return nu

class InactiveOp(LinearOperator):
    def __init__(self, K, u, tol=1e-5, dtype='float64'):
        Ku = K*u.ravel()
        nKu = pointwise_euclidean_norm(Ku)
        d = np.where(nKu >= tol,1,0)
        self.diag = np.concatenate((d,d),axis=0)
        self.shape = (len(self.diag),len(self.diag))
        self.dtype = np.dtype(dtype)
        self.matvec_count = 0
        self.rmatvec_count = 0

    def _matvec(self, x):
        y = self.diag * x
        return y

    def _rmatvec(self, x):
        y = self.diag * x
        return y

    def matrix(self):
        ncp = get_array_module(self.diag)
        densemat = ncp.diag(self.diag.squeeze())
        return densemat

    def todense(self):
        return self.matrix()