import numpy as np
from bpllib.operators.FirstDerivative import FirstDerivative

def _to_array(x,lbl):
    try:
        if isinstance(x,float):
            x = [x]
        return np.asarray_chkfinite(x)
    except ValueError:
        raise ValueError('%s contains Nan/Inf values' % lbl)
    
def pointwise_euclidean_norm(u):
    n = u.shape
    if len(n)>1:
        raise ValueError('Input must be a 1D vector...')
    u = np.reshape(u,(n[0]//2,2),order='F')
    nu = np.linalg.norm(u,axis=1)
    return nu

def tv_smooth_subdiff(u,gamma=1000):
    nx,ny = u.shape
    n = nx*ny
    Kx = FirstDerivative(n,dims=(nx,ny),dir=0)
    Ky = FirstDerivative(n,dims=(nx,ny),dir=1)
    Kxu = Kx*u.ravel()
    Kyu = Ky*u.ravel()
    nu = np.linalg.norm(np.vstack((Kxu,Kyu)).T,axis=1)
    a = np.where(gamma*nu-1 <= -1/(2*gamma),gamma,1./nu)
    b = np.where((gamma*nu-1 > -1/(2*gamma)) & (gamma*nu-1 <= 1/(2*gamma)),1/nu*(1-0.5*gamma*(1-gamma*nu+1/(2*gamma**2)**2)),a)
    return b*Kxu,b*Kyu

def spai(A, m):
    """Perform m step of the SPAI iteration."""
    from scipy.sparse import identity
    from scipy.sparse import diags
    from scipy.sparse.linalg import onenormest
    
    n = A.shape[0]
    
    ident = identity(n, format='csr')
    alpha = 2 / onenormest(A @ A.T)
    M = alpha * A
        
    for index in range(m):
        C = A @ M
        G = ident - C
        AG = A @ G
        trace = (G.T @ AG).diagonal().sum()
        alpha = trace / np.linalg.norm(AG.data)**2
        M = M + alpha * G
        
    return M

# def uzawa()