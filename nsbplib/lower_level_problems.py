
import logging
import numpy as np

from pylops import Diagonal, Identity, Block
from pylops.signalprocessing.convolve2d import Convolve2D

from nsbplib.operators.FirstDerivative import FirstDerivative
from nsbplib.operators.Gradient import Gradient
from nsbplib.operators.TOp import TOp
from nsbplib.operators.Tgamma import Tgamma
from nsbplib.operators.ActiveOp import ActiveOp
from nsbplib.operators.InactiveOp import InactiveOp
from nsbplib.operators.Patch import Patch
from nsbplib.solvers.rof.solver import ROFSolver_2D, TVDeblurring
import scipy.sparse.linalg as spla

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

def get_lower_level_problem(problem_type,data,label,px,py,Op=None):
    if problem_type == '2D_scalar_data_learning':
        return LowerScalarDataLearning_2D(data,label)
    if problem_type == '2D_scalar_reg_learning':
        return LowerScalarRegLearning_2D(data,label)
    elif problem_type == '2D_patch_data_learning':
        return LowerPatchDataLearning_2D(data,label,px,py)
    elif problem_type == '2D_patch_reg_learning':
        return LowerPatchRegLearning_2D(data,label,px,py)
    elif problem_type == '2D_scalar_data_learning_deblurring':
        return LowerScalarDataLearningDeblurring_2D(data,label,Op=Op)
    elif problem_type == '2D_patch_data_learning_deblurring':
        return LowerPatchDataLearningDeblurring_2D(data,label,Op=Op,px=px,py=py)
    else:
        raise RuntimeError('Problem Type not recognized %s' % problem_type)

class LowerLevelProblem(object):
    def __init__(self,data,label):
        self.data = data
        self.label = label
        self.recon = np.zeros_like(data)
    
    def __call__(self,param):
        pass
    
    def loss(self, true_data):
        pass
    
    def grad(self, true_data):
        pass
    
    def smooth_grad(self, true_data):
        pass
    
class LowerScalarDataLearning_2D(LowerLevelProblem):
    def __init__(self, data, label):
        self.Kx = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=0)
        self.Ky = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=1)
        self.K = Gradient(dims=(data.shape))
        self.px = 1
        self.py = 1
        self.solver = ROFSolver_2D(data,self.K)
        super().__init__(data, label)
    
    def __call__(self, param):
        """
        Get a new reconstruction from the noisy data
        """
        logging.debug(f'Solving lower level problem for {self.label}')
        data_par = Patch(param,self.px,self.py)
        reg_par = Patch(np.ones(self.px*self.py),self.px,self.py)
        self.recon = self.solver.solve(data_par=data_par,reg_par=reg_par)
    
    def loss(self, true_data, param):
        return 0.5 * np.linalg.norm(self.recon-true_data)**2
    
    def grad(self, true_data, param):
        data_par = Patch(param,self.px,self.py)
        parameter = data_par.map_to_img(true_data)
        m,n = self.K.shape
        L = Diagonal(parameter)
        T = TOp(self.Kx,self.Ky,self.recon)
        Act = ActiveOp(self.K,self.recon)
        Inact = InactiveOp(self.K,self.recon)
        A = Block([[L,self.K.adjoint()],[Act*self.K-Inact*T,Inact+1e-12*Act]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        p = spla.spsolve(A.tosparse(),b)[:n]
        #p,exit_code = spla.qmr(A,b)[:n]
        #print(exit_code)
        L2 = Diagonal(p)
        g = L2*(self.recon.ravel() - self.data.ravel())
        g = data_par.reduce_from_img(g.reshape(true_data.shape)[:-1,:-1])
        return -g
    
    def smooth_grad(self, true_data, param):
        data_par = Patch(param,self.px,self.py)
        parameter = data_par.map_to_img(true_data)
        m,n = self.K.shape
        L = Diagonal(parameter)
        T = Tgamma(self.Kx,self.Ky,self.recon)
        Id = Identity(m)
        # print(L.shape,K.adjoint().shape,T.shape,Id.shape)
        A = Block([[L,self.K.adjoint()],[-T,Id]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        p = spla.spsolve(A.tosparse(),b)[:n]
        L2 = Diagonal(p)
        grad = L2*(self.recon.ravel()-self.data.ravel())
        grad = data_par.reduce_from_img(grad.reshape(true_data.shape)[:-1,:-1])
        return -grad
    
    
class LowerPatchDataLearning_2D(LowerLevelProblem):
    def __init__(self, data, label,px,py):
        self.Kx = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=0)
        self.Ky = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=1)
        self.K = Gradient(dims=(data.shape))
        self.px = px
        self.py = py
        self.solver = ROFSolver_2D(data,self.K)
        super().__init__(data, label)
    
    def __call__(self, param):
        """
        Get a new reconstruction from the noisy data
        """
        logging.debug(f'Solving lower level problem for {self.label}')
        data_par = Patch(param,self.px,self.py)
        reg_par = Patch(np.ones(self.px*self.py),self.px,self.py)
        self.recon = self.solver.solve(data_par=data_par,reg_par=reg_par)
    
    def loss(self, true_data, param):
        return 0.5 * np.linalg.norm(self.recon-true_data)**2
    
    def grad(self, true_data, param):
        data_par = Patch(param,self.px,self.py)
        parameter = data_par.map_to_img(true_data)
        m,n = self.K.shape
        L = Diagonal(parameter)
        T = TOp(self.Kx,self.Ky,self.recon)
        Act = ActiveOp(self.K,self.recon)
        Inact = InactiveOp(self.K,self.recon)
        A = Block([[L,self.K.adjoint()],[Act*self.K-Inact*T,Inact+1e-12*Act]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        # p = spla.spsolve(A.tosparse(),b)[:n]
        p,exitcode = spla.qmr(A.tosparse(),b)
        p = p[:n]
        L2 = Diagonal(p)
        g = L2*(self.recon.ravel() - self.data.ravel())
        g = data_par.reduce_from_img(g.reshape(true_data.shape))
        return -g
    
    def smooth_grad(self, true_data, param):
        data_par = Patch(param,self.px,self.py)
        parameter = data_par.map_to_img(true_data)
        m,n = self.K.shape
        L = Diagonal(parameter)
        T = Tgamma(self.Kx,self.Ky,self.recon)
        Id = Identity(m)
        # print(L.shape,K.adjoint().shape,T.shape,Id.shape)
        A = Block([[L,self.K.adjoint()],[-T,Id]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        p = spla.spsolve(A.tosparse(),b)[:n]
        L2 = Diagonal(p)
        grad = L2*(self.recon.ravel()-self.data.ravel())
        grad = data_par.reduce_from_img(grad.reshape(true_data.shape))
        return -grad
    
    
class LowerPatchRegLearning_2D(LowerLevelProblem):
    def __init__(self, data, label,px,py):
        self.Kx = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=0)
        self.Ky = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=1)
        self.K = Gradient(dims=(data.shape))
        self.px = px
        self.py = py
        self.solver = ROFSolver_2D(data,self.K)
        super().__init__(data, label)
    
    def __call__(self, param):
        """
        Get a new reconstruction from the noisy data
        """
        logging.debug(f'Solving lower level problem for {self.label}')
        reg_par = Patch(param,self.px,self.py)
        data_par = Patch(np.ones(self.px*self.py),self.px,self.py)
        self.recon = self.solver.solve(data_par=data_par,reg_par=reg_par)
        #print(f'rec:{self.recon}\npar:{param}')
    
    def loss(self, true_data, param):
        return 0.5 * np.linalg.norm(self.recon-true_data)**2
    
    def grad(self, true_data, param):
        reg_par = Patch(param,self.px,self.py)
        parameter = reg_par.map_to_img(true_data[:-1,:-1])
        m,n = self.K.shape
        Id = Identity(n)
        L = Diagonal(np.concatenate((parameter,parameter)))
        T = TOp(self.Kx,self.Ky,self.recon)
        Act = ActiveOp(self.K,self.recon)
        Inact = InactiveOp(self.K,self.recon)
        # print(f'In:{(-T).todense()}')
        A = Block([[Id,self.K.adjoint()],[-Inact*L*T,Inact+1e-12*Act]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        # p = spla.spsolve(A.tosparse(),b)[:n]
        # print(A.todense())
        # print(np.linalg.cond(A.todense()))
        p,exitcode = spla.qmr(A.tosparse(),b)
        print(exitcode)
        if exitcode >0:
            print("Warning: bad conditioned matrix using param: %s " % param)
        p = p[:n]
        Kxu = self.Kx*self.recon.ravel()
        Kyu = self.Ky*self.recon.ravel()
        Kxp = self.Kx*p
        Kyp = self.Ky*p
        nKu = np.linalg.norm(np.vstack((Kxu,Kyu)).T,axis=1)
        mul = np.where(nKu<1e-12,0,-1/nKu)
        g = mul * (Kxu * Kxp + Kyu * Kyp)
        g = g.reshape((true_data.shape[0]-1,true_data.shape[1]-1))
        g = np.pad(g,[(0,1),(0,1)],mode='edge')
        g = reg_par.reduce_from_img(g.reshape(true_data.shape))
        # print(p,g)
        return g
    
    def smooth_grad(self, true_data, param):
        reg_par = Patch(param,self.px,self.py)
        parameter = reg_par.map_to_img(true_data[:-1,:-1])
        m,n = self.K.shape
        L = Diagonal(np.concatenate((parameter,parameter)))
        T = Tgamma(self.Kx,self.Ky,self.recon)
        Id = Identity(m)
        Id2 = Identity(n)
        # print(L.shape,self.K.adjoint().shape,T.shape,Id.shape)
        A = Block([[Id2,self.K.adjoint()],[-L*T,Id]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        p,exitcode = spla.qmr(A.tosparse(),b)
        if exitcode > 0:
            print("Warning: bad conditioned matrix using param: %s " % param)
        p = p[:n]
        Kxp = self.Kx*p
        Kyp = self.Ky*p
        hx,hy = tv_smooth_subdiff(self.recon,gamma=1000)
        grad = -(hx*Kxp + hy*Kyp)
        grad = grad.reshape((true_data.shape[0]-1,true_data.shape[1]-1))
        grad = np.pad(grad,[(0,1),(0,1)],mode='edge')
        grad = reg_par.reduce_from_img(grad)
        return grad
    
class LowerScalarRegLearning_2D(LowerLevelProblem):
    def __init__(self, data, label):
        self.Kx = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=0)
        self.Ky = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=1)
        self.K = Gradient(dims=(data.shape))
        self.px = 1
        self.py = 1
        self.solver = ROFSolver_2D(data,self.K)
        super().__init__(data, label)
    
    def __call__(self, param):
        """
        Get a new reconstruction from the noisy data
        """
        logging.debug(f'Solving lower level problem for {self.label}')
        reg_par = Patch(param,self.px,self.py)
        data_par = Patch(np.ones(self.px*self.py),self.px,self.py)
        self.recon = self.solver.solve(data_par=data_par,reg_par=reg_par)
        # print(f'rec:{self.recon}\npar:{param}')
    
    def loss(self, true_data, param):
        return 0.5 * np.linalg.norm(self.recon-true_data)**2
    
    def grad(self, true_data, param):
        reg_par = Patch(param,self.px,self.py)
        parameter = reg_par.map_to_img(true_data[:-1,:-1])
        m,n = self.K.shape
        Id = Identity(n)
        L = Diagonal(np.concatenate((parameter,parameter)))
        T = TOp(self.Kx,self.Ky,self.recon)
        Act = ActiveOp(self.K,self.recon)
        Inact = InactiveOp(self.K,self.recon)
        # print(f'In:{(-T).todense()}')
        A = Block([[Id,self.K.adjoint()],[-Inact*L*T,Inact+1e-12*Act]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        # p = spla.spsolve(A.tosparse(),b)[:n]
        # print(A.todense())
        # print(np.linalg.cond(A.todense()))
        p,exitcode,itn,normr,normar,norma,conda,normx = spla.lsmr(A.tosparse(),b)
        print(exitcode)
        if exitcode == 7:
            print("Warning: bad conditioned matrix using param: %s " % param)
        p = p[:n]
        Kxu = self.Kx*self.recon.ravel()
        Kyu = self.Ky*self.recon.ravel()
        Kxp = self.Kx*p
        Kyp = self.Ky*p
        nKu = np.linalg.norm(np.vstack((Kxu,Kyu)).T,axis=1)
        mul = np.where(nKu<1e-12,0,-1/nKu)
        g = mul * (Kxu * Kxp + Kyu * Kyp)
        g = g.reshape((true_data.shape[0]-1,true_data.shape[1]-1))
        g = np.pad(g,[(0,1),(0,1)],mode='edge')
        g = reg_par.reduce_from_img(g.reshape(true_data.shape))
        # print(f'g:{g}')
        return g
    
    def smooth_grad(self, true_data, param):
        reg_par = Patch(param,self.px,self.py)
        parameter = reg_par.map_to_img(true_data[:-1,:-1])
        m,n = self.K.shape
        L = Diagonal(np.concatenate((parameter,parameter)))
        T = Tgamma(self.Kx,self.Ky,self.recon)
        Id = Identity(m)
        Id2 = Identity(n)
        # print(L.shape,self.K.adjoint().shape,T.shape,Id.shape)
        A = Block([[Id2,self.K.adjoint()],[-L*T,Id]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        p,exitcode = spla.qmr(A.tosparse(),b)
        if exitcode > 0:
            print("Warning: bad conditioned matrix using param: %s " % param)
        p = p[:n]
        Kxp = self.Kx*p
        Kyp = self.Ky*p
        hx,hy = tv_smooth_subdiff(self.recon,gamma=1000)
        grad = -(hx*Kxp + hy*Kyp)
        grad = grad.reshape((true_data.shape[0]-1,true_data.shape[1]-1))
        grad = np.pad(grad,[(0,1),(0,1)],mode='edge')
        grad = reg_par.reduce_from_img(grad)
        # print(f'g_smooth:{grad}')
        return grad
    
class LowerScalarDataLearningDeblurring_2D(LowerLevelProblem):
    r'''
    Documentation for the LowerScalarDataLearningDeblurring_2D class
    This class is used to solve the scalar data learning problem with inpainting
    The class is initialized with the data, the discrete gradient operator K and the mask implemented as a Restriction operator from pylops.
    The solve method takes the data and regularization parameters and returns the reconstructed image using the Primal-Dual Hybrid Gradient approach.
    
    ## Usage
    ```python
    from nsbplib.solvers.rof.solver import TVInpainting
    from nsbplib.operators.FirstDerivative import FirstDerivative
    from nsbplib.operators.Patch import Patch
    
    ```
    
    '''
    def __init__(self, data, label, Op:Convolve2D):
        self.Kx = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=0)
        self.Ky = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=1)
        self.K = Gradient(dims=(data.shape))
        self.px = 1
        self.py = 1
        self.Op = Op
        # self.OpTOp = self.Op.T*self.Op
        print(f'data:{data.shape},K:{self.K.shape},Op:{self.Op.shape}')
        self.solver = TVDeblurring(data,self.K,self.Op)
        self.tik = 0.0
        super().__init__(data, label)
    
    def __call__(self, param):
        """
        Get a new reconstruction from the noisy data
        """
        logging.debug(f'Solving lower level problem for {self.label}')
        data_par = Patch(param,self.px,self.py)
        reg_par = Patch(np.ones(self.px*self.py),self.px,self.py)
        self.recon = self.solver.solve(data_par=data_par,reg_par=reg_par)
    
    def loss(self, true_data, param):
        print(f'param:{param}')
        return 0.5* np.linalg.norm(self.recon.ravel()-true_data.ravel())**2 + 0.5 * self.tik * param**2
    
    def grad(self, true_data, param):
        data_par = Patch(param,self.px,self.py)
        parameter = data_par.map_to_img(true_data)
        m,n = self.K.shape
        L = self.Op.T * Diagonal(parameter) * self.Op
        T = TOp(self.Kx,self.Ky,self.recon)
        Act = ActiveOp(self.K,self.recon)
        Inact = InactiveOp(self.K,self.recon)
        A = Block([[L,self.K.adjoint()],[Act*self.K-Inact*T,Inact+1e-12*Act]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        # p = spla.spsolve(A.tosparse(),b)[:n]
        p,exit_code = spla.gmres(A,b,maxiter=20)
        if exit_code != 0:
            print(f'Warning: GMRES for calculating hypergradient did not converge: {exit_code}')
        p = p[:n]
        L2 = Diagonal(self.Op * p)
        g = L2*(self.Op * self.recon.ravel() - self.data.ravel())
        g = data_par.reduce_from_img(g.reshape(true_data.shape)[:-1,:-1])
        return -g - self.tik * param
        # return -g
            
    def smooth_grad(self, true_data, param):
        data_par = Patch(param,self.px,self.py)
        parameter = data_par.map_to_img(true_data)
        m,n = self.K.shape
        L = self.Op.T * Diagonal(parameter) * self.Op
        T = Tgamma(self.Kx,self.Ky,self.recon)
        Id = Identity(m)
        A = Block([[L,self.K.adjoint()],[-T,Id]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        # p = spla.spsolve(A.tosparse(),b)[:n]
        p,exit_code = spla.gmres(A,b,maxiter=20)
        if exit_code != 0:
            print(f'Warning: GMRES for calculating hypergradient did not converge: {exit_code}')
        p = p[:n]
        L2 = Diagonal(self.Op * p)
        grad = L2*(self.Op * self.recon.ravel()-self.data.ravel())
        grad = data_par.reduce_from_img(grad.reshape(true_data.shape)[:-1,:-1])
        return -grad - self.tik * param
        # return -grad / np.linalg.norm(grad)
    
class LowerPatchDataLearningDeblurring_2D(LowerLevelProblem):
    def __init__(self, data, label, Op:Convolve2D, px, py):
        self.Kx = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=0)
        self.Ky = FirstDerivative(np.prod(data.shape), dims=data.shape,kind='forward',dir=1)
        self.K = Gradient(dims=(data.shape))
        self.px = px
        self.py = py
        self.Op = Op
        self.solver = TVDeblurring(data,self.K,self.Op)
        self.tik = 0.0
        super().__init__(data, label)
        
    
    def __call__(self, param):
        """
        Get a new reconstruction from the noisy data
        """
        logging.debug(f'Solving lower level problem for {self.label}')
        data_par = Patch(param,self.px,self.py)
        reg_par = Patch(np.ones(self.px*self.py),self.px,self.py)
        self.recon = self.solver.solve(data_par=data_par,reg_par=reg_par)
    
    def loss(self, true_data, param):
        return 0.5 * np.linalg.norm(self.recon.ravel()-true_data.ravel())**2 + 0.5 * self.tik * np.linalg.norm(param)**2
    
    def grad(self, true_data, param):
        data_par = Patch(param,self.px,self.py)
        parameter = data_par.map_to_img(true_data)
        m,n = self.K.shape
        L = self.Op.T * Diagonal(parameter) * self.Op
        T = TOp(self.Kx,self.Ky,self.recon)
        Act = ActiveOp(self.K,self.recon)
        Inact = InactiveOp(self.K,self.recon)
        A = Block([[L,self.K.adjoint()],[Act*self.K-Inact*T,Inact+1e-12*Act]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        # p = spla.spsolve(A.tosparse(),b)[:n]
        p,exit_code = spla.gmres(A,b,maxiter=200)
        if exit_code != 0:
            print(f'Warning: GMRES for calculating hypergradient did not converge: {exit_code}')
        p = p[:n]
        L2 = Diagonal(self.Op * p).T
        g = L2*(self.Op * self.recon.ravel() - self.data.ravel())
        g = data_par.reduce_from_img(g.reshape(true_data.shape))
        return -g - self.tik * param
        # return -100.0*g/np.linalg.norm(g)
    
    def smooth_grad(self, true_data, param):
        data_par = Patch(param,self.px,self.py)
        parameter = data_par.map_to_img(true_data)
        m,n = self.K.shape
        L = self.Op.T * Diagonal(parameter) * self.Op
        T = Tgamma(self.Kx,self.Ky,self.recon)
        Id = Identity(m)
        A = Block([[L,self.K.adjoint()],[-T,Id]])
        b = np.concatenate((self.recon.ravel()-true_data.ravel(),np.zeros(m)))
        # p = spla.spsolve(A.tosparse(),b)[:n]
        p,exit_code = spla.gmres(A,b,maxiter=200)
        if exit_code != 0:
            print(f'Warning: GMRES for calculating hypergradient did not converge: {exit_code}')
        p = p[:n]
        L2 = Diagonal(self.Op * p).T
        grad = L2*(self.Op * self.recon.ravel()-self.data.ravel())
        grad = data_par.reduce_from_img(grad.reshape(true_data.shape))
        return -grad - self.tik * param
        # return -100.0*grad/np.linalg.norm(grad)