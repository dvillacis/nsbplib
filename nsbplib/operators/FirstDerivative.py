import numpy as np
from pylops import LinearOperator
from pylops.utils.backend import get_array_module

class FirstDerivative(LinearOperator):
    '''
    Finite Differences discretization of the FirstDerivative Operator.
    This Linear Operator excludes the information on the boundaries.
    '''
    def __init__(
        self, 
        N, 
        dims=None,
        dimsd=None, 
        dir=0, 
        sampling=1.0, 
        dtype=np.float64, 
        kind='forward'
    ):
        self.N = N
        self.sampling = sampling
        if dims is None:
            self.dims = (self.N,)
            self.reshape = False
            self.shape = (self.N-1,self.N)
        else:
            if np.prod(dims) != self.N:
                raise ValueError("product of dims must equal N")
            else:
                self.dims = dims
                self.reshape = True
                self.shape = ((self.dims[0]-1)*(self.dims[1]-1),self.N)
        self.dir = dir if dir >= 0 else len(self.dims) + dir
        self.kind = kind
        self.dtype = np.dtype(dtype)
        self.matvec_count = 0
        self.rmatvec_count = 0
        
        if self.kind == 'forward':
            self._matvec = self._matvec_forward
            self._rmatvec = self._rmatvec_forward
        elif self.kind == 'backward':
            self._matvec = self._matvec_backward
            self._rmatvec = self._rmatvec_backward
        else:
            raise NotImplementedError("kind must be forward, " "or backward")

    def _matvec_forward(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N-1, self.dtype)
            y = (x[1:]-x[:-1]) / self.sampling
        else:
            x = ncp.reshape(x, self.dims)
            if self.dir > 0:
                # x = ncp.swapaxes(x,self.dir,0)
                y = (x[1:,:-1] - x[:-1,:-1]) / self.sampling
            else:
                # y = ncp.zeros(x.shape,self.dtype)
                y = (x[:-1,1:] - x[:-1,:-1]) / self.sampling
            # if self.dir > 0:
            #     y = ncp.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y
    
    def _rmatvec_forward(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N, self.dtype)
            y[:-1] -= x / self.sampling
            y[1:] += x / self.sampling
        else:
            x = ncp.reshape(x, (self.dims[0]-1,self.dims[1]-1))
            #y = ncp.zeros(self.dims, self.dtype)
            # print(y)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                #x = ncp.swapaxes(x, self.dir, 0)
                y1 = np.pad(x,[(0,1),(0,1)],mode='constant', constant_values=0)
                y2 = np.pad(x,[(1,0),(0,1)],mode='constant', constant_values=0)
                y = y2-y1
            else:
                y2 = np.pad(x,[(0,1),(1,0)],mode='constant', constant_values=0)
                y1 = np.pad(x,[(0,1),(0,1)],mode='constant', constant_values=0)
                y = y2-y1
            if self.dir == 0:
                y = ncp.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y
    
    def _matvec_backward(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N, self.dtype)
            y[1:] = (x[1:] - x[:-1]) / self.sampling
        else:
            x = ncp.reshape(x, (self.dims[0]-1,self.dims[1]-1))
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = ncp.swapaxes(x, self.dir, 0)
            y = ncp.zeros(x.shape, self.dtype)
            y[1:] = (x[1:] - x[:-1]) / self.sampling
            if self.dir > 0:
                y = ncp.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y

    def _rmatvec_backward(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N, self.dtype)
            y[:-1] -= x[1:] / self.sampling
            y[1:] += x[1:] / self.sampling
        else:
            x = ncp.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = ncp.swapaxes(x, self.dir, 0)
            y = ncp.zeros(x.shape, self.dtype)
            y[:-1] -= x[1:] / self.sampling
            y[1:] += x[1:] / self.sampling
            if self.dir > 0:
                y = ncp.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y
