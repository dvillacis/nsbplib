import numpy as np

from pylops.basicoperators import VStack

from .FirstDerivative import FirstDerivative

def Gradient(dims,sampling=1.0,dtype=np.float64,kind='forward'):
    ndims = len(dims)
    if isinstance(sampling,(int,float)):
        sampling = [sampling] * ndims
    gop = VStack(
        [
            FirstDerivative(
                np.prod(dims),
                dims=dims,
                dir=idir,
                sampling=sampling[idir],
                kind=kind,
                dtype=dtype
            )
            for idir in range(ndims)
        ]
    )
    return gop