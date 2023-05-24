import numpy as np
class Patch:
    def __init__(self,data,px,py) -> None:
        self.px = px
        self.py = py
        self.data = data
    def copy(self):
        return Patch(self.data,self.px,self.py)
    def get_matrix(self):
        return self.data.reshape((self.px,self.py))
    def map_to_img(self,img):
        nx,ny = img.shape
        if nx%self.px == 0:
            mx = nx//self.px
            my = ny//self.py
            m = self.get_matrix()
            m = np.kron(m,np.ones((mx,my)))
        elif (nx+1)%self.px == 0:
            mx = (nx+1)//self.px
            my = (ny+1)//self.py
            m = self.get_matrix()
            m = np.kron(m,np.ones((mx,my)))
            m = m[:-1,:-1]
        else:
            raise AttributeError('Cannot find even distribution for patch')
        #return (m/(nx//self.px*ny//self.py)).ravel()
        return m.ravel()
    def reduce_from_img(self,img):
        nx,ny = img.shape
        mx = nx//self.px
        my = ny//self.py
        result = np.add.reduceat(np.add.reduceat(img, np.arange(0, img.shape[0], mx), axis=0),np.arange(0, img.shape[1], my), axis=1)
        return (result/(mx*my)).ravel()
        #return result.ravel()
    def __str__(self) -> str:
        return f'Patch ({self.data},{self.px},{self.py})'
    def __repr__(self) -> str:
        return f'Patch ({self.data},{self.px},{self.py})'