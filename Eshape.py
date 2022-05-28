from loadingfun import load_ini_info
from pyfr.shapes import QuadShape

import numpy as np

class Interpolation(object):
    name = None

    def __init__(self, cfg):
        self.cfg = load_ini_info(cfg)
        self.order = self.cfg.getint('solver','order')

class HexType(Interpolation):
    name = 'hex'

    def __init__(self,cfg):
        super().__init__(cfg)

    def npts(self):
        return (self.order+1)**3


    def pre_calc(self, mesh):
        Npts = mesh.shape[0]
        Neles = mesh.shape[1]
        Nvar = mesh.shape[2]
        if Npts == (self.order+1)**3:
            #mesh = self.transfinit_mapping(mshe,pt)
            mesh = mesh
        vcenter = np.sum(mesh,axis=0)/Npts

        vpt = mesh.reshape((self.order+1,self.order+1,self.order+1, Neles, Nvar),order = 'F')
        fcenter = np.array([np.sum(vpt[:,:,0].reshape(((self.order+1)**2, Neles, Nvar),order = 'F'),axis=0),
                           np.sum(vpt[:,:,-1].reshape(((self.order+1)**2, Neles, Nvar),order = 'F'),axis=0),
                           np.sum(vpt[:,0,:].reshape(((self.order+1)**2, Neles, Nvar),order = 'F'),axis=0),
                           np.sum(vpt[:,-1,:].reshape(((self.order+1)**2, Neles, Nvar),order = 'F'),axis=0),
                           np.sum(vpt[0].reshape(((self.order+1)**2, Neles, Nvar),order = 'F'),axis=0),
                           np.sum(vpt[-1].reshape(((self.order+1)**2, Neles, Nvar),order = 'F'),axis=0)])/(self.order+1)**2

        return vcenter, fcenter



    def transfinit_mapping(self,mshe,pt = False):
        #print('this is transfinit mapping')


        if isinstance(pt,bool):
            mshres = mshe.reshape((self.order+1,self.order+1,2), order = 'F')


            st_ = np.array(QuadShape.std_ele(self.order))[0:self.order+1,0]

            xmn = mshres*0
            # Gordon-Hall Algorithm
            for m in range(self.order+1):
                for n in range(self.order+1):
                    xmn[m,n,:] = (1 - st_[m])/2*mshres[0,n,:] + (1 + st_[m])/2*mshres[-1,n,:]
                    + (1 - st_[n])/2*mshres[m,0,:] + (1 + st_[n])/2*mshres[m,-1,:]
                    - (1 - st_[m])/2*(1 - st_[n])/2*mshres[0,0,:]
                    - (1 + st_[m])/2*(1 - st_[n])/2*mshres[-1,0,:]
                    - (1 - st_[m])/2*(1 + st_[n])/2*mshres[0,-1,:]
                    - (1 + st_[m])/2*(1 + st_[n])/2*mshres[-1,-1,:]
            xmnA = xmn.reshape(((self.order+1)**2,2),order='F')

            return xmnA

        else:
            # calculate the position of unknown point after transfinite mapping
            index = np.argsort((pt[0] - mshe[:,0])**2 + (pt[1] - mshe[:,1])**2)   #resort the closest points
            incre = xmnA[index[0:2],:] - mshe[index[0:2],:]
            dx = abs(np.diff(incre,axis=0))

            incre2 = abs(np.diff(mshe[index[0:2]],axis=0) )
            for i in range(len(pt)): #loop all dimesions
                if incre2[0][i] != 0:
                    pt[i] += (pt[i] - mshe[index[1],i])/incre2[0][i] * dx[0][i] + incre[0,i]
                else:
                    pt[i] += incre[0,i]
            """obviously this can be done better"""
            return pt

    def check_curved(self,msh):
        print('check if curved')



    def facenormal(self, fcenter, vcenter, pt_noncur):
        #print('construct the face normal')
        norvec = vcenter - fcenter
        ptvec  = np.array([fcenter - pt for pt in pt_noncur])
        # using einstien notition to do tensor product and extract diagonal entries
        out    = np.einsum('ijk,lmqk->lijqm', norvec, ptvec)
        out1   = np.einsum('kijji->kji',out)
        # is that a good idea to use this loop?
        index  = np.zeros(len(pt_noncur),dtype='int')
        for i in range(len(pt_noncur)):
            for j in range(len(vcenter)):
                if np.all(out1[i,j] < 10e-10) or np.all(out1[i,j] > -10e-10):
                    index[i] = j
                    break
                """if there is a bug, it is because this point is not in the bounding box"""
                index[i] = 10000
        return index


    def facenormal_old(self, fcenter, vcenter, pt_noncur):
        #print('construct the face normal')
        normvec = fcenter - vcenter

        ptvec = fcenter - pt_noncur

        if np.all(np.diag(ptvec @ normvec.T) <= 0) or np.all(np.diag(ptvec @ normvec.T) >= 0):
            return True
        else:
            return False


    """more to do here:
    1. change transfinite interpolation to make it three D
    """
    def geteleA(self, pt, A0):
        #This function is to find element solution mapping
        #print('get element solution mapping')
        # mshn @ msho^-1 @ solno = mshn @ coeffM
        # the first two terms on the LHS I call them A


        #A = self.A1(pt) @ np.linalg.pinv(msho)
        A = self.A1(pt) @ np.linalg.pinv(A0)
        return A

    def A1(self, msh):
        if msh.ndim == 1:
            temp = np.zeros([1,(self.order+1)**3])

        else:
            temp = np.zeros([msh.shape[0],msh.shape[1],(self.order+1)**3])   #this is where the problem is, npt

        m = 0
        for k in range(self.order+1):
            for l in range(self.order+1):
                for n in range(self.order+1):
                    temp.T[m] = msh.T[0]**k * msh.T[1]**l * msh.T[2]**n
                    m += 1


        return temp
