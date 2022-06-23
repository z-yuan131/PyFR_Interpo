# -*- coding: utf-8 -*-
import numpy as np


# First-order node numbers associated with each element face
_fmap = {
    'tri': {'line': [[0, 1], [1, 2], [2, 0]]},
    'quad': {'line': [[0, 1], [1, 3], [3, 2], [2, 0]]},
    'tet': {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},
    'hex': {'quad': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
                     [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},
    'pri': {'quad': [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
            'tri': [[0, 1, 2], [3, 4, 5]]},
    'pyr': {'quad': [[0, 1, 2, 3]],
            'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]}
}

class InterpolationShape(object):
    name = None

    def __init__(self, order):
        self.order = order

class HexType(InterpolationShape):
    name = 'hex'

    def __init__(self,order):
        super().__init__(order)

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
        #print(vcenter.shape, fcenter.shape,ptvec.shape,norvec.shape)

        # using einstien notition to do tensor product and extract diagonal entries
        #out    = np.einsum('ijk,lmqk->lijqm', norvec, ptvec)
        #out1   = np.einsum('kijji->kji',out)
        #out    = np.einsum('ijk,lmqk->lijqm', norvec, ptvec)
        out1   = np.einsum('ijk,lijk->lji', norvec, ptvec)
        # is that a good idea to use this loop?
        index  = np.zeros(len(pt_noncur),dtype='int')
        for i in range(len(pt_noncur)):
            for j in range(len(vcenter)):
                if np.all(out1[i,j] < 10e-10) or np.all(out1[i,j] > -10e-10):
                    index[i] = j
                    break
                #if there is a bug, it is because this point is not in the bounding box
                index[i] = 10e10
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

    def A1(self, stdcls):
        msh = np.array(stdcls.std_ele(self.order))

        temp = np.zeros([msh.shape[0],(self.order+1)**3])   #this is where the problem is, npt

        m = 0
        for k in range(self.order+1):
            for l in range(self.order+1):
                for n in range(self.order+1):
                    temp.T[m] = msh.T[0]**k * msh.T[1]**l * msh.T[2]**n
                    m += 1


        return temp

    def original_A1(self, msh):
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


class QuadType(InterpolationShape):
    name = 'quad'

    def __init__(self,order):
        super().__init__(order)

    def npts(self):
        return (self.order+1)**2

    def nsps(self, poly_order):
        return (poly_order+1)**2


    def pre_calc(self, mesh):
        # Volume center
        vcenter = np.sum(mesh,axis=0)/self.npts()
        # Face center
        fcenter = np.array([np.sum(mesh[_fmap['quad']['line'][i]],axis=0) for i in range(4)]) / 2
        return vcenter, fcenter

        return vcenter, fcenter



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


    def A1(self, msh, poly_order = None):
        if not poly_order:
            poly_order = self.order

        if msh.ndim == 1:
            temp = np.zeros([1,int(self.npts())])
        elif msh.ndim == 2:
            temp = np.zeros([int(self.npts()),int(self.nsps(poly_order))])
        else:
            temp = np.zeros([int(self.npts()),msh.shape[1],int(self.nsps(poly_order))])   #this is where the problem is, npt

        m = 0
        for k in range(self.order+1):
            for l in range(self.order+1):
                temp.T[m] = msh.T[0]**k * msh.T[1]**l
                m += 1

        return temp


class TriType(InterpolationShape):
    name = 'tri'

    def __init__(self,order):
        super().__init__(order)

    def npts(self):
        return (self.order+1)*(self.order+2)/2

    def nsps(self, poly_order):
        return (poly_order+1)*(poly_order+2)/2


    def pre_calc(self, mesh):
        # Volume center
        vcenter = np.sum(mesh,axis=0)/self.npts()
        # Face center
        fcenter = np.array([np.sum(mesh[_fmap['tri']['line'][i]],axis=0) for i in range(3)]) / 2
        return vcenter, fcenter



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


    def A1(self, msh, poly_order = None):
        if not poly_order:
            poly_order = self.order

        if msh.ndim == 1:
            temp = np.zeros([1,int(self.npts())])
        elif msh.ndim == 2:
            temp = np.zeros([int(self.npts()),int(self.nsps(poly_order))])
        else:
            temp = np.zeros([int(self.npts()),msh.shape[1],int(self.nsps(poly_order))])   #this is where the problem is, npt

        m = 0
        for k in range(self.order+1):
            for l in range(self.order+1):
                if k+l >= self.order+1:
                    break
                else:
                    temp.T[m] = msh.T[0]**k * msh.T[1]**l
                    m += 1

        return temp
