# firstly have a class to sort out the points, i.e. which points in the
# new mesh is belong to which element in the old mesh
from pyfr.util import memoize, subclass_where
from loadingfun import load_ini_info, load_soln_info, load_mesh_info
from Eshape import Interpolation

from collections import defaultdict
import warnings

import numpy as np

class hide_och_catch(object):
    name = None

    def __init__(self, argv):
        cfg = load_ini_info(argv[0])


        self.dtype = np.dtype(float).type

        self.oldmesh = load_mesh_info(argv[1])
        self.oldname = self.loadname(self.oldmesh)

        self.soln = load_soln_info(argv[2])
        self.argv = argv


        if self.argv[-1].split('.')[-1] == 'pyfrm':
            self.newmesh = load_mesh_info(argv[-1])
            self.newname = self.loadname(self.newmesh)
        else:
            self.newname = 'some meaningful names'



    def getpartition(self):
        part = []
        mesh_inf = self.oldmesh.array_info('spt')
        for sk, (etype, shape) in mesh_inf.items():
            if sk.split('_')[-1] != [parts for parts in part]:
                part.append(sk.split('_')[-1])
        return int(max(part)[-1])


    def ptN(self, size):
        #this function is to partition the new mesh
        if self.argv[-1].split('.')[-1] == 'pyfrm':
            # check if new mesh has the same partitions with old mesh
            if self.getpartition() + 1 == size:
                return True
            else:
                raise RuntimeError('New mesh should have same partitions with old mesh')
        else:
            raise ValueError('finish the .csv job')



    def loadmesh(self,name,aname,rank):
        if aname == 'new':
            mesh = self.newmesh
        else:
            mesh = self.oldmesh


        part = []
        amesh = np.array([])
        mesh_inf = mesh.array_info('spt')
        """
        for sk, (etype, shape) in mesh_inf.items():
            if etype == name and sk.split('_')[-1] != [parts for parts in part]:
                part.append(sk.split('_')[-1])
                if amesh.size == 0:
                    amesh = mesh[sk].astype(self.dtype).swapaxes(1,2)
                else:
                    amesh = np.concatenate((amesh,mesh[sk].astype(self.dtype).swapaxes(1,2)),axis=-1)
        """
        for sk, (etype, shape) in mesh_inf.items(): # only load the mesh with same partition and type
            if etype == name and sk.split('_')[-1] == f'p{rank}':
                return mesh[sk].astype(self.dtype)    #,mesh[sk].attrs #mesh attributes can indicate if it is curved


    def loadname(self,mesh):
        name = False
        mesh_inf = mesh.array_info('spt')
        for sk, (etype, shape) in mesh_inf.items():
            if isinstance(name,bool):
                name = [etype]
            else:
                if [etype] == [aname for aname in name]:
                    continue
                else:
                    name.append(etype)

        return name

    def loadsoln(self,name):

        part = []
        asoln = np.array([])
        soln_inf = self.soln.array_info('soln')
        for sk, (etype, shape) in soln_inf.items():
            if etype == name and sk.split('_')[-1] != [parts for parts in part]:
                part.append(sk.split('_')[-1])
                if asoln.size == 0:
                    asoln = self.soln[sk].astype(self.dtype)
                else:
                    asoln = np.concatenate((asoln,self.soln[sk].astype(self.dtype)),axis=-1)
        return asoln.swapaxes(1,2)









# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
"""sub class hide"""
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------




class hide(hide_och_catch):
    name = 'sort'

    def __init__(self,argv):
        super().__init__(argv)


    def getID(self, rank, size, comm):

        #print('precalculate all data that needed during processing')
        self.ml1(rank)


        #print('step1 sort point')
        Idlist = defaultdict()
        for i in self.newname:
            sendlist, Idlist[rank] = self.sortpts(i, rank)     #step1


    def ml1(self,rank):
        #pre-load the old mesh respect to different eletype
        self.msho = defaultdict()
        self.vcenter = defaultdict()
        self.fcenter = defaultdict()

        self.mshn = defaultdict()

        self.A = defaultdict()




        for etype in self.oldname:

            tmsh = self.loadmesh(etype,'old',rank)
            #self.msho.append(list((etype, tmsh)))
            self.msho[etype] = tmsh

            #pre-calculate the old mesh center of vlume and center of face is applicatable
            etypecls = subclass_where(Interpolation, name=etype)
            #if etype == 'hex' or 'quad':
            #    self.ncmsh = etypecls.transfinite(tmsh)

            vc, fc = etypecls(self.argv[0]).pre_calc(tmsh)

            self.vcenter[etype] = vc
            self.fcenter[etype] = fc

            #pre-calculate polynomial space for each element
            self.A[etype] = etypecls(self.argv[0]).A1(tmsh)


        for etypen in self.newname:
            tmshn = self.loadmesh(etypen,'new',rank)
            #self.msho.append(list((etype, tmsh)))
            self.mshn[etypen] = tmshn

            # bug in the new mesh
            #self.mshn[etypen][...,0] = self.mshn[etypen][...,0]/2
            #self.mshn[etypen][...,2] = self.mshn[etypen][...,0]/3


            #etypecls = subclass_where(Interpolation, name=etypen)
            #if etype == 'hex' or 'quad':
            #    self.ncmsh = etypecls.transfinite(tmsh)

            #vc, fc = etypecls(self.argv[0]).pre_calc(tmshn)

            #self.vcn[etypen] = vc


            #self.Alist[etypen] = np.zeros([tmshn.shape[0],tmshn.shape[0],tmshn.shape[1]])



    def sortpts(self,etypen,rank):  #etypen is the new mesh type (looping in the default dictionary)

        storelist = defaultdict(list)
        sendlist1 = defaultdict(list)
        sendlist2 = defaultdict(list)
        for j in range(self.mshn[etypen].shape[1]):
            if rank == 0:
                print((j+1)/self.mshn[etypen].shape[1])

            # send one point to the function to decide if is in this partition
            # more prcisely, in which old element. new mesh has shape: [Npt, Nele, Nvar]

            index,locmiss = self.loc(j, self.mshn[etypen][:,j],rank)

            if len(index) > 0:
                if len(locmiss) > 0:
                    storelist[f'{etypen}_p{rank}'].append(index)
                    sendlist1[f'{etypen}_p{rank}'].append(self.mshn[etypen][:,j])
                    sendlist1[f'{etypen}_p{rank}_miss_loc'].append(locmiss)
                else:
                    storelist[f'{etypen}_p{rank}'].append(index)

            else:
                sendlist2[f'{etypen}_p{rank}'].append(self.mshn[etypen][:,j])


        print(len(sendlist1[f'{etypen}_p{rank}']),len(sendlist2[f'{etypen}_p{rank}']),len(storelist[f'{etypen}_p{rank}']),self.mshn[etypen].shape[1])

        #raise ValueError('stop2')
        return sendlist1, storelist

    def loc(self, eid, ele, rank):
        #idea first sort cloest qo element center, among them sort the cloest points
        #in case of the boundary of the different type of mesh, we have to loop dofferent type of mesh
        index = defaultdict()
        miss = defaultdict()
        temp = list()

        for etypeo in self.msho.keys():

            msho = self.msho[etypeo]

            # get bounding box for that element
            index_ele,locmiss = self.box(msho,ele, rank)



            if len(index_ele) > 0:
                """some modification here to allow multi kinds of element types"""
                temp.append(list((eid, etypeo, index_ele, f'p{rank}')))

            if np.allclose(ele,msho[:,0]):
                if rank == 0:
                    print(index_ele,locmiss)
            raise ValueError('syop1')

        return temp,locmiss


    def bounding_box(self, x, newx, pindex = []):

        xmax = np.amax(x,axis=0)
        xmaxindex = np.argsort(xmax)
        xmin = np.amin(x,axis=0)
        xminindex = np.argsort(xmin)
        bxma = np.searchsorted(xmax,newx,sorter=xmaxindex)
        bxmi = np.searchsorted(xmin,newx,sorter=xminindex)

        if len(pindex) > 0:
            """a bug here: question how to find ele for each pt?"""
            if not np.all(list(bxma),where=list(bxmi-1)):
                locmiss = np.where(bxma != bxmi-1)[0]   # record the pts which are in other ranks
                bxma[locmiss] = -1                      # replace these pts location with -1

                return xminindex[bxma],locmiss
            else:
                """pritential bug here and also ho mesh can be solved from here"""
                #if len(pindex) > len(bxma):
                return xminindex[bxma],list()
        else:
            index = np.arange(np.min(bxma),np.max(bxmi),1)
            return xminindex[index]


    def box(self,msho,ele,rank):

        # this can be rewritten into more beautiful pattern
        index1 = self.bounding_box(msho[...,0],ele[:,0])
        if index1.size > 0:
            index2 = self.bounding_box(msho[:,index1,1],ele[:,1])
            if index2.size > 0:
                index3 = self.bounding_box(msho[:,index1[index2],2],ele[:,2])
                if index3.size > 0:
                    index = index1[index2[index3]]
                    #indexq,locmiss = self.bounding_box(msho[:,index,2],ele[:,2],index)
                    #return index[indexq],locmiss
                    return index,list()
        return list(),list()



    def checkposition(self,eidx,pt,etype):
        #load fcenter and vcenter first

        vcenter = self.vcenter[etype]
        fcenter = self.fcenter[etype]


        # loop the number of elements
        for i in eidx:

            """a fuction to make pt shift to pt_nocur"""
            etypecls = subclass_where(Interpolation, name=etype)
            #if etype == 'hex' or 'quad':
            #.   pt  = etypecls().transfinite

            # chech if in the element  #fcenter [Nfaces,Nele,Nvar] vcenter [Nele,Nvar]

            if etypecls(self.argv[0]).facenormal(fcenter[:,i], vcenter[i], pt):
                return i
