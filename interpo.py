# -*- coding: utf-8 -*-
from collections import defaultdict
from mpi4py import MPI
import numpy as np

from pyfr.util import subclass_where
from pyfr.shapes import BaseShape
from base import BaseInterpo
from Eshape import InterpolationShape


class Interpo(BaseInterpo):
    def __init__(self, argv):
        super().__init__(argv)
        self.argv = argv


        #print(list(self.oldname))
        #print(self.mesh_part)




    def getID(self):
        print('-----------------------------\n')

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Get operators
        self.ops(rank)

        # Get pts location
        for etype in self.mesh_part_new:
            if self.mesh_part_new[etype][rank] != 0:
                #self.sortpts(etype, rank)
                """rewrite this part tosee if one can sort the whole partition at a time"""
                print('rewrite in progress')
    def ops(self, rank):
        #pre-load the old mesh respect to different eletype
        self.vcenter = defaultdict()
        self.fcenter = defaultdict()

        self.pspace_py = defaultdict()

        self.mop0 = defaultdict()
        self.mop1 = defaultdict()

        mesh_order, soln_order, mesh_order_new = self.order_check(rank)


        for etype in self.mesh_part:
            if self.mesh_part[etype][rank] == 0:
                continue
            else:
                mname = f'{etype}_p{rank}'

                # Pre-calculate the old mesh center of volume and center of face if applicatable
                etypecls = subclass_where(InterpolationShape, name=etype)
                #if etype == 'hex' or 'quad':
                #    self.ncmsh = etypecls.transfinite(tmsh)

                self.vcenter[mname], self.fcenter[mname] = etypecls(mesh_order).pre_calc(self.mesho[f'spt_{mname}'])

                # Pre-calculate polynomial space for each element
                # actually one can calculate coeeficient matrices by solve soln = polyspace*coeff
                #soln = np.rollaxis(self.soln[f'soln_{mname}'],2)
                cls = subclass_where(BaseShape, name=etype)

                pspace_py = etypecls(mesh_order).A1(self.mesho[f'spt_{mname}']).swapaxes(0,1)
                pspace_std = etypecls(mesh_order).A1(np.array(cls.std_ele(mesh_order)),soln_order)
                self.mop0[etype] = np.einsum('ikj,jl->ikl',self.inv(pspace_py), pspace_std)

                pspace_std = etypecls(soln_order).A1(np.array(cls.std_ele(soln_order)))
                self.mop1[etype] = np.einsum('ij,jkl->ikl',self.inv(pspace_std), self.soln[f'soln_{mname}'])


        for name in self.mesh_inf_new:
            if name.split('_')[-1] == f'p{rank}':
                etype = name.split('_')[1]
                etypecls = subclass_where(InterpolationShape, name=etype)
                self.pspace_py[etype] = etypecls(mesh_order_new).A1(self.meshn[f'spt_{etype}_p{rank}']).swapaxes(0,1)


        print(self.fcenter.keys())


    def sortpts(self,etypen,rank):  #etypen is the new mesh type (looping in the default dictionary)

        storelist = list()
        sendlist = list()
        misslist = list()
        for j in range(self.meshn[etypen].shape[1]):
            if rank == 0:
                print((j+1)/self.meshn[etypen].shape[1])

            # send one point to the function to decide if is in this partition
            # more prcisely, in which old element. new mesh has shape: [Npt, Nele, Nvar]

            index,locmiss = self.loc(self.mshn[etypen][:,j],rank)

            full_loss = list(range(len(self.mshn[etypen][:,j])))
            loccatch = list(set(full_loss).difference(set(locmiss)))

            if len(locmiss) > 0 and len(loccatch) > 0:
                storelist.append(list((etypen,j,loccatch,index)))
                misslist.append(list(locmiss))
                sendlist.append(list((etypen,j,self.mshn[etypen][locmiss,j])))
            elif len(locmiss) > 0:
                misslist.append(list(locmiss))
                sendlist.append(list((etypen,j,self.mshn[etypen][locmiss,j])))
            else:
                misslist.append(list())
                storelist.append(list((etypen,j,loccatch,index)))


        #print(len(sendlist[f'{etypen}_p{rank}']),len(storelist[f'{etypen}_p{rank}']),self.mshn[etypen].shape[1])

        return sendlist, storelist, misslist

    def sortpts2(self, revlist, rank):
        catchlist = list()
        for etypen, eid, ele in revlist:
            index,locmiss = self.loc(ele, rank)

            full_loss = list(range(len(ele)))
            loccatch = list(set(full_loss).difference(set(locmiss)))
            #if rank == 0:
            #print(rank, loccatch)
            if len(loccatch) > 0:
                #Alist = self.findA(index)
                catchlist.append(list((etypen,eid,loccatch,index)))

        return catchlist



    def loc(self, ele, rank):
        # idea first sort cloest qo element center, among them sort the cloest points
        # in case of the boundary of the different type of mesh, we have to loop dofferent type of mesh
        index = defaultdict()
        locmiss = defaultdict()
        miss = set(list(range(len(ele))))
        temp = list(range(len(ele)))

        for etypeo in self.msho.keys():

            msho = self.msho[etypeo]

            # get bounding box for that element
            index_ele = self.box(msho,ele)



            if len(index_ele) > 0:
                #temp.append(list((eid, etypeo, index_ele, f'p{rank}')))
                #index = index_ele[self.checkposition(index_ele, ele, etypeo)]
                index_tp = self.checkposition(index_ele, ele, etypeo)
                try:
                    index[etypeo] = index_ele[index_tp]
                    locmiss[etypeo] = []
                # a pssibility that point is in another partition or other etypeo
                except IndexError:
                    locmiss[etypeo] = list(np.where(index_tp >= len(index_ele))[0])
                    index_tp[locmiss[etypeo]] = 0
                    index[etypeo] = index_ele[index_tp]

                # gather all locmiss and assemble store matrix
                miss = miss.intersection(set(locmiss[etypeo]))
                for i in range(len(index[etypeo])):
                    if i not in locmiss[etypeo]:
                        temp[i] = list((etypeo,f'p{rank}',index[etypeo][i]))

            else:
                locmiss[etypeo] = list(np.arange(len(ele),dtype='int'))
                miss = miss.intersection(set(locmiss[etypeo]))

        miss = list(miss)

        #if rank == 0:
        #    print(temp,miss)


        #raise ValueError('syop1')

        return temp,miss


    def bounding_box(self, x, newx):

        xmax = np.amax(x,axis=0)
        xmaxindex = np.argsort(xmax)
        xmin = np.amin(x,axis=0)
        xminindex = np.argsort(xmin)
        bxma = np.searchsorted(xmax,newx,sorter=xmaxindex)
        bxmi = np.searchsorted(xmin,newx,sorter=xminindex)

        index = np.arange(np.min(bxma),np.max(bxmi),1)
        return xminindex[index]

    #def bounding_box_pt(self, msho, ele):



    def box(self, msho, ele):

        # this can be rewritten into more beautiful pattern
        index1 = self.bounding_box(msho[...,0],ele[:,0])
        if index1.size > 0:
            index2 = self.bounding_box(msho[:,index1,1],ele[:,1])
            if index2.size > 0:
                index3 = self.bounding_box(msho[:,index1[index2],2],ele[:,2])
                if index3.size > 0:
                    index = index1[index2[index3]]
                    return index
        return list()



    def checkposition(self,eidx,pts,etype):
        #load fcenter and vcenter first

        vcenter = self.vcenter[etype]
        fcenter = self.fcenter[etype]
        #index = np.empty_like(eidx)

        """a fuction to make pt shift to pt_nocur"""
        etypecls = subclass_where(Interpolation, name=etype)
        #if etype == 'hex' or 'quad':
        #.   pt  = etypecls().transfinite

        # chech if in the element  #fcenter [Nfaces,Nele,Nvar] vcenter [Nele,Nvar]
        return etypecls(self.argv[0]).facenormal(fcenter[:,eidx], vcenter[eidx], pts)







    def order_check(self, rank):
        soln_order = self.cfg.getint('solver','order')
        for etype in self.mesh_part:
            if self.mesh_part[etype][rank] != 0:
                name = f'spt_{etype}_p{rank}'
                if self.mesho[name].shape[-1] == 3:
                    mesh_order = np.ceil(np.power(self.mesho[name].shape[0],1/3))
                else:
                    mesh_order = np.ceil(np.power(self.mesho[name].shape[0],1/2))
                break

        for etype in self.mesh_part_new:
            if self.mesh_part_new[etype][rank] != 0:
                name = f'spt_{etype}_p{rank}'
                if self.meshn[name].shape[-1] == 3:
                    mesh_order_new = np.ceil(np.power(self.meshn[name].shape[0],1/3))
                else:
                    mesh_order_new = np.ceil(np.power(self.meshn[name].shape[0],1/2))
                break

        return int(mesh_order-1), soln_order, int(mesh_order_new-1)

    def inv(self, mat):
        return np.linalg.inv(mat)
