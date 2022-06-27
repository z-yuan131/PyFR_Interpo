# -*- coding: utf-8 -*-
from collections import defaultdict
from mpi4py import MPI
import numpy as np

from pyfr.util import subclass_where
from pyfr.shapes import BaseShape
from base import BaseInterpo
from Eshape import InterpolationShape
from Eshape import facenormal


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
        misslist = defaultdict(list)
        storelist = defaultdict(list)
        sendlist = defaultdict(list)
        for etype in self.mesh_part_new:
            if self.mesh_part_new[etype][rank] != 0:
                # Step 1: search node list inside current ranks
                sendlist[f'{etype}_p{rank}'], storelist[f'{etype}_p{rank}'], misslist[f'{etype}_p{rank}'] = self.sortpts(etype, rank)
                print(rank, len(sendlist[f'{etype}_p{rank}']), 'send')

        # STEP 2 MPI Send and Receive
        Alist = defaultdict()
        for i in range(size):
            for key in sendlist.keys():
                if len(sendlist[key]) > 0:
                    revlist = comm.bcast(sendlist[key], root = i)

                    catchlist = self.sortpts2(revlist, rank)

                    if rank == i:
                        if len(storelist) > 0:
                            Alist_temp = self.group_A(storelist[key], i, rank)
                        else:
                            Alist_temp = defaultdict(list)
                    elif len(catchlist) > 0:
                        Alist_temp = self.group_A(catchlist, i, rank)
                    else:
                        Alist_temp = defaultdict(list)

                    temp = comm.gather(Alist_temp, root = i)
                    if rank == i:
                        Alist[f'{etype}_p{i}'] = temp
                else:
                    Alist[f'{etype}_p{i}'] = self.group_A(storelist[key], i, rank)

        self.write_soln(self, Alist)



    def write_soln(self, Alist, rank):
        for i in range(len(Alist)):
            if len(Alist[i]) > 0:
                for eid, loccatch, sln in Alist[i]:
                    self.soln_new[f'soln_{i}'][loccatch,eid] = sln



    def group_A(self, catchlist, origin_rank, current_rank):
        sln = list()
        name = f'p{origin_rank}_->_p{current_rank}'

        # info[0] is eid_new, info[1] is loccatch, info[2] is index, info[3] is pspace_py
        info = list(zip(*catchlist))
        #print(info)
        for i in range(len(info[0])):
            for name in info[2][i].keys():
                op = np.einsum('ijk,kli->ijl',self.mop0[name][info[2][i][name]],self.mop1[name][...,info[2][i][name]])
                print(op.shape)
            #index = list(zip(*index))

                #sln.append(list((info[0][i], info[1][i], info[3] @ self.mop0[index[0]][:,index[1]] @ self.mop1[index[0]][:,index[1]])))

        return list()


    def write_pyfrs(self, data):
        print('write files')







    def ops(self, rank):
        #pre-load the old mesh respect to different eletype
        self.fnormal = defaultdict()
        self.fcenter = defaultdict()

        self.pspace_py = defaultdict()
        self.soln_new = defaultdict()

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

                self.fnormal[f'spt_{mname}'], self.fcenter[f'spt_{mname}'] = etypecls(mesh_order).pre_calc(self.mesho[f'spt_{mname}'])

                print(self.fnormal[f'spt_{mname}'].shape, self.fcenter[f'spt_{mname}'].shape)


                # Pre-calculate polynomial space for each element
                # actually one can calculate coeeficient matrices by solve soln = polyspace*coeff
                #soln = np.rollaxis(self.soln[f'soln_{mname}'],2)
                cls = subclass_where(BaseShape, name=etype)

                pspace_py = etypecls(mesh_order).A1(self.mesho[f'spt_{mname}']).swapaxes(0,1)
                pspace_std = etypecls(mesh_order).A1(np.array(cls.std_ele(mesh_order)),soln_order)
                self.mop0[f'spt_{etype}_p{rank}'] = np.einsum('ikj,jl->ikl',self.inv(pspace_py), pspace_std)

                pspace_std = etypecls(soln_order).A1(np.array(cls.std_ele(soln_order)))
                self.mop1[f'spt_{etype}_p{rank}'] = np.einsum('ij,jkl->ikl',self.inv(pspace_std), self.soln[f'soln_{mname}'])
                #print(self.mop0[etype].shape,'shape')


        for name in self.mesh_inf_new:
            if name.split('_')[-1] == f'p{rank}':
                etype = name.split('_')[1]
                etypecls = subclass_where(InterpolationShape, name=etype)
                self.pspace_py[etype] = etypecls(mesh_order_new).A1(self.meshn[f'spt_{etype}_p{rank}']).swapaxes(0,1)
                #print(self.pspace_py[etype].shape)

                self.soln_new[f'soln_{etype}_p{rank}'] = np.zeros([self.meshn[f'spt_{etype}_p{rank}'].shape[0],self.meshn[f'spt_{etype}_p{rank}'].shape[1],self.nvars])

        #print(self.fcenter.keys())


    def sortpts(self,etype,rank):  #etypen is the new mesh type (looping in the default dictionary)
        name = f'spt_{etype}_p{rank}'

        storelist = list()
        sendlist = list()
        misslist = list()
        for j in range(self.meshn[name].shape[1]):
            if rank == 0:
                print((j+1)/self.meshn[name].shape[1])

            # send one point to the function to decide if is in this partition
            # more prcisely, in which old element. new mesh has shape: [Npt, Nele, Nvar]

            index,locmiss = self.loc(self.meshn[name][:,j],rank)


            full_loss = list(range(len(self.meshn[name][:,j])))
            loccatch = list(set(full_loss).difference(set(locmiss)))

            if len(locmiss) > 0 and len(loccatch) > 0:
                storelist.append(list((j,loccatch,index)))
                misslist.append(list(locmiss))
                sendlist.append(list((j,self.meshn[name][locmiss,j])))
            elif len(locmiss) > 0:
                misslist.append(list(locmiss))
                sendlist.append(list((j,self.meshn[name][:,j])))
            else:
                misslist.append(list())
                storelist.append(list((j,loccatch,index)))



        #print(len(sendlist[f'{etypen}_p{rank}']),len(storelist[f'{etypen}_p{rank}']),self.mshn[etypen].shape[1])

        return sendlist, storelist, misslist

    def sortpts2(self, revlist, rank):
        catchlist = list()
        for eid, ele, pspace in revlist:
            index,locmiss = self.loc(ele, rank)

            full_loss = list(range(len(ele)))
            loccatch = list(set(full_loss).difference(set(locmiss)))
            #if rank == 0:
            #print(rank, loccatch)
            if len(loccatch) > 0:
                #Alist = self.findA(index)
                catchlist.append(list((eid,loccatch,index,pspace[loccatch])))

        return catchlist



    def loc(self, ele, rank):
        # idea first sort cloest qo element center, among them sort the cloest points
        # in case of the boundary of the different type of mesh, we have to loop dofferent type of mesh
        index = defaultdict()
        miss = set(list(range(len(ele))))
        #temp = list(range(len(ele)))
        temp = defaultdict()

        for etype in self.mesh_part:
            if self.mesh_part[etype][rank] != 0:
                name = f'spt_{etype}_p{rank}'
                msho = self.mesho[name]

                # get bounding box for that element
                index_ele = self.box(msho,ele)

                if len(index_ele) > 0:
                    index_tp = self.checkposition(index_ele, ele, name)

                    try:
                        index[etype] = index_ele[index_tp]
                        locmiss = list()
                    # a pssibility that point is in another partition or other etypes
                    except IndexError:
                        locmiss = list(np.where(index_tp >= len(index_ele))[0])
                        index_tp[locmiss] = 0
                        index[etype] = index_ele[index_tp]

                    # gather all locmiss and assemble store matrix
                    miss = miss.intersection(set(locmiss))
                    #for i in range(len(index[etype])):
                    #    if i not in locmiss:
                    #        temp[i] = list((name,index[etype][i]))

                    temp[name] = [index[etype][i] for i in range(len(index[etype])) if i not in locmiss]

                else:
                    locmiss = list(np.arange(len(ele),dtype='int'))
                    miss = miss.intersection(set(locmiss))

        miss = list(miss)
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


    def box(self, msho, ele):

        index = self.bounding_box(msho[...,0],ele[:,0])
        for dim in range(1, self.ndims+1):
            if index.size > 0 and dim < self.ndims:
                index = index[self.bounding_box(msho[:,index,dim],ele[:,dim])]
            elif index.size > 0 and dim == self.ndims:
                return index
            else:
                return list()




    def checkposition(self,eidx,pts,name):
        #load fcenter and vcenter first
        fnormal = self.fnormal[name]
        fcenter = self.fcenter[name]

        """a fuction to make pt shift to pt_nocur"""
        #if etype == 'hex' or 'quad':
        #.   pt  = etypecls().transfinite

        # chech if in the element  #fcenter [Nfaces,Nele,Nvar] vcenter [Nele,Nvar]
        return facenormal(fcenter[:,eidx], fnormal[:,eidx], pts)







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
