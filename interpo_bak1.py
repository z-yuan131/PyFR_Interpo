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

        # Check if mpi ranks are compatible
        if size != np.max(np.array([len(self.mesh_part[etype]) for etype in self.mesh_part])):
            raise ValueError('Backgroud mesh and solution do not have same partition number as MPI ranks.')


        # Get operators
        self.ops(rank)




        # Get pts location
        #misslist = defaultdict(list)
        #storelist = defaultdict(list)
        #sendlist = defaultdict(list)
        for name in self.mesh_inf_new:
            # Step 1: search node list inside current ranks
            sendlist, storelist, misslist = self.sortpts(name, rank)

            #return 0

            #print(rank, len(sendlist), 'send', name)
            #print(rank, len(storelist), 'store', name)
            #print(rank, len(storelist), 'store', name)

            #Alist = defaultdict()

            self.post_proc(storelist, name)

            #_prefix,etype,part = name.split('_')

            #print(etype,self.soln[f'soln_{etype}_{part}'].shape,self.soln_new[f'soln_{etype}_{part}'].shape)
            #print(etype,np.max(self.soln_new[f'soln_{etype}_{part}'] - self.soln[f'soln_{etype}_{part}']))

        self.write_soln()


    def post_proc(self, storelist, name):
        _prefix, _etypen, _part = name.split('_')
        temp = defaultdict(list)
        #print(storelist)
        for eid, catchlist, index in storelist:
            for etype_name in catchlist:
                # Import relevant class
                cls = subclass_where(InterpolationShape, name=etype_name.split('_')[1])
                # Polynomial space
                pspace_py = cls(self.mesh_order).A1(self.meshn[name][catchlist[etype_name],eid])
                # Rebuild solution
                self.soln_new[f'soln_{_etypen}_{_part}'][catchlist[etype_name],:,eid] = np.einsum('ij,ijk->ik',pspace_py, self.mop[etype_name][index[etype_name]])





    def write_soln(self):
        import h5py
        #self.cfg = Inifile(self.soln['config'])
        #self.stats = Inifile(self.soln['stats'])
        self.cfg.set('solver','order',self.mesh_order_new)

        f = h5py.File('newfile.pyfrs','w')
        f['config'] = self.cfg.tostr()
        f['mesh_uuid'] = self.meshn['mesh_uuid']
        for key in self.soln_new:
            f[key] = self.soln_new[key]
        f['stats'] = self.stats.tostr()
        f.close()



    def post_proc_info(self, plist, name_new):
        sln = list()
        #name = f'p{origin_rank}_->_p{current_rank}'

        # info[0] is eid_new, info[1] is loccatch, info[2] is index, info[3] is pspace_py
        info = list(zip(*plist))
        #print(info)
        for i in range(len(info[0])):
            for name in info[1][i].keys():
                # op: Nele,
                #print(info[2][i][name],info[1][i][name],self.mop[name].shape)
                #index = list(zip(*index))
                _prefix,etype,part = name.split('_')
                _prefix,etype_new,part_new = name_new.split('_')

                #sln.append(list((info[0][i], info[1][i], info[3] @ op)))

                #print(name,info[0][i],info[1][i][name],info[2][i][name])

                #print(self.pspace_py[name_new][info[0][i],info[1][i][name]].shape, self.mop[name][info[1][i][name]].shape)
                etypecls = subclass_where(InterpolationShape, name=etype)
                #print(self.meshn[name_new][info[1][i][name],info[0][i]].shape)
                pspace_py = etypecls(self.mesh_order).A1(self.meshn[name_new][info[1][i][name],info[0][i]])
                #print(pspace_py.shape,self.mop[name][info[1][i][name]].shape,etype)

                #print(list(self.soln_new.keys()))
                #print(etype_new)
                self.soln_new[f'soln_{etype_new}_{part_new}'][info[1][i][name],info[0][i]] = np.einsum('ij,ijk->ik',pspace_py, self.mop[name][info[1][i][name]])



    def write_pyfrs(self, data):
        print('write files')







    def ops(self, rank):
        #pre-load the old mesh respect to different eletype
        self.fnormal = defaultdict()
        self.fcenter = defaultdict()

        self.pspace_py = defaultdict()
        self.soln_new = defaultdict()

        self.mop = defaultdict()

        mesh_order, soln_order, mesh_order_new = self.order_check(rank)

        self.mesh_order = mesh_order
        self.mesh_order_new = mesh_order_new


        for etype in self.mesh_part:
            print(self.mesh_part)
            if self.mesh_part[etype][rank] != 0:
                mname = f'{etype}_p{rank}'

                # Pre-calculate the old mesh center of volume and center of face if applicatable
                etypecls = subclass_where(InterpolationShape, name=etype)
                #if etype == 'hex' or 'quad':
                #    self.ncmsh = etypecls.transfinite(tmsh)

                self.fnormal[f'spt_{mname}'], self.fcenter[f'spt_{mname}'] = etypecls(mesh_order).pre_calc(self.mesho[f'spt_{mname}'])

                #print(self.fnormal[f'spt_{mname}'].shape, self.fcenter[f'spt_{mname}'].shape)


                # Pre-calculate polynomial space for each element
                # actually one can calculate coeeficient matrices by solve soln = polyspace*coeff
                #soln = np.rollaxis(self.soln[f'soln_{mname}'],2)
                cls = subclass_where(BaseShape, name=etype)

                # polynomial space
                pspace_py = etypecls(mesh_order).A1(self.mesho[f'spt_{mname}']).swapaxes(0,1)
                pspace_std = etypecls(mesh_order).A1(np.array(cls.std_ele(mesh_order)),soln_order)

                # the first operator: mapping from physical space to standard space
                mop0 = np.einsum('ikj,jl->ikl',self.inv(pspace_py), pspace_std)

                pspace_std = etypecls(soln_order).A1(np.array(cls.std_ele(soln_order)))
                # the second operator: mapping from standard space to solution space
                mop1 = np.einsum('ij,jkl->ikl',self.inv(pspace_std), self.soln[f'soln_{mname}'])

                self.mop[f'spt_{etype}_p{rank}'] = np.einsum('ijk,kli->ijl',mop0,mop1)

                #print(self.mop[f'spt_{etype}_p{rank}'].shape,'mopshape')

        # this function contents are public
        for name in self.mesh_inf_new:
            #print(rank, name)
            etype = name.split('_')[1]
            #etypecls = subclass_where(InterpolationShape, name=etype)
            # physical space
            #self.pspace_py[name] = etypecls(mesh_order_new).A1(self.meshn[name]).swapaxes(0,1)
            #print(self.pspace_py[etype].shape)

            self.soln_new[f'soln_{etype}_p{rank}'] = np.zeros([self.meshn[name].shape[0],self.nvars,self.meshn[name].shape[1]])



    def sortpts(self,name,rank):  #etypen is the new mesh type (looping in the default dictionary)
        #name = f'spt_{etype}_p{rank}'

        storelist = list()#defaultdict()
        sendlist = list()#defaultdict()
        catchlist = list()#defaultdict()
        misslist = list()

        for j in range(self.meshn[name].shape[1]):
            #if rank == 0:
            #    print((j+1)/self.meshn[name].shape[1])

            # send one point to the function to decide if is in this partition
            # more prcisely, in which old element. new mesh has shape: [Npt, Nele, Nvar]

            index,loccatch,locmiss = self.loc(self.meshn[name][:,j],rank,j)


            #if len(locmiss) != 0:
            #    print(locmiss, index)
            #    return 0,0,0

            if len(locmiss) == self.meshn[name].shape[0]:
                misslist.append(list(locmiss))
                sendlist.append(list((j,self.meshn[name][:,j])))
            elif len(locmiss) == 0:
                misslist.append(list())
                storelist.append(list((j,loccatch,index)))
            else:
                storelist.append(list((j,loccatch,index)))
                misslist.append(list(locmiss))
                sendlist.append(list((j,self.meshn[name][locmiss,j])))
            """
            if len(locmiss) == self.meshn[name].shape[0]:
                sendlist[j] = locmiss
            elif len(locmiss) == 0:
                catchlist[j] = list(range(len(self.meshn[name][:,j])))
                storelist.append(list((j,loccatch,index)))
            else:
                storelist.append(list((j,loccatch,index)))
                catchlist[j] = list(set(locmiss).difference(set(list(range(len(self.meshn[name][:,j]))))))
                sendlist[j] = np.array(locmiss)
            """




        #print(len(sendlist[f'{etypen}_p{rank}']),len(storelist[f'{etypen}_p{rank}']),self.mshn[etypen].shape[1])

        return sendlist, storelist, catchlist

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
                catchlist.append(list((eid,loccatch,index)))

        return catchlist



    def loc(self, ele, rank,j):
        # idea first sort cloest qo element center, among them sort the cloest points
        # in case of the boundary of the different type of mesh, we have to loop dofferent type of mesh
        index = defaultdict()
        miss = list(range(len(ele)))
        catch = defaultdict()

        for etype in self.mesh_part:
            if self.mesh_part[etype][rank] != 0:
                name = f'spt_{etype}_p{rank}'


                # get bounding box for that element
                index_ele = self.box(self.mesho[name],ele[miss].T, j)

                if len(index_ele) > 0:
                    index_tp = self.checkposition(index_ele, ele[miss], name)

                    try:
                        index[name] = index_ele[index_tp]
                        catch[name] = miss
                        #print(name, miss, len(index_tp))
                        miss = list()

                        return index,catch,miss

                    # a pssibility that point is in another partition or other etypes
                    except IndexError:
                        loccatch = [miss[i] for i in np.where(index_tp != 1e8)[0]]
                        #print(rank, loccatch)
                        if len(loccatch) > 0:
                            #print(index_ele,index_tp,miss,loccatch)
                            index_tp = index_tp[[miss.index(i) for i in loccatch]]
                            index[name] = index_ele[index_tp]

                            catch[name] = loccatch

                            # gather all locmiss and assemble store matrix
                            #miss = set(miss).intersection(set(locmiss))
                            miss = list(set(miss).difference(set(loccatch)))


                            if len(miss) == 0:
                                break

                                #temp2[name] = list(set(list(range(len(ele)))).difference(set(miss)))
                        #raise ValueError
        #if len(miss) != 0:
        #    print(name,miss, catch,index,j)
        return index,catch,miss

    def bounding_box(self, x, newx):

        xmax = np.amax(x,axis=0)
        xmaxindex = np.argsort(xmax)
        xmin = np.amin(x,axis=0)
        xminindex = np.argsort(xmin)
        bxma = np.searchsorted(xmax,np.min(newx),sorter=xmaxindex)
        bxmi = np.searchsorted(xmin,np.max(newx),sorter=xminindex)

        #index1 = list(np.arange(bxma,len(xmaxindex),1))
        #index2 = list(np.arange(0,bxmi,1))
        index = np.array(list(set(xmaxindex[bxma:]).intersection(set(xminindex[:bxmi]))),dtype=np.int64)
        #print(index)
        return index

    def box(self, msho, ele, j):

        index = self.bounding_box(msho[...,0],ele[0])
        for dim in range(1, self.ndims + 1):
            if len(index) > 0 and dim < self.ndims:
                index = index[self.bounding_box(msho[:,index,dim],ele[dim])]
            elif len(index) > 0 and dim == self.ndims:
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

        for name in self.mesh_inf_new:
            if self.meshn[name].shape[-1] == 3:
                mesh_order_new = np.ceil(np.power(self.meshn[name].shape[0],1/3))
            else:
                mesh_order_new = np.ceil(np.power(self.meshn[name].shape[0],1/2))
            break

        return int(mesh_order-1), soln_order, int(mesh_order_new-1)

    def inv(self, mat):
        return np.linalg.inv(mat)
