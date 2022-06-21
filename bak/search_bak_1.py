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
        storelist = defaultdict(list)
        sendlist = defaultdict(list)
        misslist = defaultdict(list)
        for i in self.newname:
            sendlist[f'{i}_p{rank}'], storelist[f'{i}_p{rank}'], misslist[f'{i}_p{rank}'] = self.sortpts(i, rank)     #step1
        # step2 mpi send recv
        print(sendlist.keys())
        for i in range(size):
            if len(sendlist.keys()) > 0:
                revlist = comm.bcast(sendlist, root = i)
                #print(rank,revlist.keys())

                catchlist = self.sortpts2(revlist, rank)
                catchlist = comm.gather(catchlist, root = i)

                if rank == i:

                    # small check routine
                    for j in catchlist:
                        if len(j) > 0:
                            for etypen, eid, loccatch, index in j:
                                #print(misslist[f'{etypen}_p{rank}'][eid])
                                #misslist[f'{etypen}_p{rank}'][eid] = list(set(misslist[f'{etypen}_p{rank}'][eid]).difference(set(loccatch)))
                                #print(misslist[f'{etypen}_p{rank}'][eid])
                                print(index)
                                break
                """
                    for j in range(len(catchlist)):
                        if len(j) > 0:
                            req = comm.irecv(Alist, root = j)
                else:
                    # pick A algorithm
                    comm.isend(Alist, dest = i)
                req.WaitAll()
                """


                comm.Barrier()
                break


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

            """ bug in the new mesh """
            self.mshn[etypen][...,0] = self.mshn[etypen][...,0]/2
            self.mshn[etypen][...,2] = self.mshn[etypen][...,0]/3


            #etypecls = subclass_where(Interpolation, name=etypen)
            #if etype == 'hex' or 'quad':
            #    self.ncmsh = etypecls.transfinite(tmsh)

            #vc, fc = etypecls(self.argv[0]).pre_calc(tmshn)

            #self.vcn[etypen] = vc


            #self.Alist[etypen] = np.zeros([tmshn.shape[0],tmshn.shape[0],tmshn.shape[1]])



    def sortpts(self,etypen,rank):  #etypen is the new mesh type (looping in the default dictionary)

        storelist = list()
        sendlist = list()
        misslist = list()
        for j in range(self.mshn[etypen].shape[1]):
            if rank == 0:
                print((j+1)/self.mshn[etypen].shape[1])

            # send one point to the function to decide if is in this partition
            # more prcisely, in which old element. new mesh has shape: [Npt, Nele, Nvar]

            index,locmiss = self.loc(self.mshn[etypen][:,j],rank)

            if len(locmiss) > 0:
                storelist.append(index)
                #sendlist[f'{etypen}_p{rank}'].append(list((j,self.mshn[etypen][:,j])))
                misslist.append(list(locmiss))
                sendlist.append(list((etypen,j,self.mshn[etypen][locmiss,j])))
            else:
                misslist.append(list())
                storelist.append(index)


        #print(len(sendlist[f'{etypen}_p{rank}']),len(storelist[f'{etypen}_p{rank}']),self.mshn[etypen].shape[1])

        return sendlist, storelist, misslist

    def sortpts2(self, revlist, rank):
        catchlist = list()
        for key in revlist.keys():
            for etypen, eid, ele in revlist[key]:
                index,locmiss = self.loc(ele, rank)


                full_loss = list(range(len(ele)))
                loccatch = list(set(full_loss).difference(set(locmiss)))
                #if rank == 0:
                #print(rank, loccatch)
                if len(loccatch) > 0:
                    catchlist.append(list((etypen,eid,loccatch,index)))

        return catchlist


    def loc(self, ele, rank):
        #idea first sort cloest qo element center, among them sort the cloest points
        #in case of the boundary of the different type of mesh, we have to loop dofferent type of mesh
        index = defaultdict()
        locmiss = defaultdict()
        miss = set(list(range(len(ele))))
        temp = list(range(len(ele)))

        for etypeo in self.msho.keys():

            msho = self.msho[etypeo]

            # get bounding box for that element
            index_ele = self.box(msho,ele)



            if len(index_ele) > 0:
                """some modification here to allow multi kinds of element types"""
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
                miss = miss.intersection((locmiss[etypeo]))
                for i in range(len(index[etypeo])):
                    if i not in locmiss[etypeo]:
                        temp[i] = list((etypeo,f'p{rank}',index[etypeo][i]))

            else:
                locmiss[etypeo] = list(np.arange(len(ele),dtype='int'))
                miss = miss.intersection((locmiss[etypeo]))


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
