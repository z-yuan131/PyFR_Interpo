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


        if self.argv[3].split('.')[-1] == 'pyfrm':
            self.newmesh = load_mesh_info(argv[3])
            self.newname = self.loadname(self.newmesh)
        else:
            self.newname = 'some meaningful names'


        #self.setup['config'] = cfg



    def getpartition(self):
        part = []
        mesh_inf = self.oldmesh.array_info('spt')
        for sk, (etype, shape) in mesh_inf.items():
            if sk.split('_')[-1] != [parts for parts in part]:
                part.append(sk.split('_')[-1])
        return int(max(part)[-1])


    def ptN(self, size):
        #this function is to partition the new mesh
        if self.argv[3].split('.')[-1] == 'pyfrm':
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
        #elif aname == 'uuid':
        #    return mesh.array_info('uuid')
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

    def loadsoln(self,name,rank):

        part = []
        asoln = np.array([])
        soln_inf = self.soln.array_info('soln')
        """
        for sk, (etype, shape) in soln_inf.items():
            if etype == name and sk.split('_')[-1] != [parts for parts in part]:
                part.append(sk.split('_')[-1])
                if asoln.size == 0:
                    asoln = self.soln[sk].astype(self.dtype)
                else:
                    asoln = np.concatenate((asoln,self.soln[sk].astype(self.dtype)),axis=-1)
        return asoln.swapaxes(1,2)
        """
        for sk, (etype, shape) in soln_inf.items():
            if etype == name and sk.split('_')[-1] ==f'p{rank}':
                return self.soln[sk].astype(self.dtype)#.swapaxes(1,2)









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
        self.ml1(rank,comm)


        #print('step1 sort point')
        storelist = defaultdict(list)
        sendlist = defaultdict(list)
        misslist = defaultdict(list)
        for i in self.newname:
            sendlist, storelist, misslist[f'{i}_p{rank}'] = self.sortpts(i, rank)     #step1

        # step2 mpi send recv
        Alist = list()
        for i in range(size):
            if len(sendlist) > 0:
                revlist = comm.bcast(sendlist, root = i)
                #print(rank,revlist.keys())

                catchlist = self.sortpts2(revlist, rank)
                #print(rank,len(storelist))

                """
                # small check routine
                catchlist = self.sortpts2(revlist, rank)
                catchlist = comm.gather(catchlist, root = i)
                if rank == i:

                    for j in catchlist:
                        if len(j) > 0:
                            for etypen, eid, loccatch, index in j:
                                #print(misslist[f'{etypen}_p{rank}'][eid])
                                #misslist[f'{etypen}_p{rank}'][eid] = list(set(misslist[f'{etypen}_p{rank}'][eid]).difference(set(loccatch)))
                                #print(misslist[f'{etypen}_p{rank}'][eid])
                                print(j)
                                break
                            break
                """
                if rank == i:
                    if len(storelist) > 0:
                        Alist_temp = self.group_A(storelist, i, rank)
                elif len(catchlist) > 0:
                    Alist_temp = self.group_A(catchlist, i, rank)
                else:
                    Alist_temp = defaultdict(list)

                temp = comm.gather(Alist_temp, root = i)
                if rank == i: ## it seems it will rewrite Alist, so replace it with temp
                    Alist = temp

            else:
                Alist.append(self.group_A(storelist, i, rank))

                #comm.Barrier()
                #break
        #comm.Barrier()
        # do something to get right A
        self.write_A(Alist, rank)

        """
        for i in range(size):
            if i == rank:
                self.write_to_file(self.solnn, rank)
            comm.Barrier()
        """

    def write_A(self, Alist, rank):
        for i in range(len(Alist)):
            if len(Alist[i].keys()) > 0:

                print(rank, Alist[i].keys())
                # A_info[0] is etypen,  A_info[1] is eidn, A_info[2] is node indices, A_info[3] is A matrices
                for key in Alist[i]:
                    A_info = list(zip(*Alist[i][key]))


                    for j in range(len(A_info[0])):
                        #print(A_info[0][j],A_info[1][j],A_info[3][j].shape)
                        #print(self.Anew[A_info[0][j]][:,A_info[1][j]].shape,A_info[3][j].shape)
                        self.solnn[A_info[0][j]][A_info[2][j],:,A_info[1][j]] = self.Anew[A_info[0][j]][A_info[2][j],A_info[1][j]] @ A_info[3][j]
                        print(rank, self.Anew[A_info[0][j]][A_info[2][j],A_info[1][j]].shape)




    def group_A(self, catchlist, origin_rank, current_rank):
        Alist = defaultdict(list)
        name = f'p{origin_rank}_->_p{current_rank}'

        # info[0] is etypeo, info[1] is eid, info[2] is loccatch, info[4] is index
        info = list(zip(*catchlist))
        for i in range(len(info[0])):
            D = [info[3][i][c] for c in info[2][i]]
            info_index = list(zip(*D))

            F = list(set(info_index[0]))
            for k in F:
                index = np.where(k == np.array(info_index[0]))[0]
                E = list(set(np.array(info_index[2])[index]))
                for j in E:
                    index = np.where(j == np.array(info_index[2]))[0]
                    index = np.array(info[2][i])[index]

                    # data structure index is
                    # info[1][i] is eid of old mesh info[0][i] is new element type
                    # index is node indices of new mesh
                    # the last one is A matrix of that old mesh
                    Alist[f'{name}_{k}'].append(list((info[0][i], info[1][i], index, self.A[k][:,j] @ self.solno[k][...,j])))

        print(Alist.keys())
        #self.write_to_file(Alist)
        return Alist

    def write_to_file(self, datafile, rank):
        import h5py

        #flush to disk
        with h5py.File('newsoln.zhenyang', 'a') as f:
            for key in datafile.keys():
                print(f'soln_{key}_p{rank}')
                f.create_dataset(f'soln_{key}_p{rank}', data=np.array(datafile[key]))
            f.close()






    def ml1(self,rank,comm):
        #pre-load the old mesh respect to different eletype
        self.msho = defaultdict()
        self.solno = defaultdict()
        self.vcenter = defaultdict()
        self.fcenter = defaultdict()

        self.mshn = defaultdict()
        self.solnn = defaultdict()


        self.A = defaultdict()
        self.Anew = defaultdict()




        for etype in self.oldname:

            tmsh = self.loadmesh(etype,'old',rank)
            #self.msho.append(list((etype, tmsh)))
            self.msho[etype] = tmsh
            self.solno[etype] = self.loadsoln(etype,rank)

            #pre-calculate the old mesh center of vlume and center of face is applicatable
            etypecls = subclass_where(Interpolation, name=etype)
            #if etype == 'hex' or 'quad':
            #    self.ncmsh = etypecls.transfinite(tmsh)

            vc, fc = etypecls(self.argv[0]).pre_calc(tmsh)

            self.vcenter[etype] = vc
            self.fcenter[etype] = fc

            #pre-calculate polynomial space for each element
            """ some problem here, in notebook, it will take 10s, but here it is forever
            self.A[etype] = etypecls(self.argv[0]).A1(tmsh).swapaxes(0,1)
            print(self.A[etype].shape)

            self.A[etype] = self.fast_inverse2(self.A[etype]).swapaxes(0,1)
            print(self.A[etype].shape)
            """
            """ so I will load from a file"""
            import h5py
            with h5py.File('Alist.zhenyang', 'r') as f:
                for key in f.keys():
                    if key.split('_')[-1] == f'p{rank}':
                        self.A[etype] = np.array(f[key])
                f.close()


        for etypen in self.newname:
            tmshn = self.loadmesh(etypen,'new',rank)
            #self.msho.append(list((etype, tmsh)))
            self.mshn[etypen] = tmshn
            #self.setup['mesh_uuid'] = self.loadmesh(etypen,'uuid',rank)

            """ bug in the new mesh """
            self.mshn[etypen][...,0] = self.mshn[etypen][...,0]/2
            self.mshn[etypen][...,2] = self.mshn[etypen][...,0]/3

            self.solnn[etypen] = np.empty((self.mshn[etypen].shape[0],5,self.mshn[etypen].shape[1]))


            #etypecls = subclass_where(Interpolation, name=etypen)
            #if etype == 'hex' or 'quad':
            #    self.ncmsh = etypecls.transfinite(tmsh)

            #vc, fc = etypecls(self.argv[0]).pre_calc(tmshn)

            #self.vcn[etypen] = vc


            #pre-calculate polynomial space for each element
            self.Anew[etypen] = etypecls(self.argv[0]).A1(tmshn)


    def fast_inverse(self, A):
        identity = np.identity(A.shape[2], dtype=A.dtype)
        Ainv = np.zeros_like(A)

        for i in range(A.shape[0]):
            Ainv[i] = np.linalg.solve(A[i], identity)
        return Ainv

    def fast_inverse2(A):
        identity = np.identity(A.shape[2], dtype=A.dtype)
        return np.array([np.linalg.solve(x, identity) for x in A])

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
