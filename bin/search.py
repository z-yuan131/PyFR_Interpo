# firstly have a class to sort out the points, i.e. which points in the
# new mesh is belong to which element in the old mesh
from pyfr.util import memoize, subclass_where
from loadingfun import load_ini_info, load_soln_info, load_mesh_info
from Eshape import Interpolation

from collections import defaultdict

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

        """
        flag = 0
        if isinstance(flag,bool):
            print('something')

            #self.newmesh = Mymesh_new
            #self.newname = self.loadname(self.newmesh)
        else:
            print('something')

            #self.newmesh = diymesh
            #self.newname = ['diymesh']
            #if self.newmesh.shape[0] < self.newmesh.shape[1]:
            #    self.newmesh = self.newmesh.T

            #print(self.newmesh_othertype[0])
        """

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
            if etype == name and sk.split('_')[-1][-1] == str(rank):
                return mesh[sk].astype(self.dtype)


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





class hide(hide_och_catch):
    name = 'sort'

    def __init__(self,argv):
        super().__init__(argv)


    def getID(self, rank, size, comm):

        self.ml1(rank)


        #print(rank, self.vcn)


        Idlist = defaultdict()
        for i in self.newname:
            sendlist, Idlist[rank] = self.sortpts(i, rank)     #step1




        for i in range(size):
            if len(sendlist) != 0: # kind of useless
                revlist = comm.bcast(sendlist, root=i)         #step2
                print(str(rank)+' boardcast done')
                if rank != i:
                    Idlist[i] = self.sortpts_rev(revlist, rank)   #step3
                else:  # when the other ranks are doing something, this rank should output and write done the A
                    Idlist[i] = self.writeAlist(Idlist[i])       #step3   #simply rewrite Idlist to save memory, this Idlist is Alist
                    print(str(rank)+' Alist done')
            else:    # in case this new rank have all pts in one old rank
                Idlist[i] = self.writeAlist(Idlist[i])       #step3    #simply rewrite Idlist to save memory, this Idlist is Alist
                print(str(rank)+' Alist done')
            comm.Barrier()
        #print(list(Idlist.keys())[1])



        """
        print('update Alist')
        tempA = comm.gather(tempA, root=i)
        if i == rank:
            self.Alist.update(tempA)
        #comm.Barrier()
        print(list(self.Alist.keys()))
        """






        for i in range(1, size):                               #step4
            print('beginning '+ str(rank))
            Idlist[list(Idlist.keys())[i]] = self.writeAlist(Idlist[list(Idlist.keys())[i]])
            print('end '+ str(rank))
        comm.Barrier()

        print(Idlist.keys())



        print('update Alist')
        for i in range(size):
            temp = comm.gather(Idlist[i], root=i)
            if i == rank:
                Alist = temp
        #print(len(self.Alist[0]))


        comm.Barrier()
        self.Alist = self.nested_dict(3,float)
        for part in Alist:
            for etypen in part.keys():
                #if rank == 0:
                #    print(list(part.keys()))
                #    print(part[etypen][0:4])
                #raise ValueError('stop1')
                #for n in part[etypen]:
                for j,i,A in part[etypen]:
                    if rank == 2:
                        print(part[etypen][0:10])
                    #print(rank, j,i,A)
                    #comm.Barrier()
                    raise RuntimeError('stop1')
                    self.Alist[etypen][j][i] = A  #etypen, eid, nid

        print(rank,self.Alist['hex'][1][1])


        #if rank == 0:

        #    print(self.Alist)
        #raise ValueError('stop1')




        #return Idlist


    def ml1(self,rank):
        #pre-load the old mesh respect to different eletype
        self.msho = defaultdict()
        self.vcenter = defaultdict()
        self.fcenter = defaultdict()

        self.mshn = defaultdict()
        self.vcn = defaultdict()

        self.pspc = defaultdict()
        #self.Alist = defaultdict()




        for etype in self.oldname:

            tmsh = self.loadmesh(etype,'old',rank)
            #self.msho.append(list((etype, tmsh)))
            self.msho[etype] = tmsh

            #pre-calculate the old mesh center of vlume and center of face is applicatable
            etypecls = subclass_where(Interpolation, name=etype)
            #if etype == 'hex' or 'quad':
            #    self.ncmsh = etypecls.transfinite(tmsh)

            vc, fc = etypecls(self.argv[0]).pre_calc(tmsh)

            #self.vcenter.append(list((etype,vc)))
            #self.fcenter.append(list((etype,fc)))

            self.vcenter[etype] = vc
            self.fcenter[etype] = fc

            #pre-calculate polynomial space for each element (LU)

            pspc = etypecls(self.argv[0]).A1(tmsh)

            self.pspc[etype] = pspc


        for etypen in self.newname:
            tmshn = self.loadmesh(etypen,'new',rank)
            #self.msho.append(list((etype, tmsh)))
            self.mshn[etypen] = tmshn


            #
            etypecls = subclass_where(Interpolation, name=etypen)
            #if etype == 'hex' or 'quad':
            #    self.ncmsh = etypecls.transfinite(tmsh)

            #vc, fc = etypecls(self.argv[0]).pre_calc(tmshn)

            #self.vcn[etypen] = vc


            #self.Alist[etypen] = np.zeros([tmshn.shape[0],tmshn.shape[0],tmshn.shape[1]])


    def sortpts(self,etypen,rank):  #etypen is the new mesh type (looping in the default dictionary)

        storelist = []
        sendlist = []
        for j in range(self.mshn[etypen].shape[1]):
            if rank == 0:
                print((j+1)/self.mshn[etypen].shape[1])

            for i in range(self.mshn[etypen].shape[0]):

                # send one point to the function to decide if is in this partition
                # more prcisely, in which old element. new mesh has shape: [Npt, Nele, Nvar]

                index, etypeo = self.leastdistance(self.mshn[etypen][i,j])
                if index == None:
                    # data structure:
                    # oldmesh partition, etypen, eid, nid, x,y,z
                    sendlist.append(list((rank,etypen,j,i,self.mshn[etypen][i,j])))
                else:
                    # data structure:
                    # newmesh partition, etypen, eid, nid, oldmesh rank, etypeo, index
                    storelist.append(list((rank,etypen,j,i,rank,etypeo,index)))

        return sendlist, storelist

    def sortpts_rev(self,revlist,rank):
        # print('process the pts from other ranks')
        # revlist shape:
        # original partition, etype_new, eidn. nidn, (x,y,z)
        storelist2 = []
        for part, etype, eid, nid, coord in revlist:
            index,etypeo = self.leastdistance(coord)
            if index != None:
                # data structure:
                # newmesh partition, etypen, eid, nid, oldmesh rank, etypeo, index
                storelist2.append(list((part,etype,eid,nid,rank,etypeo,index)))
        print(len(storelist2))

        return storelist2



    def leastdistance(self, pt):
        #idea first sort cloest qo element center, among them sort the cloest points
        #in case of the boundary of the different type of mesh, we have to loop dofferent type of mesh
        for etype in list(self.msho.keys()):

            vc = self.vcenter[etype]
            msho = self.msho[etype]

            """
            if vc.shape[-1] == 2:
                r0 = (vcn[...,0] - vc[...,0])**2 + (vcn[...,1] - vc[...,1])**2 #2D
                eindex = np.argsort(r0)
            elif vc.shape[-1] == 3:
                r0 = (vcn[...,0] - vc[...,0])**2 + (vcn[...,1] - vc[...,1])**2 + (vcn[...,2] - vc[...,2])**2 #2D
                eindex = np.argsort(r0)
            else:
                raise ValueError('something wrong with old mesh: it is not 2 or 3D')

            raise ValueError('stop1')
            etypecls = subclass_where(Interpolation, name=etype)
            for pt in pts:
                eidx  = self.checkposition(eindex,pt,etype)

            if eidx == None: #in case eid is on the boudnary of two type of mesh
                continue
            else:
                raise ValueError('stop2')
                return eidx

            raise ValueError('stop1')
            """

            if vc.shape[-1] == 2:
                r0 = (pt[0] - vc[...,0])**2 + (pt[1] - vc[...,1])**2 #2D
                eindex = np.argsort(r0)[:min(10,len(vc))]   # pick first ten elements
                r0 = (pt[0] - msho[:,eindex,0])**2 + (pt[1] - msho[:,eindex,1])**2
            elif vc.shape[-1] == 3:
                r0 = (pt[0] - vc[...,0])**2 + (pt[1] - vc[...,1])**2 + (pt[2] - vc[...,2])**2 #2D
                eindex = np.argsort(r0)[:min(10,len(vc))]   # pick first ten elements
                r0 = (pt[0] - msho[:,eindex,0])**2 + (pt[1] - msho[:,eindex,1])**2 + (pt[2] - msho[:,eindex,2])**2
            else:
                raise ValueError('something wrong with old mesh: it is not 2 or 3D')


            """
            # this block of code is supposed calculate the least distance of another element type
            # but obviously this will miss some points on the boundaries
            # so a good idea can be developed to varify in every type of mesh
            if etype == self.oldname[0]:
                r1 = np.min(r0)
                index = [list((etype, eindex[np.where(r0 == r1)[1]], msho))]

            else:
                r2 = np.min(r0)
                if r2 < r1:
                    index = [list((etype, eindex[np.where(r0 == r2)[1]], msho))]
                    r1 = r2
                elif r2 == r1:    # this line will work when the interface of two different types of mesh occurs
                    index.append(list((etype, eindex[np.where(r0 == r2)[1]], msho)))
            """
            etypecls = subclass_where(Interpolation, name=etype)
            eidx  = self.checkposition(eindex,pt,etype)
            if eidx == None:
                continue
            else:
                return eidx, etype

        return None,1


        #print(index)   #for the discontinous reason, four points show up in 2D.
        #return index
    def nested_dict(self, n, type):
        if n == 1:
            return defaultdict(type)
        else:
            return defaultdict(lambda: self.nested_dict(n-1, type))

    def writeAlist(self,storelist):

        #Alist = self.nested_dict(3, float)
        Alist = defaultdict()
        for rank,etypen,j,i,rank,etypeo,index in storelist:
            etypecls = subclass_where(Interpolation, name=etypeo)
            #Alist[etypen][j][i] = etypecls(self.argv[0]).geteleA(self.mshn[etypen][i,j],self.pspc[etypeo][:,index])
            if etypen in Alist:
                Alist[etypen].append(list((j,i,etypecls(self.argv[0]).geteleA(self.mshn[etypen][i,j],self.pspc[etypeo][:,index]))))
            else:
                Alist[etypen] = [list((j,i,etypecls(self.argv[0]).geteleA(self.mshn[etypen][i,j],self.pspc[etypeo][:,index])))]


        return Alist

    def whichele(self,pt,index):
        for etype, idx, msho in index:
            etypecls = subclass_where(Interpolation, name=etype)
            #idx has shape [Neido]

            #No matter which type of element, high level operation is the same

            eidx  = self.checkposition(idx,pt,etype)


            if eidx == None: #in case eid is on the boudnary of two type of mesh
                continue
            else:
                #t = time.process_time()

                A = etypecls().geteleA(pt,msho[:,eidx])
                #A = etypecls().geteleA(pt,self.pspc[etype][:,eidx])

                #elapsed_time = time.process_time() - t

                #print(elapsed_time)
                #raise ValueError('stop1')

                return etype, eidx, A

        if eidx == None:   # in case the mesh is on the bc and there is no matching element
            A =1
            #A = etypecls().geteleA(pt,msho[:,idx[0]])
            #A = etypecls().geteleA(pt,self.pspc[etype][:,idx[0]])
            return etype, idx[0], A


    def checkposition(self,eidx,pt,etype):
        #load fcenter and vcenter first

        vcenter = self.vcenter[etype]
        fcenter = self.fcenter[etype]



        # loop the number of elements
        for i in (eidx):

            """a fuction to make pt shift to pt_nocur"""
            etypecls = subclass_where(Interpolation, name=etype)
            #if etype == 'hex' or 'quad':
            #.   pt  = etypecls().transfinite

            # chech if in the element  #fcenter [Nfaces,Nele,Nvar] vcenter [Nele,Nvar]

            if etypecls(self.argv[0]).facenormal(fcenter[:,i], vcenter[i], pt):
                return i
