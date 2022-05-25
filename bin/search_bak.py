# firstly have a class to sort out the points, i.e. which points in the
# new mesh is belong to which element in the old mesh
from pyfr.util import memoize, subclass_where
class hide_och_catch(object):
    name = None

    def __init__(self, diymesh = False):
        self.dtype = np.dtype(float).type

        self.oldmesh = Mymesh_old
        self.oldname = self.loadname(self.oldmesh)
        if isinstance(diymesh,bool):
            self.newmesh = Mymesh_new
            self.newname = self.loadname(self.newmesh)
        else:
            self.newmesh = diymesh
            self.newname = ['diymesh']
            if self.newmesh.shape[0] < self.newmesh.shape[1]:
                self.newmesh = self.newmesh.T

            #print(self.newmesh_othertype[0])

        self.soln = Mysoln

    def loadmesh(self,name,aname):
        if aname == 'new':
            mesh = self.newmesh
        else:
            mesh = self.oldmesh


        part = []
        amesh = np.array([])
        mesh_inf = mesh.array_info('spt')
        for sk, (etype, shape) in mesh_inf.items():
            if etype == name and sk.split('_')[-1] != [parts for parts in part]:
                part.append(sk.split('_')[-1])
                if amesh.size == 0:
                    amesh = mesh[sk].astype(self.dtype).swapaxes(1,2)
                else:
                    amesh = np.concatenate((amesh,mesh[sk].astype(self.dtype).swapaxes(1,2)),axis=-1)
        return amesh.swapaxes(1,2)

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

    def __init__(self, diymesh = False):
        super().__init__(diymesh)


    def getID(self):

        #pre-load the old mesh respect to different eletype
        self.msho = defaultdict
        self.vcenter = defaultdict
        self.fcenter = defaultdict
        self.pspc = defaultdict
        for etype in self.oldname:
            tmsh = self.loadmesh(etype,'old')
            #self.msho.append(list((etype, tmsh)))
            self.msho[etype] = tmsh

        #pre-calculate the old mesh center of vlume and center of face is applicatable
            etypecls = subclass_where(Interpolation, name=etype)
            #if etype == 'hex' or 'quad':
            #    self.ncmsh = etypecls.transfinite(tmsh)

            vc, fc = etypecls().pre_calc(tmsh)

            #self.vcenter.append(list((etype,vc)))
            #self.fcenter.append(list((etype,fc)))

            self.vcenter[etype] = vc
            self.fcenter[etype] = fc

        #pre-calculate polynomial space for each element

            pspc = etypecls().A1(tmsh)

            self.pspc[etype] = pspc





        Idlist = []
        for i in self.newname:
            Idlist.append(self.sortpts(i))


        #print(len(self.Alist))

        return Idlist

    def sortpts(self,name):

        if name == 'diymesh':
            mshn = self.newmesh

            idlist = []
            for i in tqdm(range(mshn.shape[0])):


                index = self.leastdistance(mshn[i])

                etype, eid, A = self.whichele(mshn[i],index)
                idlist.append(list((etype, eid, i, name, A)))



        else:
            mshn = self.loadmesh(name,'new')   #mesh shape: [elepts, Nele, Nvaribles]
            #print(mshn.shape)

            idlist = []
            for j in tqdm(range(mshn.shape[1])):
                for i in range(mshn.shape[0]):
                    index = self.leastdistance(mshn[i,j])

                    etype, eid, A = self.whichele(mshn[i,j],index)
                    idlist.append(list((etype, eid, list((i, j)), name, A)))

        return idlist

    def leastdistance(self,pt):
        #idea first sort cloest qo element center, among them sort the cloest points
        #in case of the boundary of the different type of mesh, we have to loop dofferent type of mesh
        for etype in self.oldname:

            vc = self.vcenter[etype]
            msho = self.msho[etype]


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

        #print(index)   #for the discontinous reason, four points show up in 2D.
        return index

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


    def checkposition(self,idx,pt,etype):
        #load fcenter and vcenter first

        vcenter = self.vcenter[etype]
        fcenter = self.fcenter[etype]



        # loop the number of elements
        for i in (idx):

            """a fuction to make pt shift to pt_nocur"""
            etypecls = subclass_where(Interpolation, name=etype)
            #if etype == 'hex' or 'quad':
            #.   pt  = etypecls().transfinite

            # chech if in the element  #fcenter [Nfaces,Nele,Nvar] vcenter [Nele,Nvar]

            if etypecls().facenormal(fcenter[:,i], vcenter[i], pt):
                return i
