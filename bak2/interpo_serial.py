# -*- coding: utf-8 -*-
from collections import defaultdict
from mpi4py import MPI
import numpy as np
from scipy.linalg import lu_factor, lu_solve, lu


from pyfr.util import memoize, subclass_where
from pyfr.shapes import BaseShape
from pyfr.inifile import Inifile
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
        self.post_proc_soln()

        self.write_soln()


    def post_proc(self, storelist, new_name):
        _prefix, _etypen, _part = new_name.split('_')
        temp = defaultdict(list)
        #print(storelist)
        for eid, catchlist, index in storelist:
            for old_name in catchlist:
                # Import relevant class
                #cls = subclass_where(InterpolationShape, name=etype_name.split('_')[1])
                # Polynomial space
                #pspace_py = cls(self.mesh_order).A1(self.meshn[name][catchlist[etype_name],eid])
                # Rebuild solution
                #self.soln_new[f'soln_{_etypen}_{_part}'][catchlist[etype_name],:,eid] = np.einsum('ij,ijk->ik',pspace_py, self.mop[etype_name][index[etype_name]])
                _prefix,_etypeo,_parto = old_name.split('_')

                poly_space = self._get_polyspace(_etypeo, self.soln[f'soln_{_etypeo}_{_parto}'].shape[0], self.meshn[new_name][catchlist[old_name],eid])#.swapaxes(0,1)
                #print(poly_space.shape, self.mop[old_name][index[old_name]].shape)

                self.soln_new[f'soln_{_etypen}_{_part}'][catchlist[old_name],:,eid] = np.einsum('ij,ijk -> ik', poly_space, self.mop[old_name][index[old_name]])

                if np.max(self.soln_new[f'soln_{_etypen}_{_part}'][catchlist[old_name],:,eid]) > 1000:
                    print(eid, index[old_name])
                #print(np.max(self.soln_new[f'soln_{_etypen}_{_part}']), np.max(poly_space), np.max(self.mop[old_name][index[old_name]]))
                #print(index[old_name], catchlist[old_name], eid, new_name, old_name)
                #raise ValueError

    def post_proc_soln(self):
        for key in self.soln_new:
            _prefix,etype,part = key.split('_')
            print(self.soln_op[etype].shape, self.soln_new[key].shape)
            self.soln_new[key] = np.einsum('ij,jkl -> ikl',self.soln_op[etype], self.soln_new[key])


    def write_soln(self):
        import h5py
        self.cfg.set('solver','order',self.new_mesh_order)

        f = h5py.File('newfile.pyfrs','w')
        f['config'] = self.cfg.tostr()
        f['mesh_uuid'] = self.meshn['mesh_uuid']
        for key in self.soln_new:
            f[key] = self.soln_new[key]
            print(self.soln_new[key].shape)
        f['stats'] = self.stats.tostr()
        f.close()



    def write_pyfrs(self, data):
        print('write files')


    #@memoize
    def _get_shape(self, name, nspts, cfg):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, cfg)

    #@memoize
    def _get_std_ele(self, name, nspts):
        order = int(self.cfg.get('solver','order'))
        #print(order)
        return self._get_shape(name, nspts, self.cfg).std_ele(order)

    #@memoize
    def _get_mesh_op_vis(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    #@memoize
    def _get_mesh_op_sln(self, name, nspts):
        shape = self._get_shape(name, nspts, self.cfg)
        upts = shape.upts
        return shape.sbasis.nodal_basis_at(upts).astype(self.dtype)

    def _get_ortho_basis(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.ortho_basis_at(svpts).astype(self.dtype)

    def _get_polyspace(self, name, nspts, mesh, Jacobi = True):
        if Jacobi and mesh.ndim == 3:
            return self._get_ortho_basis(name, nspts, mesh.reshape(-1,self.ndims)).reshape(nspts,nspts,-1).swapaxes(0,-1)
        elif Jacobi and mesh.ndim == 2:
            return self._get_ortho_basis(name, nspts, mesh).swapaxes(0,1)
        else:
            etypecls = subclass_where(InterpolationShape, name=name)
            return etypecls(self.order).A1(mesh).swapaxes(0,1)

    def _get_polyspace2(self, name, nspts, mesh, Jacobi = True):

        return np.array([self._get_ortho_basis(name, nspts, mesh[:,eid]).T for eid in range(mesh.shape[1])])





    def _get_npts(self, name, order):
        return self._get_shape(name, 0, self.cfg).nspts_from_order(order)

    def _get_order(self, name, nspts):
        return self._get_shape(name, nspts, self.cfg).order_from_nspts(nspts)

    def _get_soln_op(self, name, nspts):
        cfg = Inifile(self.soln['config'])

        order = self._get_order(name, nspts) - 1

        nspts = self._get_npts(name, order + 1)
        svpts = self._get_std_ele(name,nspts)
        cfg.set('solver','order',order)

        self.new_mesh_order = order

        shape = self._get_shape(name, nspts, cfg)
        return shape.sbasis.nodal_basis_at(shape.upts).astype(self.dtype)






    def ops(self, rank):
        #pre-load the old mesh respect to different eletype
        self.fnormal = defaultdict()
        self.fcenter = defaultdict()

        self.soln_op = defaultdict()
        self.soln_new = defaultdict()

        self.mop = defaultdict()

        #mesh_order, soln_order, mesh_order_new = self.order_check(rank)

        #self.mesh_order = mesh_order
        #self.mesh_order_new = mesh_order_new

        for etype in self.mesh_part:
            print(self.mesh_part)
            if self.mesh_part[etype][rank] != 0:
                mname = f'{etype}_p{rank}'

                # Pre-calculate the old mesh center of volume and center of face if applicatable
                etypecls = subclass_where(InterpolationShape, name=etype)
                #if etype == 'hex' or 'quad':
                #    self.ncmsh = etypecls.transfinite(tmsh)

                mesho_order = self._get_order(etype, self.mesho[f'spt_{mname}'].shape[0]) - 1
                self.fnormal[f'spt_{mname}'], self.fcenter[f'spt_{mname}'] = etypecls(mesho_order).pre_calc(self.mesho[f'spt_{mname}'])

                #print(self.fnormal[f'spt_{mname}'].shape, self.fcenter[f'spt_{mname}'].shape)

                # Pre-calculate matrix operators and relevant orthogonal basis
                nspts, neles = self.mesho[f'spt_{mname}'].shape[:2]
                nspts_soln = self.soln[f'soln_{mname}'].shape[0]

                svpts = self._get_std_ele(etype, nspts)

                mesh_op = self._get_mesh_op_vis(etype, nspts, svpts)
                mesh_op = self._get_mesh_op_sln(etype, nspts_soln) @ mesh_op
                #soln_op = self._get_soln_op(etype, nspts, svpts)

                mesh = np.einsum('ij, jkl -> ikl',mesh_op,self.mesho[f'spt_{mname}'])
                #soln = np.einsum('ij, jkl -> ikl',soln_op,self.soln[f'soln_{mname}'])

                #mesh_op_sln = self._get_mesh_op_sln(etype, mesh.shape[0])
                #mesh = np.einsum('ij, jkl -> ikl',mesh_op_sln,mesh)

                poly_space = self._get_polyspace2(etype, nspts_soln, mesh)

                if not np.allclose(np.linalg.matrix_rank(poly_space), np.ones(poly_space.shape[0])*poly_space.shape[1]):
                    import warnings
                    warnings.warn(f"Some polynomial space matrices in shape {etype} are not well-determined, results could be random.")
                    self.mop[f'spt_{mname}'] = np.einsum('ijk,kmi->ijm',self.pinv(poly_space),self.soln[f'soln_{mname}'])

                    print(np.min(np.linalg.cond(poly_space)),np.max(np.linalg.cond(poly_space)))
                else:
                    self.mop[f'spt_{mname}'] = np.einsum('ijk,kmi->ijm',self.inv(poly_space),self.soln[f'soln_{mname}'])


                #self.mop[f'spt_{mname}'] = self.solve(poly_space, soln)

        # this function contents are public
        for name in self.mesh_inf_new:
            etype = name.split('_')[1]
            nspts = self.meshn[name].shape[0]

            self.soln_op[etype] = self._get_soln_op(etype, nspts)
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


            #if j == 16:
            #    print(j, locmiss, loccatch, index)
            if len(locmiss) != 0:
                print(j, locmiss, loccatch, index)

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

    def pinv(self, mat):
        return np.linalg.pinv(mat)

    def solve(self, poly_space, soln):
        # Use numpy solve module
        # If a must be square and of full-rank, i.e., all rows (or, equivalently, columns) must be linearly independent;
        # if either is not true, use lstsq for the least-squares best “solution” of the system/equation.
        # return np.linalg.solve(poly_space, np.rollaxis(soln,2))


        # Use scipy lu decomposition module
        #soln = np.rollaxis(soln,2)
        return np.array([lu_solve(lu_factor(poly_space[eid]), soln[...,eid]) for eid in range(len(poly_space))])
