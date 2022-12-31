# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.util import subclass_where
from pyfr.shapes import BaseShape
from pyfr.quadrules import get_quadrule

class Base(object):
    def __init__(self, args):
        # load mesh and solution files
        mesh_old = NativeReader(args[0])
        mesh_new = NativeReader(args[2])
        soln_old = NativeReader(args[1])
        self.dir_name = args[-1]

        # Check solution and mesh are compatible
        if mesh_old['mesh_uuid'] != soln_old['mesh_uuid']:
            #raise RuntimeError('Solution "%s" was not computed on mesh "%s"' % (args.solnf, args.meshf))
            raise RuntimeError('Solution was not computed on mesh ')

        # Load the configuration and stats files
        self.cfg = Inifile(soln_old['config'])
        self.stats = Inifile(soln_old['stats'])
        self.order = self.cfg.getint('solver','order')
        self.dtype = np.dtype(self.cfg.get('backend','precision')).type
        self.uuid = mesh_new['mesh_uuid']

        # Data file prefix (defaults to soln for backwards compatibility)
        self.dataprefix = self.stats.get('data', 'prefix', 'soln')

        # Get element types and array shapes
        self.mesh_inf_old = mesh_old.array_info('spt')
        self.mesh_inf_new = mesh_new.array_info('spt')
        self.soln_inf = soln_old.array_info(self.dataprefix)

        # Get the number of elements of each type in each partition
        self.mesh_part_old = mesh_old.partition_info('spt')
        self.mesh_part_new = mesh_new.partition_info('spt')

        # Dimensions
        self.ndims = next(iter(self.mesh_inf_old.values()))[1][2]
        self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # linear elements
        #self.lin = [self.mesho.__getitem__([key,'linear']) for key in self.mesh_inf_old]

        self._search_pts(mesh_old, mesh_new, soln_old)

    def _search_pts(self, mesh_old, mesh_new, soln_old):
        # Process that only store mesh rather than other bc connectivity info
        self.mesh_old = list()
        self.mesh_old_vis = list()
        self.mesh_new = list()
        self.mesh_trans = list()
        self.mesh_op = defaultdict()
        self.soln = list()
        self.mesh_new_info = list()


        # Use a strictly interior point set
        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }

        for key in mesh_new:
            if 'spt' in key.split('_'):
                _,etype,part = key.split('_')

                # Get Operators
                nspts = mesh_new[key].shape[0]
                self.nwmsh_ord = order = self._get_order(etype, nspts) - 1
                upts = get_quadrule(etype, qrule_map[etype], nspts).pts

                mesh_op = self._get_ops(nspts, etype, upts, nspts, order)
                # Do interpolaions
                self.mesh_new.append(np.einsum('ij, jkl -> ikl',mesh_op,mesh_new[key]))
                self.mesh_new_info.append((etype, part))

        for key in mesh_old:
            if 'spt' in key.split('_'):
                _,etype,part = key.split('_')
                soln_name = f'{self.dataprefix}_{etype}_{part}'
                nupts = soln_old[soln_name].shape[0]

                # Get Operators
                nspts = mesh_old[key].shape[0]
                upts = get_quadrule(etype, qrule_map[etype], nupts).pts

                mesh_op = self._get_ops(nspts, etype, upts, nupts, self.order)
                mesh_op_vis = self._get_vis_op(nspts, etype, self.order)

                self.mesh_old.append(np.einsum('ij, jkl -> ikl',mesh_op,mesh_old[key]))
                self.mesh_old_vis.append(np.einsum('ij, jkl -> ikl',mesh_op_vis,mesh_old[key]))
                self.mesh_trans.append(upts)

                self.soln.append(soln_old[soln_name])

    # Operators
    def _get_ops(self, nspts, etype, upts, nupts, order):

        svpts = self._get_std_ele(etype, nspts, order)
        mesh_op = self._get_mesh_op_vis(etype, nspts, svpts)

        # Convert vis points to solution pts
        mesh_op = self._get_mesh_op_sln(etype, nupts, upts) @ mesh_op
        return mesh_op

    def _get_vis_op(self, nspts, etype, order):
        svpts = self._get_std_ele(etype, nspts, order)
        mesh_op = self._get_mesh_op_vis(etype, nspts, svpts)
        return mesh_op

    def _get_shape(self, name, nspts, cfg):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, cfg)

    def _get_std_ele(self, name, nspts, order):
        return self._get_shape(name, nspts, self.cfg).std_ele(order)

    def _get_mesh_op_vis(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    def _get_mesh_op_sln(self, name, nspts, upts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(upts).astype(self.dtype)

    def _get_npts(self, name, order):
        return self._get_shape(name, 0, self.cfg).nspts_from_order(order)

    def _get_order(self, name, nspts):
        return self._get_shape(name, nspts, self.cfg).order_from_nspts(nspts)
