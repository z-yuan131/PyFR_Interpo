# -*- coding: utf-8 -*-
import numpy as np

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader

class BaseInterpo(object):
    def __init__(self, args):
        # load ini files
        #self.cfg = Inifile(args[0])

        # load mesh and solution files
        self.mesho = NativeReader(args[1])
        self.meshn = NativeReader(args[-1])
        self.soln = NativeReader(args[2])

        # Check solution and mesh are compatible
        if self.mesho['mesh_uuid'] != self.soln['mesh_uuid']:
            raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (args.solnf, args.meshf))

        # Load the configuration and stats files
        self.cfg = Inifile(self.soln['config'])
        self.stats = Inifile(self.soln['stats'])

        # Data file prefix (defaults to soln for backwards compatibility)
        self.dataprefix = self.stats.get('data', 'prefix', 'soln')

        # Get element types and array shapes
        self.mesh_inf_old = self.mesho.array_info('spt')
        self.mesh_inf_new = self.meshn.array_info('spt')
        self.soln_inf = self.soln.array_info(self.dataprefix)

        # Get the number of elements of each type in each partition
        self.mesh_part = self.mesho.partition_info('spt')
        self.mesh_part_new = self.meshn.partition_info('spt')

        # Dimensions
        self.ndims = next(iter(self.mesh_inf_old.values()))[1][2]
        self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # linear elements
        self.lin = [self.mesho.__getitem__([key,'linear']) for key in self.mesh_inf_old]

        # type of meshes
        self.oldname = self.loadname(self.mesho)
        self.newname = self.loadname(self.meshn)
        #print(self.mesh_part_new)


        #print(self.lin)
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
