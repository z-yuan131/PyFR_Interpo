# -*- coding: utf-8 -*-
import numpy as np

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader

class BaseInterpo(object):
    def __init__(self, args):
        # load ini files
        #self.cfg = Inifile(args[0])

        # load mesh and solution files
        self.mesh = NativeReader(args[1])
        self.soln = NativeReader(args[2])

        # Check solution and mesh are compatible
        if self.mesh['mesh_uuid'] != self.soln['mesh_uuid']:
            raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (args.solnf, args.meshf))

        # Load the configuration and stats files
        self.cfg = Inifile(self.soln['config'])
        self.stats = Inifile(self.soln['stats'])

        # Data file prefix (defaults to soln for backwards compatibility)
        self.dataprefix = self.stats.get('data', 'prefix', 'soln')

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info('spt')
        self.soln_inf = self.soln.array_info(self.dataprefix)

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]
        self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # linear elements
        self.lin = [self.mesh.__getitem__([key,'linear']) for key in self.mesh_inf]


        #print(self.lin)
