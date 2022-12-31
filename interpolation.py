# -*- coding: utf-8 -*-
from collections import defaultdict
from mpi4py import MPI
import numpy as np

from pyfr.util import subclasses
from pyfr.shapes import BaseShape
from base import Base

class Interpolationcls(Base):
    def __init__(self, argv):
        super().__init__(argv)
        self.argv = argv



    def mainproc(self):
        print('-----------------------------\n')

        # Preparation for parallel this code
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        outpts = list()

        # For each sample point find our nearest search location
        for nelet in self.mesh_new:
            # Get shape of input new mesh array
            nupts, neles = nelet.shape[:2]
            # Reshape and input to algorithm to do finding closest point process
            pts = nelet.reshape(-1, self.ndims)
            closest = self._closest_pts(self.mesh_old, pts)

            # Sample points we're responsible for, grouped by element type + rank
            elepts = [[] for i in range(len(self.mesh_inf_old))]
            # For each sample point find our nearest search location
            for i, (dist, etype, (uidx, eidx)) in enumerate(closest):
                elepts[etype].append((i, eidx, self.mesh_trans[etype][uidx]))

            # Refine
            outpts.append(self._refine_pts(elepts, pts).reshape(nupts,
                                            neles, self.nvars).swapaxes(1,2))

        self._flash_to_disk(outpts)


    def _closest_pts(self, epts, pts):
        # Use brute force to find closest pts
        yield from self._closest_pts_bf(epts, pts)
        """More methods are on the way"""

    def _closest_pts_bf(self, epts, pts):
        for p in pts:

            # Compute the distances between each point and p
            dists = [np.linalg.norm(e - p, axis=2) for e in epts]

            # Get the index of the closest point to p for each element type and mpi rank
            amins = [np.unravel_index(np.argmin(d), d.shape) for d in dists]

            # Dereference to get the actual distances
            dmins = [d[a] for d, a in zip(dists, amins)]

            # Find the minimum across all element types and mpi ranks
            yield min(zip(dmins, range(len(epts)), amins))

    def _refine_pts(self, elepts, pts):
        # Use visualization points to do refinement to improve stability of the code
        elelist = self.mesh_old_vis
        slnlist = self.soln
        ptsinfo = []

        # Mapping from nupts to element type
        ordplusone = self.order + 1
        etype_map = {
            ordplusone**2: 'quad',
            ordplusone*(ordplusone+1)/2: 'tri',
            ordplusone**3: 'hex',
            ordplusone**2*(ordplusone+1)/2: 'pri',
            ordplusone*(ordplusone+1)*(2*ordplusone+1)/6: 'pyr',
            ordplusone*(ordplusone+1)*(ordplusone+2)/6: 'tet'
        }

        # Get basis class map
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        # Loop over all the points for each element type
        for etype, (eles, epts, soln) in enumerate(zip(elelist, elepts, slnlist)):
            if not epts:
                continue

            idx, eidx, tlocs = zip(*epts)
            spts = eles[:, eidx, :]
            eletype = etype_map.get(len(spts))
            plocs = [pts[i] for i in idx]

            # Use Newton's method to find the precise transformed locations
            basis = basismap[eletype](len(spts), self.cfg)
            ntlocs, nplocs = self._plocs_to_tlocs(basis.sbasis, spts, plocs,
                                             tlocs)

            # Form the corresponding interpolation operators
            ops = basis.ubasis.nodal_basis_at(ntlocs)

            # Do interpolation on the corresponding elements
            new_sln = np.einsum('ij,jki -> ik', ops, soln[:, :, eidx])
            # Append index and solution of each point
            ptsinfo.extend(info for info in zip(idx, new_sln))

        # Resort to the original index
        ptsinfo.sort()

        return np.array([new_sln for idx, new_sln in ptsinfo])


    def _plocs_to_tlocs(self, sbasis, spts, plocs, tlocs):
        plocs, itlocs = np.array(plocs), np.array(tlocs)

        # Set current tolerance
        tol = 10e-12

        # Evaluate the initial guesses
        iplocs = np.einsum('ij,jik->ik', sbasis.nodal_basis_at(itlocs), spts)

        # Iterates
        kplocs, ktlocs = iplocs.copy(), itlocs.copy()

        # Apply maximum ten iterations of Newton's method
        for k in range(10):
            # Get Jacobian operators
            jac_ops = sbasis.jac_nodal_basis_at(ktlocs)
            # Solve from ploc to tloc
            kjplocs = np.einsum('ijk,jkl->kli', jac_ops, spts)
            ktlocs -= np.linalg.solve(kjplocs, kplocs - plocs)
            # Transform back to ploc
            ops = sbasis.nodal_basis_at(ktlocs)
            np.einsum('ij,jik->ik', ops, spts, out=kplocs)

            # Apply check routine after three iterations of Newton's method
            if k > 2:
                kdists = np.linalg.norm(plocs - kplocs, axis=1)
                index = np.where(kdists > tol)[0]
                if len(index) == 0:
                    break
            if k == 49:
                """Currently only precise location is acceptable"""
                raise RuntimeError('warning: failed to apply Newton Method')
                # Compute the initial and final distances from the target location
                idists = np.linalg.norm(plocs - iplocs, axis=1)

                # Replace any points which failed to converge with their initial guesses
                closer = np.where(idists < kdists)
                ktlocs[closer] = itlocs[closer]
                kplocs[closer] = iplocs[closer]


        return ktlocs, kplocs

    def _flash_to_disk(self, outpts):
        # Prepare data for output
        self.cfg.set('solver','order',self.nwmsh_ord)

        import h5py

        f = h5py.File(f'{self.dir_name}','w')
        f['config'] = self.cfg.tostr()
        f['mesh_uuid'] = self.uuid
        f['stats'] = self.stats.tostr()

        for id, (etype, part) in enumerate(self.mesh_new_info):
            name = f'{self.dataprefix}_{etype}_{part}'
            f[name] = outpts[id]
        f.close()
