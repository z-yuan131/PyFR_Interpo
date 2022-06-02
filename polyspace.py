from pyfr.polys import get_polybasis
from pyfr.solvers.base import BaseSystem



class interpo_matrix(object):
    def __init__(self, cfg):
        self.dtype = np.float64
        self.cfg = cfg

        # system and elements classes
        self.systemcls = subclass_where(BaseSystem, name=self.cfg.get('solver', 'system'))
        self.elementscls = self.systemcls.elementscls

        self.etypes_div = defaultdict(lambda: self.divisor)
        self.order = self.cfg.getint('solver', 'order')

    def _get_shape(self, name, nspts):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, self.cfg)

    def _get_something(self, name, nspts, pts):
        shape = self._get_shape(name, nspts)
        return shape.upts


    def m11(self, name, mesh):
        for pts in mesh:

            ub = get_polybasis(name, self.order + 1, pts)

            n = max(sum(dd) for dd in ub.degrees)
            ncut = self.cfg.getint('soln-filter', 'cutoff')
            order = self.cfg.getint('soln-filter', 'order')
            alpha = self.cfg.getfloat('soln-filter', 'alpha')

            A = np.ones(125)
            for i, d in enumerate(sum(dd) for dd in ub.degrees):
                if d >= ncut < n:
                    A[i] = np.exp(-alpha*((d - ncut)/(n - ncut))**order)

            return np.linalg.solve(ub.vdm, A[:, None]*ub.vdm).T
