import numpy as np
from scipy.spatial import ConvexHull
# from crystal import crystalSymmetry
from orix.crystal_map.phase_list import Phase,PhaseList
from orix.quaternion import Orientation, Rotation
# from plot import plot
# from data import grainBoundary

class grain2d:
    def __init__(self, V : [np.ndarray], poly, ori: [Orientation], CSList: [str], phaseId: [int], phaseMap):
        """ 
        constructor
         
        Input
        V    - n x 2 list of vertices
        poly - cell array of the polyhedrons
        ori  - array of mean orientations
        CSList   - cell array of symmetries
        phaseId  - list of phaseId for each grain
        phaseMap - 
        """
        if V is None:
            return

        self.poly = poly
        self.inclusionId = [len(p) - np.where(p[1:] == p[0])[0][0] - 1 for p in poly]

        if ori is not None and len(ori) > 0:
            self.prop.meanRotation = ori
        else:
            self.prop.meanRotation = [Rotation.nan(len(poly), 1)]

        if CSList is not None and len(CSList) > 0:
            self.CSList = CSList
        else:
            self.CSList = ['notIndexed']

        if phaseId is not None and len(phaseId) > 0:
            self.phaseId = phaseId
        else:
            self.phaseId = np.ones(len(poly))

        if phaseMap is not None and len(phaseMap) > 0:
            self.phaseMap = phaseMap
        else:
            self.phaseMap = np.arange(1, len(self.CSList) + 1)

        self.id = np.arange(1, len(self.phaseId) + 1)
        self.grainSize = np.ones(len(poly))

        if isinstance(V, grainBoundary):  # Grain boundary already given
            self.boundary = V
        else:  # Otherwise, compute grain boundary
            # Compute boundary segments
            F = [poly[i][:len(poly[i]) - self.inclusionId[i] - 1] for i in range(len(poly))]
            F = [item for sublist in F for item in sublist]

            lBnd = [len(poly[i]) - self.inclusionId[i] - 1 for i in range(len(poly))]

            grainIds = [i + 1 for i in range(len(self))] * lBnd
            F, iF, iG = np.unique(np.sort(F, axis=1), axis=0, return_index=True, return_inverse=True)

            F = F[iF]
            grainId = np.zeros(F.shape, dtype=int)
            grainId[:, 0] = [grainIds[iF[i]] for i in range(len(F))]
            col2 = np.ones(iG.shape, dtype=bool)
            col2[iF] = False
            grainId[iG[col2], 1] = [grainIds[i] for i in range(len(F)) if not col2[iF[i]]]
            mori = [Rotation.__neg__(F.shape[0], 1)]
            isNotBoundary = np.all(grainId, axis=1)
            mori[isNotBoundary] = Rotation.__inv__(self.prop.meanRotation[grainId[isNotBoundary, 1]]) * self.prop.meanRotation[grainId[isNotBoundary, 0]]
            self.boundary = grainBoundary(V, F, grainId, np.arange(1, max(grainId) + 1), self.phaseId, mori, self.CSList, self.phaseMap)

    @property
    def meanOrientation(self):
        if len(self) == 0:
            ori = [orientation]
        else:
            ori = orientation(self.prop.meanRotation, self.CS)
            if not all(self.isIndexed):
                ori[~self.isIndexed] = np.nan
        return ori

    @meanOrientation.setter
    def meanOrientation(self, ori):
        if len(self) > 0:
            if isinstance(ori, (int, float)):
                ori = np.array([ori])
            if isinstance(ori, np.ndarray) and np.all(np.isnan(ori)):
                self.prop.meanRotation = Rotation.__neg__(self.prop.meanRotation.shape)
            else:
                self.prop.meanRotation = Rotation(ori)
                self.CS = ori.CS

    @property
    def GOS(self):
        return self.prop.GOS

    @property
    def scanUnit(self):
        return self.boundary.scanUnit

    @scanUnit.setter
    def scanUnit(self, unit):
        self.boundary.scanUnit = unit
        self.innerBoundary.scanUnit = unit

    @property
    def triplePoints(self):
        return self.boundary.triplePoints

    @triplePoints.setter
    def triplePoints(self, tP):
        self.boundary.triplePoints = tP

    @property
    def V(self):
        return self.boundary.V

    @V.setter
    def V(self, V):
        self.boundary.V = V
        self.innerBoundary.V = V

    def size(self, *args):
        return np.array(self.id).size(*args)

    def update(self):
        self.boundary.update(self)
        self.innerBoundary.update(self)
        self.triplePoints.update(self)

    def x(self):
        return self.boundary.x

    def y(self):
        return self.boundary.y

    @property
    def idV(self):
        isCell = [isinstance(p, list) for p in self.poly]
        polygons = list(self.poly)
        polygons = [x for x in polygons if isCell]
        polygons = [list(item) for sublist in polygons for item in sublist]
        return np.unique(np.array(polygons))

    @staticmethod
    def load(fname):
        # Load grain2d from a file
        pass

    def __len__(self):
        return len(self.phaseId)



