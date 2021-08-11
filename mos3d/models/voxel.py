

class Voxel:
    FREE = "free"
    OTHER = "other" #i.e. not i (same as FREE but for object observation)
    UNKNOWN = "unknown"
    def __init__(self, pose, label):
        self._pose = pose
        self._label = label
    @property
    def pose(self):
        return self._pose
    @property
    def label(self):
        return self._label
    @label.setter
    def label(self, val):
        self._label = val
    def __str__(self):
        if self._pose is None:
            return "(%s, %s)" % (None, self._label)
        else:
            return "(%d, %d, %d, %s)" % (*self._pose, self._label)
    def __repr__(self):
        return self.__str__()
    def __hash__(self):
        return hash((*self._pose, self._label))
    def __eq__(self, other):
        if not isinstance(other, Voxel):
            return False
        else:
            return self._pose == other.pose\
                and self._label == other.label


class FovVoxels:

    """Voxels in the field of view."""
    

    def __init__(self, voxels):
        """
        voxels: dictionary (x,y,z)->Voxel, or objid->Voxel
                If this is the unfactored observation, then there are UNKNOWN
                voxels in this set. Otherwise, voxels can either be labeled i or OTHER,
        """
        self._voxels = voxels

    def __contains__(self, item):
        if type(item) == tuple  or type(item) == int:
            return item in self._voxels
        elif isinstance(item, Voxel):
            return item.pose in self._voxels\
                and self._voxels[item.pose].label == item.label
        else:
            return False

    def __getitem__(self, item):
        if item not in self:
            raise ValueError("%s is not contained in this FovVoxels object." % str(item))
        else:
            if type(item) == tuple or type(item) == int:
                return self._voxels[item]
            else:  # Must be Voxel
                return self._voxels[item.pose]

    @property
    def voxels(self):
        return self._voxels

    def __eq__(self, other):
        if not isinstance(other, FovVoxels):
            return False
        else:
            return self._voxels == other.voxels
