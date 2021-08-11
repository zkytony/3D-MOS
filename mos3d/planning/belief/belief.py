import pomdp_py
from mos3d.oopomdp import M3OOState
from mos3d.planning.belief.octree_belief import update_octree_belief, OctreeBelief

class M3Belief(pomdp_py.OOBelief):
    def __init__(self, gridworld, object_beliefs):
        """
        object_beliefs (objid -> GenerativeDistribution)
            (includes robot)
        """
        super().__init__(object_beliefs)
        self._gridworld = gridworld

    def mpe(self, **kwargs):
        return M3OOState(self._gridworld.robot_id,
                         pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        return M3OOState(self._gridworld.robot_id,
                         pomdp_py.OOBelief.random(self, **kwargs).object_states)

    @property
    def gridworld(self):
        return self._gridworld
