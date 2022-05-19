# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
