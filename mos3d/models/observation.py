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

# Observation model for Mos3D
#
# As defined in the OOPOMDP for Mos3D, an observation looks like
#
#   { objid : [ ... voxels ...] }, robot_pose (x,y,z,qx,qy,qz,qw) }
#
# This observation can be factorerd by objects, or objects and voxels.
# We define one observation model for each.
#
# Note that all observations sampled from these models needs to be
# hashable. That means that are tuples.

import pomdp_py
import random
import math
import copy
import mos3d.util as util
from mos3d.oopomdp import LookAction, DetectAction
from mos3d.planning.belief.octree import DEFAULT_VAL, LOG
from mos3d.models.voxel import Voxel, FovVoxels
from mos3d.models.world.sensor_model import FrustumCamera

EPSILON=1e-9


class OOObservation(pomdp_py.OOObservation):
    """See notes on 12/8/2019

    OOObservation is an Observation that can be factored by objects."""
    T_VOLUME = "volume"
    T_VOXEL = "voxel"
    def __init__(self, voxels, ovtype):
        """
        voxels: dictionary pose->Voxel or objid->Voxel, or FovVoxels
        ovtype: 'volume' if voxels is of former format;
                'voxel' if voxels is of latter format"""
        self._hashcode = hash(frozenset(voxels.items()))
        if not isinstance(voxels, FovVoxels):
            voxels = FovVoxels(voxels)
        self._fovvoxels = voxels
        self._ovtype = ovtype
    @property
    def voxels(self):
        return self._fovvoxels.voxels
    def __str__(self):
        return "Observation(%s)" % (self.voxels)
    def __repr__(self):
        return self.__str__()
    def __hash__(self):
        return self._hashcode
    def __eq__(self, other):
        if not isinstance(other, OOObservation):
            return False
        else:
            return self.voxels == other.voxels

    def factor(self, next_state, action, *params):
        if self._ovtype == "volume":
            return self.factor_object_observation(next_state, *params)
        else:
            return self.factor_object_voxel_observation(next_state, *params)

    def factor_object_voxel_observation(self, next_state, *params):
        # This requires `self.voxels` to be a map from object id to voxel
        return {objid: self.voxels[objid]
                for objid in self.voxels}

    def factor_object_observation(self, next_state, *params):
        # Free voxels are shared by all object observations
        objids = next_state.object_states.keys() - set({next_state.robot_id})

        free_voxels = {}
        obj_voxels = {}  # observed object's voxels
        for voxel_pose in self.voxels:
            v = self.voxels[voxel_pose]
            if v.label == Voxel.UNKNOWN:
                continue
            if v.label == Voxel.FREE:
                free_voxels[v.pose] = Voxel(v.pose, v.label)
                free_voxels[v.pose].label = Voxel.OTHER
            else:
                if v.label not in obj_voxels and v.label in objids:
                    obj_voxels[v.label] = {}
                obj_voxels[v.label][v.pose] = Voxel(v.pose, v.label)
        obj_observations = {}
        for i in objids:
            voxels = free_voxels.copy()
            if i in obj_voxels: # If i is observed
                voxels.update(obj_voxels[i])
            for j in obj_voxels:
                if i != j:
                    voxels.update({voxel_pose:Voxel(voxel_pose, Voxel.OTHER)
                                   for voxel_pose in obj_voxels[j]})
            obj_observations[i] = FovVoxels(voxels)

        obj_observations[next_state.robot_id] = next_state.robot_pose
        return obj_observations


    def factor_voxel_observation(cls, observation, next_state, *params):
        """
        observation: constructed by merge_voxel_observations
        """
        observations = {}
        for voxel_pose in observation.voxels:
            voxel = observation.voxels[voxel_pose]
            observations[voxel.label] = voxel
        observations[next_state.robot_id] = next_state.robot_pose
        return observations


# Object observation model
class ObjectObservationModel(pomdp_py.ObservationModel):

    """
    The object observation model, assuming an object is contained
    only in one voxel. Then P(Mi|s',a) = P(v|s',a). If object i
    is observable, then P(v|s',a) = P(lv=i|s',a,pv) = alpha,
    P(lv=not i|s',a,pv) = beta. If i is not observable, then
    P(v|s',a) = gamma regardless of lv.
    """

    def __init__(self, objid, gridworld, epsilon=EPSILON,
                 alpha=1000., beta=0., gamma=DEFAULT_VAL):
        self._objid = objid
        self._gridworld = gridworld
        self._epsilon = epsilon
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def normalized(self):
        return False

    def probability(self, observation, next_state, action, **kwargs):
        """
        observation (ObjectObservation)
        Note that the probability is unnormalized."""
        # observation = Mi
        if not isinstance(action, LookAction):
            return self._gamma
        else:
            object_pose = next_state.object_states[self._objid]['pose']
            if object_pose in observation.voxels:
                # object is observable
                if observation.voxels[object_pose].label == self._objid:
                    return self._alpha
                else:
                    return self._beta
            else:
                return self._gamma

    def sample(self, next_state, action, **kwargs):
        """Returns observation"""
        if not isinstance(action, LookAction):
            # No observation is received when action is not look.
            # Therefore, all voxels are unknown, which translates to
            # empty observation per object.
            return FovVoxels({})

        # Obtain observation factored for this object (i.e. contains only i and not_i cells)
        observation = self._gridworld.provide_observation(next_state.robot_pose,
                                                          next_state.object_poses)
        object_observation = observation.factor(next_state, action)[self._objid]
        voxels = {voxel_pose: Voxel(voxel_pose, object_observation.voxels[voxel_pose].label)
                  for voxel_pose in object_observation.voxels}

        # See if object is observable
        object_pose = next_state.object_states[self._objid]['pose']
        if object_pose in object_observation.voxels:
            # Yes, object is observable. Then with alpha/(alpha+beta) prob., will
            # observe it; otherwise, won't.
            if FrustumCamera.sensor_functioning(self._alpha, self._beta, LOG):
                voxels[object_pose].label = self._objid
                return FovVoxels(voxels)
            else:
                return FovVoxels(voxels)
        else:
            # Object not in FOV. The observation is then uniformly distributed,
            # determined by the action and state; For simplicity, just return the
            # voxels with all OTHER (generating an actual uniform observation is
            # not easy and not useful for planning).
            return FovVoxels(voxels)

    def argmax(self, next_state, action, **kwargs):
        """Returns the most likely observation"""
        observation = self._gridworld.provide_observation(next_state.robot_pose,
                                                          next_state.object_poses)
        return observation.factor(next_state, action)[self._objid]
##################################



# Object observation model for planning
class VoxelObservationModel(ObjectObservationModel):
    # This is just a random idea. Probably not going to work,
    # since the entire observation space is still very huge.
    """The sampled observations will not be a set of voxels,
    but rather simply one voxel, with pv = si, and label lv
    equal to one of three possible values."""

    def probability(self, observation, next_state, action, **kwargs):
        """
        observation (Voxel)
        Note that the probability is unnormalized."""
        # observation = v
        if not isinstance(action, LookAction):
            return self._gamma
        else:
            if observation.label == Voxel.UNKNOWN:
                return self._gamma  # not_in_view
            else:
                if observation.pose != next_state.object_states[self._objid]['pose']:
                    # observation's pose is not corresponding to the object pose.
                    return self._epsilon  # almost zero
                elif observation.label == self._objid:
                    return self._alpha  # i
                else:
                    return self._beta  # not_i

    def sample(self, next_state, action, argmax=False, **kwargs):
        """Returns observation"""
        if not isinstance(action, LookAction):
            # No observation is received when action is not look.
            # Therefore, all voxels are unknown, which translates to
            # empty observation per object.
            return Voxel(None, None)  # voxel has literally no info.

        # See if object is observable
        object_pose = next_state.object_states[self._objid]['pose']
        voxel = Voxel(object_pose, Voxel.UNKNOWN)
        if self._gridworld.observable(self._objid, next_state.robot_pose, next_state.object_poses, next_state.situation):
            # Yes, object is observable. Then with alpha/(alpha+beta) prob., will
            # observe it; otherwise, won't.
            if argmax and self._alpha > self._beta:
                voxel.label = self._objid
            elif FrustumCamera.sensor_functioning(self._alpha, self._beta, LOG):
                voxel.label = self._objid
            else:
                voxel.label = Voxel.OTHER  # can be anything other than objid or unknown; doesn't affect its probability
            return voxel
        else:
            # Object not in FOV. The observation is then uniformly distributed;
            # For simplicity, just return the voxels with all OTHER (generating
            # an actual uniform observation is not easy and not useful for planning).
            return voxel

    def argmax(self, next_state, action, **kwargs):
        """Returns the most likely observation"""
        return self._sample(next_state, action, argmax=True)


class RobotObservationModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""
    def __init__(self, robot_id, epsilon=EPSILON):
        self._robot_id = robot_id
        self._epsilon = epsilon

    def probability(self, observation, next_state, action):
        """observation (tuple) robot pose"""
        if observation == next_state.robot_pose:
            return 1.0 - self._epsilon
        else:
            return self._epsilon

    def sample(self, next_state, action, **kwargs):
        """Returns observation"""
        self.argmax(next_state, action)

    def argmax(self, next_state, action, **kwargs):
        """Returns the most likely observation"""
        return next_state.robot_pose


class M3ObservationModel(pomdp_py.OOObservationModel):
    """Object-oriented transition model"""
    def __init__(self, gridworld, epsilon=EPSILON,
                 voxel_model=True, alpha=1000., beta=0., gamma=DEFAULT_VAL):
        self._gridworld = gridworld
        observation_models = {gridworld.robot_id: RobotObservationModel(gridworld.robot_id)}
        self._voxel_model = voxel_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if self._voxel_model:
            observation_models.update({objid: VoxelObservationModel(objid, gridworld,
                                                alpha=alpha, beta=beta, gamma=gamma,
                                                epsilon=epsilon)
                                       for objid in gridworld.target_objects})
        else:
            observation_models.update({objid: ObjectObservationModel(objid, gridworld,
                                    alpha=alpha, beta=beta, gamma=gamma,
                                    epsilon=epsilon)
                           for objid in gridworld.target_objects})
        pomdp_py.OOObservationModel.__init__(self, observation_models)

    def sample(self, next_state, action, argmax=False, **kwargs):
        if not isinstance(action, LookAction):
            return OOObservation({}, None)

        factored_observations = super().sample(next_state, action, argmax=argmax)

        if self._voxel_model:
            return M3ObservationModel.merge_voxel_observations(factored_observations, next_state, action)
        else:
            volume = self._gridworld.get_frustum_poses(next_state.robot_pose)
            return M3ObservationModel.merge_observations(factored_observations, next_state, action, volume)


    @classmethod
    def merge_voxel_observations(cls, observations, next_state, action, **kwargs):
        """
        observations: objid -> Voxel; Generated using the VoxelObservationModel.
        Just return an Observation(robot_pose, {vpose -> voxel}). Of course,
        the voxels set in this observation is NOT the field of view.
        """
        # if not isinstance(action, LookAction):
        #     robot_pose = None
        # else:
        #     robot_pose = next_state.robot_pose
        return OOObservation({objid: observations[objid]
                              for objid in observations
                              if objid != next_state.robot_id},
                              OOObservation.T_VOXEL)


    @classmethod
    def merge_observations(cls, observations, next_state, action, volume):
        voxels = {}
        for x,y,z in volume:
            p = (x,y,z)
            unknown = True
            for objid in observations:
                if objid == next_state.robot_id:
                    continue  # skip robot pose in the merged observation (no need)
                if p in observations[objid].voxels:
                    if observations[objid].voxels[p].label == objid:
                        voxels[p] = Voxel(p, objid)
                    else:
                        if p not in voxels:
                            voxels[p] = Voxel(p, Voxel.FREE)
                    unknown = False
            if unknown:
                voxels[p] = Voxel(p, Voxel.UNKNOWN)
        # # Assuming perfect observation of robot pose.
        # if not isinstance(action, LookAction):
        #     robot_pose = None
        # else:
        #     robot_pose = next_state.robot_pose
        return OOObservation(voxels, OOObservation.T_VOLUME)
