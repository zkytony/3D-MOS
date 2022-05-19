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
from mos3d.oopomdp import Actions, MotionAction, LookAction, DetectAction, RobotState, M3OOState, TargetObjectState
from mos3d.models.transition import RobotTransitionModel, M3TransitionModel, StaticObjectTransitionModel
from mos3d.models.observation import M3ObservationModel, VoxelObservationModel, RobotObservationModel
from mos3d.models.voxel import Voxel
from mos3d.models.policy import simple_path_planning
from mos3d.models.reward import GoalRewardModel
from mos3d.planning.belief.octree_belief import OctreeBelief
from mos3d.planning.belief.octree import DEFAULT_VAL, LOG
from mos3d.planning.belief.belief import M3Belief
from mos3d.planning.agent import M3Agent
import random
from collections import deque
import copy

# We assume there is state abstraction for objects, but not for robots.

"""State abstraction"""
class AbstractM3Belief(M3Belief):

    def mpe(self, res=1):
        object_states = {}
        for objid in self.object_beliefs:
            if objid == self._gridworld.robot_id:
                object_states[objid] = self.object_beliefs[objid].mpe()
            else:
                object_states[objid] = self.object_beliefs[objid].mpe(res=res)
        return M3OOState(self._gridworld.robot_id, object_states)

    def random(self, res=1):
        object_states = {}
        for objid in self.object_beliefs:
            if objid == self._gridworld.robot_id:
                object_states[objid] = self.object_beliefs[objid].random()
            else:
                object_states[objid] = self.object_beliefs[objid].random(res=res)
        return M3OOState(self._gridworld.robot_id, object_states)

    @classmethod
    def from_m3belief(self, m3belief):
        return AbstractM3Belief(m3belief.gridworld, m3belief.object_beliefs)

class AbstractM3Agent(M3Agent):
    def __init__(self,
                 object_state_resolution,
                 gridworld,
                 init_belief,
                 policy_model,
                 transition_model,
                 observation_model,
                 reward_model,
                 name="AbstractM3Agent"):
        super().__init__(gridworld,
                         AbstractM3Belief.from_m3belief(init_belief),
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model, name=name)
        self._obj_res = object_state_resolution

    def sample_belief(self):
        return self.belief.random(res=self._obj_res)


"""Action abstraction"""
class MotionOption(pomdp_py.Option, MotionAction):
    def __init__(self, motion_actions):
        self._motion_actions = motion_actions
        self._cur_index = 0
        self.name = "MotionOption(%d)" % (len(motion_actions))

    def initiation(self, state):
        raise NotImplemented

    def termination(self, state):
        """
        An option terminates when: (1) robot state is at destination.
        (2) ran out of motion actions.
        """
        if isinstance(state, RobotState):
            robot_pose = state.pose
        elif isinstance(state, M3OOState):
            robot_pose = state.robot_pose
            object_poses = state.object_poses
        else:
            raise ValueError("Cannot apply option to state %s" % str(state))
        if self._cur_index == len(self._motion_actions):
            # reset this option so it can be reused;
            self._cur_index = 0
            return True
        else:
            return False

    def sample(self, state):
        action = self._motion_actions[self._cur_index]
        self._cur_index += 1
        return action

    def reset(self):
        self._cur_index = 0

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "[%s | %s]" % (self.name, str(self._motion_actions))


class TwoPointMotionOption(MotionOption):
    def __init__(self, start, dst):
        """
        start (tuple): robot pose at the start.
        dst (tuple): robot pose at the destination.

        Note: Since there is no state abstraction for robot, the start
        and dst poses should be ground resolution level.

        Technically, motion options can be applied to any robot pose.
        """
        self._start = start
        self._dst = dst
        motion_actions = simple_path_planning(start, dst)
        super().__init__(motion_actions)
        self.name = "MotionOption(%s,%s)" % (str(self._start), str(self._dst))

    def initiation(self, state):
        return state.robot_pose[:3] == self._start

    def termination(self, state):
        """
        An option terminates when: (1) robot state is at destination.
        (2) ran out of motion actions.
        """
        if super().termination(state)\
           or robot_pose[:3] == self._dst[:3]:
            self._cur_index = 0
            return True
        else:
            return False

    def __eq__(self, other):
        if not isinstance(other, TwoPointMotionOption):
            return False
        else:
            return self._start == other._start\
                and self._dst == other._dst
    def __hash__(self):
        return hash((self._start, self._dst, len(self._motion_actions)))


class LinearMotionOption(MotionOption):
    """Just go in linear fashion for a bigger step"""
    def __init__(self, direction, step_size):
        """
        direction (str): '+x'
        step_size (int): number of primitive motion actions to take in that direction.
        """
        motion_actions = tuple([Actions.motion_action(direction)] * step_size)
        super().__init__(motion_actions)
        self.name = "LinearMotionOption(%s,%s)" % (str(direction), str(step_size))
        self._direction = direction
        self._step_size = step_size

    def initiation(self, state):
        return True

    def __eq__(self, other):
        if not isinstance(other, LinearMotionOption):
            return False
        else:
            return self._direction == other._direction\
                and self._step_size == other._step_size

    def __hash__(self):
        return hash((self._step_size, self._direction))


class LookOption(pomdp_py.Option, LookAction):
    """Starting simple. Look option is just a look action"""
    def __init__(self, look_action):
        self._look_action = look_action
        self._sampled = False
        self.name = "LookOption(%s)" % (look_action.name)
    def initiation(self, state):
        return True
    def termination(self, state, object_poses={}):
        if self._sampled:
            # reset this option
            self._sampled = False
            return True
        else:
            return False
    def sample(self, state):
        self._sampled = True
        return self._look_action
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "LookOption(%s)" % self._look_action.name
    def __eq__(self, other):
        if not isinstance(other, LookOption):
            return False
        else:
            return self._look_action == other._look_action
    def __hash__(self):
        return hash(self._look_action)


class DetectOption(pomdp_py.Option, DetectAction):
    """Starting simple. Detect option is just a detect action"""
    def __init__(self):
        self._sampled = False
        self.name = "DetectOption"
    def initiation(self, state):
        return True
    def termination(self, state, object_poses={}):
        if self._sampled:
            # reset this option
            self._sampled = False
            return True
        else:
            return False
    def sample(self, state):
        self._sampled = True
        return DetectAction()
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "DetectOption"
    def __eq__(self, other):
        return isinstance(other, DetectOption)
    def __hash__(self):
        return hash("detect")


class AbstractPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, motion_resolution=2, detect_after_look=True):
        # if true, detect can only happen after look.
        self._detect_after_look = detect_after_look
        self._motion_resolution = motion_resolution

        # Right now, the macro action simply takes a bigger step in
        # each of the 6 directions.
        self._motion_options = set({})
        for ax in ['x','y','z']:
            self._motion_options.add(LinearMotionOption("+"+ax, motion_resolution))
            self._motion_options.add(LinearMotionOption("-"+ax, motion_resolution))
        self._look_options = set({LookOption(la) for la in Actions.LOOK_ACTIONS})
        self._detect_option = set({DetectOption()})
        self._all_options = self._motion_options | self._look_options | self._detect_option
        self._all_except_detect = self._motion_options | self._look_options

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(state=state, **kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplemented

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplemented

    def get_all_actions(self, state=None, history=None):
        """note: detect can only happen after look."""
        if state is None:
            raise ValueError("state cannot be None.")
        # Build motion actions in 6 directions of size `motion_resolution`.
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                # again, use shallow copy because no option should reference objects and modify them.
                return copy.copy(self._all_options)
        if self._detect_after_look:
            return copy.copy(self._all_except_detect)
        else:
            return copy.copy(self._all_options)

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

# Abstract robot transition model
class AbstractRobotTransitionModel(RobotTransitionModel):
    """Abstract transition model (only used for planning); ignore the collisions"""
    def __init__(self, agent_belief, gridworld):
        super().__init__(gridworld)
        self._agent_belief = agent_belief

    def _expected_next_robot_pose(self, state, action):
        # IMPORTANT: If action is LookAction with motion, that means it is a look in a certain
        # direction, specified by `motion` from the default looking direction of -x. Therefore,
        # need to clear the angles of the robot; This is achieved by passing `absolute_rotation`
        # to if_move_by function.
        expected_robot_pose = self._gridworld.if_move_by(state.robot_pose, *action.motion,
                                                         object_poses={},
                                                         valid_pose_func=self._gridworld.valid_pose,
                                                         absolute_rotation=(isinstance(action, LookAction) and action.motion is not None))
        return expected_robot_pose

    def update_belief(self, agent_belief):
        if not isinstance(agent_belief, pomdp_py.OOBelief):
            raise ValueError("Object belief needs to be an octree belief")
        self._agent_belief = agent_belief

    def argmax(self, state, action):
        """Returns the most likely next robot_state"""
        if not isinstance(action, DetectAction)\
           or state.object_states[next(iter(state.object_states))]['res'] == 1:
            return super().argmax(state, action)
        else:
            # deal with detection action; This is very important since
            # it will affect the reward.
            objects_found = set(state.robot_state['objects_found'])
            for objid in self._gridworld.target_objects:
                if objid in objects_found:
                    continue
                detected_belief = 0.0
                undetected_belief = 0.0
                abstract_obj_state = state.object_states[objid]
                k = min(abstract_obj_state['res']**3, 10)
                obj_belief = self._agent_belief.object_belief(objid)
                prob_abstract = obj_belief[abstract_obj_state]
                if prob_abstract == 0:
                    continue
                # sample k ground states
                for _ in range(k):
                    ground_obj_pose = obj_belief.random_ground_child_pos(pos=abstract_obj_state['pose'],
                                                                         res=abstract_obj_state['res'])
                    prob_ground = obj_belief._probability(*ground_obj_pose, 1)
                    if self._gridworld.within_view_range(state.robot_pose, ground_obj_pose):
                        detected_belief += prob_ground / prob_abstract
                    else:
                        undetected_belief += prob_ground / prob_abstract
                if detected_belief > undetected_belief:
                    objects_found.add(objid)
            next_robot_state = copy.deepcopy(state.robot_state)
            next_robot_state['camera_direction'] = None
            next_robot_state['objects_found'] = tuple(objects_found)
            return next_robot_state


class AbstractM3TransitionModel(M3TransitionModel, pomdp_py.OOTransitionModel):
    """Object-oriented transition model"""
    def __init__(self, agent_belief, gridworld, epsilon=1e-9):
        """
        for_env (bool): True if this is a robot transition model used by the Environment.
             see RobotTransitionModel for details.
        """
        self._gridworld = gridworld
        transition_models = {objid: StaticObjectTransitionModel(objid, epsilon=epsilon)
                             for objid in gridworld.target_objects}
        transition_models[gridworld.robot_id] =\
            AbstractRobotTransitionModel(agent_belief,
                                         gridworld)
        pomdp_py.OOTransitionModel.__init__(self, transition_models)

    def update_belief(self, agent_belief):
        self.transition_models[self._gridworld.robot_id].update_belief(agent_belief)



"""Observation abstraction"""
class AbstractVoxelObservationModel(pomdp_py.ObservationModel):

    def __init__(self, agent_belief, ground_voxel_observation_model, k=None):
        """Assume object beliefs in `agent_belief` are OctreeBelief objects"""
        self._gvm = ground_voxel_observation_model
        self._objid = self._gvm._objid
        self._gridworld = self._gvm._gridworld
        if not isinstance(agent_belief, pomdp_py.OOBelief):
            raise ValueError("Agent belief needs to be OOBelief")
        self._agent_belief = agent_belief
        self._k = k  # number of samples of object poses

    def update_belief(self, agent_belief):
        if not isinstance(agent_belief, pomdp_py.OOBelief):
            raise ValueError("Object belief needs to be an octree belief")
        self._agent_belief = agent_belief

    def probability(self, observation, next_state, action, **kwargs):
        """
        observation (Voxel)
        Note that the probability is unnormalized.
        """
        raise NotImplementedError("Abstract observation model is currently only used "
                                  "for MCTS-based planning.")

    def sample(self, next_state, action, argmax=False, **kwargs):
        """Returns observation"""
        if isinstance(action, pomdp_py.Option):
            raise ValueError("Does not expect to sample observation given option")
        if not isinstance(action, LookAction):
            # No observation is received when action is not look.
            # Therefore, all voxels are unknown, which translates to
            # empty observation per object.
            return Voxel(None, None)  # voxel has literally no info.

        # sample k times from ground; Ignore occlusions (approximation)
        objclass = self._gridworld.objects[self._objid].objtype
        obj_belief = self._agent_belief.object_belief(self._objid)
        abstract_obj_state = next_state.object_states[self._objid]
        ### Taking 40 samples is good enough for our purpose if the distribution is normal
        # (assuming roughly 95%(+/-15) confidence level; of course, this is an approximation)
        k = self._k if self._k is not None else min(abstract_obj_state['res']**3, 10)
        if abstract_obj_state['res'] == 1:
            return self._gvm.sample(next_state, action, argmax=argmax)

        sampled_obj_poses = {obj_belief.random_ground_child_pos(pos=abstract_obj_state['pose'],
                                                                res=abstract_obj_state['res'])
                             for _ in range(k)}
        # For each sample, sample ground observation, also ignoring occlusions
        count_detect_correct = 0  # d(v) = i
        count_detect_free = 0     # d(v) = Voxel.OTHER
        for si in sampled_obj_poses:
            oostate = M3OOState(self._gridworld.robot_id,
                                {self._gridworld.robot_id: next_state.robot_state,
                                 self._objid: TargetObjectState(self._objid, objclass, si, res=1)})
            oi = self._gvm.sample(oostate, action)
            if oi.label == self._objid:
                count_detect_correct += 1
            elif oi.label == Voxel.OTHER:
                count_detect_free += 1

        abstract_oi = Voxel(abstract_obj_state['pose'], None)
        if count_detect_correct == 0 and count_detect_free == 0:
            abstract_oi.label = Voxel.UNKNOWN  # all samples are outside of the FOV
        elif count_detect_correct > count_detect_free:
            abstract_oi.label = self._objid
        else:
            abstract_oi.label = Voxel.OTHER
        return abstract_oi

    def argmax(self, next_state, action, **kwargs):
        """Returns the most likely observation"""
        return self._sample(next_state, action, argmax=True)

class AbstractM3ObservationModel(M3ObservationModel):
    """This is an observation model where the state can be in a different resolution"""
    def __init__(self,
                 agent_belief,
                 gridworld, epsilon=1e-9,
                 alpha=1000., beta=0., gamma=DEFAULT_VAL, k=10):
        self._gridworld = gridworld
        self._voxel_model = True
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        observation_models = {gridworld.robot_id: RobotObservationModel(gridworld.robot_id)}
        observation_models.update({objid:\
                                   AbstractVoxelObservationModel(
                                       agent_belief,
                                       VoxelObservationModel(objid, gridworld,
                                                             alpha=alpha, beta=beta, gamma=gamma,
                                                             epsilon=epsilon),
                                       k=k)
                                   for objid in gridworld.target_objects})
        pomdp_py.OOObservationModel.__init__(self, observation_models)

    def update_belief(self, agent_belief):
        for objid in self.observation_models:
            if objid != self._gridworld.robot_id:
                self.observation_models[objid].update_belief(agent_belief)
