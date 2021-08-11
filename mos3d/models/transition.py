# Transition model for Mos3D

import random
import copy
import pomdp_py
from mos3d.oopomdp import M3OOState, RobotState, Actions, MotionAction, LookAction, DetectAction

EPSILON=1e-9

class StaticObjectTransitionModel(pomdp_py.TransitionModel):
    """This model assumes the object is static."""
    def __init__(self, objid, epsilon=EPSILON):
        self._objid = objid
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action):
        if next_object_state != state.object_states[next_object_state['id']]:
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def sample(self, state, action):
        """Returns next_object_state"""
        return self.argmax(state, action)

    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        return copy.deepcopy(state.object_states[self._objid])

class RobotTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""
    def __init__(self, gridworld, epsilon=EPSILON, for_env=False):
        """
        for_env (bool): True if this is a robot transition model used by the Environment.
             The only difference is that the "detect" action will mark an object as
             detected if any voxel labeled by that object is within the viewing frustum.
             This differs from agent's RobotTransitionModel which will only mark an object
             as detected if the voxel at the object's pose is within the viewing frustum.
        """
        self._robot_id = gridworld.robot_id
        self._gridworld = gridworld
        self._epsilon = epsilon
        self._for_env = for_env

    def _expected_next_robot_pose(self, state, action):
        # IMPORTANT: If action is LookAction with motion, that means it is a look in a certain
        # direction, specified by `motion` from the default looking direction of -x. Therefore,
        # need to clear the angles of the robot; This is achieved by passing `absolute_rotation`
        # to if_move_by function.
        expected_robot_pose = self._gridworld.if_move_by(state.robot_pose, *action.motion,
                                                         object_poses=state.object_poses,
                                                         valid_pose_func=self._gridworld.valid_pose,
                                                         absolute_rotation=(isinstance(action, LookAction) and action.motion is not None))
        return expected_robot_pose

    def probability(self, next_robot_state, state, action):
        if next_robot_state != self.argmax(state, action):
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def argmax(self, state, action):
        """Returns the most likely next robot_state"""
        if isinstance(state, RobotState):
            robot_state = state
        else:
            robot_state = state.object_states[self._robot_id]
        # using shallow copy because we don't expect object state to reference other objects.
        next_robot_state = copy.deepcopy(robot_state)
        next_robot_state['camera_direction'] = None  # camera direction is only not None when looking

        if isinstance(action, MotionAction):
            # motion action
            next_robot_state['pose'] = self._expected_next_robot_pose(state, action)

        elif isinstance(action, LookAction):
            if action.motion is not None:
                # rotate the robot
                next_robot_state['pose'] = self._expected_next_robot_pose(state, action)
            next_robot_state['camera_direction'] = action.name

        elif isinstance(action, DetectAction):
            # detect;
            object_poses = {objid:state.object_states[objid]['pose']
                            for objid in state.object_states
                            if objid != self._robot_id}
            # the detect action will mark all objects within the view frustum as detected.
            #   (the RGBD camera is always running and receiving point clouds)
            objects = self._gridworld.objects_within_view_range(robot_state['pose'],
                                                                object_poses, volumetric=self._for_env)
            next_robot_state['objects_found'] = tuple(set(next_robot_state['objects_found']) | set(objects))
        return next_robot_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)

class M3TransitionModel(pomdp_py.OOTransitionModel):
    """Object-oriented transition model"""
    def __init__(self, gridworld, epsilon=EPSILON, for_env=False):
        """
        for_env (bool): True if this is a robot transition model used by the Environment.
             see RobotTransitionModel for details.
        """
        self._gridworld = gridworld
        transition_models = {objid: StaticObjectTransitionModel(objid, epsilon=epsilon)
                             for objid in gridworld.target_objects}
        transition_models[gridworld.robot_id] = RobotTransitionModel(gridworld,
                                                                     epsilon=epsilon,
                                                                     for_env=for_env)
        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return M3OOState(self._gridworld.robot_id, oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return M3OOState(self._gridworld.robot_id, oostate.object_states)
