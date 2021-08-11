# Reward model for Mos3D

import pomdp_py
from mos3d.oopomdp import Actions, MotionAction, LookAction, DetectAction

class M3RewardModel(pomdp_py.RewardModel):
    def __init__(self, gridworld, big=1000, medium=100, small=1, discount_factor=0.95):
        self.big = big
        self.medium = medium
        self.small = small
        self._gridworld = gridworld
        self._discount_factor = discount_factor

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0
    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # deterministic
        return self._reward_func(state, action, next_state)
    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        return self._reward_func(state, action, next_state)


class GuidedRewardModel(M3RewardModel):
    """
    This is a reward where the agent gets less penalty if it chooses to
    take motion actions that results in objects that have not been detected.

    This reward model can only be used for the setup where an observation
    is received after after motion/detect action.
    """
    def _reward_func(self, state, action, next_state):
        reward = 0

        if action in Actions.MOTION_ACTIONS:
            assert state.robot_state.objects_found == next_state.robot_state.objects_found,\
                    "Error: Motion action leads to difference in detected objects set."
            if next_state.robot_pose == state.robot_pose:
                # If the action is motion action, but the robot did
                # not move successfully, big penalty.
                reward -= self.big
            else:
                next_objects_within_range =\
                    self._gridworld.objects_within_view_range(next_state.robot_pose,
                                                              next_state.object_poses)
                if len(next_objects_within_range - set(next_state.robot_state.objects_found)) > 0:
                    reward -= self.small
                else:
                    reward -= self.medium
        elif action == Actions.DETECT_ACTION:
            next_objects_within_range =\
                self._gridworld.objects_within_view_range(next_state.robot_pose,
                                                          next_state.object_poses)
            new_objects_count = len(next_objects_within_range - set(next_state.robot_state.objects_found))
            if new_objects_count == 0:
                # No new detection. "detect" is a bad action.
                reward -= self.big
            else:
                # Has new detection. Award.
                reward += self.small
                # strategy2: reward += self.small*new_objects_count
        elif action == Actions.STAY_ACTION:
            next_objects_within_range =\
                self._gridworld.objects_within_view_range(next_state.robot_pose,
                                                          next_state.object_poses)
            new_objects_count = len(next_objects_within_range - set(next_state.robot_state.objects_found))
            if new_objects_count > 0:
                reward -= self.small*2
            else:
                reward -= self.big
        return reward



class GoalRewardModel(M3RewardModel):
    """
    This is a reward where the agent gets reward only for detect-related actions.

    This reward model can be used both for the setting where an observation is
    received after every action, or the setting where an observation is received
    only after detect action, in which case motion actions lead to an observation
    where voxels are labeled UNKNOWN.

    Note: Only this model is used in experiments.
    """
    def __init__(self, gridworld, big=1000, medium=100, small=1, discount_factor=0.95,
                 for_env=False):
        super().__init__(gridworld, big=big, medium=medium, small=small, discount_factor=discount_factor)
        self._for_env = for_env

    def _reward_func(self, state, action, next_state):
        reward = 0

        # If the robot has detected all objects
        if len(state.robot_state['objects_found']) == len(self._gridworld.target_objects):
            return 0  # no reward or penalty; the task is finished.

        if isinstance(action, MotionAction):
            reward = reward - self.small - action.distance_cost
        elif isinstance(action, LookAction):
            reward = reward - self.small
        elif isinstance(action, DetectAction):

            if state.robot_state['camera_direction'] is None:
                # The robot didn't look before detect. So nothing is in the field of view.
                reward -= self.big
            else:
                # transition function should've taken care of the detection.
                new_objects_count = len(set(next_state.robot_state.objects_found) - set(state.robot_state.objects_found))
                if new_objects_count == 0:
                    # No new detection. "detect" is a bad action.
                    reward -= self.big
                else:
                    # Has new detection. Award.
                    reward += self.big
                    # strategy2: reward += self.small*new_objects_countc
        return reward
