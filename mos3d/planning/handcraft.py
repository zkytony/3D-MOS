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

from pomdp_py import Planner, BeliefDistribution
from mos3d.hierarchical.env import ACTION_NAMES_REVERSE
import mos3d.util as util
import random

def safe_shift(pose, ranges):
    x,y,z = pose[:3]
    for i in range(3):
        final_pose = None
        shifted_pose = list(pose[:3])
        shifted_pose[i] = pose[i] + 1
        if util.in_region(shifted_pose, ranges):
            final_pose = tuple(shifted_pose + list(pose[3:]))
        else:
            shifted_pose[i] = pose[i] - 1
            if util.in_region(shifted_pose, ranges):
                final_pose = tuple(shifted_pose + list(pose[3:]))
        if final_pose is not None:
            return final_pose
    return pose

def compute_move_actions(pose1, pose2, motion_model="AXIS"):
    x1,y1,z1 = pose1[:3]
    x2,y2,z2 = pose2[:3]
    diffx = x2 - x1
    diffy = y2 - y1
    diffz = z2 - z1

    actions = []

    if diffx != 0:
        dx_sign = "+" if diffx > 0 else "-"
        actions += [ACTION_NAMES_REVERSE[motion_model][dx_sign + "x"]]*abs(diffx)
    if diffy != 0:
        dy_sign = "+" if diffy > 0 else "-"
        actions += [ACTION_NAMES_REVERSE[motion_model][dy_sign + "y"]]*abs(diffy)
    if diffz != 0:
        dz_sign = "+" if diffz > 0 else "-"
        actions += [ACTION_NAMES_REVERSE[motion_model][dz_sign + "z"]]*abs(diffz)
    return actions


class HandCraftedPlanner(Planner):

    def __init__(self, env, oopomdp, num_particles=2000):
        self.visited_poses = {}
        self._oopomdp = oopomdp
        self._env = env  # for convenience. Never access groundtruth.
        self._pending_actions = []
        self._moving = False
        self._num_particles = num_particles

    def plan_next_action(self):
        robot_id = self._env.gridworld.robot_id
        if hasattr(self._oopomdp.cur_belief, "get_distribution"):
            robot_state = self._oopomdp.cur_belief.get_distribution(robot_id).mpe()
        else:
            robot_state = self._oopomdp.cur_belief.distribution.mpe().get_object_state(robot_id)
        if len(self._pending_actions) == 0:

            if self._moving:
                # Just reached the previous desired robot pose. Now, start rotating
                self._pending_actions = [
                    ACTION_NAMES_REVERSE[self._env.motion_model]["+thx"],
                    ACTION_NAMES_REVERSE[self._env.motion_model]["+thx"],
                    ACTION_NAMES_REVERSE[self._env.motion_model]["-thx"],
                    ACTION_NAMES_REVERSE[self._env.motion_model]["+thy"],
                    ACTION_NAMES_REVERSE[self._env.motion_model]["+thy"],
                    ACTION_NAMES_REVERSE[self._env.motion_model]["-thy"],
                    ACTION_NAMES_REVERSE[self._env.motion_model]["+thz"],
                    ACTION_NAMES_REVERSE[self._env.motion_model]["+thz"],
                    ACTION_NAMES_REVERSE[self._env.motion_model]["-thz"]
                ]
                self._moving = False
            else:
                # Just finished rotating

                # randomly pick an undetected object
                undetected = set(self._env.gridworld.objects.keys()) - set(robot_state["observed"])
                objid = random.choice(tuple(undetected))

                # pick the MPE pose in the distribution
                if hasattr(self._oopomdp.cur_belief, "get_distribution"):
                    obj_mpe_pose = self._oopomdp.cur_belief.get_distribution(objid).mpe()["pose"]
                else:
                    obj_mpe_pose = self._oopomdp.cur_belief.distribution.mpe().get_object_state(objid)["pose"]
                next_robot_pose = safe_shift(obj_mpe_pose, self._oopomdp.ground_search_region())

                self._pending_actions = compute_move_actions(robot_state["pose"], next_robot_pose,
                                                             motion_model=self._env.motion_model)
                if len(self._pending_actions) == 0:
                    self._pending_actions.append("nothing")
                self._moving = True

        # If not moving,
        # if not self._moving:
        # check if it's a good idea to "detect"
        observation = self._oopomdp.observation_func(self._env.state, "detect")
        for objid, cube_poses in observation:
            if len(cube_poses) == 0:
                continue
            if objid not in robot_state["observed"]:
                self._pending_actions.append("detect")
                print("****************************DETECT!")
                break
        action = self._pending_actions[-1]
        return action

    def update(self, real_action, real_observation):
        if real_action == self._pending_actions[-1]:
            self._pending_actions.pop()
        else:
            print("**IS THIS EXPECTED?**")

    @property
    def params(self):
        return {"num_particles":self._num_particles}
