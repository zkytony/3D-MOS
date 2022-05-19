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

# The experiment is essentially about generating the configurations
# for a bunch of trials, and run all trials as parallel as possible.

from mos3d import *
from mos3d.tests.experiments.runner import *
import random
import math

VIZ = False

def make_domain(n, k, d):
    objcounts = {}
    total_count = 0
    while total_count < k:  # +1 to account for the robot
        objtype = random.choice(OBJECT_MANAGER.all_object_types(get_str=True))
        if objtype == "robot":
            continue
        if objtype not in objcounts:
            objcounts[objtype] = 0
        objcounts[objtype] += 1
        total_count += 1
    robot_camera = "occlusion 45 1.0 0.1 %d" % d

    worldstr = world_config(objcounts=objcounts,
                            robot_camera=robot_camera,
                            size=n)
    return worldstr


def make_trial(trial_name, worldstr, planner, belief_type,
               prior_type="uniform", prior_region_res=1,
               prior_confidence=ALPHA, num_particles=1000,
               alpha=ALPHA, beta=BETA, gamma=DEFAULT_VAL,
               setting=None, porollout_policy=None,
               max_depth=10, discount_factor=0.95, planning_time=1.0,
               exploration_const=50, max_steps=100, max_time=120,
               plot_belief=False, plot_tree=False,
               plot_analysis=False, viz=VIZ, anonymize=True,
               detect_after_look=True, num_simulations=100,
               big=1000, medium=100, small=1):
    model_cfg = model_config(alpha=alpha, beta=beta, gamma=gamma,
                             detect_after_look=detect_after_look,
                             big=big, medium=medium, small=small)
    planner_cfg = planner_config(planner, max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 exploration_const=exploration_const,
                                 setting=setting, rollout_policy=porollout_policy,
                                 num_simulations=num_simulations)
    belief_cfg = belief_config(belief_type, prior_type=prior_type,
                                  prior_region_res=prior_region_res,
                                  prior_confidence=prior_confidence,
                                  num_particles=num_particles)
    exec_cfg = exec_config(max_steps=max_steps, max_time=max_time, plot_belief=plot_belief,
                           plot_tree=plot_tree, plot_analysis=plot_analysis, viz=viz, anonymize=anonymize)
    return M3Trial("%s_%s-%s-%s" % (trial_name, planner, belief_type, prior_type),
                   config={"model_config": model_cfg,
                           "planner_config": planner_cfg,
                           "belief_config": belief_cfg,
                           "world_config": worldstr,
                           "exec_config": exec_cfg})
