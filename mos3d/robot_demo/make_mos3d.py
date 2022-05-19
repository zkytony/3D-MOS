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

# Creates an instance of MOS3D problem.
from mos3d.tests.experiments.runner import *
from mos3d import *
from mos3d.tests.trial import RunStep, ExecuteAction, print_step_info
import time

##### Change configuration here #####
worldstr=\
"""
16
16
16

cube 0 5 1
cube 5 5 1
cube 10 2 4 obstacle
---
robot 2 0 0 0 0 0 occlusion 45 1.0 0.1 10
"""

# Observation parameters
alpha =1e5
beta = 0
gamma = 1

# action parameters
detect_after_look=True

# reward parametrs
big = 1000
medium = 100
small = 1

# planning parameters
max_depth = 10
discount_factor = 0.99
planning_time = 3.0
exploration_const = 1000
setting = [(1,1,max_depth), (2,2,max_depth), (4,4,max_depth)]# for hierarchical planning

# execution config
max_steps = 500
max_time = 120
plot_belief = True
plot_tree = False
plot_analysis = False
viz = True
anonymize = True


##### Create an MOS3D Instance, with Hierarchical Planner.
model_cfg = model_config(alpha=alpha, beta=beta, gamma=gamma,
                         detect_after_look=detect_after_look,
                         big=big, medium=medium, small=small)
planner_cfg = planner_config("hierarchical", max_depth=max_depth,
                             discount_factor=discount_factor,
                             planning_time=planning_time,
                             exploration_const=exploration_const,
                             setting=setting)
exec_cfg = exec_config(max_steps=max_steps, max_time=max_time, plot_belief=plot_belief,
                       plot_tree=plot_tree, plot_analysis=plot_analysis, viz=viz, anonymize=anonymize)

# grid world search space
gridworld, init_state = parse_worldstr(worldstr)

# OOPOMDP models
Ov = M3ObservationModel(gridworld, **model_cfg['O'], voxel_model=True)
Om = M3ObservationModel(gridworld, **model_cfg['O'], voxel_model=False)
T = M3TransitionModel(gridworld, **model_cfg['T'])
Tenv = M3TransitionModel(gridworld, for_env=True, **model_cfg['T'])
R = GoalRewardModel(gridworld, **model_cfg['R'])
Renv = GoalRewardModel(gridworld, for_env=True, **model_cfg['R'])
pi = PolicyModel(**model_cfg['Pi'])

# belief (if want prior, then set the `prior` dictionary to be {objid -> {(x,y,z,r): Probability}}
prior = {}
prior_belief = AbstractM3Belief(gridworld, init_octree_belief(gridworld,
                                                              init_state.robot_state,
                                                              prior=prior))
# environment
env = Mos3DEnvironment(init_state, gridworld, Tenv, Renv)

# agent
agent = M3Agent(gridworld, prior_belief, pi, T, Ov, R)

# Hierarchical planner
planner = MultiResPlanner(setting, agent, gridworld, **planner_cfg['init_kwargs'])

# OpenGL
viz = None
if exec_cfg['viz']:
    viz = Mos3DViz(env, gridworld, fps=15)
    if viz.on_init() == False:
        raise Exception("Environment failed to initialize")
    viz.on_render()

############ Run #########
execute_action = True
step = 0
while True:
    run = input("Run step? [y] ")
    while not run.lower().startswith("y"):
        run = input("Run step? [y] ")
        time.sleep(0.5)
    # Run step
    print("planning...")
    real_action, env_reward, action_value = RunStep(gridworld, agent, env, Om,
                                                    planner, exec_cfg, viz=None)
    print_step_info(step, env, real_action, env_reward, planner, action_value)

    print("executing...")
    if execute_action:
        ExecuteAction(real_action, gridworld, agent, env, Om, planner, exec_cfg)

    step += 1

    # Visualize
    if exec_cfg['viz']:
        viz.update(real_action, env.state.robot_pose, env.state.object_poses)
        viz.on_loop()
        viz.on_render(rerender=True)
