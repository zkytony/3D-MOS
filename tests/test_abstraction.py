from mos3d import *
from mos3d.experiments.trial import BuildProblemInstance, RunTrial, change_res
from mos3d.models.abstraction import AbstractM3Belief
from mos3d.planning.multires import MultiResPlanner
import random
import time

random.seed(129)

if LOG:
    ALPHA = math.log(10000000)
    BETA = -10
else:
    ALPHA = 10000000
    BETA = 0


world1=\
"""
32
32
32

hero 9 17 2
hero 2 4 13
smashboy 22 15 16
cube 14 3 23
---
robot 8 10 9 0 0 0 occlusion 45 1.0 0.1 16"""

worldocc=\
"""
4
4
4

cube 0 0 0 hidden
cube 1 0 0 obstacle
cube 0 1 0 obstacle
cube 0 0 1 obstacle
---
robot 2 2 2 0 0 0 occlusion 65 1.0 0.1 5
"""

world_config = {
    # forworld generation
    'objtypes': {'cube': 4 }, # to be filled
    'robot_camera': "occlusion 45 1.0 0.1 20",
    'width': 8,
    'length': 8,
    'height': 8,
}
worldstr = random_3dworld(world_config)
worldstr=worldocc

model_config = {
    'O': {
        'epsilon': 1e-9,
        'alpha': ALPHA,
        'beta': BETA,
        'gamma': DEFAULT_VAL
    },
    'T': {
        'epsilon': 1e-9,
    },
    'R': {
        'big': 1000,
        'medium': 100,
        'small': 10
    }
}

gridworld, init_state = parse_worldstr(worldstr)
print(init_state.robot_state)

Tr = RobotTransitionModel(gridworld)
# abs_Tr = AbstractRobotTransitionModel(Tr)

# om = MotionOption(init_state.robot_pose, (5, 5, 5), gridworld)
# next_state = abs_Tr.sample(init_state, om)
# print(next_state)

prior_type = "uniform"
prior_region_res = 2
if prior_type == "uniform":
    prior = None
elif prior_type == "informed":
    prior = {}
    for objid in gridworld.target_objects:
        true_pose = init_state.object_poses[objid]
        prior_region = list(change_res(true_pose, 1, prior_region_res))
        prior[objid] = {(*prior_region, prior_region_res): ALPHA}
elif prior_type == "ambiguous":
    num_ambiguous = 2
    prior = {}
    # True pose
    for objid in gridworld.target_objects:
        true_pose = init_state.object_poses[objid]
        prior_region = list(change_res(true_pose, 1, prior_region_res))
        prior[objid] = {(*prior_region, prior_region_res): ALPHA}
    # Some other locations
    for objid in gridworld.target_objects:
        for i in range(num_ambiguous):
            ambiguous_pose = util.uniform(3,
                                          [(0, gridworld.width),
                                           (0, gridworld.length),
                                           (0, gridworld.height)])
            prior_region = list(change_res(ambiguous_pose, 1, prior_region_res))
            prior[objid] = {(*prior_region, prior_region_res): ALPHA}

# # Give a wrong prior
# if prior_region[0] == 0:
#     prior_region[0] = 1
# else:
#     prior_region[0] = 0
# prior = {1: {(*prior_region, prior_region_res): ALPHA / 2}}
# prior = None
belief_config = {
    'prior': M3Belief(gridworld, init_octree_belief(gridworld,
                                                    init_state.robot_state,
                                                    prior=prior))
}

alg_config = {'planner': "pouct"}

exec_config = {
    'max_steps': 100,
    'max_time': 120,  # seconds
    'plot_belief': True,
    'plot_tree': False,
    'plot_analysis': True,
    'viz': True,
    'anonymize': True
}

test = "state_action_abstraction"
######### TESTING rollout time #########
if test == "rollout_time":
    api = AbstractPolicyModel(motion_resolution=8)
    pi = MemoryPolicyModel(detect_after_look=True)

    start = time.time()
    for i in range(100000):
        api.rollout(init_state)
    time_api = time.time() - start

    start = time.time()
    for i in range(100000):
        pi.rollout(init_state)
    time_pi = time.time() - start

    print("Total rollout time for AbstractPolicyModel: %.3f" % (time_api))
    print("Total rollout time for MemoryPolicyModel: %.3f" % (time_pi))


######### TESTING Action Abstraction #########
if test == "action_abstraction":
    # Run a trial
    alg_config['init_args'] = {
        'max_depth': 40,
        'discount_factor': 0.95,
        'planning_time': 1.0,
        'exploration_const': 2000
    }

    Ov = M3ObservationModel(gridworld, **model_config['O'], voxel_model=True)
    Om = M3ObservationModel(gridworld, **model_config['O'], voxel_model=False)
    T = M3TransitionModel(gridworld, **model_config['T'])
    Tenv = M3TransitionModel(gridworld, for_env=True, **model_config['T'])
    pi = AbstractPolicyModel(motion_resolution=8)
    R = GoalRewardModel(gridworld, **model_config['R'])

    print(pi.get_all_actions(state=init_state))
    print(len(pi.get_all_actions(state=init_state)))


    planner = pomdp_py.POUCT(rollout_policy=pi, **alg_config['init_args'])
    env = Mos3DEnvironment(init_state, gridworld, Tenv, R)
    agent = M3Agent(gridworld, belief_config['prior'], pi, T, Ov, R)

    # action = planner.plan(agent)
    # print(action)

    RunTrial(gridworld, agent, env, Om, planner, exec_config)
    exit(0)


######### TESTING Action Abstraction #########
if test == "state_action_abstraction":
    belief_config = {
        'prior': AbstractM3Belief(gridworld, init_octree_belief(gridworld,
                                                                init_state.robot_state,
                                                                prior=prior))
    }
    alg_config['init_args'] = {
        'discount_factor': 0.99,
        'planning_time': 3,
        # I have empirically tried many times. It seems that exploration=50 is good enough.
        # This
        'exploration_const': 1000
    }

    Ov = M3ObservationModel(gridworld, **model_config['O'], voxel_model=True)
    Om = M3ObservationModel(gridworld, **model_config['O'], voxel_model=False)
    T = M3TransitionModel(gridworld, **model_config['T'])
    Tenv = M3TransitionModel(gridworld, for_env=True, **model_config['T'])
    pi = MemoryPolicyModel(detect_after_look=True)
    R = GoalRewardModel(gridworld, **model_config['R'])

    agent = M3Agent(gridworld, belief_config['prior'], pi, T, Ov, R)
    env = Mos3DEnvironment(init_state, gridworld, Tenv, R)
    # planner = MultiResPlanner([(1,1,10),(2,2,10),(4,4,10)],
    #                           agent, gridworld, **alg_config['init_args'])
    # planner = MultiResPlanner([(1,1,10), (2,2,10), (4,4,10)], #(8,8,10),(2,2,10)],
    #                           agent, gridworld, **alg_config['init_args'])
    # planner = MultiResPlanner([(1,2,20),(1,1,10), (1,4,40)], agent, gridworld, **alg_config['init_args'])

    planner = pomdp_py.POUCT(rollout_policy=pi, exploration_const=1000, discount_factor=0.99,
                             max_depth=10, planning_time=1)
    RunTrial(gridworld, agent, env, Om, planner, exec_config, forced_actions=None)
