# One experiment trial; For a given world, build an environment,
# runs the planner to plan the agent. Records key statistics.

import logging   # standard python package
import pomdp_py
import time
import copy
import math
import matplotlib.pyplot as plt
from collections import deque
from mos3d import *
import mos3d.util_viz

# OUR GOAL IS JUST TO SOLVE THE PROBLEM. AS GOOD AS WE CAN.
# DONT CARE ABOUT BASELINE UNTIL YOU CAN SOLVE THE PROBLEM
# WELL ENOUGH.

def init_particles_belief(num_particles, gridworld, init_state, belief="uniform"):

    w, l, h = gridworld.width, gridworld.length, gridworld.height
    particles = []
    for _ in range(num_particles):
        object_states = {}
        for objid in gridworld.target_objects:
            objclass = gridworld.objects[objid].objtype
            if belief == "uniform":
                objstate = TargetObjectState(objid, objclass,
                                             util.uniform(3, [(0, w), (0, l), (0, h)]),
                                             res=1)
            elif belief == "informed":
                objstate = TargetObjectState(objid, objclass,
                                             init_state.object_poses[objid],
                                             res=1)
            object_states[objid] = objstate
        object_states[gridworld.robot_id] = copy.deepcopy(init_state.robot_state)
        oostate = M3OOState(gridworld.robot_id, object_states)
        particles.append(oostate)
    return pomdp_py.Particles(particles)


def belief_update(agent, real_action, real_observation, next_robot_state, planner):
    """Updates agent's belief in place"""
    O = agent.observation_model
    T = agent.transition_model

    if isinstance(agent.cur_belief, pomdp_py.OOBelief):
        for objid in agent.cur_belief.object_beliefs:
            # If the planner is bruteforce or random, there is no need to update object beliefs.
            if (isinstance(planner, BruteForcePlanner) or isinstance(planner, RandomPlanner))\
               and objid != agent._gridworld.robot_id:
                continue

            belief_obj = agent.cur_belief.object_belief(objid)
            if isinstance(belief_obj, pomdp_py.Histogram):
                if objid == agent._gridworld.robot_id:
                    # we assume robot has perfect knowledge of itself.
                    new_belief = pomdp_py.Histogram({next_robot_state: 1.0})
                else:
                    new_belief = pomdp_py.update_histogram_belief(belief_obj,
                                                                  real_action,
                                                                  real_observation.robot_pose,
                                                                  O[objid], T[objid])
            elif isinstance(belief_obj, OctreeBelief):
                new_belief = update_octree_belief(belief_obj,
                                                  real_action, real_observation,
                                                  alpha=O[objid].alpha, beta=O[objid].beta, gamma=O[objid].gamma)
            else:
                raise ValueError("Cannot update the belief of type %s" % type(belief_obj))
            agent.cur_belief.set_object_belief(objid, new_belief)
    else:
        if isinstance(agent.cur_belief, pomdp_py.Particles):
            if isinstance(planner, RandomPlanner) or isinstance(planner, PurelyRandomPlanner):
                # Nothing to do for random planner
                return
            if not isinstance(planner, pomdp_py.POMCP):
                raise ValueError("Particle belief currently only supported for POMCP")
            # POMCP automatically udpates the belief, when the planner is updated.
        else:
            raise ValueError("Cannot update the belief of type %s" % type(agent.cur_belief))

def print_step_info(i, env, real_action, env_reward, planner, action_value):
    if isinstance(planner, pomdp_py.POUCT):
        info = ("Step %d: robot: %s   action: %s   reward: %.3f   NumSims: %d    ActionVal: %.3f"
                % (i+1, env.state.robot_state, real_action.name, env_reward, planner.last_num_sims, action_value))
    elif isinstance(planner, pomdp_py.PORollout):
        info = ("Step %d: robot: %s   action: %s   reward: %.3f   BestReward: %.2f"
                % (i+1, env.state.robot_state, real_action.name, env_reward, planner.last_best_reward))
    else:
        info = ("Step %d: robot: %s   action: %s   reward: %.3f"
                % (i+1, env.state.robot_state, real_action.name, env_reward))
        if isinstance(planner, MultiResPlanner):
            info += "\n"
            for key in planner._last_results:
                if type(key) == tuple:
                    action, action_value, num_sims = planner._last_results[key]
                    info += ("     Planner %s    Value: %.3f    Num Sim: %d   [%s]\n"\
                             % (str(key), action_value, num_sims, str(action)))
            info += ("Best chocie: %s" % str(planner._last_results['__chosen__']))
    print(info)

def BuildProblemInstance(worldstr, belief_config, model_config, alg_config):
    """
    worldstr: string description of the gridworld.
    belief_config: configurations for the belief
    model_config: configurations for the models.
    alg_config: configurations for the planning algorithm.
    """
    gridworld, init_state = parse_worldstr(worldstr)

    # models
    Ov = M3ObservationModel(gridworld, **model_config['O'], voxel_model=True)
    Om = M3ObservationModel(gridworld, **model_config['O'], voxel_model=False)
    T = M3TransitionModel(gridworld, **model_config['T'])
    Tenv = M3TransitionModel(gridworld, for_env=True, **model_config['T'])
    R = GoalRewardModel(gridworld, **model_config['R'])
    pi = MemoryPolicyModel(detect_after_look=True)  # no config for policy model.

    # Planner
    if alg_config['planner'].lower() == "pomcp":
        planner = pomdp_py.POMCP(rollout_policy=pi, **alg_config['init_args'])
    elif alg_config['planner'].lower() == "pouct":
        planner = pomdp_py.POUCT(rollout_policy=pi, **alg_config['init_args'])
    elif alg_config['planner'].lower().startswith("porollout"):
        assert "rollout_policy" in alg_config['init_args']
        planner = pomdp_py.PORollout(**alg_config['init_args'])
        pi = alg_config['init_args']['rollout_policy']
    elif alg_config['planner'].lower() == "greedy":
        planner = GreedyPlanner(GreedyPolicyModel(gridworld, belief_config['prior']))
    else:
        # Hierarchical?
        raise ValueError("Planner (%s) not specified correctly."
                         % alg_config['planner'])

    # environment
    env = Mos3DEnvironment(init_state, gridworld, Tenv, R)

    # agent
    prior = belief_config['prior']  # prior belief distribution
    agent = M3Agent(gridworld, prior, pi, T, Ov, R)
    return gridworld, agent, env, Om, planner


def ExecuteAction(real_action, gridworld, agent, env, Om, planner, exec_config, forced_actions=deque()):
    # state transition
    state = copy.deepcopy(env.state)
    env_reward = env.state_transition(real_action, execute=True)

    # receive observation
    real_observation_m = env.provide_observation(Om, real_action)
    real_observation_v = agent.convert_real_observation_to_planning_observation(real_observation_m, real_action)

    # updates
    agent.clear_history()  # for optimization
    agent.update_history(real_action, real_observation_m)
    belief_update(agent, real_action, real_observation_m, env.state.robot_state, planner)

    next_state = copy.deepcopy(env.state)
    agent.policy_model.update(state.robot_state, next_state.robot_state, real_action, belief=agent.belief)  # only robot state is accessed.

    if exec_config['plot_tree']:
        plt.figure(1)
        plt.clf()
        ax_tree = plt.gca()
        pomdp_py.visual.visualize_pouct_search_tree(agent.tree,
                                                    max_depth=4,
                                                    anonymize_observations=True,
                                                    anonymize_actions=exec_config.get("anonymize",False),
                                                    ax=ax_tree)
        import pdb; pdb.set_trace()

    # Planner update
    if forced_actions is not None:
        time.sleep(1)
        return  # taking forced actions. No planner update.

    try:
        planner.update(agent, real_action, real_observation_v)
    except Exception:
        raise ValueError("Planner update failed.\n"\
                         "Real action: %s\nReal observation: %s"
                         % (str(real_action), str(real_observation_v)))


def RunStep(gridworld, agent, env, Om, planner, exec_config, viz=None, execute=False,
            forced_actions=deque()):
    while True:
        if forced_actions is not None and len(forced_actions) > 0:
            real_action = forced_actions.popleft()
            print("Taking a forced action: %s" % str(real_action))
        else:
            real_action = planner.plan(agent)
        if env.action_valid(real_action):
            break
        else:
            print("%s is invalid. Will replan" % str(real_action))
            # To replan, update the planner with ReplanAction and NullObservation.
            if forced_actions is not None:
                print("Taking the next forced action")
                continue
            try:
                planner.update(agent, ReplanAction(), NullObservation())
            except Exception:
                raise ValueError("Planner update failed.\n"\
                                 "Real action: %s\nReal observation: %s"
                                 % (str(real_action), str(real_observation_v)))

    _, env_reward = env.state_transition(real_action, execute=False)
    if isinstance(planner, pomdp_py.POUCT):
        action_value = agent.tree[real_action].value
    else:
        action_value = env_reward  # just a placeholder. We only care about the case above.
    if execute:
        ExecuteAction(real_action, gridworld, agent, env, Om, planner, exec_config, forced_actions=forced_actions)
    return real_action, env_reward, action_value


def RunTrial(gridworld, agent, env, Om, planner, exec_config, forced_actions=deque()):
    """
    Om: Observation model which actually receives volumetric observation
        from the environment.
    exec_config: configurations for executing the trial.
    """
    # Run the trial
    _Rewards = []
    _Values = []  # action values
    _States = []
    _TotDist = []  # total distance to undetected objects over time.
    __colors = []

    if exec_config['plot_belief']:
        plt.figure(0)
        plt.ion()
        fig = plt.gcf()
        axes = {}
        if len(gridworld.objects) == 1 or gridworld.width > 8:
            nobj_plot = 1
            shape = (1,1)
        # elif len(gridworld.objects) <= 4:
        #     nobj_plot = len(gridworld.objects)
        #     shape = (2, 2)
        else:
            nobj_plot = min(4, len(gridworld.target_objects))
            shape = (2, 2)
        for i, objid in enumerate(gridworld.target_objects):
            if i >= nobj_plot:
                break
            ax = fig.add_subplot(*shape,
                                 i+1,projection="3d")
            ax.set_xlim([0, gridworld.width])
            ax.set_ylim([0, gridworld.length])
            ax.set_zlim([0, gridworld.height])
            ax.grid(False)
            axes[objid] = ax
        plt.show(block=False)

    viz = None
    if exec_config['viz']:
        viz = Mos3DViz(env, gridworld, fps=15)
        if viz.on_init() == False:
            raise Exception("Environment failed to initialize")
        viz.on_render()
        bar_shown = False

    start_time = time.time()
    for i in range(exec_config['max_steps']):
        real_action, env_reward, action_value = RunStep(gridworld, agent, env, Om, planner, exec_config, viz=viz,
                                                        execute=True, forced_actions=forced_actions)

        # Record
        _Rewards.append(env_reward)
        _States.append(copy.deepcopy(env.state))
        _TotDist.append(env.total_distance_to_undetected_objects())
        if isinstance(planner, pomdp_py.POUCT):
            _Values.append(action_value)

        # Plot belief
        if exec_config['plot_belief']:
            plt.figure(0)
            for objid in gridworld.target_objects:
                belief_obj = agent.cur_belief.object_belief(objid)
                if isinstance(belief_obj, OctreeBelief) and objid in axes:
                    for artist in axes[objid].collections:
                        artist.remove()
                    m = plot_octree_belief(axes[objid], belief_obj, robot_pose=env.state.robot_pose,
                                           alpha="clarity", edgecolor="black", linewidth=0.1)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        # Visualize
        if exec_config['viz']:
            viz.update(real_action, env.state.robot_pose, env.state.object_poses)
            viz.on_loop()
            viz.on_render(rerender=True)

        # Plot analysis related
        if exec_config['plot_analysis']:
            plt.figure(2)
            plt.clf()
            fig = plt.gcf()
            ax = plt.gca()
            if isinstance(real_action, LookAction):
                __colors.append("blue")
            elif isinstance(real_action, DetectAction):
                __colors.append("orange")
            else:
                __colors.append("green")
            ax.plot(np.arange(len(_TotDist)), _TotDist, 'go--', linewidth=2, markersize=12, zorder=1)
            ax.scatter(np.arange(len(_TotDist)), _TotDist, c=__colors, zorder=2)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # other analysis
            if hasattr(agent, 'tree'):
                pomdp_py.print_preferred_actions(agent.tree)

        if forced_actions is None:
            print_step_info(i, env, real_action, env_reward, planner, action_value)

        if time.time() - start_time > exec_config['max_time']:
            break
        if len(env.state.robot_state.objects_found) >= len(gridworld.target_objects):
            print("Task finished!")
            break
        if forced_actions is not None and len(forced_actions) == 0:
            print("All forced actions taken. Done.")
            import pdb; pdb.set_trace()
            break

    return _Rewards, _States, agent.history

def change_res(point, r1, r2):
    x,y,z = point
    return (x // (r2 // r1), y // (r2 // r1), z // (r2 // r1))
