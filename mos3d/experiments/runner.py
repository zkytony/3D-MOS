from sciex import Experiment, Trial, Event, Result, YamlResult, PklResult, PostProcessingResult
from mos3d import *
import mos3d.util_viz
from mos3d.experiments.trial import init_particles_belief, belief_update
from mos3d.experiments.result_types import *
import matplotlib.pyplot as plt
import numpy as np
import pomdp_py
import os
import yaml
import pickle
import time
from pprint import pprint
import re
import math
import pandas as pd

##### PERFECT SENSOR #####
if LOG:  # LOG comes from moos3d
    ALPHA = math.log(100000)
    BETA = -10
else:
    ALPHA = 100000
    BETA = 0

# Configurations
def model_config(epsilon=1e-9, alpha=ALPHA, beta=BETA, gamma=DEFAULT_VAL,
                 big=1000, medium=100, small=1, detect_after_look=True):
    return {
        'O': {
            'epsilon': epsilon,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        },
        'T': {
            'epsilon': epsilon,
        },
        'R': {
            'big': big,
            'medium': medium,
            'small': small
        },
        'Pi': {
            'detect_after_look': detect_after_look
        }
    }

def planner_config(planner, max_depth=10, discount_factor=0.95, planning_time=1.0,
                   exploration_const=50, setting=None, rollout_policy=None,
                   num_simulations=100):
    kwconfig = {
        'max_depth': max_depth,
        'discount_factor': discount_factor,
    }
    if planner.lower() in {"hierarchical", "pomcp", "pouct", "options"}:
        kwconfig['planning_time'] = planning_time
        kwconfig['exploration_const'] = exploration_const

    argconfig = {}
    if planner == "porollout":
        kwconfig['rollout_policy'] = rollout_policy
        kwconfig['num_simulations'] = num_simulations
    elif planner in {"hierarchical", "options"}:
        argconfig['setting'] = setting
    return {"planner": planner, "init_kwargs": kwconfig, "init_args": argconfig}

def belief_config(belief_type, prior_type="uniform",
                  prior_region_res=1, prior_confidence=ALPHA,
                  num_particles=1000):
    if belief_type.lower() == "octree":
        priorconfig = {'type': prior_type, 'prior_region_res': prior_region_res,
                       'prior_confidence': prior_confidence}
    elif belief_type.lower() == "particles":
        priorconfig = {'type': prior_type, 'num_particles': num_particles}
    return {'type': belief_type, 'prior': priorconfig}

def world_config(objcounts={'cube':1},
                 robot_camera="occlusion 45 1.0 0.1 10",
                 width=8, length=8, height=8, size=None):
    # Returns a world string
    if size is not None:
        width, length, height = size, size, size
    world_config = {
        # for world generation
        'objtypes': objcounts, # to be filled
        'robot_camera': robot_camera,
        'width': width,
        'length': length,
        'height': height,
    }
    return random_3dworld(world_config)

def exec_config(max_steps=100, max_time=120, plot_belief=False, plot_tree=False,
                plot_analysis=False, viz=True, anonymize=True):
    return {
        'max_steps': max_steps,
        'max_time': max_time,  # seconds
        'plot_belief': plot_belief,
        'plot_tree': plot_tree,
        'plot_analysis': plot_analysis,
        'viz': viz,
        'anonymize': anonymize
    }


class M3Trial(Trial):

    RESULT_TYPES = [RewardsResult, StatesResult]
                    # HistoryResult,
                    # TotDistResult, ExplorationRatioResult]

    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)

    def _build_problem_instance(self):
        # Configurations
        model_config = self._config['model_config']
        planner_config = self._config['planner_config']
        belief_config = self._config['belief_config']
        world_config = self._config['world_config']
        if type(world_config) == str:
            # Treat world config as worldstr
            worldstr = world_config
        else:
            # Treat world config as a dictionary
            worldstr = random_3dworld(world_config)
        gridworld, init_state = parse_worldstr(worldstr)

        # models
        Ov = M3ObservationModel(gridworld, **model_config['O'], voxel_model=True)
        Om = M3ObservationModel(gridworld, **model_config['O'], voxel_model=False)
        T = M3TransitionModel(gridworld, **model_config['T'])
        Tenv = M3TransitionModel(gridworld, for_env=True, **model_config['T'])
        R = GoalRewardModel(gridworld, **model_config['R'])
        Renv = GoalRewardModel(gridworld, for_env=True, **model_config['R'])
        pi = PolicyModel(**model_config['Pi'])

        # belief
        belief_type = belief_config['type'].lower()
        prior_type = belief_config['prior']['type'].lower()
        assert prior_type in {"informed", "uniform"}, "Prior type %s unsupported" % prior_yype
        if belief_type == "octree":
            if prior_type == "informed":
                prior = {}
                prior_region_res = belief_config['prior']['prior_region_res']
                prior_confidence = belief_config['prior']['prior_confidence']
                for objid in gridworld.target_objects:
                    true_pose = init_state.object_poses[objid]
                    prior_region = list(change_res(true_pose, 1, prior_region_res))
                    prior[objid] = {(*prior_region, prior_region_res): prior_confidence}
            elif prior_type == "uniform":
                prior = None
            prior_belief = AbstractM3Belief(gridworld, init_octree_belief(gridworld,
                                                                          init_state.robot_state,
                                                                          prior=prior))
        elif belief_type == "particles":
            num_particles = belief_config['prior']['num_particles']
            prior_belief = init_particles_belief(num_particles, gridworld, init_state, belief=prior_type)
        else:
            raise ValueError("Unsupported belief type %s" % belief_type)

        # environment
        env = Mos3DEnvironment(init_state, gridworld, Tenv, Renv)

        # agent
        agent = M3Agent(gridworld, prior_belief, pi, T, Ov, R)

        # planners
        if planner_config['planner'].lower() == "pomcp":
            planner = pomdp_py.POMCP(rollout_policy=pi, **planner_config['init_kwargs'])
        elif planner_config['planner'].lower() == "pouct":
            planner = pomdp_py.POUCT(rollout_policy=pi, **planner_config['init_kwargs'])
        elif planner_config['planner'].lower().startswith("porollout"):
            assert "rollout_policy" in planner_config['init_kwargs']
            planner = pomdp_py.PORollout(**planner_config['init_kwargs'])
            pi = planner_config['init_kwargs']['rollout_policy']
        elif planner_config['planner'].lower() == "greedy":
            # TODO: This should be changed.
            planner = GreedyPlanner(GreedyPolicyModel(gridworld, belief_config['prior']))
        elif planner_config['planner'].lower() in {"hierarchical", "options"}:
            setting = planner_config['init_args']['setting']
            planner = MultiResPlanner(setting, agent, gridworld, **planner_config['init_kwargs'])
        elif planner_config['planner'].lower() == "bruteforce":
            planner = BruteForcePlanner(gridworld, init_state.robot_pose)
        elif planner_config['planner'].lower() == "random":
            planner = RandomPlanner()
        elif planner_config['planner'].lower() == "purelyrandom":
            planner = PurelyRandomPlanner()
        else:
            raise ValueError("Planner (%s) not specified correctly."
                             % planner_config['planner'])
        return gridworld, agent, env, Om, planner

    def step_info(self, i, env, real_action, env_reward, planner, action_value, total_reward):
        if isinstance(planner, pomdp_py.POUCT):
            info = ("Step %d: robot: %s   action: %s   reward: %.3f   cum_reward: %.3f    NumSims: %d    ActionVal: %.3f"
                    % (i+1, env.state.robot_state, real_action.name, env_reward, total_reward, planner.last_num_sims, action_value))
        elif isinstance(planner, pomdp_py.PORollout):
            info = ("Step %d: robot: %s   action: %s   reward: %.3f   cum_reward: %.3f    BestReward: %.2f"
                    % (i+1, env.state.robot_state, real_action.name, env_reward, total_reward, planner.last_best_reward))
        else:
            info = ("Step %d: robot: %s   action: %s   reward: %.3f   cum_reward: %.3f"
                    % (i+1, env.state.robot_state, real_action.name, env_reward, total_reward))
            if isinstance(planner, MultiResPlanner):
                info += "\n"
                for key in planner._last_results:
                    if type(key) == tuple:
                        action, action_value, num_sims = planner._last_results[key]
                        info += ("     Planner %s    Value: %.3f    Num Sim: %d   [%s]\n"\
                                 % (str(key), action_value, num_sims, str(action)))
                info += ("Best chocie: %s" % str(planner._last_results['__chosen__']))
        return info

    def _plan(self, planner, agent, env, _time_used, max_time, logging=False):
        # plan action; Keep replanning until the action is valid.
        while True:
            _start = time.time()
            real_action = planner.plan(agent)
            _time_used += time.time() - _start

            if _time_used > max_time:
                return None, _time_used

            if env.action_valid(real_action):
                break
            else:
                if logging:
                    self.log_event(Event("%s is invalid. Will replan" % str(real_action),
                                         kind=Event.WARNING))
                # To replan, update the planner with ReplanAction and NullObservation.
                self._planner_update(planner, agent, ReplanAction(), NullObservation())
        return real_action, _time_used

    def _planner_update(self, planner, agent, real_action, real_observation_v, logging=False):
        # Planner update
        try:
            planner.update(agent, real_action, real_observation_v)
        except Exception as ex:
            if logging:
                self.log_event(Event("Trial %s | Planner update failed. Reason: %s" % (self.name, str(ex)),
                                     kind=Event.WARNING))

            if isinstance(planner, pomdp_py.POMCP):
                if logging:
                    self.log_event(Event("Trial %s | Switching planner to Random" % (self.name),
                                         kind=Event.WARNING))
                planner = PurelyRandomPlanner()
        return planner

    ##### RUN ######
    def run(self, logging=False):
        # Build problem instance
        if logging:
            self.log_event(Event("initializing trial %s." % self.name))
        gridworld, agent, env, Om, planner = self._build_problem_instance()
        if logging:
            self.log_event(Event("Trial %s initialized." % self.name,
                                 kind=Event.SUCCESS))

        exec_config = self.config['exec_config']

        # Run the trial
        _Rewards = []
        _Values = []  # action values
        _States = [copy.deepcopy(env.state)]
        _TotDist = []  # total distance to undetected objects over time.
        _ExplorationRatios = []  # ratio of the gridworld that has been observed
        _History = []
        __colors = []

        if exec_config['plot_belief']:
            plt.figure(0)
            plt.ion()
            fig = plt.gcf()
            axes = {}
            if len(gridworld.objects) == 1 or gridworld.width > 8:
                nobj_plot = 1
                shape = (1,1)
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

        if exec_config['viz']:
            viz = Mos3DViz(env, gridworld, fps=15)
            if viz.on_init() == False:
                raise Exception("Environment failed to initialize")
            viz.on_render()
            bar_shown = False

        # Start running
        _time_used = 0  # Records the time used effectively by the agent for planning and belief update
        _detect_actions_count = 0  # does not allow > |#obj| number of detect actions.
        for i in range(exec_config['max_steps']):
            state = copy.deepcopy(env.state)

            # Plan action
            real_action, _time_used = self._plan(planner, agent, env, _time_used, exec_config['max_time'], logging=logging)
            if _time_used > exec_config['max_time']:
                break

            # Execute action
            env_reward = env.state_transition(real_action, execute=True)
            if isinstance(planner, pomdp_py.POUCT):
                action_value = agent.tree[real_action].value
            else:
                action_value = None  # just a placeholder. We only care about the case above.
            if isinstance(real_action, DetectAction):
                _detect_actions_count += 1  # count detect action

            # receive observation
            _start = time.time()
            real_observation_m = env.provide_observation(Om, real_action)
            real_observation_v = agent.convert_real_observation_to_planning_observation(real_observation_m, real_action)

            # updates
            agent.clear_history()  # for optimization
            agent.update_history(real_action, real_observation_m)
            belief_update(agent, real_action, real_observation_m, env.state.robot_state, planner)
            next_state = copy.deepcopy(env.state)
            # update policy model with real robot state (observable)
            agent.policy_model.update(state.robot_state, next_state.robot_state, real_action)

            _time_used += time.time() - _start

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

            planner = self._planner_update(planner, agent, real_action, real_observation_v, logging=logging)

            # Record
            _Rewards.append(env_reward)
            _States.append(copy.deepcopy(env.state))
            _TotDist.append(env.total_distance_to_undetected_objects())
            _ExplorationRatios.append(agent.exploration_ratio(real_observation_m))
            _History += ((real_action, real_observation_m),)
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
                        m = plot_octree_belief(axes[objid], belief_obj, alpha="clarity", edgecolor="black", linewidth=0.1)
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

            info = self.step_info(i, env, real_action, env_reward, planner, action_value, sum(_Rewards))
            if logging:
                self.log_event(Event("Trial %s | %s" % (self.name, info)))

            if len(env.state.robot_state.objects_found) >= len(gridworld.target_objects):
                if logging:
                    self.log_event(Event("Trial %s | Task finished!\n\n" % (self.name)))
                break

            if _detect_actions_count > len(gridworld.target_objects):
                if logging:
                    self.log_event(Event("Trial %s | Task ended; Used up detect actions.\n\n" % (self.name)))
                break

            if _time_used > exec_config['max_time']:
                break


        results = [
            RewardsResult(_Rewards),
            StatesResult(_States),
            HistoryResult(_History),
            TotDistResult(_TotDist),
            ExplorationRatioResult(_ExplorationRatios)
        ]
        return results
