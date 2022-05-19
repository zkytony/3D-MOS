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
from mos3d import AbstractPolicyModel, AbstractM3Belief, M3Agent,\
    AbstractM3ObservationModel, AbstractM3Agent,\
    AbstractM3TransitionModel, DetectAction
from mos3d.planning.belief.octree import DEFAULT_VAL, LOG
# import multiprocessing
import concurrent.futures

class MultiResPlanner(pomdp_py.Planner):

    def __init__(self,
                 settings,
                 agent,
                 gridworld,
                 planning_time=1.0,
                 exploration_const=2000,
                 discount_factor=0.95,
                 abstract_policy=None,
                 k=None, **kwargs):
        """
        settings: list of tuples (rs, ra, max_depth), each
            represents a POUCT planner module, where ro is
            resolution of object state and ra is resolution
            of action step size, `max_depth` is the maximum planning
            depth of this planner.
        """
        self._history = agent.history
        self._agents = {}
        self._planners = {}
        self._last_results = {}
        for rs, ra, max_depth in settings:
            if abstract_policy is None:
                abstract_policy = AbstractPolicyModel(motion_resolution=ra,
                                                      detect_after_look=agent.policy_model._detect_after_look)
            else:
                print("Abstract policy model already provided; Motion resolution %d will be ignored." % ra)

            planner = pomdp_py.POUCT(rollout_policy=abstract_policy, max_depth=max_depth,
                                     discount_factor=discount_factor,
                                     exploration_const=exploration_const,
                                     planning_time=planning_time)
            self._planners[(rs,ra)] = planner
            abstract_transition_model = AbstractM3TransitionModel(agent.belief, gridworld)
            abstract_observation_model = AbstractM3ObservationModel(agent.belief,
                                                                    gridworld,
                                                                    alpha=agent.observation_model.alpha,
                                                                    beta=agent.observation_model.beta,
                                                                    gamma=agent.observation_model.gamma,
                                                                    k=k)
            self._agents[(rs,ra)] = AbstractM3Agent(rs, gridworld, agent.belief,
                                                    abstract_policy,
                                                    abstract_transition_model,
                                                    abstract_observation_model,
                                                    agent.reward_model,
                                                    name="M3Agent(%d,%d)" % (rs, ra))

    def plan(self, agent):
        if len(agent.history) != len(self._history):
            raise ValueError("Agent history and planner history mismatch. "
                             "Did you update the planner?")

        # Plan with all planners
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._plan_single, list(self._planners.keys()))
        # results = [self._plan_single(key) for key in self._planners]

        # update agent trees, pick best action;
        # Does not allow higher-level planner to "detect";
        best_action = None
        best_value = float("-inf")
        best_key = None
        for key, action, action_value, agent, num_sims in results:
            # if key[0] > 1 and isinstance(action, DetectAction):
            #     print("DETECT NOT ALLOWED FOR %s" % str(key))
            #     continue  # skip state-abstraction detect action -- doesn't work well.
            self._agents[key].tree = agent.tree
            if action_value > best_value:
                best_value = action_value
                best_action = action
                best_key = key
            self._last_results[key] = (action, action_value, num_sims)
        self._last_results['__chosen__'] = best_key
        return best_action

    def print_last_results(self):
        for key in self._last_results:
            if not key.startswith("_"):
                action, action_value, num_sims = self._last_results[key]
                print("     Planner %s    Value: %.3f    Num Sim: %d   [%s]"
                      % (str(key), action_value, num_sims, str(action)))
        print("Best chocie: %s" % self._last_results['__chosen__'])

    def _plan_single(self, key):
        action = self._planners[key].plan(self._agents[key])
        action_value = self._agents[key].tree[action].value
        return key, action, action_value, self._agents[key], self._planners[key].last_num_sims

    def update(self, agent, real_action, real_observation):
        if len(agent.history) != len(self._history):
            # Update the belief and observation models of all agents
            for key in self._agents:
                self._agents[key].set_belief(agent.belief)
                self._agents[key].observation_model.update_belief(agent.belief)
                # NOTE: even though this function below is not called (i.e. commented out),
                # the agent belief that is used in argmax is still updated, because in
                # MultiResPlanner, the agent belief object is passed as a reference when
                # constructing AbstractRobotTransitionModel; So changes that happens inside
                # the agent belief object, which happens in belief_update() through set_object_belief
                # in tests/trial.py, will be reflected here as well. This means the existing
                # code works as expected. Nevertheless, the code is easier to follow if this
                # function is called explicitly, and the behavior of argmax will be independent
                # from all these other classes. Hence, we have a modification in MultiResPlanner.update(),
                # which will not result in actual change in the planning behavior.
                self._agents[key].transition_model.update_belief(agent.belief)
            self._history = agent.history

        for key in self._planners:
            self._planners[key].update(self._agents[key], real_action, real_observation)
            self._agents[key].update_history(real_action, real_observation)
