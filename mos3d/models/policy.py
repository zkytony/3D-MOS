# Policy model for Mos3D

import pomdp_py
from mos3d.oopomdp import Actions, MotionAction, LookAction, DetectAction
import mos3d.util as util
import random
from collections import deque

class PolicyModel(pomdp_py.RolloutPolicy):
    # Simplest policy model. All actions are possible at any state.

    def __init__(self, detect_after_look=True):
        self._all_actions = set(Actions.ALL_ACTIONS)
        self._all_except_detect = self._all_actions - set({Actions.DETECT_ACTION})
        # if true, detect can only happen after look.
        self._detect_after_look = detect_after_look

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplemented

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplemented

    def get_all_actions(self, state=None, history=None):
        """note: detect can only happen after look."""
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                return self._all_actions
        if self._detect_after_look:
            return self._all_except_detect
        else:
            return self._all_actions

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(history=history), 1)[0]

# We'd like to have a policy model that has "memory". That is
# if some motion action were not executed successfully because
# of collision, then next time this action should not be
# available for planning when the robot is in that state
# again. This is just a convenience for our particular simulation
# setting. OOPOMDP paper built a topological graph before
# hand which means no motion action will result in collision.
# But there's extra work there.
class MemoryPolicyModel(PolicyModel):
    def __init__(self, detect_after_look=True):
        """Prior is a function s->{a}"""
        super().__init__(detect_after_look=detect_after_look)
        self._memory = {}  # map from robot_pose to actions

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(state=state, **kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplemented

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplemented

    def get_all_actions(self, state=None, history=None):
        last_action = None
        if history is not None and len(history) >= 1:
            # last action
            last_action = history[-1][0]
        no_detect = self._detect_after_look and (history is not None and not isinstance(last_action, LookAction))
        if state is None or state.robot_pose not in self._memory:
            if no_detect:
                return self._all_except_detect
            else:
                return self._all_actions
        else:
            if no_detect:
                return self._memory[state.robot_pose] - set({Actions.DETECT_ACTION})
            else:
                return self._memory[state.robot_pose]

    def _record_invalid_action(self, robot_state, action):
        if robot_state.pose not in self._memory:
            self._memory[robot_state.pose] = self._all_actions - set({action})
        else:
            self._memory[robot_state.pose] -= set({action})

    def update(self, robot_state, next_robot_state, action, **kwargs):
        if isinstance(action, MotionAction) and next_robot_state.pose == robot_state.pose:
            self._record_invalid_action(robot_state, action)

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

def simple_path_planning(pointA, pointB):
    # Naive path planning. TODO: use Astar?
    diffx = pointB[0] - pointA[0]
    diffy = pointB[1] - pointA[1]
    diffz = pointB[2] - pointA[2]
    signx = "+" if diffx > 0 else "-"
    signy = "+" if diffy > 0 else "-"
    signz = "+" if diffz > 0 else "-"
    motion_actions = [Actions.motion_action(signx+"x")] * abs(diffx)\
        + [Actions.motion_action(signy+"y")] * abs(diffy)\
        + [Actions.motion_action(signz+"z")] * abs(diffz)
    return motion_actions

class GreedyPolicyModel(PolicyModel):
    """
    This is a baseline policy model used as a rollout policy for PO-Rollout.

    It attempts to do the following: Move to a previously unvisited location,
    then look in all directions; detect after fov of look contains an object
    in the given state.

    This policy model uses the prior belief over object locations, and tries
    to move the robot to more probable locations. It attempts to iteratively
    search all objects one by one, unless multiple objects appeared coincidentally
    in the field of view. Therefore, this model should be updated at each time
    step by the agent's current belief.

    This policy can work with OOBelief(i.e. factored) or non-factored particle belief.
    """
    def __init__(self, gridworld, init_belief):
        """
        `init_belief`: The prior of the agent.
        """
        self._agent_belief = init_belief
        self._target_objids= list(set(gridworld.target_objects))
        self._action_queue = deque()
        self._gridworld = gridworld
        super().__init__(detect_after_look=True)

    def _point_next_to(self, point):
        """returns a 3d point that is next to given `point`.
        i.e. it has a coordinate off by 1 in either x, y or z"""
        result = list(point)
        for i in range(len(point)):
            if point[i] > 0:
                result[i] -= 1
                return result
        # still didn't return. so...
        if result[0] < self._gridworld.width - 1:
            result[0] += 1
        elif result[1] < self._gridworld.length - 1:
            result[1] += 1
        elif result[2] < self._gridworld.height - 1:
            result[2] += 1
        return result

    @property
    def action_queue(self):
        return self._action_queue

    def _prepare(self):
        # Generates a sequence of actions in the form of
        # move -> look -> detect for the current object
        # using current agent belief.
        mpe_state = self._agent_belief.mpe()
        robot_pose = mpe_state.robot_pose

        # Path planning.
        dst = self._point_next_to(mpe_state.object_states[self._target_objids[0]]['pose'])

        ## making destination off by 1 from the object's actual location in mpe state
        motion_actions = simple_path_planning(robot_pose, dst)
        look_actions = list(Actions.LOOK_ACTIONS)
        self._action_queue = deque(motion_actions + look_actions)

    def rollout(self, state, history=None):
        # If we just took a look action, then take a detect action if
        # there are objects in the field of view that are not yet detected.
        # This is basically exploiting the knowledge of the reward function.
        print("Warning: PORollout with Greedy Policy doesn't really work."\
              "You should use GreedyPlanner.")
        if history is not None and len(history) > 0\
           and isinstance(history[-1][0], LookAction):
            objects_within_range =\
                self._gridworld.objects_within_view_range(state.robot_pose,
                                                          state.object_poses)
            new_objects_count = len(objects_within_range - set(state.robot_state.objects_found))
            if new_objects_count > 0:
                return DetectAction()
        # Otherwise, return the next action in queue. If no more actions, prepare.
        if len(self._action_queue) == 0:
            self._prepare()
        action = self._action_queue.popleft()
        return action

    def update(self, state, next_state, action, belief=None):
        if isinstance(action, LookAction) and belief is not None:
            self._agent_belief = belief
        if next_state.robot_state.objects_found > state.robot_state.objects_found:
            # Remove found objects from target objids
            new_found = set(next_state.robot_state.objects_found) - set(state.robot_state.objects_found)
            self.mark_objects_found(new_found)

    def mark_objects_found(self, objects_found):
        self._target_objids = list(set(self._target_objids) - objects_found)


class GreedyPlanner(pomdp_py.Planner):
    """This planner is a new algorithm, instead of a baseline;
    it relies on bayesian belief update."""
    def __init__(self, greedy_rollout):
        self._should_detect = False
        self._greedy_rollout = greedy_rollout
        self._agent = None
        self._objects_detecting = None

    def plan(self, agent):
        """The agent carries the information:
        Bt, ht, O,T,R/G, pi, necessary for planning"""
        self._agent = agent
        # Used up action queue. Update belief and replan
        if len(self._greedy_rollout.action_queue) == 0:
            self._greedy_rollout._agent_belief = agent.belief
            self._greedy_rollout._prepare()
        if self._should_detect:
            return DetectAction()
        else:
            return self._greedy_rollout.action_queue.popleft()

    def update(self, agent, real_action, real_observation):
        """real_observation is real_observation_v in trial.py"""
        # This is basically exploiting the knowledge of the reward function.
        robot_state = agent.belief.mpe().robot_state
        if isinstance(real_action, LookAction):
            objects_observing = set({objid
                                     for objid in real_observation.voxels
                                     if real_observation.voxels[objid].label == objid})
            objects_found = robot_state['objects_found']
            if len(objects_observing - set(objects_found)) > 0:
                self._should_detect = True
                self._objects_detecting = objects_observing - set(objects_found)
                return
        elif isinstance(real_action, DetectAction):
            self._greedy_rollout.mark_objects_found(set(self._objects_detecting))
            self._objects_detecting = None
        self._should_detect = False

    def updates_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return False

class BruteForcePlanner(pomdp_py.Planner):
    def __init__(self, gridworld, init_robot_pose):
        self._gridworld = gridworld
        # All valid robot poses to visit, ranked by distance from
        # the initial robot pose;
        self._poses_to_visit = []
        for x in range(self._gridworld.width):
            for y in range(self._gridworld.length):
                for z in range(self._gridworld.height):
                    if self._gridworld.in_boundary((x,y,z)):
                        self._poses_to_visit.append((x,y,z))
        self._poses_to_visit = deque(sorted(self._poses_to_visit,
                                            key=lambda pose: util.euclidean_dist(pose,
                                                                                 init_robot_pose[:3])))
        self._action_queue = deque()
        self._should_detect = False
        self._agent = None

    def _prepare(self, robot_pose):
        # Tries to go to an unvisited location and look around
        pose = self._poses_to_visit.popleft()
        motion_actions = simple_path_planning(robot_pose, pose)
        look_actions = list(Actions.LOOK_ACTIONS)
        self._action_queue = deque(motion_actions + look_actions)


    def plan(self, agent):
        self._agent = agent
        if len(self._action_queue) == 0:
            robot_pose = self._agent.belief.mpe().robot_pose
            self._prepare(robot_pose)
        if self._should_detect:
            return DetectAction()
        else:
            return self._action_queue.popleft()

    def update(self, agent, real_action, real_observation, **kwargs):
        robot_state = agent.belief.mpe().robot_state
        if isinstance(real_action, LookAction):
            objects_observing = set({objid
                                     for objid in real_observation.voxels
                                     if real_observation.voxels[objid].label == objid})
            objects_found = robot_state['objects_found']
            if len(objects_observing - set(objects_found)) > 0:
                self._should_detect = True
                return
        self._should_detect = False

class RandomPlanner(pomdp_py.Planner):
    """Randomly plan, but still detect after look"""
    def __init__(self):
        self._should_detect = False

    def plan(self, agent):
        if self._should_detect:
            return DetectAction()
        else:
            return random.sample(agent.policy_model.get_all_actions(history=agent.history), 1)[0]

    def update(self, agent, real_action, real_observation, **kwargs):
        robot_state = agent.belief.mpe().robot_state
        if isinstance(real_action, LookAction):
            objects_observing = set({objid
                                     for objid in real_observation.voxels
                                     if real_observation.voxels[objid].label == objid})
            objects_found = robot_state['objects_found']
            if len(objects_observing - set(objects_found)) > 0:
                self._should_detect = True
                return
        self._should_detect = False


class PurelyRandomPlanner(pomdp_py.Planner):
    """PurelyRandomPlanner"""
    def plan(self, agent):
        return random.sample(agent.policy_model.get_all_actions(history=agent.history), 1)[0]
