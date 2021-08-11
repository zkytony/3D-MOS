# from moos3d.environment.env import Mos3D
# from pomdp_py import OOPOMDP, Environment
from pomdp_py import ObjectState, OOPOMDP, OOState, Action, Observation
import random

class TargetObjectState(ObjectState):
    def __init__(self, objid, objclass, pose, res=1):
        super().__init__(objclass, {"pose":pose, "id":objid, "res":res})
    def __str__(self):
        return '%s%s' % (str(self.objclass), str(self.pose))
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def res(self):
        return self.attributes['res']
    @property
    def resolution(self):
        return self.res
    @property
    def objid(self):
        return self.attributes['id']

class RobotState(ObjectState):
    def __init__(self, robot_id, pose, objects_found, camera_direction, res=1):
        """Note: camera_direction is None unless the robot is looking at a direction,
        in which case camera_direction is the string e.g. look+thz, or 'look'"""
        super().__init__("robot", {"id":robot_id,
                                   "pose":pose,
                                   "res": res,
                                   "objects_found": objects_found,
                                   "camera_direction": camera_direction})
    def __str__(self):
        return 'RobotState(%s%s|%s)' % (str(self.objclass), str(self.pose), str(self.objects_found))
    def __repr__(self):
        return str(self)
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def robot_pose(self):
        return self.attributes['pose']
    @property
    def objects_found(self):
        return self.attributes['objects_found']
    @property
    def res(self):
        return self.attributes['res']
    @property
    def resolution(self):
        return self.res

class M3OOState(OOState):
    def __init__(self, robot_id, object_states):
        self._robot_id = robot_id
        super().__init__(object_states)
    def get_robot_state(self):
        return self.object_states[self._robot_id]
    @property
    def robot_id(self):
        return self._robot_id
    @property
    def robot_pose(self):
        return self.object_states[self._robot_id]['pose']
    @property
    def object_poses(self):
        return {objid:self.object_states[objid]['pose']
                for objid in self.object_states
                if objid != self._robot_id}
    @property
    def robot_state(self):
        return self.object_states[self._robot_id]
    def __str__(self):
        return 'M3OOState(%d)%s' % (self._robot_id, str(self.object_states))
    def __repr__(self):
        return str(self)


############# Actions #############3
MOTION_ACTIONS = {
    "TRANS": {
        0: ((-1,0,0), (0,0,0)),
        1: ((1,0,0), (0,0,0)),
        2: ((0,-1,0), (0,0,0)),
        3: ((0,1,0), (0,0,0)),
        4: ((0,0,-1), (0,0,0)),
        5: ((0,0,1), (0,0,0)),
    },
    "AXIS":{
        # AXIS motion model
        0: ((-1,0,0), (0,0,0)),
        1: ((1,0,0), (0,0,0)),
        2: ((0,-1,0), (0,0,0)),
        3: ((0,1,0), (0,0,0)),
        4: ((0,0,-1), (0,0,0)),
        5: ((0,0,1), (0,0,0)),
        6: ((0,0,0), (90,0,0)),
        7: ((0,0,0), (-90,0,0)),
        8: ((0,0,0), (0,90,0)),
        9: ((0,0,0), (0,-90,0)),
        10: ((0,0,0), (0,0,90)),
        11: ((0,0,0), (0,0,-90))
    },

    "FORWARD":{
        # FORWARD/BACKWORD motion model
        0:  ( 1, (0,0,0)  ),
        1:  (-1, (0,0,0)  ),
        2:  ( 0, (90,0,0) ),
        3:  ( 0, (-90,0,0)),
        4:  ( 0, (0,90,0) ),
        5:  ( 0, (0,-90,0)),
        6:  ( 0, (0,0,90) ),
        7:  ( 0, (0,0,-90))
    }
}
MOTION_MODEL="TRANS"  # All other code assumes TRANS motion model.

# Look actions only used if motion model is TRANSLATIONAL;
# 6D camera pose relative to the robot. If all 0, then
# the camera points to the robot's -z direction. In sim, point to
# -x by default. LOOK_ACTIONS are configured based on this camrea pose.
CAMERA_INSTALLATION_POSE = (0, 0, 0, 0, 90, 0)  # points to -x
LOOK_ACTIONS = {
    "look+thx": ((0,0,0), (0,180,0)), # point to +x direction
    "look-thx": ((0,0,0), (0,0,0)),   # point to -x direction
    "look+thy": ((0,0,0), (0,0,-90)),  # point to +y
    "look-thy": ((0,0,0), (0,0,90)), # point to -y
    "look+thz": ((0,0,0), (0,90,0)),  # point to +z
    "look-thz": ((0,0,0), (0,-90,0))  # point to -z
}

MOTION_ACTION_NAMES = {
    "TRANS": {
        0: "-x",
        1: "+x",
        2: "-y",
        3: "+y",
        4: "-z",
        5: "+z",
    },
    "AXIS": {
        0: "-x",
        1: "+x",
        2: "-y",
        3: "+y",
        4: "-z",
        5: "+z",
        6: "+thx",
        7: "-thx",
        8: "+thy",
        9: "-thy",
        10: "+thz",
        11: "-thz"
    },
    "FORWARD": {
        0: "forward",
        1: "backward",
        2: "+thx",
        3: "-thx",
        4: "+thy",
        5: "-thy",
        6: "+thz",
        7: "-thz"
    }
}

ACTION_NAMES_REVERSE = {
    "TRANS": {MOTION_ACTION_NAMES["TRANS"][i]:i for i in MOTION_ACTION_NAMES["TRANS"]},
    "AXIS": {MOTION_ACTION_NAMES["AXIS"][i]:i for i in MOTION_ACTION_NAMES["AXIS"]},
    "FORWARD": {MOTION_ACTION_NAMES["FORWARD"][i]:i for i in MOTION_ACTION_NAMES["FORWARD"]}
}

class M3Action(Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
        else:
            return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "M3Action(%s)" % self.name

class MotionAction(M3Action):
    def __init__(self, motion, name, distance_cost=0.1):
        self.motion = motion
        self.distance_cost = distance_cost
        super().__init__(name)

class SimMotionAction(MotionAction):
    def __init__(self, action_num, motion_model=MOTION_MODEL):
        super().__init__(MOTION_ACTIONS[motion_model][action_num],
                         MOTION_ACTION_NAMES[motion_model][action_num])

class LookAction(M3Action):
    def __init__(self, motion=None, look_direction=None):
        if look_direction is None:
            # This is just a look action
            self.motion = None
            super().__init__("look")
        else:
            self.motion = motion
            super().__init__(look_direction)
    @property
    def direction(self):
        return self.motion

class SimLookAction(LookAction):
    def __init__(self, look_direction=None):
        if look_direction is None:
            super().__init__(motion=None,
                             look_direction=None)
        else:
            super().__init__(motion=LOOK_ACTIONS[look_direction],
                             look_direction=look_direction)
    @property
    def direction(self):
        return self.motion

class DetectAction(M3Action):
    def __init__(self):
        super().__init__("detect")

class ReplanAction(M3Action):
    """Extra action, used for replan;
    will lead to MCTS tree truncation for replan"""
    def __init__(self):
        super().__init__("replan")

class NullObservation(Observation):
    """Just an observation used to trigger MCTS tree
    truncation for replanning; No substance"""
    def __eq__(self, other):
        return isinstance(other, NullObservation)
    def __hash__(self):
        return hash("null_observation")


def build_motion_actions():
    action_names = MOTION_ACTION_NAMES[MOTION_MODEL]
    motion_actions = MOTION_ACTIONS[MOTION_MODEL]

    result = set({})
    for action_num in motion_actions:
        action = SimMotionAction(action_num)
        result.add(action)
        if action.motion[0] == (0,0,0):
            # motion is rotational; no distance cost
            action.distance_cost = 0
        else:
            action.distance_cost = 1 * abs(max(action.motion[0]))
    return result

def build_look_actions():
    return {SimLookAction(look_direction)
            for look_direction in LOOK_ACTIONS}

class Actions:
    # simulation
    MOTION_ACTIONS = build_motion_actions()
    DETECT_ACTION = DetectAction()
    LOOK_ACTION = LookAction()
    MOTION_MODEL = MOTION_MODEL
    MOTION_ACTION_NAMES = MOTION_ACTION_NAMES[MOTION_MODEL]

    if MOTION_MODEL == "TRANS":
        LOOK_ACTIONS = build_look_actions()
        ALL_ACTIONS = MOTION_ACTIONS | LOOK_ACTIONS | set({DETECT_ACTION})
    else:
        ALL_ACTIONS = MOTION_ACTIONS | set({LOOK_ACTION, DETECT_ACTION})

    @classmethod
    def motion_action(self, name):
        return SimMotionAction(ACTION_NAMES_REVERSE[MOTION_MODEL][name])
