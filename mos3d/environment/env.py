import pomdp_py
from moos3d.oopomdp import RobotState, TargetObjectState, M3OOState,\
    Actions, CAMERA_INSTALLATION_POSE, MotionAction
from moos3d.models.world.sensor_model import FrustumCamera
from moos3d.models.world.world import GridWorld, OBJECT_MANAGER
from moos3d.models.world.robot import Robot
import moos3d.util as util
import copy
import time
import random


def parse_worldstr(worldstr, robot_id=0):
    """
    Parse the given world string, with the following format:
    The first three lines are width, length and height of the gridworld.
    Then there is a white line.
    Then, each line specifies an object:
        <objtype>-<objid;optional> <x> <y> <z> <"obstacle";optional>
    If the 'objid' is unspecified, then the id for this object will be:
        `robot_id` + L + 1 where L is the number of objects specified before this one.
    After all objects, there is a line "---"
    Then, it is the robot specification:
        "robot" <x> <y> <z> <row> <pitch> <yaw> <camera_config>
    where the <camera_config> is:
        <camera_type|{frustum, occlusion}> <fov> <aspect_ratio> <near> <far>
    Example:
    '''
    4
    4
    4

    teewee-5 0 0 0
    cube 1 0 0
    hero-19 0 0 1
    cube 0 1 0 obstacle
    ---
    robot 2 0 0 0 0 0 occlusion 45 1.0 0.1 4
    '''

    Returns a tuple (gridworld, init_true_state)"""
    # Things to return
    # w, l, h, robot, objects, robot_pose, object_poses
    objects = {}
    object_poses = {}
    obstacles = set({})  # set of object ids that are just obstacles
    obstacle_color = (0.7, 0.7, 0.7)  # color for obstacles are the same
    hidden = set({})  # hidden grids will never show up in observation.

    lines = [l for l in worldstr.splitlines()
             if len(l) > 0]
    state = "w"
    for i, line in enumerate(lines):
        try:
            line = line.rstrip()
            if len(line) == 0:
                continue # blank line
            if line.startswith("#"):
                continue # comment

            if state == "w":
                w = int(line) + 1  # +1 so that actually w squares will be drawn on x axis
                state = "l"
            elif state == "l":
                l = int(line) + 1
                state = "h"
            elif state == "h":
                h = int(line) + 1
                state = "obj"
            elif state == "obj":
                if line == "---":
                    state = "robot"
                    continue
                tokens = line.split()
                objtype = tokens[0]  # string representation of the object class
                if "-" in objtype:
                    # This object is given an id.
                    objid = int(objtype.split("-")[1])
                    objtype = objtype.split("-")[0]
                    if objid in objects:
                        raise ValueError("Invalid object id %d; It already exists" % objid)
                else:
                    objid = robot_id + 1 + len(objects)
                x = int(tokens[1])
                y = int(tokens[2])
                z = int(tokens[3])
                info = OBJECT_MANAGER.info(objtype)
                if info is None:
                    raise ValueError("Error in parsing world string. "\
                                     "%s is not a valid object type" % objtype)

                if len(tokens) == 5 and tokens[4] == "obstacle":
                    obstacles.add(objid)
                    objects[objid] = info[0](objid, objtype=objtype, color=obstacle_color)
                elif len(tokens) == 5 and tokens[4] == "hidden":
                    hidden.add((x,y,z))
                    objects[objid] = info[0](objid, objtype=objtype)
                else:
                    objects[objid] = info[0](objid, objtype=objtype)
                object_poses[objid] = (x,y,z)
            elif state == "robot":
                tokens = line.split()
                objtype = tokens[0]
                x = int(tokens[1])
                y = int(tokens[2])
                z = int(tokens[3])
                thx = int(tokens[4])
                thy = int(tokens[5])
                thz = int(tokens[6])
                if tokens[7] == "frustum":
                    fov, asp, near, far = float(tokens[8]), float(tokens[9]),\
                        float(tokens[10]), float(tokens[11])
                    camera_model = FrustumCamera(fov, asp, near, far)
                    occlusion_enabled = False
                elif tokens[7] == "frustum-occlusion" or tokens[7] == "occlusion":
                    fov, asp, near, far = float(tokens[8]), float(tokens[9]),\
                        float(tokens[10]), float(tokens[11])
                    camera_model = FrustumCamera(fov, asp, near, far)
                    occlusion_enabled = True
                robot = Robot(robot_id,
                              CAMERA_INSTALLATION_POSE, 
                              camera_model,
                              objtype=objtype)  # only supports one robot per world now.
                # use quaternion
                qx, qy, qz, qw = util.euler_to_quat(thx, thy, thz)
                robot_pose = (x, y, z, qx, qy, qz, qw) #(x,y,z,thx,thy,thz)
        except Exception as ex:
            print("!!!!! Line %d caused an error! !!!!!" % i)
            print(ex)
            raise ex
    # Build gridworld
    gridworld = GridWorld(w, l, h, robot, objects, robot_id=robot_id,
                          occlusion_enabled=occlusion_enabled,
                          obstacles=obstacles, hidden=hidden)
    # Build initial state
    robot_state = RobotState(gridworld.robot_id,
                             robot_pose,
                             (), # objects found
                             None)
    object_states = {gridworld.robot_id: robot_state}
    # The true state of the world includes both target objects and obstacles, so that
    # correct occlusion can be simulated by the environment.
    for objid in objects:
        object_state = TargetObjectState(objid,
                                         objects[objid].objtype,
                                         object_poses[objid])
        object_states[objid] = object_state
    init_state = M3OOState(gridworld.robot_id, object_states)
    return gridworld, init_state


def random_3dworld(config):
    """config contains:
    width,
    length,
    height,
    objtypes: {objtype: amount}
    robot_camera: "string"  e.g. "frustum 60 1.0 1.0 10"
    """
    def random_good_pose(obj, occupied_voxels, config, seconds=15):
        pose = None
        starttime = time.time()
        while time.time() - starttime < seconds:
            pose = (random.randint(0, config['width']-1),
                    random.randint(0, config['length']-1),
                    random.randint(0, config['height']-1))
            pose_ok = True
            for cube_pose in objects[objid].cube_poses(*pose):
                if tuple(cube_pose) in occupied_voxels\
                   or not(util.in_range(cube_pose[0], (0, config['width']))\
                          and util.in_range(cube_pose[1], (0, config['length']))\
                          and util.in_range(cube_pose[2], (0, config['height']))):
                    pose_ok = False
                    break
            if pose_ok:
                break
        return pose

    objm = OBJECT_MANAGER
    
    # generate a world string
    occupied_voxels = set({})
    objects = {}
    world_str = "%s\n%s\n%s\n\n" % (config['width'], config['length'], config['height'])
    objid = 0
    seconds = 10  # time to try generating a pose    
    for objtype in config['objtypes']:
        info = objm.info(objtype)
        objects[objid] = info[0](objid, objtype=objtype)

        pose = None
        for i in range(config['objtypes'][objtype]):
            pose = random_good_pose(objects[objid], occupied_voxels, config, seconds=seconds)
            if pose is None:
                raise ValueError("Cannot find a place to put a %s after %d seconds of trying!"
                                 % (objtype, seconds))
        
            for cube_pose in objects[objid].cube_poses(*pose):
                occupied_voxels.add(tuple(cube_pose))

            x, y, z = pose
            world_str += "%s %d %d %d\n" % (objtype, x, y, z)
    world_str += "---\n"

    # generate robot pose
    pose = random_good_pose(objects[objid], occupied_voxels, config, seconds=seconds)
    if pose is not None:
        rx, ry, rz = pose
    else:
        raise ValueError("Cannot find a place to put robot after %d seconds of trying!"
                                 % (seconds))        
    world_str += "robot %d %d %d 0 0 0 %s" % (rx, ry, rz, config['robot_camera'])
    return world_str



class Mos3DEnvironment(pomdp_py.Environment):

    def __init__(self, init_state, gridworld, transition_model, reward_model):
        self._gridworld = gridworld
        pomdp_py.Environment.__init__(self, init_state,
                                      transition_model, reward_model) # NOT READY YET

    @property
    def robot_pose(self):
        return self.state.object_states[self._gridworld.robot_id]['pose']

    @property
    def object_poses(self):
        return {objid:self.state.object_states[objid]['pose']
                for objid in self.state.object_states
                if objid != self._gridworld.robot_id}

    def move_robot(self, action):
        if not hasattr(action, "motion")\
           or action.motion is None:
            raise ValueError("Cannot move robot with action %s" % action)
        self.state_transition(action, execute=True)

    def total_distance_to_undetected_objects(self):
        totaldist = 0
        for objid in self._gridworld.target_objects:
            if objid not in self.state.robot_state['objects_found']:
                totaldist += util.euclidean_dist(self.state.robot_pose[:3],
                                                 self.state.object_states[objid]['pose'])
        return totaldist

    def action_valid(self, action):
        """Returns true if the given action is allowed to be performed
        on the current environment state. This is used to trigger agent
        replan. The rationale is that, since the agent has no access
        to environment state, its planner may produce a bad action (e.g.
        bumping into the wall); However, such action cannot be performed.
        """
        # See what state this action may lead to
        if isinstance(action, MotionAction):
            next_state, _ = self.state_transition(action, execute=False)
            # If action is motion and robot didn't move, then the action is not valid.
            if next_state.robot_pose[:3] == self.state.robot_pose[:3]:
                return False
        return True

