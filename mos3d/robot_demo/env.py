from mos3d import Mos3DEnvironment
from mos3d.oopomdp import RobotState, TargetObjectState, M3OOState,\
    Actions, CAMERA_INSTALLATION_POSE, MotionAction
from mos3d.models.world.sensor_model import FrustumCamera
from mos3d.models.world.world import GridWorld, OBJECT_MANAGER
from mos3d.models.world.robot import Robot
from mos3d.models.world.objects import Cube
from topo_policy_model import CAMERA_INSTALLATION_POSE

class SearchRegionEnvironment(Mos3DEnvironment):

    """Different from simulation, in the real world demo,
    we don't know the groundtruth location of the target
    objects; We know their types - which we'll simply
    represent as cubes internally. Hence, we don't model
    the transition function or reward model; We do obtain
    the real robot pose though, as published by localization
    modules."""

    def __init__(self,
                 # search region dimensions
                 w, l, h,
                 target_object_ids,  # list or set of marker ids for targets
                 init_robot_pose, # initial robot pose
                 # camera settings
                 fov=60,
                 aspect_ratio=1.0,
                 near=1.0,
                 far=7):
        camera_model = FrustumCamera(fov, aspect_ratio, near, far)
        robot = Robot(0,
                      CAMERA_INSTALLATION_POSE,
                      camera_model,
                      objtype="robot")
        objects = {objid: Cube(objid)
                   for objid in target_object_ids}
        gridworld = GridWorld(w+1, l+1, h+1, robot, objects, # reason for +1 is in mos3d/environment/env.py
                              robot_id=938,
                              occlusion_enabled=True)

        robot_state = RobotState(gridworld.robot_id,
                                 init_robot_pose,
                                 (), # objects found
                                 None)
        object_states = {gridworld.robot_id: robot_state}
        for objid in objects:
            object_state = TargetObjectState(objid,
                                             objects[objid].objtype,
                                             None)  # we don't know object's pose
            object_states[objid] = object_state
        init_state = M3OOState(gridworld.robot_id, object_states)
        super().__init__(init_state, gridworld, None, None)  # no transition or reward models
