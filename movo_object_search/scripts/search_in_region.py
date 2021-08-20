#!/usr/bin/env python
# Object search within a region
#
# NOTE: This script should be run in a Python2 shell
# which has access to ROS packages. It expects launch
# files that will serve the rosparams to be already
# running.

import argparse
import rospy
import time
import sensor_msgs.point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from aruco_msgs.msg import MarkerArray as ArMarkerArray
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, Point
import message_filters
import tf
import subprocess
import os
import yaml
import math
import json
from action.waypoint import WaypointApply
from action.head_and_torso import TorsoJTAS, HeadJTAS
from action.action_type import ActionType
from scipy.spatial.transform import Rotation as scipyR
from topo_marker_publisher import PublishTopoMarkers, PublishSearchRegionMarkers
from ros_util import get_param, get_if_has_param
from pprint import pprint

# Start a separate process to run POMDP; Use the virtualenv
VENV_PYTHON = "/home/kaiyuzh/pyenv/py37/bin/python"
POMDP_SCRIPT = "/home/kaiyuzh/repo/3d-moos-pomdp/moos3d/robot_demo/build_pomdp.py"

def to_rad(deg):
    return math.pi * deg / 180.0

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def read_pose_msg(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    qx = msg.pose.pose.orientation.x
    qy = msg.pose.pose.orientation.y
    qz = msg.pose.pose.orientation.z
    qw = msg.pose.pose.orientation.w
    return (x,y,z,qx,qy,qz,qw)


def list_arg(l):
    return " ".join(map(str,l))

def execute_action(action_info,
                   robot_state,
                   last_action_observation):
    """Exxecute the action described in action_info.
    navclient (SimpleActionClient) used to send goal
        to the navigation components."""
    obs_info = {"action_type": action_info["type"]}
    if action_info["type"] == ActionType.TopoMotionAction:
        # check source
        cur_robot_pose = wait_for_robot_pose()
        goal_tolerance = get_param('goal_tolerance')
        _gap = euclidean_dist(cur_robot_pose[:3], action_info["src_pose"])
        if _gap > goal_tolerance:
            rospy.logwarn("Robot is at %s, far from topo node location %s (dist=%.3f > %.3f); But won't stop moving."\
                          % (str(cur_robot_pose[:3]), str(action_info["src_pose"]), _gap, goal_tolerance))
        
        # navigate to destination
        position = action_info["dst_pose"]  # x,y,z
        # default rotation is (0,0,0) --> will look "forward"
        orientation = (0,0,0,1)  # equivalent to (0,0,0) in euler.
        if WaypointApply(position, orientation).status == WaypointApply.Status.SUCCESS:
            # Successfully moved robot; Return an observation about the robot state.
            rospy.loginfo("APPLIED MOTION ACTION SUCCESSFULLY")
            obs_info["status"] = "success"
        else:
            rospy.logwarn("FAILED TO APPLY MOTION ACTION")
            obs_info["status"] = "failed"
        obs_info["robot_pose"] = wait_for_robot_pose()
        # obs_info["torso_height"] = wait_for_torso_height()  # provides z pose.
        obs_info["objects_found"] = robot_state["objects_found"]
        obs_info["camera_direction"] = None  # consistent with transition model.

    elif action_info["type"] == ActionType.TorsoAction:
        torso_height = wait_for_torso_height()
        desired_height = torso_height + action_info["displacement"]
        TorsoJTAS.move(desired_height, current_height=torso_height)

        # If torso up, tilt camera up. If torso down, tilt camera down.
        if action_info["displacement"] > 0:
            rospy.loginfo("Tilting camera up")
            HeadJTAS.move(0, to_rad(5), v=0.8)
        if action_info["displacement"] < 0:
            rospy.loginfo("Tilting camera down")
            HeadJTAS.move(0, to_rad(-20), v=0.8)            

        # Verify
        actual_height = wait_for_torso_height()
        if abs(actual_height - desired_height) > 1e-2:
            rospy.logerr("Torso did not move to desired height. (Desired: %.3f ; Actual: %.3f)"
                         % (actual_height, desired_height))
        else:
            rospy.loginfo("Torso motion complete. Now torso at %.3f" % actual_height)
        # Get observation about robot state
        obs_info["status"] = "success"
        obs_info["robot_pose"] = wait_for_robot_pose()
        # obs_info["torso_height"] = actual_height  # provides z pose.
        obs_info["objects_found"] = robot_state["objects_found"]
        obs_info["camera_direction"] = None  # consistent with transition model.        

    elif action_info["type"] == ActionType.LookAction:
        rotation = action_info["rotation"]
        cur_robot_pose = wait_for_robot_pose()
        # Rotate the robot
        position = cur_robot_pose[:3]
        orientation = tuple(map(float, scipyR.from_euler("xyz", rotation, degrees=True).as_quat()))
        if WaypointApply(position, orientation).status == WaypointApply.Status.SUCCESS:
            # Successfully moved robot; Return an observation about the robot state.
            rospy.loginfo("Robot rotate successfully")
            obs_info["status"] = "success"
        else:
            rospy.logwarn("Failed to rotate robot")
            obs_info["status"] = "failed"

        # robot state
        rospy.loginfo("Finished rotation. Now observing...")        
        obs_info["robot_pose"] = wait_for_robot_pose()
        rospy.loginfo("Finished rotation. hello...")        
        # obs_info["torso_height"] = wait_for_torso_height()  # provides z pose.
        obs_info["objects_found"] = robot_state["objects_found"]
            
        if obs_info["status"] == "success":
            # Project field of view; Start the point cloud processing script,
            # one with ar tag detection, one without. Upon publishing the
            # volumetric observation, these scripts will save the observation
            # to a file.
            point_cloud_topic = get_param("point_cloud_topic")
            rospy.loginfo("Projecting field of view; Processing point cloud from %s." % point_cloud_topic)
            start_time = rospy.Time.now()
            voxels_dir = os.path.dirname(get_param('observation_file'))
            vpath_ar = os.path.join(voxels_dir, "voxels_ar.yaml")
            vpath = os.path.join(voxels_dir, "voxels.yaml")
            vdone_path = os.path.join(voxels_dir, "vdone.txt")  # signals voxel file save is complete
            vdone_path_ar = os.path.join(voxels_dir, "vdone_ar.txt")            

            start_pcl_process(save_path=vpath,
                              detect_ar=False,
                              step=action_info["step"])                
            start_pcl_process(save_path=vpath_ar,
                              detect_ar=True,
                              step=action_info["step"])

            # Remove existing
            if os.path.exists(vpath):
                os.remove(vpath)
            if os.path.exists(vpath_ar):
                os.remove(vpath_ar)
            
            # wait until files are present
            wait_time = max(1, get_param('point_cloud_wait_time'))
            ar_extra_wait_time = max(1, get_param('ar_extra_wait_time'))
            observation_saved = False
            while rospy.Time.now() - start_time < rospy.Duration(wait_time):
                if os.path.exists(vpath):
                    observation_saved = True
                    while rospy.Time.now() - start_time < rospy.Duration(wait_time + 5):
                        if os.path.exists(vdone_path):
                            os.remove(vdone_path)
                            break
                        else:
                            rospy.sleep(0.2)
                    break
                else:
                    rospy.loginfo("Waiting for voxel observation file.")
                    rospy.sleep(1.0)

            if observation_saved:
                rospy.loginfo("Voxel observation saved.")

            if not os.path.exists(vpath_ar):
                rospy.loginfo("Waiting for several more seconds for AR tag detection.")
                start_ar_wait = rospy.Time.now()
                while rospy.Time.now() - start_ar_wait < rospy.Duration(ar_extra_wait_time):
                    if os.path.exists(vpath_ar):
                        while rospy.Time.now() - start_ar_wait < rospy.Duration(ar_extra_wait_time + 5):
                            if os.path.exists(vdone_path_ar):
                                # we are good
                                os.remove(vdone_path_ar)
                                break
                            else:
                                rospy.sleep(0.3)
                        break
                    else:
                        rospy.loginfo("Waiting for voxel AR observation file.")
                        rospy.sleep(0.5)
            if os.path.exists(vpath_ar):
                # use ar tag detection observation
                rospy.loginfo("Using the voxels that may contain AR tag labels.")
                with open(vpath_ar) as f:
                    voxels = yaml.safe_load(f)
            elif os.path.exists(vpath):
                rospy.loginfo("Loading saved voxels..")
                with open(vpath) as f:
                    voxels = yaml.safe_load(f)
            else:
                rospy.logwarn("No volumetric observation was ever recieved within wait time=%ds. Action FAILED." % wait_time)
                obs_info["status"] = "failed"
                voxels = {}
                
            obs_info["camera_direction"] = orientation  # consistent with transition model.
            obs_info["voxels"] = voxels # volumetric observation; (from voxel id (int) to (voxel_pose, label)!
        else:
            # robot didn't reach the desired rotation; no observation received
            obs_info["status"] = "failed"
            obs_info["camera_direction"] = None  # consistent with transition model.
            obs_info["voxels"] = {}
            rospy.logerr("No observation received because desired rotation not reached.")
        rospy.set_param("pcl_process_%d_done" % action_info["step"], True)


    elif action_info["type"] == ActionType.DetectAction:
        # Based on last observation, mark objects as detected
        last_action, last_observation = last_action_observation
        if last_action is not None and last_observation is not None:
            target_object_ids = set(get_param('target_object_ids'))
            new_objects_found = set({})            
            if last_action["type"] == ActionType.LookAction:
                voxels = last_observation["voxels"]
                for voxel_pose in voxels:
                    _, label = voxels[voxel_pose]
                    if label == "free" or label == "occupied" or label == "unknown":
                        continue
                    if int(label) in target_object_ids:
                        new_objects_found.add(label)
                obs_info["objects_found"] = robot_state["objects_found"] | new_objects_found
        else:
            obs_info["objects_found"] = robot_state["objects_found"]

        # Nod head to indicate Detect action called.
        rospy.loginfo("Detection made. Nodding head...head tilting down")
        HeadJTAS.move(0, to_rad(0), v=0.8)
        rospy.loginfo("Detection made. Nodding head...head tilting back up")
        HeadJTAS.move(0, to_rad(-20), v=0.8)
        # robot state
        obs_info["robot_pose"] = wait_for_robot_pose()
        # obs_info["torso_height"] = wait_for_torso_height()  # provides z pose.
        obs_info["camera_direction"] = obs_info["robot_pose"][3:]  # consistent with transition model.
    return obs_info
        
        
def start_pcl_process(save_path, detect_ar=False, step=1):  # step: the planning step
    search_space_dimension = get_param('search_space_dimension')
    search_space_resolution = get_param('search_space_resolution')    
    fov = get_param('fov')
    asp = get_param('aspect_ratio')
    near = get_param('near')
    far = get_param('far')
    sparsity = get_param("sparsity")
    occupied_threshold = get_param("occupied_threshold")
    mark_nearby = get_param("mark_nearby")
    assert type(mark_nearby) == bool
    marker_topic = get_param("marker_topic") #"/movo_pcl_processor/observation_markers"
    point_cloud_topic = get_param("point_cloud_topic")
    target_object_ids = set(get_param('target_object_ids'))
    
    optional_args = []
    if detect_ar:
        optional_args.append("-M")
        marker_topic += "_ar"
        if mark_nearby:
            optional_args.append("-N")
    
    subprocess.Popen(["rosrun", "movo_object_search", "process_pcl.py",
                      "--plan-step", str(step),
                      "--save-path", str(save_path),
                      "--quit-when-saved",
                      "--point-cloud-topic", str(point_cloud_topic),
                      "--marker-topic", str(marker_topic),
                      "--resolution", str(search_space_resolution),
                      "--target-ids", list_arg(target_object_ids),
                      "--fov", str(fov),
                      "--asp", str(asp),
                      "--near", str(near),
                      "--far", str(far),
                      "--sparsity", str(sparsity),
                      "--occupied-threshold", str(occupied_threshold)]\
                     + optional_args)

def wait_for_robot_pose():
    robot_pose_topic = get_param('robot_pose_topic')
    msg = rospy.wait_for_message(robot_pose_topic, PoseWithCovarianceStamped, timeout=15)
    robot_pose = list(read_pose_msg(msg))
    torso_height = wait_for_torso_height()
    # Use torso height as the z coordinate
    robot_pose[2] = torso_height
    return tuple(robot_pose)

def wait_for_torso_height():
    torso_topic = get_param('torso_height_topic')  # /movo/linear_actuator/joint_states
    return TorsoJTAS.wait_for_torso_height(torso_topic=torso_topic)

########### SEARCH REGION FUNCTION ##########
def search_region(region_name, regions_file):
    with open(regions_file) as f:
        # This is the json region file
        data = json.load(f)
        region_data = data["regions"][region_name]
    region_origin = tuple(map(float, region_data["origin"][:2]))
    access_point = tuple(map(float, region_data["access"][:2]))
    search_space_dimension = int(region_data["dimension"])
    search_space_resolution = float(region_data["resolution"])
    rospy.set_param("search_space_dimension", search_space_dimension)
    rospy.set_param("search_space_resolution", search_space_resolution)
    rospy.set_param("region_name", region_name)
    
    # This parameter should be loaded by the regions_info.yaml file
    target_object_ids = get_param("%s_target_ids" % (region_name.replace("-", "_")))
    rospy.set_param("target_object_ids", target_object_ids)
    
    # target_object_ids = get_param('target_object_ids')  # a list
    _size = search_space_dimension * search_space_resolution
    rospy.loginfo("Total search space area: %.3f x %.3f x %.3f m^3"
                  % (_size, _size, _size))

    # files
    topo_map_file = get_param('topo_map_file')
    action_file = get_param('action_file')
    observation_file = get_param('observation_file')
    prior_file = get_param('prior_file')
    done_file = get_param('done_file')

    # clear exisiting action/observation files
    if os.path.exists(action_file):
        os.remove(action_file)
    if os.path.exists(observation_file):        
        os.remove(observation_file)  

    # other config
    observation_wait_time = get_param('observation_wait_time')
    action_wait_time = get_param('action_wait_time')    
    fov = get_param('fov')
    asp = get_param('aspect_ratio')
    near = get_param('near')
    far = get_param('far')
    torso_min = get_param('torso_min')
    torso_max = get_param('torso_max')

    # for volumetric observation
    sparsity = get_param("sparsity")
    occupied_threshold = get_param("occupied_threshold")
    mark_nearby = get_param("mark_nearby")

    # execution
    max_planning_steps = get_param("max_planning_steps")
    
    # publish topo markers
    PublishTopoMarkers(topo_map_file, search_space_resolution)
    PublishSearchRegionMarkers(region_origin, search_space_dimension, search_space_resolution)

    # Move to access point
    rospy.loginfo("Navigating to access point of region %s at %s"
                  % (region_name, str(access_point)))
    posit, orien = access_point, (0,0,0,1)
    if WaypointApply(posit, orien).status != WaypointApply.Status.SUCCESS:
        rospy.logerr("Waypoint to %d failed" % nid)
        return

    # tilt head
    HeadJTAS.move(0, to_rad(-20), v=0.8)    

    # Listen to robot pose
    robot_pose = wait_for_robot_pose()

    cmd = [VENV_PYTHON, POMDP_SCRIPT,
           # arguments
           topo_map_file,
           list_arg(robot_pose),
           str(search_space_dimension),
           list_arg(target_object_ids),
           list_arg(region_origin),
           str(search_space_resolution),
           action_file,
           observation_file,
           done_file,           
           "--torso-min", str(torso_min),
           "--torso-max", str(torso_max),
           "--wait-time", str(observation_wait_time),
           "--prior-file", prior_file,
           "--fov", str(fov),
           "--asp", str(asp),
           "--near", str(near),
           "--far", str(far)]
    rospy.loginfo("Starting POMDP with the following command:\n%s"
                  % subprocess.list2cmdline(cmd))
    subprocess.Popen(cmd)
    
    # Wait for an action and execute this action
    robot_state = {"objects_found": set({})}
    last_action_observation = (None, None)
    step = 0
    while not rospy.is_shutdown():
        rospy.loginfo("Waiting for action...(t=%d)" % step)
        start_time = rospy.Time.now()
        observation = None
        observation_issued = False
        while rospy.Time.now() - start_time < rospy.Duration(action_wait_time):
            if os.path.exists(action_file):
                rospy.loginfo("Got action! Executing action...")
                with open(action_file) as f:
                    action_info = yaml.load(f)

                    # observation obtained from robot                    
                    obs_info = execute_action(action_info,  
                                              robot_state,
                                              last_action_observation)
                    robot_state["objects_found"] = obs_info["objects_found"]

                    with open(observation_file, "w") as f:
                        yaml.safe_dump(obs_info, f)                    
                    rospy.loginfo("Action executed. Observation written to file %s" % observation_file)
                    with open(done_file, "w") as f:
                        f.write("done")

                    last_action_observation = (action_info, obs_info)
                    observation_issued = True

                    # remove action file and done file
                    os.remove(action_file)
                    break  # break the loop
            else:
                rospy.loginfo("Waiting for POMDP action...")
                rospy.sleep(0.5)
        if not observation_issued:
            rospy.logerr("Timed out waiting for POMDP action.")
            break
        if robot_state["objects_found"] == set(target_object_ids):
            rospy.loginfo("Done! All objects found in this region (Num Steps: %d)" % (step+1))
            break
        step += 1
        if step >= max_planning_steps:
            rospy.loginfo("Maximum planning step reached for this region. "
                          "Found %d objects (%s)" % (len(robot_state["objects_found"]),
                                                     robot_state["objects_found"]))
    return robot_state["objects_found"], step


########### SEQUENTIALLY SEARCH MULTIPLE REGIONS ##########            
def main():
    rospy.init_node("movo_object_search_in_region",
                    anonymous=True)

    regions_file = get_if_has_param("regions_file")
    regions = get_if_has_param("regions")
    if regions is None:
        region_name = get_param("region_name")
        regions = [region_name]
    else:
        regions = get_param("regions").split(",")  # list of region names separated by comma.

    results = {}

    # Multiple region search
    for region_name in regions:
        rospy.loginfo("[start region search] Start searching in region %s" % region_name)
        region_name = region_name.strip()
        objects_found, step = search_region(region_name, regions_file)
        results[region_name] = {"objects_found": objects_found,
                                "steps_taken": step}

    pprint(results)
    
if __name__ == "__main__":    
    main()
