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

# In the robot demo, this script is called from the
# search_in_region script as a subprocess; It is not
# expected to start running by itself.
import argparse
import pickle
import os
from collections import deque
from action_type import ActionType
import time
import matplotlib.pyplot as plt
from mos3d.robot_demo.conversion import convert, Frame
from mos3d.tests.experiments.runner import *
from mos3d import *
from mos3d.robot_demo.env import SearchRegionEnvironment
from mos3d.robot_demo.topo_policy_model import TopoPolicyModel, TopoMotionAction, TorsoAction, look_action_for
from mos3d.robot_demo.topo_maps.topological_graph import TopoMap
from mos3d.util import print_info, print_error, print_warning, print_success,\
    print_info_bold, print_error_bold, print_warning_bold, print_success_bold, print_note_bold, print_note
import mos3d.util as util
from mos3d.tests.trial import RunTrial
import yaml

##### Change configuration here #####
# Observation parameters
alpha =1e6
beta = 0.5
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
setting = [(1,1,max_depth), (2,1,max_depth), (4,1,max_depth)]# for hierarchical planning

plot_belief = True

##### Create an MOS3D Instance, with Hierarchical Planner.
model_cfg = model_config(alpha=alpha, beta=beta, gamma=gamma,
                         detect_after_look=detect_after_look,
                         big=big, medium=medium, small=small)
planner_cfg = planner_config("hierarchical", max_depth=max_depth,
                             discount_factor=discount_factor,
                             planning_time=planning_time,
                             exploration_const=exploration_const,
                             setting=setting)

def get_action_type(action):
    if isinstance(action, TopoMotionAction):
        return ActionType.TopoMotionAction
    elif isinstance(action, TorsoAction):
        return ActionType.TorsoAction
    elif isinstance(action, LookAction):
        return ActionType.LookAction
    elif isinstance(action, DetectAction):
        return ActionType.DetectAction
    else:
        raise ValueError("action %s is of unknown type." % str(action))

def get_action_info(action, action_file, region_origin, search_space_resolution, step):
    action_info = {}
    action_info["type"] = get_action_type(action)
    action_info["name"] = action.name
    if isinstance(action, TopoMotionAction):
        action_info["src_pose"] = convert(action.src, Frame.POMDP_SPACE, Frame.WORLD,
                                          region_origin=region_origin,
                                          search_space_resolution=search_space_resolution)
        action_info["dst_pose"] = convert(action.dst, Frame.POMDP_SPACE, Frame.WORLD,
                                          region_origin=region_origin,
                                          search_space_resolution=search_space_resolution)
    elif isinstance(action, TorsoAction):
        action_info["direction"] = action.name
        action_info["displacement"] = action.motion[0][2] * search_space_resolution

    elif isinstance(action, LookAction):
        action_info["rotation"] = action.motion[1]
        action_info["direction"] = action.name

    elif action == ActionType.DetectAction:
        pass  # nothing
    action_info["step"] = step
    return action_info

def load_prior_belief(prior_file,
                      region_origin, search_space_resolution):
    """
    The prior file should be formatted as:

    objid:
        regions:
            resolution_level: 1     // resolution level of this region
            pose: [x, y, z]         // the center of the region, in world frame, (at the ground level)
            belief: e.g. 10000      // unnormalized belief about this region.
    """
    if os.path.exists(prior_file):
        with open(prior_file) as f:
            prior_data = yaml.load(f)
            if prior_data is None:
                return {}
    else:
        return {}
    prior = {}   # objid -> {(x,y,z,r) -> value}
    for objid in prior_data:
        prior[objid] = {}
        for region in prior_data[objid]["regions"]:
            world_pos = region["pose"]
            reslevel = region["resolution_level"]
            belief = region["belief"]

            # Convert world pose to POMDP pose
            pomdp_pos = list(convert(world_pos, Frame.WORLD, Frame.POMDP_SPACE,
                                     region_origin=region_origin,
                                     search_space_resolution=search_space_resolution))
            # scale by resolution level
            pomdp_pos[0] = pomdp_pos[0] // reslevel
            pomdp_pos[1] = pomdp_pos[1] // reslevel
            pomdp_pos[2] = pomdp_pos[2] // reslevel
            octree_voxel = tuple(pomdp_pos[:3] + [reslevel])
            prior[objid][octree_voxel] = belief
    return prior

def verify_topo_map(topo_map, region_origin):
    # Verify that all nodes in the topo_map have positive indices with respect to the region_origin
    for nid in topo_map.nodes:
        region_pose = convert(topo_map.nodes[nid].pose,
                              Frame.WORLD, Frame.REGION,
                              region_origin)
        if region_pose[0] < 0 or region_pose[1] < 0 or region_pose[2] < 0:
            return False
    return True

def subgraph_in_region(topo_map, region_origin,
                       search_region_dimension, search_region_resolution):
    def good(pomdp_pose, pomdp_dimension):
        return (pomdp_pose[0] >= 0 and pomdp_pose[1] >= 0 and pomdp_pose[2] >= 0)\
            and (pomdp_pose[0] < pomdp_dimension and pomdp_pose[1] < pomdp_dimension and pomdp_pose[2] < pomdp_dimension)

    # region_size = search_region_dimension * search_region_resolution

    valid_edges = set({})
    for eid in topo_map.edges:
        edge = topo_map.edges[eid]
        nid1, nid2 = edge.nodes[0].id, edge.nodes[1].id
        pomdp_pose1 = convert(topo_map.nodes[nid1].pose,
                               Frame.WORLD, Frame.POMDP_SPACE,
                               region_origin, search_region_resolution)
        pomdp_pose2 = convert(topo_map.nodes[nid2].pose,
                              Frame.WORLD, Frame.POMDP_SPACE,
                              region_origin, search_region_resolution)
        if good(pomdp_pose1, search_region_dimension) and good(pomdp_pose2, search_region_dimension):
            valid_edges.add(edge)
    # Get the biggest connected component:
    components = TopoMap(valid_edges).connected_components()
    return max(components, key=lambda c: len(c.nodes))

def print_last_planning_step_info(planner):
    assert type(planner) == MultiResPlanner, "Planner is not MultiRes!"
    info = ""
    for key in planner._last_results:
        if type(key) == tuple:
            action, action_value, num_sims = planner._last_results[key]
            info += ("     Planner %s    Value: %.3f    Num Sim: %d   [%s]\n"\
                     % (str(key), action_value, num_sims, str(action)))
    info += ("Best chocie: %s" % str(planner._last_results['__chosen__']))
    print_note(info)

def plot_oo_octree_belief(belief, gridworld, fig, axes, next_robot_state):
    for objid in gridworld.target_objects:
        belief_obj = belief.object_belief(objid)
        if isinstance(belief_obj, OctreeBelief) and objid in axes:
            for artist in axes[objid].collections:
                artist.remove()
            m = plot_octree_belief(axes[objid], belief_obj, robot_pose=next_robot_state.robot_pose,
                                   alpha="clarity", edgecolor="black", linewidth=0.1)
            fig.canvas.draw()
            fig.canvas.flush_events()

########## Local sanity check (comment out when really running) ##########
LOCAL_CHECK = False

def local_check(topo_map, pomdp_init_robot_pose,
                region_origin, torso_range, prior, args):
    ## Basically, with real object / robot initial locations, but just
    ## a different action space compared to simulation.
    print("Assuming the world is test_simple")
    exec_config = {
        'max_steps': 500,
        'max_time': 120,  # seconds
        'plot_belief': True,
        'plot_tree': False,
        'plot_analysis': False,
        'viz': True,
        'anonymize': True
    }

    world_str = "%s\n%s\n%s\n\n" % (str(args.dim), str(args.dim), str(args.dim))
    object_poses = {
        0: [1.35, 1.66, 1.02],
        1: [2.8, 0, 0.9],
        2: [0, 1.8, 1.30]
    }
    for objid in object_poses:
        search_space_pose = convert(object_poses[objid], Frame.WORLD, Frame.POMDP_SPACE,
                                    region_origin=region_origin,
                                    search_space_resolution=args.search_space_resolution)
        world_str += "cube-%d %d %d %d\n" % (objid, search_space_pose[0],
                                             search_space_pose[1],
                                             search_space_pose[2])
    world_str += "---\n"
    rx, ry, rz = pomdp_init_robot_pose[:3]
    robot_camera = "occlusion %.3f %.3f %.3f %.3f" % (args.fov, args.asp, args.near, args.far)
    world_str += "robot %d %d %d 0 0 0 %s" % (rx, ry, rz, robot_camera)
    gridworld, init_state = parse_worldstr(world_str, robot_id=100)

    Ov = M3ObservationModel(gridworld, **model_cfg['O'], voxel_model=True)
    Om = M3ObservationModel(gridworld, **model_cfg['O'], voxel_model=False)
    T = M3TransitionModel(gridworld, **model_cfg['T'])
    Tenv = M3TransitionModel(gridworld, for_env=True, **model_cfg['T'])
    R = GoalRewardModel(gridworld, **model_cfg['R'])
    pi = TopoPolicyModel(topo_map,
                         region_origin=region_origin,
                         search_space_resolution=args.search_space_resolution,
                         torso_range=torso_range,
                         **model_cfg['Pi'])
    prior_belief = AbstractM3Belief(gridworld, init_octree_belief(gridworld,
                                                                  init_state.robot_state,
                                                                  prior=prior))
    # agent
    agent = M3Agent(gridworld, prior_belief, pi, T, Ov, R)

    # Hierarchical planner
    planner = MultiResPlanner(setting, agent, gridworld,
                              abstract_policy=pi,  # for now just use the same motion policy
                              **planner_cfg['init_kwargs'])
    # env, local.
    env = Mos3DEnvironment(init_state, gridworld, Tenv, R)

    RunTrial(gridworld, agent, env, Om, planner, exec_config,
             forced_actions=deque())
################# sanity check ends ###################

###  For debugging purposes  ###
USING_FORCED_ACTIONS = False
FORCED_ACTIONS = deque([DetectAction()])  #look_action_for("look-left"),
################################

def main():
    parser = argparse.ArgumentParser(description='Build a POMDP; Outputs an action to a file')
    parser.add_argument('topo_map_file', type=str,
                        help='Path to topolocial map file')
    parser.add_argument('init_robot_pose', type=str,
                        help='initial robot pose; Format, floats separated by a space'\
                        '"x y z qx qy qz qw"')
    parser.add_argument('dim', type=int,
                        help='Search region dimension')
    parser.add_argument('target_object_ids', type=str,
                        help='object ids of target objects. Format, "id1 id2 id3..."')
    parser.add_argument('region_origin', type=str,
                        help='origin of region in world frame. Format, "x y z"')
    parser.add_argument('search_space_resolution', type=float,
                        help='resolution of search region. Format, float')
    parser.add_argument('action_file', type=str,
                        help='Plans an action and outputs the action to this file location.')
    parser.add_argument('observation_file', type=str,
                        help="Path to observation file; The format depends on type of action.")
    parser.add_argument('done_file', type=str,
                        help="Path to done file which indicates observation saving is done.")
    parser.add_argument('--torso-min', type=float,
                        help="Minimum height of torso in meters",
                        default=0.1)
    parser.add_argument('--torso-max', type=float,
                        help="Maximum height of torso in meters",
                        default=1.5)
    parser.add_argument('-w', '--wait-time', type=int,
                        help="Time (s) to wait for the observation file to show up.",
                        default=60)
    parser.add_argument('-p', '--prior-file', type=str,
                        help="path to a file (yaml) that specifies the prior belief.")
    parser.add_argument('--fov', type=float,
                        help="FOV angle",
                        default=60)
    parser.add_argument('--asp', type=float,
                        help="aspect ratio",
                        default=1.0)
    parser.add_argument('--near', type=float,
                        help="near plane",
                        default=0.1)
    parser.add_argument('--far', type=float,
                        help="far plane",
                        default=7)
    args = parser.parse_args()

    target_object_ids = tuple(map(int, args.target_object_ids.split(" ")))
    region_origin = tuple(map(float, args.region_origin.split(" ")))
    torso_range = (args.torso_min, args.torso_max)

    # process args
    init_robot_pose = tuple(map(float, args.init_robot_pose.split(" ")))
    pomdp_pos = convert(init_robot_pose, Frame.WORLD, Frame.POMDP_SPACE,
                        region_origin=region_origin,
                        search_space_resolution=args.search_space_resolution)
    pomdp_init_robot_pose = tuple(pomdp_pos + init_robot_pose[3:])
    # Building POMDP; Then obtain a subgraph of this topological graph
    # where all nodes have positive region pose, which are then valid
    # robot poses for this search region.
    topo_map = subgraph_in_region(TopoMap.load(args.topo_map_file), region_origin,
                                  args.dim, args.search_space_resolution)
    if len(topo_map.nodes) == 0:
        raise ValueError("Region contains no topological graph nodes.")
        return

    # Load prior; # objid -> {(x,y,z,r) -> value}
    prior = load_prior_belief(args.prior_file,
                              region_origin, args.search_space_resolution)

    if LOCAL_CHECK:
        local_check(topo_map, pomdp_init_robot_pose,
                    region_origin, torso_range, prior, args)
        return

    env = SearchRegionEnvironment(args.dim, args.dim, args.dim,
                                  target_object_ids,
                                  pomdp_init_robot_pose,  # should be in pomdp space already
                                  fov=args.fov,
                                  aspect_ratio=args.asp,
                                  near=args.near,
                                  far=args.far)

    init_state = env.state
    gridworld = env._gridworld
    prior_belief = AbstractM3Belief(gridworld, init_octree_belief(gridworld,
                                                                  init_state.robot_state,
                                                                  prior=prior))

    # OOPOMDP models
    Ov = M3ObservationModel(gridworld, voxel_model=True, **model_cfg['O'])
    T = M3TransitionModel(gridworld, **model_cfg['T'])
    R = GoalRewardModel(gridworld, **model_cfg['R'])
    pi = TopoPolicyModel(topo_map,
                         region_origin=region_origin,
                         search_space_resolution=args.search_space_resolution,
                         torso_range=torso_range,
                         **model_cfg['Pi'])

    # agent
    agent = M3Agent(gridworld, prior_belief, pi, T, Ov, R)

    # Hierarchical planner
    planner = MultiResPlanner(setting, agent, gridworld,
                              abstract_policy=pi,  # for now just use the same motion policy
                              **planner_cfg['init_kwargs'])
    print_info("POMDP and planner initialized")

    # plot belief init
    if plot_belief:
        plt.figure(0)
        plt.ion()
        fig = plt.gcf()
        axes = {}
        if len(gridworld.objects) == 1:# or gridworld.width > 8:
            nobj_plot = 1
            shape = (1,1)
        elif len(gridworld.objects) <= 4:
            nobj_plot = len(gridworld.objects)
            shape = (2, 2)
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

        # plot initial belief
        plot_oo_octree_belief(agent.cur_belief, gridworld, fig, axes, init_state.robot_state)

    # Execution starts
    next_robot_state = None
    step = 0
    while True:
        print_info("Planning...(t=%d)" % step)
        if USING_FORCED_ACTIONS and len(FORCED_ACTIONS) > 0:
            print_warning("Taking FORCED ACTION (only for debugging purposes!)")
            real_action = FORCED_ACTIONS.popleft()
        else:
            if USING_FORCED_ACTIONS:
                print_success_bold("FORCED ACTIONS all executed. Done.")
                break
            real_action = planner.plan(agent)
            print_last_planning_step_info(planner)
        print_note_bold("    Action selected: %s" % str(real_action))
        print_info("Executing action...")
        action_info = get_action_info(real_action, get_action_type(real_action),
                                      region_origin, args.search_space_resolution,
                                      step)
        with open(args.action_file, "w") as f:
            yaml.dump(action_info, f)
        print_info("    Written action to file %s" % args.action_file)
        print_info("    Waiting for action to be executed...")
        action_status = "executing"  # {"executing", "done", "replan"}
        start_time = time.time()
        while time.time() - start_time < args.wait_time:
            if os.path.exists(args.observation_file):
                print_info("Action executed!")
                while time.time() - start_time < args.wait_time:
                    if os.path.exists(args.done_file):
                        print_info("Observation save complete.")
                        os.remove(args.done_file)
                        break
                    else:
                        time.sleep(0.2)

                # This loaded observation may contain additional information such
                # as robot pose or detected objects, depending on the type of action.
                with open(args.observation_file) as f:
                    obs_info = yaml.safe_load(f)

                # Check status; If failed, then replan
                if "status" in obs_info\
                   and obs_info["status"] == "failed":
                    action_status = "replan"
                    break  # break this wait-loop and replan.

                next_robot_state = update(agent, planner, real_action, obs_info,
                                          region_origin, args.search_space_resolution)
                action_status = "done"
                print_info("Agent and planner updated!")
                print_info_bold("Robot state %s" % str(next_robot_state))

                # Plot belief
                if plot_belief:
                    plt.figure(0)
                    plot_oo_octree_belief(agent.cur_belief, gridworld, fig, axes, next_robot_state)
                    # for objid in gridworld.target_objects:
                    #     belief_obj = agent.cur_belief.object_belief(objid)
                    #     if isinstance(belief_obj, OctreeBelief) and objid in axes:
                    #         for artist in axes[objid].collections:
                    #             artist.remove()
                    #         m = plot_octree_belief(axes[objid], belief_obj, robot_pose=next_robot_state.robot_pose,
                    #                                alpha="clarity", edgecolor="black", linewidth=0.1)
                    #         fig.canvas.draw()
                    #         fig.canvas.flush_events()

                # remove observation file
                os.remove(args.observation_file)
                break  # break the loop
            else:
                time.sleep(0.5)

            if int(time.time() - start_time) % 4 == 0:
                print_info("    ...Waiting for action to be executed...")

        if action_status == "replan":
            print_warning("Action did not complete successfully. Replan.")

        elif action_status == "executing":  # still executing but already timed out
            raise ValueError("Action execution timed out. Didn't receive observation within %.3fs."
                             % args.wait_time)

        if next_robot_state is not None\
           and set(next_robot_state.objects_found) == set(target_object_ids):
            print_info("Done!")
            break  # Task finished!

        step += 1


def read_volumetric_observation(gridworld, real_action, pomdp_robot_pose, voxels):
    """Given voxels which map from int to (voxel_pose, label),
    where voxel pose is with respect to the robot camera frame,
    return the OOObservation object representing these voxels."""
    # This may be confusing so I must explain here.
    # The robot's camera looks at +x with respect to the world frame.
    #    Therefore, the robot's camera in the POMDP also looks at +x
    #    with respect to the POMDP search space. Checkout CAMERA_INSTALLATION_POSE
    #    in the topo_policy_model.py
    # The robot's camera itself looks at -z with respect to the camera's own
    # frame in POMDP. However, on the real robot, the frame "movo_camera_ir_optical_frame"
    # makes the robot look at +z direction. I manually inverted the voxel z coordinates
    # when projecting the voxels in RViz - but the voxels saved are still from
    # the POMDP camera which is looking at -z
    #
    # To sum up, gridworld.robot.camera_model looks at +x, and the voxels
    # obtained from obs_info["voxels"] have their coordinates according
    # to a "vanilla" camera which looks at -z.
    #
    # Hence, to sync these two frames, I need, given a voxel_pose(rx,ry,rz)
    # from obs_info["voxels"] ('r' indicates it comes from the robot):
    #
    # px = rz
    # py = ry
    # pz = rx
    #
    # Just swap x and z. This should then be correct --> I was correct.

    gridworld.robot.camera_model.print_info()

    # Match voxels in volume to the voxels.
    pomdp_voxels = {}

    # volume = set(map(tuple, gridworld.robot.camera_model.volume[:,:3].astype(int)))
    # print(volume)

    # turn voxels into {voxel_pose -> label}
    proc_voxels = {tuple(map(int, voxels[i][0])):voxels[i][1] for i in voxels}
    for voxel_pose in proc_voxels:
        # Note that in POMDP, the camera by default looks at +x,
        # but the robot's camera looks at +z. Therefore, we
        # swap x and z of the homo_voxel_pose.
        pomdp_cam_voxel_pose = (
            int(voxel_pose[2]),  # z -> x
            int(voxel_pose[1]),  # y
            int(voxel_pose[0]),  # x -> z
            1  # homo-geneous
        )
        # assert pomdp_cam_voxel_pose[:3] in volume, "Unknown pomdp camera pose %s" % (str(pomdp_cam_voxel_pose[:3]))
        label = proc_voxels[voxel_pose]
        # Transform the voxel_pose from camera space to world space in POMDP
        x, y, z, qx, qy, qz, qw = pomdp_robot_pose
        R = util.R_quat(qx, qy, qz, qw, affine=True)
        pomdp_voxel_pose = np.transpose(np.matmul(util.T(x, y, z),
                                                  np.matmul(R, np.array(pomdp_cam_voxel_pose)))).astype(int)
        pomdp_voxel_pose = tuple(pomdp_voxel_pose[:3])

        if not gridworld.in_boundary(pomdp_voxel_pose):
            print_info("   Pose %s is not in gridworld boundary (%d^3)"
                       % (str(pomdp_voxel_pose), gridworld.width))
            continue   # filter out voxels not in the search region.

        print_note("   Pose %s is observed to be %s"
                       % (str(pomdp_voxel_pose), label))

        if label == "free" or label == "occupied":
            voxel = Voxel(pomdp_voxel_pose, Voxel.FREE)
        elif label == "unknown":
            voxel = Voxel(pomdp_voxel_pose, Voxel.UNKNOWN)
        else:
            assert label in gridworld.target_objects,\
                ("Unrecognized object label %d" % label)
            voxel = Voxel(pomdp_voxel_pose, label)
            print_note_bold("Observed object %d at location (pomdp: %s)"
                            % (label, str(pomdp_voxel_pose)))
        pomdp_voxels[pomdp_voxel_pose] = voxel

    return OOObservation(pomdp_voxels, OOObservation.T_VOLUME)


def update(agent, planner, real_action, obs_info,
           region_origin, search_space_resolution):
    print_info("Processing saved observation info...")

    # Process observation info; z should already be torso height
    robot_pose = list(map(float, obs_info["robot_pose"]))
    # torso_height = float(obs_info["torso_height"])
    # robot_pose[2] = torso_height  # set z to be torso height
    # Convert pose to POMDP space
    robot_pose[:3] = convert(robot_pose[:3], Frame.WORLD, Frame.POMDP_SPACE,
                             region_origin=region_origin,
                             search_space_resolution=search_space_resolution)

    # Convert voxels to POMDP space observation
    if "voxels" in obs_info:
        assert isinstance(real_action, LookAction)
        print_info_bold("Volumetric observation received by POMDP agent.")

        pomdp_observation = read_volumetric_observation(agent._gridworld, real_action,
                                                        robot_pose, obs_info["voxels"])

    else:
        pomdp_observation = OOObservation({}, None)


    objects_found = set(map(int, obs_info["objects_found"]))
    if obs_info["camera_direction"] != None:
        camera_direction = tuple(map(float, obs_info["camera_direction"]))
    else:
        camera_direction = None

    # next robot state
    next_robot_state = RobotState(agent._gridworld.robot_id,
                                  tuple(robot_pose),
                                  tuple(objects_found),
                                  camera_direction)
    # belief update
    print_info("Updating belief...")
    O = agent.observation_model
    for objid in agent.cur_belief.object_beliefs:
        belief_obj = agent.cur_belief.object_belief(objid)
        if objid == agent._gridworld.robot_id:
            # we assume robot has perfect knowledge of itself.
            new_belief = pomdp_py.Histogram({next_robot_state: 1.0})
        elif isinstance(belief_obj, OctreeBelief):
            new_belief = update_octree_belief(belief_obj,
                                              real_action, pomdp_observation,
                                              alpha=O[objid].alpha,
                                              beta=O[objid].beta,
                                              gamma=O[objid].gamma)
        else:
            raise ValueError("Cannot update the belief of type %s" % type(belief_obj))
        agent.cur_belief.set_object_belief(objid, new_belief)

    # Update planner
    if not USING_FORCED_ACTIONS:
        print_info("Updating planner...")
        planning_observation = agent.convert_real_observation_to_planning_observation(pomdp_observation, real_action)
        planner.update(agent, real_action, planning_observation)

    return next_robot_state


if __name__ == "__main__":
    main()
