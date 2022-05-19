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

# Test models
import numpy as np

import random
from mos3d import *
from pomdp_py import Environment, ObjectState, OOState
import time
import matplotlib.pyplot as plt
import sys

world_config = {
    # for world generation
    'objtypes': {'cube': 5 }, # to be filled
    'robot_camera': "occlusion 45 1.0 0.1 5",
    'width': 32,
    'length': 32,
    'height': 32,
}

def random_robot_pose(gridworld):
    xyz = (random.randint(0, gridworld.width-1),
           random.randint(0, gridworld.length-1),
           random.randint(0, gridworld.height-1))
    ang = (random.choice([0, 90, 180, 270]),
           random.choice([0, 90, 180, 270]),
           random.choice([0, 90, 180, 270]))
    q = util.euler_to_quat(*ang)
    return xyz + tuple(q)


def run(gridworld, belief, O, aO, T, R, count=30):
    # Perform `count` number of samplings for B, O, T, R.
    timeB, timeO, timeaO, timeTm, timeTd, timeRm, timeRd = 0, 0, 0, 0, 0, 0, 0
    for i in range(count):
        tb = time.time()
        s = belief.random()
        timeB += time.time() - tb

        # random robot pose
        pose = random_robot_pose(gridworld)
        s.robot_state['pose'] = pose
        to = time.time()
        o = O.sample(s, Actions.LOOK_ACTION)
        timeO += time.time() - to

        # abstract observation model
        to = time.time()
        ao = aO.sample(s, Actions.LOOK_ACTION)
        timeaO += time.time() - to

        # motion action transition
        action = random.choice(list(Actions.MOTION_ACTIONS))
        ttm = time.time()
        spm = T.sample(s, action)
        timeTm += time.time() - ttm

        # detect action transition
        ttd = time.time()
        spd = T.sample(s, Actions.DETECT_ACTION)
        timeTd += time.time() - ttd

        # reward sampling
        trm = time.time()
        r = R.sample(s, action, spm)
        timeRm += time.time() - trm

        trd = time.time()
        r = R.sample(s, Actions.DETECT_ACTION, spd)
        timeRd += time.time() - trd

        sys.stdout.write("[%d/%d]\r" % (i+1, count))
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return timeB, timeO, timeaO, timeTm, timeTd, timeRm, timeRd

def plot(x, avgB, avgO, avgaO, avgTm, avgTd, avgRm, avgRd, stdB, stdO, stdaO, stdTm, stdTd, stdRm, stdRd):
    plt.plot(x, avgB, label="Belief")
    plt.fill_between(x, np.array(avgB) - np.array(stdB), np.array(avgB) + np.array(stdB), alpha=0.5)

    plt.plot(x, avgO, label="O")
    plt.fill_between(x, np.array(avgO) - np.array(stdO), np.array(avgO) + np.array(stdO), alpha=0.5)

    plt.plot(x, avgaO, label="abstract-O")
    plt.fill_between(x, np.array(avgaO) - np.array(stdaO), np.array(avgaO) + np.array(stdaO), alpha=0.5)

    plt.plot(x, avgTm, label="T-motion")
    plt.fill_between(x, np.array(avgTm) - np.array(stdTm), np.array(avgTm) + np.array(stdTm), alpha=0.5)

    plt.plot(x, avgTd, label="T-detect")
    plt.fill_between(x, np.array(avgTd) - np.array(stdTd), np.array(avgTd) + np.array(stdTd), alpha=0.5)

    plt.plot(x, avgRm, label="R-motion")
    plt.fill_between(x, np.array(avgRm) - np.array(stdRm), np.array(avgRm) + np.array(stdRm), alpha=0.5)

    plt.plot(x, avgRd, label="R-detect")
    plt.fill_between(x, np.array(avgRd) - np.array(stdRd), np.array(avgRd) + np.array(stdRd), alpha=0.5)

def sampling_times_vs_nobj(world_config):
    size = 32
    Nmax = 7
    trials = 5
    count = 10

    avgB, avgO, avgaO, avgTm, avgTd, avgRm, avgRd = [], [], [], [], [], [], []
    stdB, stdO, stdaO, stdTm, stdTd, stdRm, stdRd = [], [], [], [], [], [], []

    for num_objects in range(1, Nmax+1):
        world_config['width'] = size
        world_config['length'] = size
        world_config['height'] = size
        print("**N OBJ: %d **" % num_objects)
        world_config['objtypes']['cube'] = num_objects
        worldstr = random_3dworld(world_config)

        allB, allO, allaO, allTm, allTd, allRm, allRd = [], [], [], [], [], [], []
        for t in range(trials):
            gridworld, init_state = parse_worldstr(worldstr)

            belief = M3Belief(gridworld, init_octree_belief(gridworld, init_state.robot_state))
            O = M3ObservationModel(gridworld, voxel_model=True)
            aO = AbstractM3ObservationModel(belief, gridworld)
            T = M3TransitionModel(gridworld)
            R = GoalRewardModel(gridworld)

            timeB, timeO, timeaO, timeTm, timeTd, timeRm, timeRd = run(gridworld, belief, O, aO, T, R, count=count)
            allB.append(timeB / count)
            allO.append(timeO / count)
            allaO.append(timeaO / count)
            allTm.append(timeTm / count)
            allTd.append(timeTd / count)
            allRm.append(timeRm / count)
            allRd.append(timeRd / count)
        avgB.append(np.mean(allB))
        stdB.append(np.std(allB))
        avgO.append(np.mean(allO))
        stdO.append(np.std(allO))
        avgaO.append(np.mean(allaO))
        stdaO.append(np.std(allaO))
        avgTm.append(np.mean(allTm))
        stdTm.append(np.std(allTm))
        avgTd.append(np.mean(allTd))
        stdTd.append(np.std(allTd))
        avgRm.append(np.mean(allRm))
        stdRm.append(np.std(allRm))
        avgRd.append(np.mean(allRd))
        stdRd.append(np.std(allRd))

    x = np.arange(Nmax)
    plot(x, avgB, avgO, avgaO, avgTm, avgTd, avgRm, avgRd,
         stdB, stdO, stdaO, stdTm, stdTd, stdRm, stdRd)
    plt.legend(loc="upper left")
    plt.xlabel('num objects')
    plt.ylabel('time (sec/sample)')
    plt.show()


def sampling_times_vs_worldsize(world_config):
    sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    nobj = 5
    trials = 5
    count = 10
    depth = 5

    avgB, avgO, avgTm, avgTd, avgRm, avgRd = [], [], [], [], [], []
    stdB, stdO, stdTm, stdTd, stdRm, stdRd = [], [], [], [], [], []

    for size in sizes:
        print("**size: %d **" % size)
        world_config['width'] = size
        world_config['length'] = size
        world_config['height'] = size
        world_config['objtypes']['cube'] = nobj
        world_config['robot_camera'] = "occlusion 45 1.0 0.1 %d" % depth
        worldstr = random_3dworld(world_config)

        allB, allO, allTm, allTd, allRm, allRd = [], [], [], [], [], []
        for t in range(trials):
            gridworld, init_state = parse_worldstr(worldstr)

            belief = init_octree_belief(gridworld, init_state.robot_state)
            O = M3ObservationModel(gridworld, voxel_model=True, observe_when_look=False)
            T = M3TransitionModel(gridworld, observe_when_look=False)
            R = GoalRewardModel(gridworld)

            timeB, timeO, timeTm, timeTd, timeRm, timeRd = run(gridworld, belief, O, T, R, count=count)
            allB.append(timeB / count)
            allO.append(timeO / count)
            allTm.append(timeTm / count)
            allTd.append(timeTd / count)
            allRm.append(timeRm / count)
            allRd.append(timeRd / count)
        avgB.append(np.mean(allB))
        stdB.append(np.std(allB))
        avgO.append(np.mean(allO))
        stdO.append(np.std(allO))
        avgTm.append(np.mean(allTm))
        stdTm.append(np.std(allTm))
        avgTd.append(np.mean(allTd))
        stdTd.append(np.std(allTd))
        avgRm.append(np.mean(allRm))
        stdRm.append(np.std(allRm))
        avgRd.append(np.mean(allRd))
        stdRd.append(np.std(allRd))

    x = sizes
    plot(x, avgB, avgO, avgTm, avgTd, avgRm, avgRd,
         stdB, stdO, stdTm, stdTd, stdRm, stdRd)

    plt.legend(loc="lower right")
    plt.xlabel('world dimension')
    plt.ylabel('time (sec/sample)')
    plt.show()


def sampling_times_vs_fovdepth(world_config):
    depths = [10, 20]
    size = 32
    nobj = 5
    trials = 5
    count = 10

    avgB, avgO, avgTm, avgTd, avgRm, avgRd = [], [], [], [], [], []
    stdB, stdO, stdTm, stdTd, stdRm, stdRd = [], [], [], [], [], []

    for d in depths:
        print("**depth: %d **" % d)
        world_config['width'] = size
        world_config['length'] = size
        world_config['height'] = size
        world_config['objtypes']['cube'] = nobj
        world_config['robot_camera'] = "occlusion 45 1.0 0.1 %d" % d
        worldstr = random_3dworld(world_config)

        allB, allO, allTm, allTd, allRm, allRd = [], [], [], [], [], []
        ratios = []
        num_voxels = []
        for t in range(trials):
            gridworld, init_state = parse_worldstr(worldstr)
            num_voxels.append(len(gridworld.robot.camera_model.get_volume(init_state.robot_pose)))
            ratios.append(num_voxels[-1] / (size**3))

            belief = init_octree_belief(gridworld, init_state.robot_state)
            O = M3ObservationModel(gridworld, voxel_model=True, observe_when_look=False)
            T = M3TransitionModel(gridworld, observe_when_look=False)
            R = GoalRewardModel(gridworld)

            timeB, timeO, timeTm, timeTd, timeRm, timeRd = run(gridworld, belief, O, T, R, count=count)
            allB.append(timeB / count)
            allO.append(timeO / count)
            allTm.append(timeTm / count)
            allTd.append(timeTd / count)
            allRm.append(timeRm / count)
            allRd.append(timeRd / count)
            print("timeO: %.5f" % allO[-1])
        print(" -- n_vox: %.7f" % max(num_voxels))
        print(" -- ratio: %.7f" % max(ratios))


        avgB.append(np.mean(allB))
        stdB.append(np.std(allB))
        avgO.append(np.mean(allO))
        stdO.append(np.std(allO))
        avgTm.append(np.mean(allTm))
        stdTm.append(np.std(allTm))
        avgTd.append(np.mean(allTd))
        stdTd.append(np.std(allTd))
        avgRm.append(np.mean(allRm))
        stdRm.append(np.std(allRm))
        avgRd.append(np.mean(allRd))
        stdRd.append(np.std(allRd))

    x = depths
    plot(x, avgB, avgO, avgTm, avgTd, avgRm, avgRd,
         stdB, stdO, stdTm, stdTd, stdRm, stdRd)

    plt.legend(loc="lower right")
    plt.xlabel('fov depth')
    plt.ylabel('time (sec/sample)')
    plt.show()




if __name__ == '__main__':
    sampling_times_vs_nobj(world_config)
    # sampling_times_vs_worldsize(world_config)
    # sampling_times_vs_fovdepth(world_config)
