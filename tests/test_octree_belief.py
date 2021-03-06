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

from mos3d.planning.belief.octree_belief import OctreeBelief, update_octree_belief
from mos3d.planning.belief.octree import OctNode, Octree, LOG, DEFAULT_VAL
from mos3d.planning.belief.visual import plot_octree_belief
import mos3d.util as util
import time
import math
from mos3d.oopomdp import TargetObjectState, LookAction
from mos3d.models.transition import M3TransitionModel
from mos3d.models.reward import GoalRewardModel, GuidedRewardModel
from mos3d.environment.env import parse_worldstr, Mos3DEnvironment
from mos3d.models.observation import OOObservation, M3ObservationModel

world1 =\
"""
16
16
16

orange_ricky 2 2 0
hero 0 4 0
teewee 2 4 3
teewee 5 6 5
smashboy 3 3 4
smashboy 6 6 0
cube 9 5 9
teewee 2 5 9
cube 8 5 0
cube 7 0 9
hero 0 2 5
cube 5 9 0
cube 7 8 0
---
robot 8 1 0 0 0 0 occlusion 45 1.0 1.0 10
"""

world_basic =\
"""
4
4
4

cube 0 0 0
cube 1 0 0
cube 0 0 1
cube 0 1 0
---
robot 2 0 0 0 0 0 occlusion 45 1.0 0.1 4
"""

if LOG:
    TEST_ALPHA = math.log(100000)
    TEST_BETA = math.log(0.00001)
else:
    TEST_ALPHA = 100000
    TEST_BETA = 0.00001

def init(worldstr=world1):
    gridworld, init_state = parse_worldstr(worldstr)
    T = M3TransitionModel(gridworld)
    R = GoalRewardModel(gridworld)
    env = Mos3DEnvironment(init_state, gridworld, T, R)

    # Test constructors
    octree = Octree(1, (gridworld.width,
                        gridworld.length,
                        gridworld.height))
    octree_belief = OctreeBelief(gridworld.width,
                                 gridworld.length,
                                 gridworld.height,
                                 1, gridworld.objects[1].objtype, octree)
    oom = M3ObservationModel(gridworld, alpha=TEST_ALPHA, beta=TEST_BETA, voxel_model=False)
    return octree_belief, init_state, oom

def test_basics(octree_belief):
    # Test probability; __getitem__
    print("** Testing Basics")
    print(octree_belief._probability(0,0,1,1))
    print(octree_belief._probability(0,0,0,2))
    print(octree_belief._probability(0,0,0,4))
    print(octree_belief._probability(0,0,0,8))
    print(octree_belief._probability(0,0,0,16))

    octree_belief[TargetObjectState(1, "cube", (0,0,1), res=1)] = TEST_ALPHA
    print(octree_belief._probability(0,0,1,1))
    print(octree_belief._probability(0,0,0,2))

    octree_belief[TargetObjectState(1, "cube", (0,0,1), res=1)] = DEFAULT_VAL
    print(octree_belief._probability(0,0,1,1))
    print(octree_belief._probability(0,0,0,2))
    print(octree_belief._probability(1,1,1,2))

def test_assign_prior1(octree_belief):
    print("** Testing Prior assignment (1)")
    print("[start]")
    state = TargetObjectState(1, "cube", (1,1,1), res=2)
    print("Probability at")
    print("(1,1,1,2): %.5f" % octree_belief._probability(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief._probability(0,0,0,1))
    print("MPE: %s" % octree_belief.mpe())
    print("assigning high probability to (1,1,1,2)...")
    if LOG:
        octree_belief.assign(state, TEST_ALPHA - math.log(10))
    else:
        octree_belief.assign(state, TEST_ALPHA/10)
    print("MPE: %s" % octree_belief.mpe())
    print("(1,1,1,2): %.5f" % octree_belief._probability(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief._probability(0,0,0,1))
    print("(3,3,3,1): %.5f" % octree_belief._probability(3,3,3,1))
    print("**** Sub test:")
    test_mpe_random(octree_belief, res=2)  # MPE/random at resolution 2
    print("[end]")

def test_assign_prior2(octree_belief):
    print("** Testing Prior assignment (2)")
    print("[start]")
    state = TargetObjectState(1, "cube", (5,6,7), res=2)
    print("Probability at")
    print("(1,1,1,2): %.5f" % octree_belief._probability(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief._probability(0,0,0,1))
    print("MPE: %s" % octree_belief.mpe())
    print("assigning high probability to (1,1,1,2)...")
    if LOG:
        octree_belief.assign(state, TEST_ALPHA - math.log(10))
    else:
        octree_belief.assign(state, TEST_ALPHA/10)
    print("MPE: %s" % octree_belief.mpe())
    print("(1,1,1,2): %.5f" % octree_belief._probability(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief._probability(0,0,0,1))
    print("(3,3,3,1): %.5f" % octree_belief._probability(3,3,3,1))
    print("**** Sub test:")
    test_mpe_random(octree_belief, res=2)  # MPE/random at resolution 2
    print("[end]")

def test_assign_prior3(octree_belief):
    print("** Testing Prior assignment (3) ZERO Prior")
    print("[start]")
    state = TargetObjectState(1, "cube", (5,6,7), res=2)
    print("Probability at")
    print("(5, 6, 7, 2): %.5f" % octree_belief._probability(5, 6, 7, 2))
    print("(11,13,15,1): %.5f" % octree_belief._probability(11,13,15,1))
    print("(1,1,1,2): %.5f" % octree_belief._probability(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief._probability(0,0,0,1))
    print("MPE: %s" % octree_belief.mpe())
    print("assigning high probability to (1,1,1,2)...")
    if LOG:
        octree_belief.assign(state, float("-inf"))
    else:
        octree_belief.assign(state, 0)
    print("MPE: %s" % octree_belief.mpe())
    print("(5, 6, 7, 2): %.5f" % octree_belief._probability(5, 6, 7, 2))
    print("(11,13,15,1): %.5f" % octree_belief._probability(11,13,15,1))
    print("(1,1,1,2): %.5f" % octree_belief._probability(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief._probability(0,0,0,1))
    print("(3,3,3,1): %.5f" % octree_belief._probability(3,3,3,1))
    print("**** Sub test:")
    test_mpe_random(octree_belief, res=2)  # MPE/random at resolution 2
    print("[end]")

def test_mpe_random(octree_belief, res=1):
    def test_round(octree_belief):
        print("-- Round --")
        mpe_state = octree_belief.mpe(res=res)
        mpe_prob = octree_belief[mpe_state]
        print(mpe_state)

        results = []
        num_same = 0
        for i in range(10000):
            rnd_state = octree_belief.random(res=res)
            results.append(rnd_state)
            if mpe_state == rnd_state:
                num_same += 1
        if LOG:
            mpe_prob = math.exp(mpe_prob)
        print("Expected probability: %.5f; Actual frequency: %.5f"
              % (mpe_prob, num_same / len(results)))
        assert abs(num_same / len(results) - mpe_prob) <= 1e-2
    print("** Testing MPE and Random (res=%d)" % res)
    test_round(octree_belief)
    octree_belief[TargetObjectState(1, "cube", (0,0,1), res=1)] = TEST_ALPHA
    test_round(octree_belief)
    octree_belief[TargetObjectState(1, "cube", (0,0,1), res=1)] = DEFAULT_VAL

def test_time(octree_belief):
    """Try setting cells to be BETA and see if it affects the
    likelihood of sampling from the remaining cells."""
    start = time.time()
    for i in range(1000):
        octree_belief.random(res=1)
    tot_res1 = time.time() - start

    start = time.time()
    for i in range(1000):
        octree_belief.random(res=2)
    tot_res2 = time.time() - start

    start = time.time()
    for i in range(1000):
        octree_belief.random(res=4)
    tot_res4 = time.time() - start
    print("Avg sample time (res=1): %.3f" % (tot_res1 % 1000))
    print("Avg sample time (res=2): %.3f" % (tot_res2 % 1000))
    print("Avg sample time (res=4): %.3f" % (tot_res4 % 1000))



def test_belief_update(octree_belief, state, observation_model):
    print("** Testing belief update")
    mpe = octree_belief.mpe()
    print(mpe)
    print(octree_belief[mpe])

    objo = observation_model.sample(state, LookAction("look-thx"))
    print(objo)
    octree_belief = update_octree_belief(octree_belief, LookAction("look-thx"), objo,
                                         alpha=TEST_ALPHA, beta=TEST_BETA)  # this setting is for log space

    mpe = octree_belief.mpe()
    print(mpe)
    print(octree_belief[mpe])
    return octree_belief


def test_visualize(octree_belief):
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    ax = fig.add_subplot(1,1,1,projection="3d")
    m = plot_octree_belief(ax, octree_belief,
                           alpha="clarity", edgecolor="black", linewidth=0.1)
    ax.set_xlim([0, octree_belief._width])
    ax.set_ylim([0, octree_belief._length])
    ax.set_zlim([0, octree_belief._height])
    ax.grid(False)
    fig.colorbar(m, ax=ax)
    plt.show()


if __name__ == '__main__':
    octree_belief, init_state, oom = init()
    test_basics(octree_belief)
    test_mpe_random(octree_belief)
    test_assign_prior1(octree_belief)
    test_assign_prior2(octree_belief)
    test_assign_prior3(octree_belief)
    test_visualize(octree_belief)
    octree_belief = test_belief_update(octree_belief, init_state, oom)
    test_visualize(octree_belief)
    test_time(octree_belief)
