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

from sciex import Experiment, Trial, Event, Result
from moos3d.tests.experiments.runner import *
from moos3d.tests.experiments.experiment import make_domain, make_trial
from moos3d import *
import matplotlib.pyplot as plt
import os
import random

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(ABS_PATH, "results")
prior_type = "uniform"
discount_factor = 0.99
detect_after_look = True

"""
This experiment investigates sensor quality. Fix alpha, alter beta.
Or fix beta, alter alpha.

The alpha will be fixed at 1e6. The beta will be fixed at 0. To
separate the possible interaction between the two numbers.

We run the experiments on 16x16x16 worlds with 4 objects.
"""

def main():
    # Check experiment_scalability.py for comments on `num_trials`
    num_trials = 14  # running on 3 computers. so 14*3 = 42 > 40.

    domain = (16, 2, 10, 10, 3.0, 500, 360)
    n, k, d, max_depth, planning_time, max_steps, max_time = domain
    if n == 16:
        setting_hier = [(1,1,max_depth), (2,2,max_depth), (4,4,max_depth)]
        setting_op = [(1,1,max_depth), (1,2,max_depth), (1,4,max_depth)]

    ## parameters
    big = 1000
    small = 1
    exploration_const = 1000
    params = {"prior_type": prior_type,
              "discount_factor": discount_factor,
              "max_depth": max_depth,
              "planning_time": planning_time,
              "max_steps": max_steps,
              "max_time": max_time,
              "detect_after_look": detect_after_look,
              "big": big,
              "small": small,
              "exploration_const": exploration_const}

    alpha_fixed = 1e5
    beta_fixed = 0

    # SIMPLER to understand!
    scenarios = [(1e1, 0.3),   # severe noise
                 (1e1, 0.8),
                 (1e2, 0.3),
                 (1e2, 0.8),
                 (5e2, 0.3),
                 (5e2, 0.8),
                 (1e3, 0.3),
                 (1e3, 0.8),
                 (1e4, 0.3),
                 (1e4, 0.8),
                 (1e5, 0.3),
                 (1e5, 0.8)]  # no noise
    all_trials = []

    # Generate a world. For the same world, run different sensors & baselines.
    # Do this for #num_trials number of worlds
    for t in range(num_trials):
        seed = random.randint(1, 1000000)

        # build world
        worldstr = make_domain(n, k, d)

        # Run different sensors and baselines
        for i in range(len(scenarios)):
            alpha, beta = scenarios[i]
            params['alpha'] = alpha
            params['beta'] = beta

            trial_name = "quality%s_%s" % (str(scenarios[i]).replace(", ", "-"), str(seed))
            pouct_trial = make_trial(trial_name, worldstr,
                                     "pouct", "octree", **params)
            multires_trial = make_trial(trial_name, worldstr,
                                        "hierarchical", "octree",
                                        setting=setting_hier, **params)
            options_trial = make_trial(trial_name, worldstr,
                                        "options", "octree",
                                       setting=setting_op, **params)
            pomcp_trial = make_trial(trial_name, worldstr,
                                     "pomcp", "particles",
                                     num_particles=1000, **params)
            random_trial = make_trial(trial_name, worldstr,
                                      "purelyrandom", "octree", **params)
            porollout_trial = make_trial(trial_name, worldstr,
                                         "porollout", "octree",
                                         porollout_policy=PolicyModel(detect_after_look=detect_after_look),
                                         **params)
            all_trials.extend([pouct_trial,
                               multires_trial,
                               options_trial,
                               pomcp_trial,
                               porollout_trial,
                               random_trial])
    # Generate scripts to run experiments and gather results
    exp = Experiment("QualitySensorCC", all_trials, output_dir, verbose=True)
    exp.generate_trial_scripts(split=5)
    print("Find multiple computers to run these experiments.")

if __name__ == "__main__":
    main()
