#!/usr/bin/env python
#
# Replays a trial.
from mos3d.tests.experiments.runner import StatesResult, HistoryResult
from mos3d.environment.env import parse_worldstr
from mos3d.util import print_info, print_error, print_warning, print_success,\
    print_info_bold, print_error_bold, print_warning_bold, print_success_bold, print_note_bold, print_note
from mos3d import M3TransitionModel, GoalRewardModel, Mos3DEnvironment, Mos3DViz
import pickle
import yaml
import time
import os
from collections import deque

# experiments dir path
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_protected") #"Scalability_Additional")#"results_protected", "used_as_examples")
FRM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")

def play_trial(trial_path, wait=0.3):
    """full path to trial directory"""
    state_res_path = os.path.join(trial_path, StatesResult.FILENAME())
    states = StatesResult.collect(state_res_path)

    history_res_path = os.path.join(trial_path, HistoryResult.FILENAME())
    history = StatesResult.collect(history_res_path)

    with open(os.path.join(trial_path, "config.yaml")) as f:
        config = yaml.load(f)
        worldstr = config["world_config"]
        gridworld, init_state = parse_worldstr(worldstr)

    assert init_state == states[0]
    _replay(gridworld, init_state, states, history, config["model_config"],
            wait=wait, frame_dirpath=os.path.join(FRM_DIR, os.path.basename(trial_path)))

def _replay(gridworld, init_state, states, history,
            model_config, wait=0.3, frame_dirpath="./frames"):
    """Replay results of trial with visualization"""
    if frame_dirpath is not None and\
       not os.path.exists(frame_dirpath):
        os.makedirs(frame_dirpath)
    T = M3TransitionModel(gridworld, **model_config['T'])
    R = GoalRewardModel(gridworld, **model_config['R'])
    env = Mos3DEnvironment(init_state, gridworld, T, R)
    viz = Mos3DViz(env, gridworld, fps=15)
    if viz.on_init() == False:
        raise Exception("Environment failed to initialize")

    viz.on_render()
    viz.save_frame(os.path.join(frame_dirpath, "frame-0.png"))
    for i in range(len(history)):
        action, _ = history[i]
        env_reward = env.state_transition(action, execute=True)
        real_observation = gridworld.provide_render_observation(env.state.robot_pose,
                                                                env.state.object_poses)
        viz.update(action, env.state.robot_pose, env.state.object_poses,
                   observation=real_observation)
        print("Step %d: robot: %s   action: %s   reward: %.3f"
              % (i+1, env.state.robot_state, action.name, env_reward))

        viz.on_loop()
        viz.on_render(rerender=True)
        viz.save_frame(os.path.join(frame_dirpath, "frame-%d.png" % (i+1)))
        time.sleep(wait)

def _ask(prompt, accepted_strings):
    """The response could be a comma separated list of accepted strings as well"""
    while True:
        token = input(prompt)
        if "," in token:
            tokens = deque(token.split(","))
        else:
            tokens = deque([token])

        valid = True
        while len(tokens) > 0:
            t = tokens.popleft()
            if t.strip() in accepted_strings:
                break
            elif read == '':
                exit(0)
            else:
                print_error("Unacceptable input %s" % read)
                valid = False
                break
        if valid:
            break
    return token

if __name__ == "__main__":
    # Interactively obtain trial path
    extype = _ask("Quality [q] or Scalability [s]? ", {"q", "s"})
    if extype == "q":
        exdir = "Quality"
    elif extype == "s":
        exdir = "Scalability_Additional"
    else:
        raise ValueError("Unexpected program state due to extype == %s" % extype)

    # List a couple of trial paths (numbered)
    nummap = {}
    i = 1
    expath = os.path.join(RES_DIR, exdir)
    for fname in sorted(os.listdir(expath)):
        if os.path.isdir(os.path.join(expath, fname)):
            nummap[i] = fname
            i += 1
    for i in sorted(nummap):
        print_info("[%d] %s" % (i, nummap[i]))

    token = _ask("Which trial [%d-%d] ? " % (min(nummap), max(nummap)),
                 {str(i) for i in range(min(nummap), max(nummap)+1)})
    if "," in token:
        tokens = token.split(",")
    else:
        tokens = [token]

    wait = float(input("wait (default 0.3s): "))

    for t in tokens:
        fname = nummap[int(t)]
        trial_path = os.path.join(expath, fname)
        print_note_bold("Playing trial %s" % fname)
        play_trial(trial_path, wait=wait)
