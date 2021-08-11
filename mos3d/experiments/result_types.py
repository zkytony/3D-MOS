from sciex import Experiment, Trial, Event, Result, YamlResult, PklResult, PostProcessingResult
from mos3d import *
import numpy as np
import pomdp_py
import os
import yaml
import pickle
import time
from pprint import pprint
import re
import math
import pandas as pd

#### Actual results for experiments ####
class RewardsResult(YamlResult):
    def __init__(self, rewards):
        """rewards: a list of reward floats"""
        super().__init__(rewards)
    @classmethod
    def FILENAME(cls):
        return "rewards.yaml"

    @classmethod
    def discounted_reward(cls, rewards, gamma=0.99):
        discount = 1.0
        cum_disc = 0.0
        for reward in rewards:
            cum_disc += discount * reward
            discount *= gamma
        return cum_disc

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # compute cumulative rewards
        myresult = {}
        for specific_name in results:
            all_rewards = []
            all_disc_rewards = []
            seeds = []
            for seed in results[specific_name]:
                rewards = list(results[specific_name][seed])
                cum_reward = sum(rewards)
                # 0.99 is the gamma used in the experiments
                disc_reward = cls.discounted_reward(rewards, gamma=0.99)
                all_rewards.append(cum_reward)
                all_disc_rewards.append(disc_reward)
                seeds.append(seed)

            myresult[specific_name] = {'mean': np.mean(all_rewards),
                                       'std': np.std(all_rewards),
                                       'ci-95': util.ci_normal(all_rewards),
                                       '_size': len(results[specific_name]),
                                       '_all_rewards': all_rewards,
                                       '_all_disc_rewards': all_disc_rewards,
                                       '_seeds': seeds}
        return myresult


    @classmethod
    def save_gathered_results(cls, gathered_results, path):

        # Save the pandas table
        rows = []
        for global_name in gathered_results:
            if global_name.startswith("domain"):
                # scalability
                size, nobj, depth = global_name.split("domain(")[1].split("-")[:3]
                case = [size, nobj, depth]
                columns = ["size", "nobj", "depth"]
            elif global_name.startswith("quality"):
                alpha, beta = global_name.split("quality(")[1][:-1].split("-")[:2]
                case = [alpha, beta]
                columns = ["alpha", "beta"]

            for method in gathered_results[global_name]:
                results = gathered_results[global_name][method]
                for i in range(len(results["_all_rewards"])):
                    seed = results["_seeds"][i]
                    cum_reward = results["_all_rewards"][i]
                    disc_reward = results["_all_disc_rewards"][i]
                    row = case + [seed, method, cum_reward, disc_reward]
                    rows.append(row)
        df = pd.DataFrame(rows,
                          columns=columns+["seed", "method", "cum_reward", "disc_reward"])

        df.to_csv(os.path.join(path, "rewards_results.csv"))
        return True


class TotDistResult(YamlResult):
    def __init__(self, totdists):
        """total euclidean distances to undetected objects (floats)"""
        super().__init__(totdists)
    @classmethod
    def FILENAME(cls):
        return "totdists.yaml"

class ExplorationRatioResult(YamlResult):
    """ratios should be floats"""
    def __init__(self, ratios):
        super().__init__(ratios)
    @classmethod
    def FILENAME(cls):
        return "exploration_ratios.yaml"

class HistoryResult(PklResult):
    def __init__(self, history):
        """list of state objects"""
        super().__init__(history)

    @classmethod
    def FILENAME(cls):
        return "history.pkl"

class StatesResult(PklResult):
    def __init__(self, states):
        """list of state objects"""
        super().__init__(states)

    @classmethod
    def FILENAME(cls):
        return "states.pkl"

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # Returns the number of objects detected at the end.
        myresult = {}
        for specific_name in results:
            all_counts = []
            seeds = []
            for seed in results[specific_name]:
                result = results[specific_name][seed]
                count = len(result[-1].robot_state.objects_found)
                all_counts.append(count)
                seeds.append(seed)
            myresult[specific_name] = {'mean': np.mean(all_counts),
                                       'std': np.std(all_counts),
                                       'ci-95': util.ci_normal(all_counts),
                                       '_size': len(all_counts),
                                       '_all': all_counts,
                                       '_seeds': seeds}
        return myresult

    @classmethod
    def save_gathered_results(cls, gathered_results, path):

        # Save the pandas table
        rows = []
        for global_name in gathered_results:
            if global_name.startswith("domain"):
                # scalability
                size, nobj, depth = global_name.split("domain(")[1].split("-")[:3]
                case = [size, nobj, depth]
                columns = ["size", "nobj", "depth"]
            elif global_name.startswith("quality"):
                alpha, beta = global_name.split("quality(")[1][:-1].split("-")[:2]
                case = [alpha, beta]
                columns = ["alpha", "beta"]

            for method in gathered_results[global_name]:
                results = gathered_results[global_name][method]
                for i in range(len(results["_all"])):
                    seed = results["_seeds"][i]
                    num_detected = results["_all"][i]
                    row = case + [seed, method, num_detected]
                    rows.append(row)
        df = pd.DataFrame(rows,
                          columns=columns+["seed", "method", "num_detected"])

        df.to_csv(os.path.join(path, "detections_results.csv"))
        return True
