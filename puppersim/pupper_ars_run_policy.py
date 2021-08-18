"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse. 
Example usage:
python3 pupper_ars_run_policy.py --expert_policy_file=data/lin_policy_plus_best_10.npz --json_file=data/params.json

"""
import numpy as np
import gym
import time
import pybullet_envs

try:
    import tds_environments
except:
    pass
import json
from arspb.policies import *
import time
import arspb.trained_policies as tp
import os

# temp hack to create an envs_v2 pupper env

import os
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd


def create_pupper_env(args):
    CONFIG_DIR = puppersim.getPupperSimPath() + "/"
    if args.run_on_robot:
        _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg_robot.gin")
    else:
        _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    gin.bind_parameter("SimulationParameters.enable_rendering", True)
    env = env_loader.load()

    return env


def main(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_policy_file", type=str, default="")
    parser.add_argument("--nosleep", action="store_true")

    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of expert rollouts"
    )
    parser.add_argument("--json_file", type=str, default="")
    parser.add_argument("--run_on_robot", action="store_true")
    if len(argv):
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()

    print("loading and building expert policy")
    if len(args.json_file) == 0:
        args.json_file = tp.getDataPath() + "/" + args.envname + "/params.json"
    with open(args.json_file) as f:
        params = json.load(f)
    print("params=", params)
    if len(args.expert_policy_file) == 0:
        args.expert_policy_file = (
            tp.getDataPath() + "/" + args.envname + "/nn_policy_plus.npz"
        )
        if not os.path.exists(args.expert_policy_file):
            args.expert_policy_file = (
                tp.getDataPath() + "/" + args.envname + "/lin_policy_plus.npz"
            )
    data = np.load(args.expert_policy_file, allow_pickle=True)

    print("create gym environment:", params["env_name"])
    env = create_pupper_env(args)  # gym.make(params["env_name"])

    lst = data.files
    weights = data[lst[0]][0]
    mu = data[lst[0]][1]
    print("mu=", mu)
    std = data[lst[0]][2]
    print("std=", std)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    ac_lb = env.action_space.low
    ac_ub = env.action_space.high

    policy_params = {
        "type": params["policy_type"],
        "ob_filter": params["filter"],
        "ob_dim": ob_dim,
        "ac_dim": ac_dim,
        "action_lower_bound": ac_lb,
        "action_upper_bound": ac_ub,
    }
    policy_params["weights"] = weights
    policy_params["observation_filter_mean"] = mu
    policy_params["observation_filter_std"] = std
    if params["policy_type"] == "nn":
        print("FullyConnectedNeuralNetworkPolicy")
        policy_sizes_string = params["policy_network_size_list"].split(",")
        print("policy_sizes_string=", policy_sizes_string)
        policy_sizes_list = [int(item) for item in policy_sizes_string]
        print("policy_sizes_list=", policy_sizes_list)
        policy_params["policy_network_size"] = policy_sizes_list
        policy = FullyConnectedNeuralNetworkPolicy(policy_params, update_filter=False)
    else:
        print("LinearPolicy2")
        policy = LinearPolicy2(policy_params, update_filter=False)
    policy.get_weights()

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print("iter", i)
        obs = env.reset()
        done = False
        totalr = 0.0
        steps = 0
        while not done:
            action = policy.act(obs)
            observations.append(obs)
            actions.append(action)

            # time.sleep(1)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1

            # if steps % 100 == 0: print("%i/%i"%(steps, env.spec.timestep_limit))
            # if steps >= env.spec.timestep_limit:
            #    break
        # print("steps=",steps)
        returns.append(totalr)

    print("returns", returns)
    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
