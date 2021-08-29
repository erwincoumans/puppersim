import numpy as np
import gym
import time
import torch
import random
import pybullet_envs

try:
    import tds_environments
except:
    pass
import json
import time
import os

# temp hack to create an envs_v2 pupper env

import os
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd


from dc_train_aac import create_pupper_env
from deep_control.nets import StochasticActor


def main(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_policy_file", type=str, default="")
    parser.add_argument("--nosleep", action="store_true")

    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of expert rollouts"
    )
    args = parser.parse_args()
    env = create_pupper_env(render=True)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    policy = StochasticActor(ob_dim, ac_dim, -10.0, 2.0, hidden_size=64)
    policy.load_state_dict(torch.load(args.expert_policy_file))

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
            with torch.no_grad():
                action = (
                    policy(torch.from_numpy(obs).unsqueeze(0).float())
                    .mean.squeeze()
                    .numpy()
                )
                # action = env.action_space.sample()
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
