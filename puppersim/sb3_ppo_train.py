import os
import argparse

import gym
import numpy as np
from wrappers import make_sb3_env

import torch
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--clip_range", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_dir = os.path.join("dc_saves", args.name)
    i = 0
    while os.path.exists(log_dir + f"_{i}"):
        i += 1
    log_dir = log_dir + f"_{i}"
    print(log_dir)
    os.makedirs(log_dir)

    num_cpu = 31
    env = SubprocVecEnv([make_sb3_env(i, log_dir) for i in range(num_cpu)])
    eval_env = SubprocVecEnv([make_sb3_env(num_cpu + 1, log_dir)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "model_save/"),
        log_path=log_dir,
        eval_freq=100_000,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        learning_rate=1e-4,
        n_steps=1000,
        verbose=1,
        tensorboard_log=log_dir,
        gamma=0.55,
    )
    model.learn(total_timesteps=1_000_000_000, callback=eval_callback)
    model.save(log_dir)
