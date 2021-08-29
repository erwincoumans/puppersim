import argparse
import random
import os
import sys

import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

import deep_control as dc


def create_pupper_env(render=False):
    CONFIG_DIR = puppersim.getPupperSimPath() + "/"
    # _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_with_imu.gin")
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    if render:
        gin.bind_parameter("SimulationParameters.enable_rendering", True)
    env = env_loader.load()
    env = dc.envs.ScaleReward(env, scale=10.0)
    env = dc.envs.NormalizeContinuousActionSpace(env)
    env = dc.envs.PersistenceAwareWrapper(env)
    return env


def create_parser():
    parser = argparse.ArgumentParser()
    dc.aac.add_args(parser)
    parser.add_argument("--hidden_size", type=int, default=256)
    return parser.parse_args()


def train_gym(args):
    aac = dc.aac.aac(create_pupper_env, **vars(args))


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    args = create_parser()
    args.steps_per_epoch = 4000
    args.epochs = 300
    args.max_episode_steps = 2000

    train_gym(args)
