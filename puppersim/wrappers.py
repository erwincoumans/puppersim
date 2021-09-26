import os

import gym
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

import deep_control as dc
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor


def create_pupper_env(render=False):
    CONFIG_DIR = puppersim.getPupperSimPath() + "/"
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg_slow.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    if render:
        gin.bind_parameter("SimulationParameters.enable_rendering", True)
    env = env_loader.load()
    env = dc.envs.ScaleReward(env, scale=100.0)
    env = dc.envs.NormalizeContinuousActionSpace(env)
    env = dc.envs.PersistenceAwareWrapper(env)
    return env

class SqueezeRew(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)


    def reward(self, r):
        return r[0]


def make_sb3_env(rank, log_dir, seed=0):
    def _init():
        env = Monitor(
            SqueezeRew(gym.wrappers.TimeLimit(create_pupper_env(), 10_000)),
            log_dir,
        )
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


