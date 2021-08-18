import argparse
import random
import os
import sys
from collections import deque

import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd
import gym
import numpy as np

import deep_control as dc


class StateStack(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int):
        gym.Wrapper.__init__(self, env)
        self._k = num_stack
        self._frames = deque([], maxlen=num_stack)
        shp = env.observation_space.shape[0]
        low = np.array([env.observation_space.low for _ in range(num_stack)]).flatten()
        high = np.array(
            [env.observation_space.high for _ in range(num_stack)]
        ).flatten()
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(shp * num_stack,),
            dtype=env.observation_space.dtype,
        )
        self._OBS_MAX = 0

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        obs = np.concatenate(list(self._frames), axis=0)
        if abs(obs).max() > self._OBS_MAX:
            self._OBS_MAX = abs(obs).max()
            print(self._OBS_MAX)
        return obs


def create_pupper_env(args, seed):
    CONFIG_DIR = puppersim.getPupperSimPath() + "/"
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_with_imu.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    env = env_loader.load()

    env.seed(seed)
    env = StateStack(env, num_stack=args.f_stack)
    test = env.reset()
    env = dc.envs.NormalizeContinuousActionSpace(env)
    env = dc.envs.PersistenceAwareWrapper(
        env, k=args.k, return_history=False, discount=args.gamma
    )
    return env


def create_parser():
    alg = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("alg")
    if alg == "ddpg":
        dc.ddpg.add_args(parser)
    elif alg == "sac":
        dc.sac.add_args(parser)
    elif alg == "redq":
        dc.redq.add_args(parser)
    elif alg == "grac":
        dc.grac.add_args(parser)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--f_stack", type=int, default=1)
    return parser.parse_args()


def create_buffer(args, env):
    # create replay buffer
    if args.prioritized_replay:
        buffer_type = dc.replay.PrioritizedReplayBuffer
    else:
        buffer_type = dc.replay.ReplayBuffer

    buffer = buffer_type(
        args.buffer_size,
        state_shape=env.observation_space.shape,
        state_dtype=float,
        action_shape=env.action_space.shape,
    )
    return buffer


def train_gym(args, train_env, test_env, buffer):
    if args.alg == "ddpg":
        # create agent
        agent = dc.ddpg.DDPGAgent(
            obs_space_size=train_env.observation_space.shape[0],
            action_space_size=train_env.action_space.shape[0],
            hidden_size=args.hidden_size,
        )
        # run ddpg
        dc.ddpg.ddpg(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=buffer,
            **vars(args),
        )
    elif args.alg == "sac":
        agent = dc.sac.SACAgent(
            train_env.observation_space.shape[0],
            train_env.action_space.shape[0],
            log_std_low=args.log_std_low,
            log_std_high=args.log_std_high,
            hidden_size=args.hidden_size,
        )
        agent = dc.sac.sac(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=buffer,
            **vars(args),
        )
    elif args.alg == "redq":
        agent = dc.redq.REDQAgent(
            train_env.observation_space.shape[0],
            train_env.action_space.shape[0],
            log_std_low=args.log_std_low,
            log_std_high=args.log_std_high,
            critic_ensemble_size=10,
            hidden_size=args.hidden_size,
        )

        agent = dc.redq.redq(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=buffer,
            **vars(args),
        )
    elif args.alg == "grac":
        agent = dc.grac.GRACAgent(
            train_env.observation_space.shape[0],
            train_env.action_space.shape[0],
            log_std_low=args.log_std_low,
            log_std_high=args.log_std_high,
            hidden_size=args.hidden_size,
        )
        agent = dc.grac.grac(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=buffer,
            **vars(args),
        )
    elif args.alg == "aac":
        aac = dc.aac.aac(_create_pupper_env, **vars(args))
    else:
        raise ValueError(f"Unrecognized algorithm `{args.alg}`")


if __name__ == "__main__":
    args = create_parser()
    args.max_episode_steps = 1000
    args.eval_interval = 10_000

    seed = random.randint(1, 100)
    train_env = create_pupper_env(args, seed)
    test_env = create_pupper_env(args, seed)

    buffer = create_buffer(args, train_env)

    train_gym(args, train_env, test_env, buffer)
