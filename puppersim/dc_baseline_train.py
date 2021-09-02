import argparse
import random
import os
import sys

import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

import deep_control as dc


def create_pupper_env(args, seed):
    CONFIG_DIR = puppersim.getPupperSimPath() + "/"
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg_slow.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    env = env_loader.load()

    env.seed(seed)
    env = dc.envs.ScaleReward(env, scale=args.r_scale)
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
    parser.add_argument("--r_scale", type=float, default=1.0)
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
    args.max_episode_steps = 2000
    args.eval_interval = 10_000

    # seed = random.randint(1, 100) if not args
    seed = 37
    train_env = create_pupper_env(args, seed)
    test_env = create_pupper_env(args, seed)

    buffer = create_buffer(args, train_env)

    train_gym(args, train_env, test_env, buffer)
