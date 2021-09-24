import argparse
import random
import os
import sys

import torch
import deep_control as dc

from wrappers import create_pupper_env


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
    elif alg == "sunrise":
        dc.sunrise.add_args(parser)
    elif alg == "aac":
        dc.aac.add_args(parser)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--r_scale", type=float, default=10.0)
    return parser.parse_args()


def create_buffer(args):
    env = create_pupper_env()
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


def create_agent(args):
    train_env = create_pupper_env()
    if args.alg == "ddpg":
        agent = dc.ddpg.DDPGAgent(
            obs_space_size=train_env.observation_space.shape[0],
            action_space_size=train_env.action_space.shape[0],
            hidden_size=args.hidden_size,
        )
    elif args.alg == "sac":
        agent = dc.sac.SACAgent(
            train_env.observation_space.shape[0],
            train_env.action_space.shape[0],
            log_std_low=args.log_std_low,
            log_std_high=args.log_std_high,
            hidden_size=args.hidden_size,
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
    elif args.alg == "grac":
        agent = dc.grac.GRACAgent(
            train_env.observation_space.shape[0],
            train_env.action_space.shape[0],
            log_std_low=args.log_std_low,
            log_std_high=args.log_std_high,
            hidden_size=args.hidden_size,
        )
    elif args.alg == "sunrise":
        agent = dc.sunrise.SunriseAgent(
            train_env.observation_space.shape[0],
            train_env.action_space.shape[0],
            log_std_low=args.log_std_low,
            log_std_high=args.log_std_high,
            ensemble_size=args.ensemble_size,
            hidden_size=args.hidden_size,
        )
    elif args.alg == "aac":
        agent = None  # (N/A - created inside alg)
    else:
        raise ValueError(f"Unrecognized algorithm `{args.alg}`")
    return agent


def train(args):
    args.max_episode_steps = 10_000
    args.eval_episodes = 1
    train_env = create_pupper_env()
    test_env = create_pupper_env()

    agent = create_agent(args)
    if args.alg == "ddpg":
        dc.ddpg.ddpg(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=create_buffer(args),
            **vars(args),
        )
    elif args.alg == "sac":
        agent = dc.sac.sac(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=create_buffer(args),
            **vars(args),
        )
    elif args.alg == "redq":
        agent = dc.redq.redq(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=create_buffer(args),
            **vars(args),
        )
    elif args.alg == "grac":
        agent = dc.grac.grac(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=create_buffer(args),
            **vars(args),
        )
    elif args.alg == "sunrise":
        agent = dc.sunrise.sunrise(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=create_buffer(args),
            **vars(args),
        )
    elif args.alg == "aac":
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")
        args.steps_per_epoch = 10_000
        args.epochs = 300
        agent = dc.aac.aac(create_pupper_env, **vars(args))
    else:
        raise ValueError(f"Unrecognized algorithm `{args.alg}`")


if __name__ == "__main__":
    args = create_parser()
    args.eval_interval = 10_000
    args.max_episode_steps = 10_000
    args.eval_episodes = 1

    if args.alg == "aac":
        # fill in N/A args
        args.gamma = 0.99
        args.prioritized_replay = True
        args.buffer_size = 2_000_000

    train(args)
