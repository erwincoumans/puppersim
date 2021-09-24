import argparse
import random
import os
import sys

import torch
import deep_control as dc

from wrappers import create_pupper_env, make_sb3_env


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
    elif alg == "sb3ppo":
        parser.add_argument("--name", type=str, required=True)
        parser.add_argument("--batch_size", type=int, default=4096)
        parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--r_scale", type=float, default=10.0)
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


def train_gym(args):
    args.max_episode_steps = 10_000
    args.eval_episodes = 1
    train_env = create_pupper_env()
    test_env = create_pupper_env()

    if args.alg == "ddpg":
        agent = dc.ddpg.DDPGAgent(
            obs_space_size=train_env.observation_space.shape[0],
            action_space_size=train_env.action_space.shape[0],
            hidden_size=args.hidden_size,
        )

        dc.ddpg.ddpg(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=create_buffer(args, train_env),
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
            buffer=create_buffer(args, train_env),
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
            buffer=create_buffer(args, train_env),
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
            buffer=create_buffer(args, train_env),
            **vars(args),
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
        agent = dc.sunrise.sunrise(
            agent=agent,
            train_env=train_env,
            test_env=test_env,
            buffer=create_buffer(args, train_env),
            **vars(args),
        )
    elif args.alg == "aac":
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")
        aac = dc.aac.aac(create_pupper_env, **vars(args))
        args.steps_per_epoch = 10_000
        args.epochs = 300
    elif args.alg == "sb3ppo":
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.utils import set_random_seed
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
            from stable_baselines3.common.callbacks import EvalCallback
            from stable_baselines3 import PPO
        except ImportError:
            print("Missing stable_baselines3 installtion. Exiting.")
            exit(0)

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

    train_gym(args)
