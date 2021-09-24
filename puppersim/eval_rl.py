import argparse
import random
import pickle

import numpy as np
import gym
import torch

import deep_control as dc
from wrappers import create_pupper_env
from train_rl import create_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alg")
    parser.add_argument("--expert_policy_file", type=str, required=True)
    parser.add_argument("--track_motors", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log_std_low", type=float, default=-10.0)
    parser.add_argument("--log_std_high", type=float, default=2.0)
    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of expert rollouts"
    )
    args = parser.parse_args()
    env = create_pupper_env(render=args.render)
    env.set_k(args.k)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    if args.alg != "aac":
        policy = create_agent(args)
        policy.load(args.expert_policy_file)
    else:
        policy = dc.nets.StochasticActor(
            ob_dim,
            ac_dim,
            args.log_std_low,
            args.log_std_high,
            hidden_size=args.hidden_size,
        )
        policy.load_state_dict(torch.load(args.expert_policy_file))
    policy.to(dc.device)

    returns = []
    observations = []
    actions = []
    motor_angle_histories = []
    for i in range(args.num_rollouts):
        motor_angles = []
        obs = env.reset()
        done = False
        totalr = 0.0
        steps = 0
        while not done and steps < args.max_steps:
            with torch.no_grad():
                if args.alg != "aac":
                    action = policy.forward(obs, from_cpu=True)
                else:
                    action = (
                        policy(torch.from_numpy(obs).unsqueeze(0).float().to(dc.device))
                        .mean.squeeze()
                        .cpu()
                        .numpy()
                    )
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            motor_angles.append(obs[:-2])
            totalr += r
            steps += 1
        returns.append(totalr)
        motor_angle_histories.append(motor_angles)

    if args.track_motors:
        with open(args.track_motors, "wb") as f:
            motor_angle_histories = np.array(motor_angle_histories)
            returns = np.array(returns)
            pickle.dump({"returns": returns, "motor_angles": motor_angle_histories}, f)

    print("returns", returns)
    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))


if __name__ == "__main__":
    main()
