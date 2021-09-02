import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor_file", type=str, required=True)
    parser.add_argument("--actor_idx", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    with open(args.motor_file, "rb") as f:
        history = pickle.load(f)

    motors = history["motor_angles"]
    mean_motors = motors.mean(0)
    time = np.arange(args.timesteps) * 0.002

    fig, axs = plt.subplots(12, figsize=(8, 10), sharex=True)

    def draw_plot(i, text=False):
        axs[i].plot(time, mean_motors[:, i + 1][: args.timesteps])
        if text:
            axs[i].set_xlabel("Time (s)")

    for i in range(0, 12):
        draw_plot(i, i == 11)

    fig.suptitle(
        f"Motor Angles for Actor Network {args.actor_idx}. Mean Return = {history['returns'].mean():.1f}",
        y=1.0,
    )
    plt.subplots_adjust(hspace=0.7)
    plt.tight_layout()
    if args.show:
        plt.show()
    else:
        plt.savefig(f"motor_plots/motors_actor_{args.actor_idx}")


if __name__ == "__main__":
    main()
