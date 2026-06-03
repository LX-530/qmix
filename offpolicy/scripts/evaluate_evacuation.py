import argparse
import random
from pathlib import Path

import numpy as np

from offpolicy.envs.envs.CA.robot_env import RobotEnvironment


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "envs" / "envs" / "CA" / "map.json"
TARGET_AREA = (3, 32, 2, 7)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate evacuation-time baselines.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-persons", type=int, default=140)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--target-rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def make_env(args, seed, policy):
    random.seed(seed)
    np.random.seed(seed)
    env = RobotEnvironment(
        MAP_PATH,
        TARGET_AREA,
        args.num_persons,
        args.max_steps,
        use_health=False,
        evacuation_target_rate=args.target_rate,
    )
    env.set_robot_repulsion_factor(1.0)

    if policy == "no_robot":
        env.robot_positions = []
    elif policy == "static_good":
        env.robot_positions = [(15, 15), (14, 18)]

    return env


def run_episode(args, seed, policy):
    env = make_env(args, seed, policy)
    done = False
    info = None
    while not done:
        actions = [] if policy == "no_robot" else [4, 4]
        _, _, done, info = env.step(actions)
    return info


def summarize(name, rows):
    steps = np.array([row["step"] for row in rows], dtype=float)
    escaped = np.array([row["persons_escaped"] for row in rows], dtype=float)
    rates = np.array([row["evacuation_rate"] for row in rows], dtype=float)
    return {
        "policy": name,
        "mean_steps": round(float(steps.mean()), 3),
        "std_steps": round(float(steps.std()), 3),
        "mean_escaped": round(float(escaped.mean()), 3),
        "mean_rate": round(float(rates.mean()), 3),
    }


def main():
    args = parse_args()
    policies = ["no_robot", "default_stay", "static_good"]
    print(
        "health=off, "
        f"target_rate={args.target_rate}, "
        "doorway_conflict_friction=on(base_service_time=2,fixed_service_time,capacity=1), "
        "robot_effect=repulsion_only(no direct service-time or friction-probability reduction)"
    )
    for policy in policies:
        rows = [run_episode(args, args.seed + idx, policy) for idx in range(args.episodes)]
        print(summarize(policy, rows))


if __name__ == "__main__":
    main()
