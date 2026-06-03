import argparse
import csv
import random
from pathlib import Path

import numpy as np

from offpolicy.envs.envs.CA.robot_env import RobotEnvironment
from offpolicy.scripts.evaluate_evacuation import MAP_PATH, TARGET_AREA


POSITION_PRESETS = {
    "default": None,
    "exit_default": [(15, 15), (17, 33)],
    "exit_upper": [(15, 15), (15, 34)],
    "exit_lower": [(15, 15), (18, 34)],
    "upstream_exit": [(15, 15), (16, 30)],
    "behind_exit": [(15, 15), (16, 28)],
    "behind_crowd_pair": [(12, 8), (24, 8)],
    "push_left_pair": [(10, 5), (24, 5)],
}


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item]


def parse_args():
    parser = argparse.ArgumentParser(description="Scan pure robot-repulsion field settings.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--block-probs", default="0.7")
    parser.add_argument("--factors", default="0,0.5,0.8,1.0,1.2,1.5")
    parser.add_argument("--cutoffs", default="3,4,5,6")
    parser.add_argument("--decays", default="0.35,0.5,0.65")
    parser.add_argument("--positions", default="default,exit_default,exit_upper,exit_lower,upstream_exit")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def configure_env(seed, block_prob, factor, cutoff, decay, position_name, with_robot):
    random.seed(seed)
    np.random.seed(seed)
    env = RobotEnvironment(
        MAP_PATH,
        TARGET_AREA,
        140,
        200,
        use_health=False,
        evacuation_target_rate=0.8,
    )
    env.use_arch_breaking_field = True
    env.arch_lane_attraction = 0.0
    env.arch_side_repulsion = 0.0
    env.arch_block_probability = block_prob
    env.set_robot_repulsion_factor(factor)
    env.robot_repulsion_cutoff_radius = cutoff
    env.robot_repulsion_decay = decay

    if not with_robot:
        env.robot_positions = []
    else:
        preset = POSITION_PRESETS[position_name]
        if preset is not None:
            env.robot_positions = [
                pos for pos in preset
                if env._is_valid_position(pos) and pos not in env.map_loader.exits
            ]
    return env


def run_episode(seed, block_prob, factor, cutoff, decay, position_name, with_robot):
    env = configure_env(seed, block_prob, factor, cutoff, decay, position_name, with_robot)
    actions = [4 for _ in env.robot_positions]
    done = False
    info = None
    while not done:
        _, _, done, info = env.step(actions)
    return info


def summarize_steps(rows):
    steps = np.array([row["step"] for row in rows], dtype=float)
    escaped = np.array([row["persons_escaped"] for row in rows], dtype=float)
    rates = np.array([row["evacuation_rate"] for row in rows], dtype=float)
    return float(steps.mean()), float(steps.std()), float(escaped.mean()), float(rates.mean())


def main():
    args = parse_args()
    block_probs = parse_float_list(args.block_probs)
    factors = parse_float_list(args.factors)
    cutoffs = parse_float_list(args.cutoffs)
    decays = parse_float_list(args.decays)
    positions = [item for item in args.positions.split(",") if item]
    output = Path(args.output) if args.output else Path("offpolicy/scripts/results/repulsion_scan.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "block_prob", "position", "factor", "cutoff", "decay",
        "baseline_mean_steps", "robot_mean_steps", "robot_std_steps",
        "mean_escaped", "mean_rate", "reduction_percent",
    ]
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for block_prob in block_probs:
            baseline_rows = [
                run_episode(args.seed + idx, block_prob, 0.0, 4.0, 0.5, "default", False)
                for idx in range(args.episodes)
            ]
            baseline_mean, _, _, _ = summarize_steps(baseline_rows)

            for position in positions:
                for factor in factors:
                    for cutoff in cutoffs:
                        for decay in decays:
                            robot_rows = [
                                run_episode(
                                    args.seed + idx,
                                    block_prob,
                                    factor,
                                    cutoff,
                                    decay,
                                    position,
                                    True,
                                )
                                for idx in range(args.episodes)
                            ]
                            mean_steps, std_steps, mean_escaped, mean_rate = summarize_steps(robot_rows)
                            reduction = (baseline_mean - mean_steps) / baseline_mean * 100.0
                            row = {
                                "block_prob": block_prob,
                                "position": position,
                                "factor": factor,
                                "cutoff": cutoff,
                                "decay": decay,
                                "baseline_mean_steps": round(baseline_mean, 3),
                                "robot_mean_steps": round(mean_steps, 3),
                                "robot_std_steps": round(std_steps, 3),
                                "mean_escaped": round(mean_escaped, 3),
                                "mean_rate": round(mean_rate, 3),
                                "reduction_percent": round(reduction, 3),
                            }
                            writer.writerow(row)
                            f.flush()
                            print(row, flush=True)


if __name__ == "__main__":
    main()
