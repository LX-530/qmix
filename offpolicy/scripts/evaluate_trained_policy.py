import argparse
import json
from pathlib import Path

import numpy as np
import torch

from offpolicy.config import get_config
from offpolicy.scripts.train.train import make_eval_env, make_train_env, parse_args as parse_train_args
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved MQMIX evacuation policy.")
    parser.add_argument("--model-dir", required=True, help="Directory containing mixer.pt and policy_*/q_network.pt.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=150)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def build_runner(args):
    train_cli = [
        "--algorithm_name", "mqmix",
        "--experiment_name", "eval_saved",
        "--seed", str(args.seed),
        "--episode_length", str(args.episode_length),
        "--buffer_size", str(args.buffer_size),
        "--batch_size", str(args.batch_size),
        "--hidden_size", str(args.hidden_size),
        "--model_dir", str(Path(args.model_dir).resolve()),
        "--num_eval_episodes", str(args.episodes),
    ]
    all_args = parse_train_args(train_cli, get_config())

    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(all_args.n_training_threads)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    env = make_train_env(all_args)
    eval_env = make_eval_env(all_args)
    num_agents = all_args.num_agents
    policy_info = {
        "policy_0": {
            "cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
            "cent_act_dim": get_cent_act_dim(env.action_space),
            "obs_space": env.observation_space[0],
            "share_obs_space": env.share_observation_space[0],
            "act_space": env.action_space[0],
        }
    }

    def policy_mapping_fn(_agent_id):
        return "policy_0"

    from offpolicy.runner.mlp.sumo_runner import SUMORunner

    return SUMORunner(
        config={
            "args": all_args,
            "policy_info": policy_info,
            "policy_mapping_fn": policy_mapping_fn,
            "env": env,
            "eval_env": eval_env,
            "num_agents": num_agents,
            "device": device,
            "use_same_share_obs": all_args.use_same_share_obs,
            "run_dir": Path("offpolicy/scripts/results/eval_saved"),
        }
    )


def summarize(rows):
    steps = np.array([row["average_episode_steps"] for row in rows], dtype=float)
    rewards = np.array([row["average_episode_rewards"] for row in rows], dtype=float)
    rates = np.array([row["average_evacuation_rate"] for row in rows], dtype=float)
    escaped = np.array([row["average_persons_escaped"] for row in rows], dtype=float)
    return {
        "policy": "trained_mqmix_best",
        "episodes": int(len(rows)),
        "mean_steps": round(float(steps.mean()), 3),
        "std_steps": round(float(steps.std()), 3),
        "mean_reward": round(float(rewards.mean()), 3),
        "mean_escaped": round(float(escaped.mean()), 3),
        "mean_rate": round(float(rates.mean()), 3),
    }


def main():
    args = parse_args()
    runner = build_runner(args)
    rows = [
        runner.collecter(explore=False, training_episode=False, warmup=False)
        for _ in range(args.episodes)
    ]
    result = {"summary": summarize(rows), "episodes": rows}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")

    runner.env.close()
    runner.eval_env.close()


if __name__ == "__main__":
    main()
