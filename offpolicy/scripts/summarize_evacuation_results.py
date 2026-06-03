import argparse
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt

from offpolicy.scripts.evaluate_evacuation import run_episode, summarize


RESULT_ROOT = Path(__file__).resolve().parent / "results"
RUN_ROOT = RESULT_ROOT / "two_robots" / "2_robots" / "mqmix" / "check"


def parse_args():
    parser = argparse.ArgumentParser(description="Create paper-ready evacuation result artifacts.")
    parser.add_argument("--long-run", default="run13")
    parser.add_argument("--best-run", default="run13")
    parser.add_argument("--trained-eval", default=None)
    parser.add_argument("--repulsion-scan", default=str(RESULT_ROOT / "repulsion_scan_stage2.csv"))
    parser.add_argument("--baseline-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def read_scalar_series(run_dir, tag):
    summary_path = Path(run_dir) / "logs" / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    suffix = f"/{tag}/{tag}"
    for key, values in data.items():
        if key.replace("\\", "/").endswith(suffix):
            return [{"step": int(step), "value": float(value)} for _, step, value in values]
    return []


def run_baselines(episodes, seed):
    baseline_args = SimpleNamespace(
        num_persons=140,
        max_steps=200,
        target_rate=0.8,
    )
    rows = []
    for policy in ["no_robot", "default_stay", "static_good"]:
        infos = [run_episode(baseline_args, seed + idx, policy) for idx in range(episodes)]
        rows.append(summarize(policy, infos))
    return rows


def load_trained_eval(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    summary_row = data["summary"]
    return {
        "policy": summary_row["policy"],
        "mean_steps": summary_row["mean_steps"],
        "std_steps": summary_row["std_steps"],
        "mean_escaped": summary_row["mean_escaped"],
        "mean_rate": summary_row["mean_rate"],
    }


def read_repulsion_scan(path):
    scan_path = Path(path)
    if not scan_path.exists():
        return []
    with open(scan_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda row: float(row["reduction_percent"]), reverse=True)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curve(path, long_series, best_series, no_robot_steps, trained_steps):
    plt.figure(figsize=(7.2, 4.2), dpi=160)
    if long_series:
        plt.plot(
            [x["step"] for x in long_series],
            [x["value"] for x in long_series],
            marker="o",
            label="GPU training",
        )
    if best_series and best_series != long_series:
        plt.plot(
            [x["step"] for x in best_series],
            [x["value"] for x in best_series],
            marker="s",
            label="Best-checkpoint run",
        )
    plt.axhline(no_robot_steps, color="#6b7280", linestyle="--", linewidth=1.2, label="No robot baseline")
    plt.axhline(trained_steps, color="#059669", linestyle=":", linewidth=1.5, label="Best policy re-eval")
    plt.xlabel("Environment steps")
    plt.ylabel("Steps to reach 80% evacuation")
    plt.title("Evacuation-Time Convergence")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_comparison(path, rows):
    labels = {
        "no_robot": "No robot",
        "default_stay": "Default stay",
        "static_good": "Static hand-set",
        "trained_mqmix_best": "MQMIX best",
    }
    colors = ["#4b5563", "#2563eb", "#7c3aed", "#059669"]
    x = list(range(len(rows)))
    means = [float(row["mean_steps"]) for row in rows]
    errors = [float(row.get("std_steps") or 0.0) for row in rows]

    plt.figure(figsize=(7.2, 4.2), dpi=160)
    plt.bar(x, means, yerr=errors, color=colors[:len(rows)], capsize=4)
    plt.xticks(x, [labels.get(row["policy"], row["policy"]) for row in rows], rotation=12, ha="right")
    plt.ylabel("Mean steps to reach 80% evacuation")
    plt.title("Policy Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_repulsion_scan(path, rows):
    top_rows = rows[:10]
    if not top_rows:
        return
    labels = [
        f"{row['position']}\nF={row['factor']},R={row['cutoff']},D={row['decay']}"
        for row in top_rows
    ]
    values = [float(row["reduction_percent"]) for row in top_rows]
    plt.figure(figsize=(8.4, 4.8), dpi=160)
    plt.bar(range(len(top_rows)), values, color="#0f766e")
    plt.xticks(range(len(top_rows)), labels, rotation=35, ha="right")
    plt.ylabel("Reduction vs no-robot baseline (%)")
    plt.title("Robot Repulsion Field Scan")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_literature_basis(path):
    lines = [
        "# 科学依据：只通过机器人排斥力减少门口摩擦",
        "",
        "## 建模逻辑",
        "",
        "当前模型不再让机器人直接把出口服务时间从 2 步降到 1 步，也不再让机器人直接降低门口阻滞概率。出口容量、出口服务时间、门口高密度冲突摩擦都是独立环境约束；机器人唯一可学习作用是通过自身占位和排斥力改变局部人群密度、来流方向和排队结构，从而削弱门口拱形堵塞。",
        "",
        "## 可引用依据",
        "",
        "1. Helbing 与 Molnár 的社会力模型把行人与障碍、其他行人之间的避让表示为排斥相互作用，可作为机器人排斥势场的基础：D. Helbing and P. Molnár, Social force model for pedestrian dynamics, Physical Review E, 1995. https://doi.org/10.1103/PhysRevE.51.4282",
        "",
        "2. Helbing、Farkas 与 Vicsek 的逃生恐慌模型讨论了出口处拱形结构、堵塞和 faster-is-slower 等现象，支持“出口附近高密度冲突会增加疏散时间”的假设：D. Helbing, I. Farkas, and T. Vicsek, Simulating dynamical features of escape panic, Nature, 2000. https://doi.org/10.1038/35035023",
        "",
        "3. Kirchner、Nishinari 与 Schadschneider 在元胞自动机行人模型中引入 friction/conflict 参数，用来描述多人竞争同一格点导致的阻滞和堵塞，直接支持本项目中的门口高密度阻滞概率：A. Kirchner, K. Nishinari, and A. Schadschneider, Friction effects and clogging in a cellular automaton model for pedestrian dynamics, Physical Review E, 2003. https://doi.org/10.1103/PhysRevE.67.056122",
        "",
        "4. Yanagisawa 等研究了出口前障碍物对瓶颈流出的影响，指出障碍物设置会改变出口处冲突与转向结构。本文中机器人不提高出口处理能力，而是作为可学习的移动排斥源和占位体，对门口拱形结构进行扰动：D. Yanagisawa et al., Introduction of frictional and turning function for pedestrian outflow with an obstacle, Physical Review E, 2009. https://doi.org/10.1103/PhysRevE.80.036110",
        "",
        "## 对本文实验的含义",
        "",
        "- 机器人排斥力代表人群对机器人占位的避让和路径重排。",
        "- 门口摩擦代表多人竞争出口前沿时产生的冲突、互相阻挡和拱形堵塞。",
        "- 若 QMIX 学到把机器人放在出口上游合适位置，疏散时间减少应解释为队列组织和拱形结构破坏，而不是出口服务能力被人为提高。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path, rows, long_series, best_series, best_metrics_path, top_scan_rows):
    no_robot = next(row for row in rows if row["policy"] == "no_robot")
    trained = next(row for row in rows if row["policy"] == "trained_mqmix_best")
    default = next(row for row in rows if row["policy"] == "default_stay")
    reduction_vs_no_robot = (float(no_robot["mean_steps"]) - float(trained["mean_steps"])) / float(no_robot["mean_steps"]) * 100
    reduction_vs_default = (float(default["mean_steps"]) - float(trained["mean_steps"])) / float(default["mean_steps"]) * 100

    lines = [
        "# 火灾疏散强化学习实验结果摘要",
        "",
        "## 实验设置",
        "",
        "- 目标：只优化达到 80% 人员疏散所需步数，不启用健康值目标。",
        "- 出口约束：每个出口每步最多完成 1 人疏散；人员到达出口后固定等待 2 步；机器人不直接降低出口服务时间。",
        "- 门口摩擦：高密度人员竞争出口前沿时，按局部密度触发阻滞概率；该概率不因机器人靠近而直接降低。",
        "- 机器人作用：只通过占位与排斥势场改变局部密度、来流和排队结构，用于表示组织排队、破坏门口拱形拥堵。",
        "- 训练：MQMIX，2 个机器人，CUDA PyTorch，RTX 3050 Laptop GPU。",
        f"- 最佳模型指标文件：`{best_metrics_path}`。",
        "",
        "## 主要结果",
        "",
        f"- 无机器人基线：{float(no_robot['mean_steps']):.2f}±{float(no_robot['std_steps']):.2f} 步。",
        f"- 默认静止机器人：{float(default['mean_steps']):.2f}±{float(default['std_steps']):.2f} 步。",
        f"- MQMIX 最佳策略复评：{float(trained['mean_steps']):.2f}±{float(trained['std_steps']):.2f} 步，平均疏散率 {float(trained['mean_rate']):.3f}。",
        f"- 相比无机器人，MQMIX 最佳策略将疏散时间减少 {reduction_vs_no_robot:.2f}%。",
        f"- 相比默认静止机器人，MQMIX 最佳策略再减少 {reduction_vs_default:.2f}%。",
        "",
        "## 排斥力场差异",
        "",
        "阶段 2 扫描固定门口阻滞概率为 0.9，只改变机器人排斥强度、影响半径、衰减系数和初始位置。较优配置集中在出口上游，而不是直接堵在出口格点上，说明效果来自对来流和拱形结构的整理。",
        "",
        "| 位置 | 强度 | 半径 | 衰减 | 机器人步数 | 相比无机器人减少 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_scan_rows[:8]:
        lines.append(
            f"| {row['position']} | {float(row['factor']):.2f} | {float(row['cutoff']):.1f} | "
            f"{float(row['decay']):.2f} | {float(row['robot_mean_steps']):.2f} | "
            f"{float(row['reduction_percent']):.2f}% |"
        )

    lines += [
        "",
        "## 收敛记录",
        "",
        f"- 训练评估点：{[(x['step'], round(x['value'], 3)) for x in best_series]}。",
        f"- 独立复评文件：`{path.parent / 'policy_comparison.csv'}`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    long_run_dir = RUN_ROOT / args.long_run
    best_run_dir = RUN_ROOT / args.best_run
    trained_eval = args.trained_eval or str(best_run_dir / "models_best" / "trained_eval_repulsion_only_50.json")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / "paper_outputs" / f"{args.best_run}_repulsion_only"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = run_baselines(args.baseline_episodes, args.seed)
    trained_row = load_trained_eval(trained_eval)
    rows = baseline_rows + [trained_row]
    no_robot_steps = next(row["mean_steps"] for row in rows if row["policy"] == "no_robot")

    for row in rows:
        row["reduction_vs_no_robot_percent"] = round(
            (float(no_robot_steps) - float(row["mean_steps"])) / float(no_robot_steps) * 100,
            3,
        )

    long_series = read_scalar_series(long_run_dir, "eval_average_episode_steps")
    best_series = read_scalar_series(best_run_dir, "eval_average_episode_steps")
    curve_rows = (
        [{"run": args.long_run, **row} for row in long_series]
        + ([] if args.best_run == args.long_run else [{"run": args.best_run, **row} for row in best_series])
    )

    comparison_csv = output_dir / "policy_comparison.csv"
    curve_csv = output_dir / "training_eval_curve.csv"
    write_csv(
        comparison_csv,
        rows,
        ["policy", "mean_steps", "std_steps", "mean_escaped", "mean_rate", "reduction_vs_no_robot_percent"],
    )
    write_csv(curve_csv, curve_rows, ["run", "step", "value"])

    scan_rows = read_repulsion_scan(args.repulsion_scan)
    if scan_rows:
        shutil.copyfile(args.repulsion_scan, output_dir / "repulsion_scan_stage2.csv")
        write_csv(
            output_dir / "top_repulsion_configs.csv",
            scan_rows[:20],
            [
                "block_prob",
                "position",
                "factor",
                "cutoff",
                "decay",
                "baseline_mean_steps",
                "robot_mean_steps",
                "robot_std_steps",
                "mean_escaped",
                "mean_rate",
                "reduction_percent",
            ],
        )

    plot_training_curve(
        output_dir / "training_curve.png",
        long_series,
        best_series,
        float(no_robot_steps),
        float(trained_row["mean_steps"]),
    )
    plot_comparison(output_dir / "policy_comparison.png", rows)
    plot_repulsion_scan(output_dir / "repulsion_scan_top10.png", scan_rows)
    write_literature_basis(output_dir / "literature_basis.md")
    write_markdown(
        output_dir / "experiment_summary.md",
        rows,
        long_series,
        best_series,
        best_run_dir / "models_best" / "best_eval_metrics.json",
        scan_rows,
    )

    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
