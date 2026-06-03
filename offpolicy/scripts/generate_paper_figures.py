import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from offpolicy.envs.envs.CA.robot_env import RobotEnvironment
from offpolicy.scripts.evaluate_evacuation import MAP_PATH, TARGET_AREA
from offpolicy.scripts.evaluate_trained_policy import build_runner


RESULT_ROOT = Path(__file__).resolve().parent / "results"
PAPER_ROOT = RESULT_ROOT / "paper_outputs" / "run13_repulsion_only"
RUN13_BEST = RESULT_ROOT / "two_robots" / "2_robots" / "mqmix" / "check" / "run13" / "models_best"

COLORS = {
    "no_robot": "#4E79A7",
    "default_stay": "#F28E2B",
    "static_good": "#E15759",
    "trained_mqmix_best": "#59A14F",
}

POLICY_LABELS = {
    "no_robot": "无机器人",
    "default_stay": "默认静止机器人",
    "static_good": "手工静态位置",
    "trained_mqmix_best": "MQMIX最佳策略",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready evacuation figures.")
    parser.add_argument("--output-dir", default=str(PAPER_ROOT / "figures"))
    parser.add_argument("--model-dir", default=str(RUN13_BEST))
    parser.add_argument("--policy-comparison", default=str(PAPER_ROOT / "policy_comparison.csv"))
    parser.add_argument("--training-curve", default=str(PAPER_ROOT / "training_eval_curve.csv"))
    parser.add_argument("--repulsion-scan", default=str(PAPER_ROOT / "top_repulsion_configs.csv"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=200)
    return parser.parse_args()


def configure_fonts():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            font = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = font.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def setup_style():
    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, output_dir, name):
    png = output_dir / f"{name}.png"
    svg = output_dir / f"{name}.svg"
    fig.savefig(png, dpi=450)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def make_env(seed, policy):
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
    env.set_robot_repulsion_factor(1.0)
    if policy == "no_robot":
        env.robot_positions = []
    elif policy == "static_good":
        env.robot_positions = [(15, 15), (14, 18)]
    return env


def record_environment_state(env, density, robot_density):
    for person in env.persons:
        if not person.escaped and not person.is_dead:
            r, c = person.position
            density[r, c] += 1
    for r, c in env.robot_positions:
        robot_density[r, c] += 1


def run_direct_policy_episode(seed, policy, max_steps):
    env = make_env(seed, policy)
    rows, cols = env.map_loader.rows, env.map_loader.cols
    density = np.zeros((rows, cols), dtype=float)
    robot_density = np.zeros((rows, cols), dtype=float)
    rates = np.full(max_steps + 1, np.nan, dtype=float)
    escaped = np.full(max_steps + 1, np.nan, dtype=float)
    positions = []

    rates[0] = 0.0
    escaped[0] = 0.0
    record_environment_state(env, density, robot_density)
    done = False
    info = None
    step = 0
    while not done and step < max_steps:
        actions = [] if policy == "no_robot" else [4 for _ in env.robot_positions]
        _, _, done, info = env.step(actions)
        step = info["step"]
        record_environment_state(env, density, robot_density)
        rates[step] = info["evacuation_rate"]
        escaped[step] = info["persons_escaped"]
        positions.append({"step": step, "robot_positions": list(env.robot_positions)})

    final_rate = rates[~np.isnan(rates)][-1]
    final_escaped = escaped[~np.isnan(escaped)][-1]
    rates[np.isnan(rates)] = final_rate
    escaped[np.isnan(escaped)] = final_escaped
    return {
        "policy": policy,
        "seed": seed,
        "final_step": step,
        "final_rate": float(final_rate),
        "final_escaped": float(final_escaped),
        "rates": rates,
        "escaped": escaped,
        "density": density,
        "robot_density": robot_density,
        "positions": positions,
    }


def build_trained_runner(model_dir, episodes, seed, max_steps):
    runner_args = SimpleNamespace(
        model_dir=model_dir,
        episodes=episodes,
        seed=seed,
        episode_length=max_steps,
        hidden_size=64,
        buffer_size=10000,
        batch_size=64,
    )
    return build_runner(runner_args)


@torch.no_grad()
def run_trained_episode(runner, seed, max_steps):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = runner.eval_env
    policy = runner.policies["policy_0"]
    runner.trainer.prep_rollout()
    obs = env.reset()
    base_env = env.envs[0].env.robot_env
    rows, cols = base_env.map_loader.rows, base_env.map_loader.cols
    density = np.zeros((rows, cols), dtype=float)
    robot_density = np.zeros((rows, cols), dtype=float)
    rates = np.full(max_steps + 1, np.nan, dtype=float)
    escaped = np.full(max_steps + 1, np.nan, dtype=float)
    positions = []

    rates[0] = 0.0
    escaped[0] = 0.0
    record_environment_state(base_env, density, robot_density)
    info = None
    final_step = max_steps
    for _ in range(max_steps):
        obs_batch = np.concatenate(obs)
        acts_batch, _ = policy.get_actions(obs_batch, t_env=0, explore=False)
        if not isinstance(acts_batch, np.ndarray):
            acts_batch = acts_batch.cpu().detach().numpy()
        env_acts = np.split(acts_batch, env.num_envs)
        obs, _, dones, infos = env.step(env_acts)
        info = infos[0].item() if hasattr(infos[0], "item") else infos[0]
        final_step = int(info["step"])
        record_environment_state(base_env, density, robot_density)
        rates[final_step] = info["evacuation_rate"]
        escaped[final_step] = info["persons_escaped"]
        positions.append({"step": final_step, "robot_positions": list(base_env.robot_positions)})
        if np.all(dones):
            break

    final_rate = rates[~np.isnan(rates)][-1]
    final_escaped = escaped[~np.isnan(escaped)][-1]
    rates[np.isnan(rates)] = final_rate
    escaped[np.isnan(escaped)] = final_escaped
    return {
        "policy": "trained_mqmix_best",
        "seed": seed,
        "final_step": final_step,
        "final_rate": float(final_rate),
        "final_escaped": float(final_escaped),
        "rates": rates,
        "escaped": escaped,
        "density": density,
        "robot_density": robot_density,
        "positions": positions,
    }


def collect_trajectory_data(args):
    policies = ["no_robot", "default_stay", "static_good"]
    records = {policy: [] for policy in policies + ["trained_mqmix_best"]}
    for policy in policies:
        for idx in range(args.episodes):
            records[policy].append(run_direct_policy_episode(args.seed + idx, policy, args.max_steps))

    runner = build_trained_runner(args.model_dir, args.episodes, args.seed, args.max_steps)
    try:
        for idx in range(args.episodes):
            records["trained_mqmix_best"].append(run_trained_episode(runner, args.seed + idx, args.max_steps))
    finally:
        runner.env.close()
        runner.eval_env.close()
    return records


def aggregate_records(records, max_steps):
    aggregated = {}
    for policy, episodes in records.items():
        rates = np.stack([row["rates"] for row in episodes])
        escaped = np.stack([row["escaped"] for row in episodes])
        density = np.sum([row["density"] for row in episodes], axis=0) / len(episodes)
        robot_density = np.sum([row["robot_density"] for row in episodes], axis=0) / len(episodes)
        steps = np.array([row["final_step"] for row in episodes], dtype=float)
        aggregated[policy] = {
            "mean_rate": rates.mean(axis=0),
            "std_rate": rates.std(axis=0),
            "mean_escaped": escaped.mean(axis=0),
            "std_escaped": escaped.std(axis=0),
            "density": density,
            "robot_density": robot_density,
            "episode_steps": steps,
        }
    aggregated["steps"] = np.arange(max_steps + 1)
    return aggregated


def write_matrix_csv(path, matrix):
    pd.DataFrame(matrix).to_csv(path, index=False, header=False)


def write_trajectory_data(output_dir, records, aggregated, args):
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    curve_rows = []
    for policy, episodes in records.items():
        for episode_idx, row in enumerate(episodes):
            episode_rows.append(
                {
                    "policy": policy,
                    "episode": episode_idx,
                    "seed": row["seed"],
                    "final_step": row["final_step"],
                    "final_rate": round(row["final_rate"], 6),
                    "final_escaped": row["final_escaped"],
                }
            )
        for step in aggregated["steps"]:
            curve_rows.append(
                {
                    "policy": policy,
                    "step": int(step),
                    "mean_evacuation_rate": round(float(aggregated[policy]["mean_rate"][step]), 6),
                    "std_evacuation_rate": round(float(aggregated[policy]["std_rate"][step]), 6),
                    "mean_escaped": round(float(aggregated[policy]["mean_escaped"][step]), 3),
                    "std_escaped": round(float(aggregated[policy]["std_escaped"][step]), 3),
                }
            )
        write_matrix_csv(data_dir / f"person_density_{policy}.csv", aggregated[policy]["density"])
        write_matrix_csv(data_dir / f"robot_density_{policy}.csv", aggregated[policy]["robot_density"])

    pd.DataFrame(episode_rows).to_csv(data_dir / "trajectory_episode_summaries.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(data_dir / "cumulative_evacuation_curves.csv", index=False)

    no_robot_density = aggregated["no_robot"]["density"]
    qmix_density = aggregated["trained_mqmix_best"]["density"]
    write_matrix_csv(data_dir / "person_density_no_robot_minus_qmix.csv", no_robot_density - qmix_density)

    return data_dir


def load_map_data():
    env = RobotEnvironment(MAP_PATH, TARGET_AREA, 1, 1, use_health=False)
    return env.map_loader.map_data, env.map_loader.exits, env.map_loader.fires


def overlay_map(ax, map_data, exits, fires, crop=None, show_legend=True):
    if crop is None:
        r0, r1, c0, c1 = 0, map_data.shape[0], 0, map_data.shape[1]
    else:
        r0, r1, c0, c1 = crop

    for r in range(r0, r1):
        for c in range(c0, c1):
            if map_data[r, c] == 1:
                ax.add_patch(
                    Rectangle(
                        (c - c0 - 0.5, r - r0 - 0.5),
                        1,
                        1,
                        facecolor="#d1d5db",
                        edgecolor="none",
                        zorder=3,
                    )
                )
    plotted = {}
    fire_labeled = False
    exit_labeled = False
    for r, c in fires:
        if r0 <= r < r1 and c0 <= c < c1:
            label = "火源" if not fire_labeled else "_nolegend_"
            plotted["fire"] = ax.scatter(c - c0, r - r0, s=26, marker="s", color="#C73E1D", label=label, zorder=4)
            fire_labeled = True
    for r, c in exits:
        if r0 <= r < r1 and c0 <= c < c1:
            label = "出口" if not exit_labeled else "_nolegend_"
            plotted["exit"] = ax.scatter(c - c0, r - r0, s=30, marker=">", color="#059669", label=label, zorder=4)
            exit_labeled = True
    ax.set_xlim(-0.5, c1 - c0 - 0.5)
    ax.set_ylim(r1 - r0 - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("列")
    ax.set_ylabel("行")
    ax.grid(False)
    if show_legend and plotted:
        ax.legend(loc="upper right", frameon=True, fontsize=7)


def masked_matrix(matrix, map_data, crop=None):
    if crop is not None:
        r0, r1, c0, c1 = crop
        matrix = matrix[r0:r1, c0:c1]
        map_data = map_data[r0:r1, c0:c1]
    return np.ma.array(matrix, mask=(map_data == 1))


def plot_policy_comparison(output_dir, policy_csv):
    df = pd.read_csv(policy_csv)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = [POLICY_LABELS[p] for p in df["policy"]]
    colors = [COLORS[p] for p in df["policy"]]
    bars = ax.bar(
        range(len(df)),
        df["mean_steps"],
        yerr=df["std_steps"],
        color=colors,
        capsize=4,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=14, ha="right")
    ax.set_ylabel("达到80%疏散所需步数")
    ax.set_title("不同策略的疏散时间对比")
    for bar, reduction in zip(bars, df["reduction_vs_no_robot_percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f"{reduction:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, max(df["mean_steps"] + df["std_steps"]) + 25)
    return save_figure(fig, output_dir, "fig1_evacuation_time_comparison")


def plot_cumulative_curve(output_dir, aggregated):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    steps = aggregated["steps"]
    for policy in ["no_robot", "default_stay", "static_good", "trained_mqmix_best"]:
        mean = aggregated[policy]["mean_rate"]
        std = aggregated[policy]["std_rate"]
        color = COLORS[policy]
        ax.plot(steps, mean, color=color, linewidth=1.8, label=POLICY_LABELS[policy])
        ax.fill_between(steps, np.maximum(mean - std, 0), np.minimum(mean + std, 1), color=color, alpha=0.12)
    ax.axhline(0.8, color="#374151", linestyle="--", linewidth=1.0, label="80%目标")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.86)
    ax.set_xlabel("时间步")
    ax.set_ylabel("累计疏散率")
    ax.set_title("累计疏散率随时间变化")
    ax.legend(loc="lower right", frameon=True)
    return save_figure(fig, output_dir, "fig2_cumulative_evacuation_curve")


def plot_density_heatmaps(output_dir, aggregated, map_data, exits, fires):
    no_robot = np.log1p(aggregated["no_robot"]["density"])
    qmix = np.log1p(aggregated["trained_mqmix_best"]["density"])
    vmax = max(float(np.nanmax(no_robot)), float(np.nanmax(qmix)))
    cmap = matplotlib.colormaps["YlOrRd"].copy()
    cmap.set_bad("#f3f4f6")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), constrained_layout=True)
    for ax, data, title in [
        (axes[0], no_robot, "无机器人"),
        (axes[1], qmix, "MQMIX最佳策略"),
    ]:
        im = ax.imshow(masked_matrix(data, map_data), cmap=cmap, vmin=0, vmax=vmax, origin="upper")
        overlay_map(ax, map_data, exits, fires, show_legend=(ax is axes[1]))
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02)
    cbar.set_label("log(1 + 人员占用次数/回合)")
    fig.suptitle("人员空间密度热力图")
    return save_figure(fig, output_dir, "fig3_person_density_heatmaps")


def plot_exit_density_difference(output_dir, aggregated, map_data, exits, fires):
    crop = (8, 24, 24, 37)
    diff = aggregated["no_robot"]["density"] - aggregated["trained_mqmix_best"]["density"]
    data = masked_matrix(diff, map_data, crop)
    max_abs = float(np.nanmax(np.abs(data))) if data.size else 1.0
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    im = ax.imshow(data, cmap=cmap, vmin=-max_abs, vmax=max_abs, origin="upper")
    overlay_map(ax, map_data, exits, fires, crop=crop)
    ax.set_title("出口区域人员密度差异")
    ax.set_xlabel("列（出口区域裁剪）")
    ax.set_ylabel("行（出口区域裁剪）")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("无机器人 - MQMIX（人员步/回合）")
    return save_figure(fig, output_dir, "fig4_exit_density_difference")


def plot_robot_density(output_dir, aggregated, map_data, exits, fires):
    crop = (8, 24, 20, 37)
    data = masked_matrix(aggregated["trained_mqmix_best"]["robot_density"], map_data, crop)
    cmap = matplotlib.colormaps["PuBuGn"].copy()
    cmap.set_bad("#f3f4f6")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(data, cmap=cmap, origin="upper")
    overlay_map(ax, map_data, exits, fires, crop=crop)
    ax.set_title("MQMIX策略下机器人位置热力图")
    ax.set_xlabel("列（出口区域裁剪）")
    ax.set_ylabel("行（出口区域裁剪）")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("机器人占用次数/回合")
    return save_figure(fig, output_dir, "fig5_robot_position_heatmap")


def plot_training_convergence(output_dir, curve_csv, policy_csv):
    curve = pd.read_csv(curve_csv)
    comparison = pd.read_csv(policy_csv).set_index("policy")
    no_robot_steps = float(comparison.loc["no_robot", "mean_steps"])
    best_steps = float(comparison.loc["trained_mqmix_best", "mean_steps"])
    target_steps = no_robot_steps * 0.92
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(curve["step"], curve["value"], color="#0077BB", marker="o", linewidth=1.8, label="训练期评估")
    ax.axhline(no_robot_steps, color="#4b5563", linestyle="--", linewidth=1.1, label="无机器人基线")
    ax.axhline(target_steps, color="#EE7733", linestyle=":", linewidth=1.4, label="8%减少目标")
    ax.axhline(best_steps, color="#009988", linestyle="-.", linewidth=1.4, label="best复评")
    best_row = curve.loc[curve["value"].idxmin()]
    ax.scatter([best_row["step"]], [best_row["value"]], color="#CC3311", s=42, zorder=4)
    ax.annotate(
        "最佳checkpoint",
        xy=(best_row["step"], best_row["value"]),
        xytext=(18, 16),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 0.8},
    )
    ax.set_xlabel("环境交互步数")
    ax.set_ylabel("达到80%疏散所需步数")
    ax.set_title("MQMIX强化学习收敛曲线")
    ax.set_xlim(0, max(curve["step"]) + 2500)
    ax.legend(loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "fig6_training_convergence")


def plot_repulsion_scan(output_dir, scan_csv):
    df = pd.read_csv(scan_csv).head(10)
    labels = [
        f"{row.position}\nF={row.factor}, R={row.cutoff}, D={row.decay}"
        for row in df.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(range(len(df)), df["reduction_percent"], color="#0077BB", edgecolor="white", linewidth=0.8)
    ax.axhline(8.0, color="#EE7733", linestyle="--", linewidth=1.2, label="8%目标")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel("相比无机器人减少比例（%）")
    ax.set_title("排斥力场参数扫描Top10")
    ax.legend(loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "fig7_repulsion_field_scan")


def write_manifest(output_dir, data_dir, outputs):
    rows = [
        {
            "Figure": "Fig.1 Evacuation-time comparison",
            "Data file": str(data_dir / "policy_comparison_for_figures.csv"),
            "Real/mock": "Real",
            "Source": "run13 independent 50-episode evaluation",
            "Script": "offpolicy/scripts/generate_paper_figures.py",
            "Outputs": "fig1_evacuation_time_comparison.png/svg",
        },
        {
            "Figure": "Fig.2 Cumulative evacuation curves",
            "Data file": str(data_dir / "cumulative_evacuation_curves.csv"),
            "Real/mock": "Real",
            "Source": "20 trajectory episodes per policy",
            "Script": "offpolicy/scripts/generate_paper_figures.py",
            "Outputs": "fig2_cumulative_evacuation_curve.png/svg",
        },
        {
            "Figure": "Fig.3 Person density heatmaps",
            "Data file": "person_density_no_robot.csv; person_density_trained_mqmix_best.csv",
            "Real/mock": "Real",
            "Source": "20 trajectory episodes per policy",
            "Script": "offpolicy/scripts/generate_paper_figures.py",
            "Outputs": "fig3_person_density_heatmaps.png/svg",
        },
        {
            "Figure": "Fig.4 Exit-region density difference",
            "Data file": str(data_dir / "person_density_no_robot_minus_qmix.csv"),
            "Real/mock": "Real",
            "Source": "20 trajectory episodes per policy",
            "Script": "offpolicy/scripts/generate_paper_figures.py",
            "Outputs": "fig4_exit_density_difference.png/svg",
        },
        {
            "Figure": "Fig.5 Robot position heatmap",
            "Data file": str(data_dir / "robot_density_trained_mqmix_best.csv"),
            "Real/mock": "Real",
            "Source": "20 trained-policy trajectory episodes",
            "Script": "offpolicy/scripts/generate_paper_figures.py",
            "Outputs": "fig5_robot_position_heatmap.png/svg",
        },
        {
            "Figure": "Fig.6 MQMIX convergence curve",
            "Data file": str(data_dir / "training_convergence_for_figures.csv"),
            "Real/mock": "Real",
            "Source": "run13 TensorBoard scalar export",
            "Script": "offpolicy/scripts/generate_paper_figures.py",
            "Outputs": "fig6_training_convergence.png/svg",
        },
        {
            "Figure": "Fig.7 Repulsion-field scan",
            "Data file": str(data_dir / "repulsion_scan_top10_for_figures.csv"),
            "Real/mock": "Real",
            "Source": "repulsion_scan_stage2.csv",
            "Script": "offpolicy/scripts/generate_paper_figures.py",
            "Outputs": "fig7_repulsion_field_scan.png/svg",
        },
    ]

    manifest = output_dir / "data-manifest.md"
    lines = [
        "# Data Manifest",
        "",
        "| Figure | Data file | Real/mock | Source | Script | Outputs |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Figure']} | {row['Data file']} | {row['Real/mock']} | "
            f"{row['Source']} | {row['Script']} | {row['Outputs']} |"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = output_dir / "figure_outputs.json"
    summary.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    policy_csv = Path(args.policy_comparison)
    curve_csv = Path(args.training_curve)
    scan_csv = Path(args.repulsion_scan)
    shutil.copyfile(policy_csv, data_dir / "policy_comparison_for_figures.csv")
    shutil.copyfile(curve_csv, data_dir / "training_convergence_for_figures.csv")
    shutil.copyfile(scan_csv, data_dir / "repulsion_scan_top10_for_figures.csv")

    records = collect_trajectory_data(args)
    aggregated = aggregate_records(records, args.max_steps)
    write_trajectory_data(output_dir, records, aggregated, args)
    map_data, exits, fires = load_map_data()

    outputs = {}
    for name, paths in [
        ("fig1", plot_policy_comparison(output_dir, data_dir / "policy_comparison_for_figures.csv")),
        ("fig2", plot_cumulative_curve(output_dir, aggregated)),
        ("fig3", plot_density_heatmaps(output_dir, aggregated, map_data, exits, fires)),
        ("fig4", plot_exit_density_difference(output_dir, aggregated, map_data, exits, fires)),
        ("fig5", plot_robot_density(output_dir, aggregated, map_data, exits, fires)),
        ("fig6", plot_training_convergence(output_dir, data_dir / "training_convergence_for_figures.csv", data_dir / "policy_comparison_for_figures.csv")),
        ("fig7", plot_repulsion_scan(output_dir, data_dir / "repulsion_scan_top10_for_figures.csv")),
    ]:
        outputs[name] = [str(path) for path in paths]

    write_manifest(output_dir, data_dir, outputs)
    print(json.dumps({"output_dir": str(output_dir), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
