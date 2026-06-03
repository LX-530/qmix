# 多机器人火灾疏散环境 - MQMIX 强化学习

## 当前目标

本项目当前复现目标只优化疏散时间：默认关闭健康值/死亡对奖励的影响，以 80% 人员完成疏散所需步数作为主要指标。当前版本不允许机器人直接提高出口通行能力，机器人只通过占位和排斥势场改变门口附近的人群排队结构，用于表示组织排队、破坏门口拱形堵塞、减少出口摩擦。

## 关键文件

- `offpolicy/envs/envs/CA/robot_env.py`：火灾元胞自动机、人群移动、机器人排斥力、出口服务队列和门口高密度阻滞。
- `offpolicy/envs/envs/CA/qmix_env.py`：MQMIX 环境封装，设置人员数量、最大步数、疏散目标比例和健康值开关。
- `offpolicy/envs/envs/CA/map.json`：疏散场景地图。
- `offpolicy/scripts/evaluate_evacuation.py`：无机器人、默认机器人、手工静态机器人基线评估。
- `offpolicy/scripts/evaluate_trained_policy.py`：加载已保存 MQMIX checkpoint 做独立复评。
- `offpolicy/scripts/scan_repulsion_fields.py`：扫描机器人排斥力场强度、半径、衰减和位置差异。
- `offpolicy/scripts/summarize_evacuation_results.py`：生成论文用 CSV、曲线图、对比图、排斥力场图和文献依据说明。

## 当前环境机制

默认疏散目标：

```python
target_area = (3, 32, 2, 7)
num_persons = 140
max_steps = 200
use_health = False
evacuation_target_rate = 0.8
```

出口约束与门口摩擦：

```python
self.exit_capacity_per_step = 1
self.base_service_time = 2
self.arch_block_probability = 0.9
self.arch_density_threshold = 5
self.robot_repulsion_factor = 0.5
self.robot_repulsion_cutoff_radius = 4.0
self.robot_repulsion_decay = 0.35
```

含义：

- 未引导时，每个出口每步最多完成 1 人疏散。
- 人员到达出口后固定等待 2 步。
- 机器人不再把出口等待时间从 2 步降为 1 步。
- 机器人不再直接降低门口阻滞概率。
- 门口高密度区域会触发冲突摩擦和拱形拥堵；机器人只能通过自身占位和排斥力改变局部密度、来流方向和排队结构，从而间接缩短疏散时间。

奖励函数只围绕疏散时间：

```python
reward = newly_escaped * escape_reward - step_penalty
```

达到 80% 疏散目标时奖励剩余步数，超时惩罚。健康值默认关闭，`avg_health` 固定为 100，仅作为兼容旧日志的字段保留。

## 运行命令

基线评估：

```bash
python -m offpolicy.scripts.evaluate_evacuation --episodes 50 --target-rate 0.8
```

GPU 训练示例：

```bash
python -m offpolicy.scripts.train.train --algorithm_name mqmix --num_env_steps 50000 --episode_length 200 --buffer_size 15000 --batch_size 64 --hidden_size 64 --save_interval 5000 --log_interval 5000 --eval_interval 5000 --num_eval_episodes 20 --epsilon_anneal_time 15000
```

注意：本项目原始参数中 `--cuda` 是 `store_false`，传入 `--cuda` 会关闭 CUDA；GPU 训练时不要带这个参数。当前验证环境为 CUDA PyTorch，设备为 `NVIDIA GeForce RTX 3050 Laptop GPU`。

最佳 checkpoint 复评：

```bash
python -m offpolicy.scripts.evaluate_trained_policy --model-dir "D:\github\qmix\offpolicy\scripts\results\two_robots\2_robots\mqmix\check\run13\models_best" --episodes 50 --episode-length 200 --output "D:\github\qmix\offpolicy\scripts\results\two_robots\2_robots\mqmix\check\run13\models_best\trained_eval_repulsion_only_50.json"
```

排斥力场扫描：

```bash
python -m offpolicy.scripts.scan_repulsion_fields --episodes 10 --seed 1200 --block-probs 0.9 --factors 0.5,0.8,1.0,1.2 --cutoffs 3,4,5 --decays 0.35,0.5 --positions upstream_exit,behind_exit,default --output "D:\github\qmix\offpolicy\scripts\results\repulsion_scan_stage2.csv"
```

生成论文用结果：

```bash
python -m offpolicy.scripts.summarize_evacuation_results --long-run run13 --best-run run13 --trained-eval "D:\github\qmix\offpolicy\scripts\results\two_robots\2_robots\mqmix\check\run13\models_best\trained_eval_repulsion_only_50.json" --baseline-episodes 50 --output-dir "D:\github\qmix\offpolicy\scripts\results\paper_outputs\run13_repulsion_only"
```

## 当前结果

旧 `run9` 结果使用了“机器人靠近出口时直接把服务时间从 2 步降到 1 步”的机制，已经作废，不能作为当前结论使用。

当前 `run13` 在“固定出口服务时间 + 独立门口冲突摩擦 + 机器人排斥力”机制下，50 回合复评结果如下：

| 策略 | 平均步数 | 标准差 | 平均疏散人数 | 平均疏散率 | 相比无机器人减少 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无机器人 | 172.38 | 10.59 | 111.90 | 0.799 | 0.00% |
| 默认静止机器人 | 155.02 | 5.03 | 112.16 | 0.801 | 10.07% |
| 手工静态位置 | 171.72 | 9.57 | 112.04 | 0.800 | 0.38% |
| MQMIX 最佳策略（run13） | 151.04 | 2.78 | 112.08 | 0.801 | 12.38% |

结论：在不直接改变出口服务时间、不直接降低门口阻滞概率的条件下，QMIX 最佳策略仍能通过机器人排斥力和占位使 80% 疏散时间从 172.38 步降到 151.04 步，减少约 12.38%，达到“减少 8% 左右”的目标。
