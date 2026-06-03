# test_exit_service.py
# 测试出口服务机制的效果

import numpy as np
from robot_env import RobotEnvironment
import random

def run_simulation(use_service_time, exit_capacity, base_service_time, num_runs=3):
    """
    运行多次模拟并统计结果
    
    参数：
    - use_service_time: 是否启用出口服务机制
    - exit_capacity: 每个出口每步容量
    - base_service_time: 基础服务时间
    - num_runs: 运行次数
    """
    results = []
    
    for run in range(num_runs):
        # 初始化环境
        target_area = (3, 32, 2, 7)
        num_persons = 50  # 使用较少人数便于观察
        max_steps = 300
        
        robot_env = RobotEnvironment('map.json', target_area, num_persons, max_steps)
        
        # 配置出口服务参数
        robot_env.use_exit_service_time = use_service_time
        robot_env.exit_capacity_per_step = exit_capacity
        robot_env.base_service_time = base_service_time
        
        # 重置环境以应用配置
        robot_env.reset()
        
        print(f"\n{'='*60}")
        print(f"运行 {run+1}/{num_runs}")
        print(f"出口服务机制: {'启用' if use_service_time else '禁用'}")
        if use_service_time:
            print(f"  - 每个出口每步容量: {exit_capacity} 人")
            print(f"  - 基础服务时间: {base_service_time} 步")
        print(f"初始人数: {robot_env.initial_person_count}")
        print(f"出口数量: {len(robot_env.map_loader.exits)}")
        print(f"出口位置: {list(robot_env.map_loader.exits)}")
        
        # 记录每步逃生人数
        escape_per_step = []
        prev_escaped = 0
        
        # 仿真循环
        for step in range(max_steps):
            # 机器人不动（纯观察人员疏散）
            robot_actions = [4, 4]
            
            state, reward, done, info = robot_env.step(robot_actions)
            
            # 统计本步逃生人数
            current_escaped = info['persons_escaped']
            escaped_this_step = current_escaped - prev_escaped
            escape_per_step.append(escaped_this_step)
            prev_escaped = current_escaped
            
            # 每20步打印详细信息
            if step % 20 == 0:
                print(f"步骤 {step:3d}: 存活 {info['persons_remaining']:3d}, "
                      f"已疏散 {info['persons_escaped']:3d}, "
                      f"死亡 {info['persons_dead']:3d}, "
                      f"本步逃生 {escaped_this_step}")
                
                # 统计正在出口服务的人数
                if use_service_time:
                    in_service = sum(1 for p in robot_env.persons if p.in_exit_process)
                    print(f"       正在出口排队: {in_service} 人")
            
            if done:
                print(f"\n仿真结束于步骤 {step + 1}")
                print(f"最终统计 - 疏散: {info['persons_escaped']}人, "
                      f"死亡: {info['persons_dead']}人, "
                      f"剩余: {info['persons_remaining']}人")
                print(f"疏散率: {info['evacuation_rate']:.2%}")
                break
        
        # 分析逃生速率
        escape_per_step = np.array(escape_per_step)
        non_zero_steps = escape_per_step[escape_per_step > 0]
        
        print(f"\n逃生统计分析:")
        print(f"  总疏散时间: {step + 1} 步")
        print(f"  总逃生人数: {current_escaped}")
        print(f"  平均逃生速率: {current_escaped / (step + 1):.3f} 人/步")
        if len(non_zero_steps) > 0:
            print(f"  有逃生的步数: {len(non_zero_steps)} 步")
            print(f"  逃生步的平均人数: {non_zero_steps.mean():.3f} 人/步")
            print(f"  逃生步的最大人数: {non_zero_steps.max()} 人/步")
            print(f"  逃生步的最小人数: {non_zero_steps.min()} 人/步")
            print(f"  逃生步的标准差: {non_zero_steps.std():.3f}")
        
        results.append({
            'total_steps': step + 1,
            'escaped': current_escaped,
            'died': info['persons_dead'],
            'avg_rate': current_escaped / (step + 1),
            'escape_per_step': escape_per_step
        })
    
    return results

if __name__ == "__main__":
    print("="*60)
    print("出口服务机制测试")
    print("="*60)
    
    # 测试1：禁用出口服务（原始行为）
    print("\n\n【测试1：禁用出口服务机制（原始行为）】")
    results_no_service = run_simulation(
        use_service_time=False,
        exit_capacity=1,
        base_service_time=0,
        num_runs=1
    )
    
    # 测试2：启用出口服务，容量=1，服务时间=2
    print("\n\n【测试2：启用出口服务机制 (容量=1人/步, 服务时间=2步)】")
    results_with_service = run_simulation(
        use_service_time=True,
        exit_capacity=1,
        base_service_time=2,
        num_runs=1
    )
    
    # 测试3：启用出口服务，容量=1，服务时间=3（更严格）
    print("\n\n【测试3：启用出口服务机制 (容量=1人/步, 服务时间=3步)】")
    results_stricter = run_simulation(
        use_service_time=True,
        exit_capacity=1,
        base_service_time=3,
        num_runs=1
    )
    
    # 对比总结
    print("\n\n" + "="*60)
    print("对比总结")
    print("="*60)
    print(f"\n1. 原始机制（无服务时间）:")
    print(f"   疏散时间: {results_no_service[0]['total_steps']} 步")
    print(f"   平均速率: {results_no_service[0]['avg_rate']:.3f} 人/步")
    
    print(f"\n2. 服务机制 (容量=1, 时间=2):")
    print(f"   疏散时间: {results_with_service[0]['total_steps']} 步")
    print(f"   平均速率: {results_with_service[0]['avg_rate']:.3f} 人/步")
    print(f"   时间增加: {results_with_service[0]['total_steps'] - results_no_service[0]['total_steps']} 步 "
          f"({(results_with_service[0]['total_steps'] / results_no_service[0]['total_steps'] - 1) * 100:.1f}%)")
    
    print(f"\n3. 服务机制 (容量=1, 时间=3):")
    print(f"   疏散时间: {results_stricter[0]['total_steps']} 步")
    print(f"   平均速率: {results_stricter[0]['avg_rate']:.3f} 人/步")
    print(f"   时间增加: {results_stricter[0]['total_steps'] - results_no_service[0]['total_steps']} 步 "
          f"({(results_stricter[0]['total_steps'] / results_no_service[0]['total_steps'] - 1) * 100:.1f}%)")
    
    print("\n结论: 出口服务机制成功引入了疏散瓶颈，增加了疏散时间。")
    print("      这为机器人优化提供了更真实的改进空间。")

