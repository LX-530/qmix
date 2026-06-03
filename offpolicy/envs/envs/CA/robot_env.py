import numpy as np
import random
import matplotlib.pyplot as plt
from offpolicy.envs.envs.CA.map_loader import MapLoader
from offpolicy.envs.envs.CA.person_behavior import Person
from offpolicy.envs.envs.CA.fire_model import FireModel

class RobotEnvironment:
    def __init__(self, map_file, target_area, num_persons, max_steps=300,
                 use_health=False, evacuation_target_rate=1.0):
        self.map_loader = MapLoader(map_file)
        self.target_area = target_area
        self.num_persons = num_persons
        self.max_steps = max_steps
        self.use_health = use_health
        self.evacuation_target_rate = evacuation_target_rate
        self.current_step = 0

        self.robot_positions = []
        self.robot_repulsion_factor = 0.5
        self.robot_repulsion_cutoff_radius = 4.0
        self.robot_repulsion_decay = 0.35
        self.step_penalty = 1.0
        self.escape_reward = 2.0

        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (0, 0)    # stay
        }

        # Doorway bottleneck model. Robots do not directly reduce either
        # service time or friction probability; they only affect pedestrians
        # through occupancy and the repulsive potential field below.
        self.use_exit_service_time = True  # 是否启用出口摩擦机制（True=出口有排队/通过延迟）
        self.exit_capacity_per_step = 1  # 每个出口每步最多完成疏散的人数
        self.base_service_time = 2  # 固定出口摩擦时间：到达出口后还需等待的步数
        self.use_arch_breaking_field = True
        self.arch_breaking_activation_radius = 6
        self.arch_corridor_depth = 8
        self.arch_lane_attraction = 0.0
        self.arch_side_repulsion = 0.0
        self.arch_zone_radius = 4
        self.arch_density_radius = 2
        self.arch_density_threshold = 5
        self.arch_block_probability = 0.9
        self.exit_service_mode = 'fixed'  # 服务时间模式：'fixed'=固定时间, 'density'=受密度影响（未来扩展）

        self._reset_environment()

    def _reset_environment(self):
        self.current_step = 0
        self._initialize_persons()
        self._initialize_robots()
        self.fire_model = FireModel(self.map_loader.fires, self.map_loader.rows, self.map_loader.cols)
        self.fire_field = self.fire_model.compute_dynamic_field()
        self.initial_person_count = len(self.persons)
        self.escaped_persons = 0
        self.prev_escaped_persons = 0  # 重置增量计数器


        return self._get_state()

    def _alive_persons(self):
        if self.use_health:
            return [p for p in self.persons if not p.is_dead]
        return [p for p in self.persons if not p.escaped]

    def _dead_count(self):
        if not self.use_health:
            return 0
        return len([p for p in self.persons if p.is_dead])

    def _avg_health(self, persons):
        if not self.use_health:
            return 100.0
        return sum(p.health for p in persons) / len(persons) if persons else 100.0

    def _evacuation_complete(self):
        if self.initial_person_count <= 0:
            return True
        required = int(np.ceil(self.initial_person_count * self.evacuation_target_rate))
        return self.escaped_persons >= required

    def _time_only_reward(self, done):
        newly_escaped = self.escaped_persons - self.prev_escaped_persons
        reward_value = newly_escaped * self.escape_reward - self.step_penalty
        self.prev_escaped_persons = self.escaped_persons

        if done:
            if self._evacuation_complete():
                reward_value += max(0, self.max_steps - self.current_step)
            else:
                reward_value -= self.max_steps

        return [reward_value for _ in self.robot_positions]

    def _initialize_persons(self):
        target_empty_positions = []
        for r in range(self.target_area[0], self.target_area[1]):
            for c in range(self.target_area[2], self.target_area[3]):
                if self.map_loader.map_data[r, c] == 0:
                    target_empty_positions.append((r, c))

        self.persons = []
        for _ in range(min(self.num_persons, len(target_empty_positions))):
            pos = random.choice(target_empty_positions)
            target_empty_positions.remove(pos)
            self.persons.append(Person(pos, self.map_loader))

    def _initialize_robots(self):
        """根据角色初始化机器人位置"""
        empty_positions = []
        for r in range(self.map_loader.rows):
            for c in range(self.map_loader.cols):
                if self.map_loader.map_data[r, c] == 0:
                    occupied = any(person.position == (r, c) for person in self.persons)
                    if not occupied:
                        empty_positions.append((r, c))

        self.robot_positions = []

        # 机器人1：火源防护机器人 - 初始化在火源附近
        if self.map_loader.fires and len(empty_positions) > 0:
            fire_pos = list(self.map_loader.fires)[0]  # 选择第一个火源
            robot1_pos = self._find_best_position_near_target(fire_pos, empty_positions, 3, 6)
            if robot1_pos:
                self.robot_positions.append(robot1_pos)
                empty_positions.remove(robot1_pos)

        # 机器人2：出口疏散机器人 - 初始化在出口附近
        if self.map_loader.exits and len(empty_positions) > 0:
            exit_pos = list(self.map_loader.exits)[0]  # 选择第一个出口
            robot2_pos = self._find_best_position_near_target(exit_pos, empty_positions, 5, 8)
            if robot2_pos:
                self.robot_positions.append(robot2_pos)
                empty_positions.remove(robot2_pos)

        # 如果没有找到合适位置，随机选择
        while len(self.robot_positions) < 2 and empty_positions:
            pos = random.choice(empty_positions)
            self.robot_positions.append(pos)
            empty_positions.remove(pos)

    def _find_best_position_near_target(self, target, available_positions, min_dist, max_dist):
        """在目标附近找到最佳位置"""
        tx, ty = target
        candidates = []

        # 寻找在理想距离范围内的位置
        for pos in available_positions:
            px, py = pos
            dist = max(abs(px - tx), abs(py - ty))
            if min_dist <= dist <= max_dist:
                candidates.append((pos, dist))

        if candidates:
            # 选择距离最接近理想值的位置（理想值为min_dist + 1）
            ideal_dist = min_dist + 1
            best_pos = min(candidates, key=lambda x: abs(x[1] - ideal_dist))[0]
            return best_pos

        # 如果没有理想位置，选择最近的可用位置
        if available_positions:
            nearest_pos = min(available_positions, 
                            key=lambda pos: max(abs(pos[0] - tx), abs(pos[1] - ty)))
            return nearest_pos

        return None

    def _compute_robot_repulsion_field(self):
        robot_field = np.zeros((self.map_loader.rows, self.map_loader.cols))
        if not self.robot_positions:
            return robot_field

        A = self.fire_model.A * self.robot_repulsion_factor
        B = self.robot_repulsion_decay
        cutoff_radius = self.robot_repulsion_cutoff_radius
        row_idx = np.arange(self.map_loader.rows)[:, None]
        col_idx = np.arange(self.map_loader.cols)[None, :]

        for rx, ry in self.robot_positions:
            dist = np.sqrt((row_idx - rx) ** 2 + (col_idx - ry) ** 2)
            active = (dist > 0) & (dist <= cutoff_radius)
            robot_field += np.where(active, A * np.exp(-B * dist), 0.0)
            robot_field[rx, ry] = np.inf

        return robot_field

    def _exit_inward_direction(self, exit_pos):
        er, ec = exit_pos
        candidates = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = er + dr, ec + dc
            if 0 <= nr < self.map_loader.rows and 0 <= nc < self.map_loader.cols:
                if self.map_loader.map_data[nr, nc] != 1 and (nr, nc) not in self.map_loader.exits:
                    candidates.append((dr, dc, self.map_loader.static_field[nr, nc]))
        if not candidates:
            return None
        dr, dc, _ = min(candidates, key=lambda item: item[2])
        return dr, dc

    def _compute_arch_breaking_field(self):
        arch_field = np.zeros((self.map_loader.rows, self.map_loader.cols))
        if not self.use_arch_breaking_field or not self.robot_positions:
            return arch_field

        for exit_pos in self.map_loader.exits:
            robot_near_exit = any(
                max(abs(exit_pos[0] - rx), abs(exit_pos[1] - ry)) <= self.arch_breaking_activation_radius
                for rx, ry in self.robot_positions
            )
            if not robot_near_exit:
                continue

            inward = self._exit_inward_direction(exit_pos)
            if inward is None:
                continue
            dr, dc = inward
            side_dirs = [(-dc, dr), (dc, -dr)]

            for depth in range(1, self.arch_corridor_depth + 1):
                rr = exit_pos[0] + dr * depth
                cc = exit_pos[1] + dc * depth
                if not self._is_valid_position((rr, cc)):
                    break

                strength = (self.arch_corridor_depth - depth + 1) / self.arch_corridor_depth
                arch_field[rr, cc] -= self.arch_lane_attraction * strength

                for sr, sc in side_dirs:
                    side_pos = (rr + sr, cc + sc)
                    if self._is_valid_position(side_pos):
                        arch_field[side_pos] += self.arch_side_repulsion * strength

        return arch_field

    def _nearest_exit(self, pos):
        if not self.map_loader.exits:
            return None, float('inf')
        nearest = min(
            self.map_loader.exits,
            key=lambda exit_pos: max(abs(pos[0] - exit_pos[0]), abs(pos[1] - exit_pos[1]))
        )
        distance = max(abs(pos[0] - nearest[0]), abs(pos[1] - nearest[1]))
        return nearest, distance

    def _robot_breaks_arch_at_exit(self, exit_pos):
        if not self.robot_positions:
            return False
        return any(
            max(abs(exit_pos[0] - rx), abs(exit_pos[1] - ry)) <= self.arch_breaking_activation_radius
            for rx, ry in self.robot_positions
        )

    def _local_crowd_count(self, center_pos):
        cr, cc = center_pos
        count = 0
        for person in self.persons:
            if person.escaped or person.is_dead:
                continue
            pr, pc = person.position
            if max(abs(cr - pr), abs(cc - pc)) <= self.arch_density_radius:
                count += 1
        return count

    def _arch_blocks_move(self, current_pos, next_pos):
        if next_pos == current_pos or not self.use_arch_breaking_field:
            return False

        nearest_exit, exit_distance = self._nearest_exit(next_pos)
        if nearest_exit is None or exit_distance > self.arch_zone_radius:
            return False

        crowd_count = self._local_crowd_count(next_pos)
        if crowd_count <= self.arch_density_threshold:
            return False

        # This friction probability is independent of robot proximity.
        # Robots reduce blockage only indirectly by changing local density.
        base_probability = self.arch_block_probability
        density_scale = min(1.0, (crowd_count - self.arch_density_threshold) / 8.0)
        return random.random() < base_probability * density_scale

    def _process_persons_at_exits(self):
        """
        处理正在出口进行疏散服务的人员
        每个时间步开始时调用，用于：
        1. 减少正在出口等待的人员的剩余服务时间
        2. 将服务完成的人员标记为已逃生并移除
        3. 统计本步完成疏散的人数（受出口容量限制）
        """
        if not self.use_exit_service_time:
            return  # 如果未启用出口服务机制，直接返回

        # 按出口分组统计正在服务的人员
        exit_groups = {}  # {exit_pos: [persons_list]}
        for person in self.persons:
            if person.in_exit_process and not person.escaped and not person.is_dead:
                exit_pos = tuple(person.position)
                if exit_pos not in exit_groups:
                    exit_groups[exit_pos] = []
                exit_groups[exit_pos].append(person)

        # 处理每个出口的疏散队列
        escaped_persons = []
        for exit_pos, persons_at_exit in exit_groups.items():
            completed_count = 0  # 本步该出口已完成的人数

            for person in persons_at_exit:
                # 减少剩余服务时间
                person.service_time_left -= 1

                # 检查是否服务完成且未超出本步容量
                if person.service_time_left <= 0 and completed_count < self.exit_capacity_per_step:
                    person.escaped = True
                    escaped_persons.append(person)
                    completed_count += 1
                    self.escaped_persons += 1

        # 从人员列表中移除已逃生的人员
        for person in escaped_persons:
            self.persons.remove(person)

    def _is_valid_position(self, pos):
        r, c = pos
        if r < 0 or r >= self.map_loader.rows or c < 0 or c >= self.map_loader.cols:
            return False
        if self.map_loader.map_data[r, c] == 1:  # 墙壁
            return False
        if (r, c) in self.map_loader.fires:  # 火源位置
            return False
        return True

    def _service_time_for_exit(self, _exit_pos):
        # Robots no longer reduce exit service time directly. They can only
        # affect evacuation by changing the local crowd field near the doorway.
        return self.base_service_time

    def step(self, robot_actions):
        if self.current_step >= self.max_steps:
            # 当达到最大步数时，也要提供完整的info结构
            alive_persons_list = self._alive_persons()
            dead_persons = self._dead_count()
            avg_health = self._avg_health(alive_persons_list)

            info = {
                "step": self.current_step,
                "persons_remaining": len(alive_persons_list),
                "persons_escaped": self.escaped_persons,
                "persons_dead": dead_persons,
                "evacuation_rate": self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0,
                "avg_health": avg_health,
                "reason": "max_steps_reached"
            }
            return self._get_state(), [0, 0], True, info

        # === 步骤1：处理出口服务中的人员（每步开始时先处理正在排队的人） ===
        # 这会让服务时间倒计时，并让服务完成的人逃生
        self._process_persons_at_exits()

        # === 步骤2：更新机器人位置 ===
        self._update_robot_positions(robot_actions)

        # === 步骤3：计算动态场（火源+机器人斥力+门口拱形破坏场） ===
        fire_field = self.fire_field
        robot_field = self._compute_robot_repulsion_field()
        arch_breaking_field = self._compute_arch_breaking_field()
        combined_dynamic_field = fire_field + robot_field + arch_breaking_field

        # === 步骤4：更新人员位置（新到达出口的人开始服务流程） ===
        self._update_person_positions(combined_dynamic_field)

        # === 步骤5：更新所有人员的健康值 ===
        self._update_persons_health()

        # === 步骤6：计算奖励 ===
        # reward = self._calculate_reward()

        # 修改结束条件：所有人员死亡或撤离，或达到最大步数
        alive_persons = self._alive_persons()
        done = (
            self._evacuation_complete()
            or len(alive_persons) == 0
            or (self.current_step + 1) >= self.max_steps
        )

        self.current_step += 1

        dead_persons = self._dead_count()
        alive_persons_list = self._alive_persons()
        alive_persons_count = len(alive_persons_list)

        reward = self._time_only_reward(done)

        # 计算当前时间步所有人员的平均健康值
        avg_health = self._avg_health(alive_persons_list)

        info = {
            "step": self.current_step,
            "persons_remaining": alive_persons_count,
            "persons_escaped": self.escaped_persons,
            "persons_dead": dead_persons,
            "evacuation_rate": self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0,
            "avg_health": avg_health
        }

        return self._get_state(), reward, done, info

    def _update_robot_positions(self, robot_actions):
        new_positions = []
        for i, action in enumerate(robot_actions):
            if i >= len(self.robot_positions):
                break

            current_pos = self.robot_positions[i]
            dr, dc = self.actions[action]
            new_pos = (current_pos[0] + dr, current_pos[1] + dc)

            if self._is_valid_position(new_pos) and new_pos not in new_positions:
                new_positions.append(new_pos)
            else:
                new_positions.append(current_pos)

        self.robot_positions = new_positions

    def _update_person_positions(self, dynamic_field):
        """
        更新所有人员的位置
        改进后的逻辑：
        1. 正在出口服务中的人员不参与移动，占用出口格
        2. 其他人员正常移动
        3. 到达出口的人员不立即逃生，而是开始出口服务流程（需等待T步）
        """
        next_occupancy = np.zeros((self.map_loader.rows, self.map_loader.cols), dtype=int)

        # 标记机器人占用的格子
        for rx, ry in self.robot_positions:
            next_occupancy[rx, ry] = 1

        # 标记正在出口服务中的人员占用的格子（他们暂时不移动，占据出口格）
        for person in self.persons:
            if person.in_exit_process and not person.escaped and not person.is_dead:
                px, py = person.position
                next_occupancy[px, py] = 1

        next_positions = {}

        # 只有未逃生、未死亡、且不在出口服务中的人员才参与移动决策
        for person in self.persons:
            if not person.escaped and not person.is_dead and not person.in_exit_process:
                next_pos = person.get_possible_moves(next_occupancy, dynamic_field)
                if self._arch_blocks_move(person.position, next_pos):
                    next_pos = person.position
                # 处理多人争夺同一格子的冲突
                if next_pos in next_positions:
                    if random.random() > 0.5:
                        next_positions[next_pos] = person
                else:
                    next_positions[next_pos] = person
                    next_occupancy[next_pos[0], next_pos[1]] = 1

        # 更新位置并处理到达出口的情况
        for person in self.persons:
            if person in next_positions.values():
                new_pos = [pos for pos, p in next_positions.items() if p == person][0]
                person.position = new_pos

                # 如果到达出口格，开始出口服务流程（而非立即逃生）
                if new_pos in self.map_loader.exits and not person.in_exit_process:
                    if self.use_exit_service_time:
                        # 启用出口服务机制：设置服务状态和等待时间
                        person.in_exit_process = True
                        person.exit_since_step = self.current_step
                        service_time = self._service_time_for_exit(new_pos)
                        person.service_time_needed = service_time
                        person.service_time_left = service_time
                    else:
                        # 未启用服务机制：保持原有的立即逃生逻辑
                        person.escaped = True
                        self.escaped_persons += 1

        # 如果未启用服务机制，移除已逃生的人员（保持向后兼容）
        if not self.use_exit_service_time:
            escaped = [p for p in self.persons if p.escaped]
            for p in escaped:
                self.persons.remove(p)

    def _update_persons_health(self):
        """更新所有人员的健康值"""
        if not self.use_health:
            return
        for person in self.persons:
            if not person.escaped and not person.is_dead:
                person.update_health(self.map_loader.fires)

    def _calculate_reward(self):
        """计算双机器人个体奖励"""
        alive_persons = [p for p in self.persons if not p.is_dead]

        # 共享奖励部分
        evacuation_rate = self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0
        time_efficiency = max(0, (self.max_steps - self.current_step) / self.max_steps)
        shared_evacuation_reward = evacuation_rate * 10 + time_efficiency * 2

        # 健康保护奖励（共享）
        if alive_persons:
            avg_health = sum(p.health for p in alive_persons) / len(alive_persons)
            shared_health_reward = (avg_health / 100) * 6
        else:
            shared_health_reward = 6

        # 协作效果奖励（共享）
        shared_cooperation_reward = self._calculate_cooperation_reward(alive_persons)

        # 完成奖励（共享）
        shared_completion_bonus = 0
        if len(alive_persons) == 0:
            if self.escaped_persons == self.initial_person_count:
                shared_completion_bonus = 20  # 完美撤离
            else:
                shared_completion_bonus = 8   # 至少结束了

        # 计算每个机器人的个体奖励
        robot1_reward, robot2_reward = self._calculate_individual_rewards(alive_persons)

        # 总奖励 = 共享奖励 + 个体奖励
        total_robot1_reward = (shared_evacuation_reward + shared_health_reward + 
                              shared_cooperation_reward + shared_completion_bonus + robot1_reward)
        total_robot2_reward = (shared_evacuation_reward + shared_health_reward + 
                              shared_cooperation_reward + shared_completion_bonus + robot2_reward)

        # 存储奖励分解供调试使用
        self.reward_breakdown = {
            "shared_evacuation": round(shared_evacuation_reward, 3),
            "shared_health": round(shared_health_reward, 3),
            "shared_cooperation": round(shared_cooperation_reward, 3),
            "shared_completion": round(shared_completion_bonus, 3),
            "robot1_individual": round(robot1_reward, 3),
            "robot2_individual": round(robot2_reward, 3),
            "robot1_total": round(total_robot1_reward, 3),
            "robot2_total": round(total_robot2_reward, 3)
        }

        return [total_robot1_reward, total_robot2_reward]

    def _calculate_individual_rewards(self, alive_persons):
        """计算每个机器人的个体奖励"""
        robot1_reward = 0
        robot2_reward = 0

        if len(self.robot_positions) >= 2:
            # 机器人1（火源防护）个体奖励
            robot1_reward = self._calculate_robot1_individual_reward(alive_persons)

            # 机器人2（出口疏散）个体奖励
            robot2_reward = self._calculate_robot2_individual_reward(alive_persons)

        return robot1_reward, robot2_reward

    def _calculate_robot1_individual_reward(self, alive_persons):
        """计算机器人1的个体奖励"""
        reward = 0
        robot1_pos = self.robot_positions[0]

        # 1. 位置奖励 - 是否在火源附近的合适位置
        if self.map_loader.fires:
            min_fire_dist = min([max(abs(robot1_pos[0] - fx), abs(robot1_pos[1] - fy)) 
                               for fx, fy in self.map_loader.fires])
            # 距离火源3-5格最佳，给予位置奖励
            if 3 <= min_fire_dist <= 5:
                reward += 3
            elif 2 <= min_fire_dist <= 6:
                reward += 1

        # 2. 火源防护效果奖励
        persons_near_fire = 0
        persons_driven_away = 0  # 被驱赶远离火源的人员

        for person in alive_persons:
            px, py = person.position
            # 统计火源3格内的危险人员
            for fx, fy in self.map_loader.fires:
                if max(abs(px - fx), abs(py - fy)) <= 3:
                    persons_near_fire += 1
                    break

            # 统计机器人3格内但远离火源的人员（防护成功）
            robot_dist = max(abs(px - robot1_pos[0]), abs(py - robot1_pos[1]))
            if robot_dist <= 3:
                fire_dist = min([max(abs(px - fx), abs(py - fy)) 
                               for fx, fy in self.map_loader.fires]) if self.map_loader.fires else float('inf')
                if fire_dist > 4:  # 人员在机器人附近但远离火源
                    persons_driven_away += 1

        # 火源附近人员越少越好
        total_persons = max(len(alive_persons), 1)
        fire_safety_ratio = 1 - (persons_near_fire / total_persons)
        reward += fire_safety_ratio * 4

        # 成功驱赶人员的奖励
        drive_away_ratio = persons_driven_away / total_persons
        reward += drive_away_ratio * 2

        return reward

    def _calculate_robot2_individual_reward(self, alive_persons):
        """计算机器人2的个体奖励"""
        reward = 0
        robot2_pos = self.robot_positions[1]

        # 1. 位置奖励 - 是否在出口附近的合适位置
        if self.map_loader.exits:
            min_exit_dist = min([max(abs(robot2_pos[0] - ex), abs(robot2_pos[1] - ey)) 
                               for ex, ey in self.map_loader.exits])
            # 距离出口2-4格最佳，给予位置奖励
            if 2 <= min_exit_dist <= 4:
                reward += 3
            elif 1 <= min_exit_dist <= 5:
                reward += 1

        # 2. 疏散引导效果奖励
        persons_near_exit = 0
        persons_guided = 0  # 被引导到出口的人员

        for person in alive_persons:
            px, py = person.position
            # 统计出口5格内的人员
            for ex, ey in self.map_loader.exits:
                if max(abs(px - ex), abs(py - ey)) <= 5:
                    persons_near_exit += 1
                    break

            # 统计机器人3格内且接近出口的人员（引导成功）
            robot_dist = max(abs(px - robot2_pos[0]), abs(py - robot2_pos[1]))
            if robot_dist <= 3:
                exit_dist = min([max(abs(px - ex), abs(py - ey)) 
                               for ex, ey in self.map_loader.exits]) if self.map_loader.exits else float('inf')
                if exit_dist <= 6:  # 人员在机器人附近且接近出口
                    persons_guided += 1

        # 出口附近人员越多越好（说明疏散进行中）
        total_persons = max(len(alive_persons), 1)
        exit_gathering_ratio = persons_near_exit / total_persons
        reward += exit_gathering_ratio * 4

        # 成功引导人员的奖励
        guide_ratio = persons_guided / total_persons
        reward += guide_ratio * 2

        # 疏散进度奖励
        evacuation_progress = self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0
        reward += evacuation_progress * 2

        return reward

    def _calculate_position_reward(self):
        """计算机器人位置奖励"""
        if len(self.robot_positions) < 2:
            return 0

        reward = 0

        # 机器人1（火源防护）应该靠近火源
        robot1_pos = self.robot_positions[0]
        if self.map_loader.fires:
            min_fire_dist = min([max(abs(robot1_pos[0] - fx), abs(robot1_pos[1] - fy)) 
                               for fx, fy in self.map_loader.fires])
            # 距离火源2-4格最佳
            if 2 <= min_fire_dist <= 4:
                reward += 3
            elif min_fire_dist <= 6:
                reward += 1

        # 机器人2（出口疏散）应该靠近出口
        robot2_pos = self.robot_positions[1]  
        if self.map_loader.exits:
            min_exit_dist = min([max(abs(robot2_pos[0] - ex), abs(robot2_pos[1] - ey)) 
                               for ex, ey in self.map_loader.exits])
            # 距离出口1-3格最佳
            if 1 <= min_exit_dist <= 3:
                reward += 3
            elif min_exit_dist <= 5:
                reward += 1

        return reward

    def _calculate_cooperation_reward(self, alive_persons):
        """计算协作效果奖励"""
        if not alive_persons or len(self.robot_positions) < 2:
            return 0

        reward = 0

        # 统计火源附近和出口附近的人员
        persons_near_fire = 0
        persons_near_exit = 0

        for person in alive_persons:
            px, py = person.position

            # 检查是否靠近火源
            for fx, fy in self.map_loader.fires:
                if max(abs(px - fx), abs(py - fy)) <= 3:
                    persons_near_fire += 1
                    break

            # 检查是否靠近出口
            for ex, ey in self.map_loader.exits:
                if max(abs(px - ex), abs(py - ey)) <= 5:
                    persons_near_exit += 1
                    break

        # 火源附近人员越少越好
        if persons_near_fire == 0:
            reward += 4
        elif persons_near_fire <= 2:
            reward += 2

        # 出口附近人员较多说明疏散进行中
        if persons_near_exit >= len(alive_persons) * 0.3:  # 30%以上的人在出口附近
            reward += 3
        elif persons_near_exit >= len(alive_persons) * 0.1:  # 10%以上
            reward += 1

        return reward

    def _get_state(self):
        """为两个机器人提供10维一维向量状态"""
        alive_persons = self._alive_persons()

        # 机器人1状态：火源防护机器人 [10维]
        robot1_state = self._get_robot1_state_vector(alive_persons)

        # 机器人2状态：出口疏散机器人 [10维]
        robot2_state = self._get_robot2_state_vector(alive_persons)
        global_info = {
            "step": self.current_step,
            "total_persons": len(alive_persons),
            "avg_health": sum(p.health for p in alive_persons) / len(alive_persons) if alive_persons else 100
        }

        return {
            "robot1_state": robot1_state,
            "robot2_state": robot2_state,

        }

    def _get_robot_state(self, robot_id, alive_persons, fire_positions, exit_positions, role):
        """获取特定机器人的状态"""
        if robot_id >= len(self.robot_positions):
            return {}

        robot_pos = self.robot_positions[robot_id]

        # 基础状态
        state = {
            "position": robot_pos,
            "role": role
        }

        # 周围人员信息（5x5范围内）
        nearby_persons = []
        for person in alive_persons:
            px, py = person.position
            rx, ry = robot_pos
            if abs(px - rx) <= 2 and abs(py - ry) <= 2:
                nearby_persons.append({
                    "position": person.position,
                    "health": person.health,
                    "distance_to_robot": max(abs(px - rx), abs(py - ry))
                })

        state["nearby_persons"] = nearby_persons

        if role == "fire_guard":
            # 火源防护机器人关注：
            # 1. 到最近火源的距离
            # 2. 火源周围的人员数量
            # 3. 危险区域的人员健康状况
            min_fire_dist = min([max(abs(robot_pos[0] - fx), abs(robot_pos[1] - fy)) 
                               for fx, fy in fire_positions]) if fire_positions else float('inf')

            # 统计火源3格内的人员
            persons_near_fire = 0
            for person in alive_persons:
                px, py = person.position
                for fx, fy in fire_positions:
                    if max(abs(px - fx), abs(py - fy)) <= 3:
                        persons_near_fire += 1
                        break

            state.update({
                "distance_to_nearest_fire": min_fire_dist,
                "persons_near_fire": persons_near_fire,
                "fire_positions": fire_positions
            })

        elif role == "exit_guide":
            # 出口疏散机器人关注：
            # 1. 到最近出口的距离
            # 2. 出口周围的人员数量
            # 3. 疏散效率
            min_exit_dist = min([max(abs(robot_pos[0] - ex), abs(robot_pos[1] - ey)) 
                               for ex, ey in exit_positions]) if exit_positions else float('inf')

            # 统计出口5格内的人员
            persons_near_exit = 0
            for person in alive_persons:
                px, py = person.position
                for ex, ey in exit_positions:
                    if max(abs(px - ex), abs(py - ey)) <= 5:
                        persons_near_exit += 1
                        break

            state.update({
                "distance_to_nearest_exit": min_exit_dist,
                "persons_near_exit": persons_near_exit,
                "exit_positions": exit_positions
            })

        return state

    def _get_robot1_state_vector(self, alive_persons):
        """获取机器人1（火源防护）的10维状态向量"""
        if len(self.robot_positions) < 1:
            return [0.0] * 10

        robot_pos = self.robot_positions[0]
        rx, ry = robot_pos

        # 归一化参数
        max_distance = max(self.map_loader.rows, self.map_loader.cols)
        total_persons = max(len(alive_persons), 1)  # 避免除零

        state_vector = []

        # 1-2. 基础位置信息 (归一化坐标)
        state_vector.extend([
            rx / self.map_loader.rows,      # 归一化x坐标
            ry / self.map_loader.cols       # 归一化y坐标
        ])

        # 3-5. 火源相关信息
        if self.map_loader.fires:
            # 找到最近火源
            min_fire_dist = float('inf')
            nearest_fire = None
            for fx, fy in self.map_loader.fires:
                dist = max(abs(rx - fx), abs(ry - fy))
                if dist < min_fire_dist:
                    min_fire_dist = dist
                    nearest_fire = (fx, fy)

            # 到最近火源的距离和方向
            fire_distance = min_fire_dist / max_distance
            if nearest_fire and min_fire_dist > 0:
                fx, fy = nearest_fire
                fire_dir_x = (fx - rx) / min_fire_dist
                fire_dir_y = (fy - ry) / min_fire_dist
            else:
                fire_dir_x, fire_dir_y = 0.0, 0.0
        else:
            fire_distance, fire_dir_x, fire_dir_y = 1.0, 0.0, 0.0

        state_vector.extend([fire_distance, fire_dir_x, fire_dir_y])

        # 6-9. 人员相关信息
        nearby_persons_count = 0
        nearby_healths = []
        persons_near_fire = 0

        for person in alive_persons:
            px, py = person.position

            # 统计周围3格内人员
            if max(abs(px - rx), abs(py - ry)) <= 3:
                nearby_persons_count += 1
                nearby_healths.append(person.health)

            # 统计火源3格内人员
            for fx, fy in self.map_loader.fires:
                if max(abs(px - fx), abs(py - fy)) <= 3:
                    persons_near_fire += 1
                    break

        # 人员统计信息
        nearby_ratio = nearby_persons_count / total_persons
        fire_danger_ratio = persons_near_fire / total_persons
        avg_health_nearby = (sum(nearby_healths) / len(nearby_healths) / 100) if nearby_healths else 1.0
        min_health_nearby = (min(nearby_healths) / 100) if nearby_healths else 1.0

        state_vector.extend([nearby_ratio, fire_danger_ratio, avg_health_nearby, min_health_nearby])

        # 10. 时间进度
        time_progress = self.current_step / self.max_steps
        state_vector.append(time_progress)

        # 格式化到小数点后三位
        state_vector = [round(float(x), 3) for x in state_vector]

        return state_vector

    def _get_robot2_state_vector(self, alive_persons):
        """获取机器人2（出口疏散）的10维状态向量"""
        if len(self.robot_positions) < 2:
            return [0.0] * 10

        robot_pos = self.robot_positions[1]
        rx, ry = robot_pos

        # 归一化参数
        max_distance = max(self.map_loader.rows, self.map_loader.cols)
        total_persons = max(len(alive_persons), 1)  # 避免除零

        state_vector = []

        # 1-2. 基础位置信息 (归一化坐标)
        state_vector.extend([
            rx / self.map_loader.rows,      # 归一化x坐标
            ry / self.map_loader.cols       # 归一化y坐标
        ])

        # 3-5. 出口相关信息
        if self.map_loader.exits:
            # 找到最近出口
            min_exit_dist = float('inf')
            nearest_exit = None
            for ex, ey in self.map_loader.exits:
                dist = max(abs(rx - ex), abs(ry - ey))
                if dist < min_exit_dist:
                    min_exit_dist = dist
                    nearest_exit = (ex, ey)

            # 到最近出口的距离和方向
            exit_distance = min_exit_dist / max_distance
            if nearest_exit and min_exit_dist > 0:
                ex, ey = nearest_exit
                exit_dir_x = (ex - rx) / min_exit_dist
                exit_dir_y = (ey - ry) / min_exit_dist
            else:
                exit_dir_x, exit_dir_y = 0.0, 0.0
        else:
            exit_distance, exit_dir_x, exit_dir_y = 1.0, 0.0, 0.0

        state_vector.extend([exit_distance, exit_dir_x, exit_dir_y])

        # 6-9. 人员相关信息
        nearby_persons_count = 0
        nearby_healths = []
        persons_near_exit = 0

        for person in alive_persons:
            px, py = person.position

            # 统计周围3格内人员
            if max(abs(px - rx), abs(py - ry)) <= 3:
                nearby_persons_count += 1
                nearby_healths.append(person.health)

            # 统计出口5格内人员
            for ex, ey in self.map_loader.exits:
                if max(abs(px - ex), abs(py - ey)) <= 5:
                    persons_near_exit += 1
                    break

        # 人员统计信息
        nearby_ratio = nearby_persons_count / total_persons
        exit_gathering_ratio = persons_near_exit / total_persons
        avg_health_nearby = (sum(nearby_healths) / len(nearby_healths) / 100) if nearby_healths else 1.0
        evacuation_rate = self.escaped_persons / self.initial_person_count if self.initial_person_count > 0 else 1.0

        state_vector.extend([nearby_ratio, exit_gathering_ratio, avg_health_nearby, evacuation_rate])

        # 10. 时间进度
        time_progress = self.current_step / self.max_steps
        state_vector.append(time_progress)

        # 格式化到小数点后三位
        state_vector = [round(float(x), 3) for x in state_vector]

        return state_vector

    def reset(self):
        self._reset_environment()
        return self._get_state()

    def set_robot_repulsion_factor(self, factor):
        self.robot_repulsion_factor = factor

    def render(self, mode='human'):
        """渲染环境，进行可视化显示"""
        vis_map = np.copy(self.map_loader.map_data)

        # 添加人员标记（区分活着和死亡）
        for person in self.persons:
            x, y = person.position
            if person.is_dead:
                vis_map[x, y] = 6  # 死亡人员标记
            else:
                vis_map[x, y] = 4  # 活着的人员标记

        # 添加火源标记
        for fx, fy in self.map_loader.fires:
            vis_map[fx, fy] = 3  # Fire

        # 添加机器人标记
        for rx, ry in self.robot_positions:
            vis_map[rx, ry] = 5  # Robot marker

        if mode == 'human':
            # 执行可视化显示
            plt.imshow(vis_map, cmap='hot', interpolation='nearest')
            plt.title(f'Step {self.current_step}')
            plt.show(block=False)
            plt.pause(0.05)
            plt.clf()
        elif mode == 'rgb_array':
            # 返回vis_map数组（用于其他用途）
            return vis_map

        return vis_map
