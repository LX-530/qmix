# 科学依据：只通过机器人排斥力减少门口摩擦

## 建模逻辑

当前模型不再让机器人直接把出口服务时间从 2 步降到 1 步，也不再让机器人直接降低门口阻滞概率。出口容量、出口服务时间、门口高密度冲突摩擦都是独立环境约束；机器人唯一可学习作用是通过自身占位和排斥力改变局部人群密度、来流方向和排队结构，从而削弱门口拱形堵塞。

## 可引用依据

1. Helbing 与 Molnár 的社会力模型把行人与障碍、其他行人之间的避让表示为排斥相互作用，可作为机器人排斥势场的基础：D. Helbing and P. Molnár, Social force model for pedestrian dynamics, Physical Review E, 1995. https://doi.org/10.1103/PhysRevE.51.4282

2. Helbing、Farkas 与 Vicsek 的逃生恐慌模型讨论了出口处拱形结构、堵塞和 faster-is-slower 等现象，支持“出口附近高密度冲突会增加疏散时间”的假设：D. Helbing, I. Farkas, and T. Vicsek, Simulating dynamical features of escape panic, Nature, 2000. https://doi.org/10.1038/35035023

3. Kirchner、Nishinari 与 Schadschneider 在元胞自动机行人模型中引入 friction/conflict 参数，用来描述多人竞争同一格点导致的阻滞和堵塞，直接支持本项目中的门口高密度阻滞概率：A. Kirchner, K. Nishinari, and A. Schadschneider, Friction effects and clogging in a cellular automaton model for pedestrian dynamics, Physical Review E, 2003. https://doi.org/10.1103/PhysRevE.67.056122

4. Yanagisawa 等研究了出口前障碍物对瓶颈流出的影响，指出障碍物设置会改变出口处冲突与转向结构。本文中机器人不提高出口处理能力，而是作为可学习的移动排斥源和占位体，对门口拱形结构进行扰动：D. Yanagisawa et al., Introduction of frictional and turning function for pedestrian outflow with an obstacle, Physical Review E, 2009. https://doi.org/10.1103/PhysRevE.80.036110

## 对本文实验的含义

- 机器人排斥力代表人群对机器人占位的避让和路径重排。
- 门口摩擦代表多人竞争出口前沿时产生的冲突、互相阻挡和拱形堵塞。
- 若 QMIX 学到把机器人放在出口上游合适位置，疏散时间减少应解释为队列组织和拱形结构破坏，而不是出口服务能力被人为提高。
