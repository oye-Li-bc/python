# 07 Isaac Lab 风格

> Isaac Lab 把 gymnasium 接口拓展成"大规模并行 + GPU + 模块化配置"。
> 这一篇带你**看懂官方任务模板**，套用着写自己的任务。

---

## 1. Isaac Lab 与 gymnasium 的关系

Isaac Lab 在 gymnasium 的基础上加了三个"超能力"：

1. **大规模并行**：一次开 4096 个机器人在同一张 GPU 上
2. **物理仿真**：基于 NVIDIA PhysX 的接触、关节、刚体动力学
3. **模块化配置**：把"任务"拆成 Scene/Observation/Reward/Termination... 各模块

→ 你看到的 `ManagerBasedRLEnv` 就是这套抽象的核心。

---

## 2. Manager-Based 架构（核心思想）

老式 RL 环境：所有逻辑塞在一个类里，又长又乱。
Isaac Lab 的方案：**把任务拆成多个 manager**，每个负责一件事：

```
ManagerBasedRLEnv
├── SceneManager           ── 加载机器人、地面、障碍物
├── ObservationManager     ── 算观测（关节角、IMU…）
├── ActionManager          ── 把策略输出解析成关节命令
├── RewardManager          ── 算每一步奖励（多项组合）
├── TerminationManager     ── 判断 done（摔倒、超时）
├── EventManager           ── 域随机化（重力扰动、摩擦…）
├── CurriculumManager      ── 课程学习（难度递增）
└── CommandManager         ── 生成目标命令（目标速度…）
```

> 💡 **类比**：传统 RL 是手写"全栈"，Isaac Lab 像 React/Vue 那样组件化。

---

## 3. `@configclass`：所有配置的基础

Isaac Lab 几乎每个东西都有一个对应的"Cfg"类，用 `@configclass` 装饰。

```python
from isaaclab.utils import configclass

@configclass
class MyTaskCfg:
    num_envs: int = 4096
    episode_length_s: float = 20.0
    decimation: int = 4

cfg = MyTaskCfg()
cfg.num_envs = 1024              # 可以改字段
```

行为几乎等于上一篇讲的 `@dataclass`，但加了 Isaac Lab 自己的特性（嵌套配置、可序列化等）。

---

## 4. 一个 Cfg 文件的典型骨架

以你将要写的"轮腿机器人速度跟踪"为例：

```python
# wheel_legged_env_cfg.py
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import (
    ObservationGroupCfg, ObservationTermCfg,
    RewardTermCfg, TerminationTermCfg, EventTermCfg
)
import isaaclab.envs.mdp as mdp


@configclass
class SceneCfg(InteractiveSceneCfg):
    """场景：地面、机器人、灯光。"""
    num_envs: int = 4096
    env_spacing: float = 2.5
    # robot = ArticulationCfg(...)   ← 这里放你的 URDF/USD 配置
    # ground = TerrainImporterCfg(...)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        actions = ObservationTermCfg(func=mdp.last_action)
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    track_lin_vel = RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel = RewardTermCfg(func=mdp.track_ang_vel_z_exp,  weight=0.5)
    ang_vel_xy   = RewardTermCfg(func=mdp.ang_vel_xy_l2,         weight=-0.05)
    torques      = RewardTermCfg(func=mdp.joint_torques_l2,       weight=-2e-5)
    alive        = RewardTermCfg(func=mdp.is_alive,               weight=0.1)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_fell = TerminationTermCfg(func=mdp.bad_orientation)


@configclass
class WheelLeggedEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    episode_length_s: float = 20.0
    decimation: int = 4
```

**关键观察：**
- 每个模块都是一个**嵌套的 configclass**
- 每个奖励项 = 一个**函数 + 权重**
- 框架自动加权求和、自动构建 buffer

---

## 5. 自定义函数（你最常写的部分）

奖励函数、观测函数、终止函数都是**普通 Python 函数**，签名要按规定来。

### 5.1 奖励函数

```python
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation

def track_lin_vel_xy_exp(env: ManagerBasedRLEnv, std: float = 0.5,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """跟踪 xy 平面线速度命令（高斯型）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 命令速度（来自 CommandManager）
    cmd_vel = env.command_manager.get_command("base_velocity")[:, :2]    # (N, 2)
    
    # 当前速度（base 坐标系下）
    curr_vel = asset.data.root_lin_vel_b[:, :2]                            # (N, 2)
    
    # 二范数误差
    err = torch.sum((cmd_vel - curr_vel) ** 2, dim=1)                       # (N,)
    
    # 高斯奖励：误差越小奖励越接近 1
    return torch.exp(-err / std ** 2)
```

**记忆要点：**
- 第一个参数永远是 `env`
- 返回 shape `(num_envs,)` 的 tensor
- 数据来自 `env.scene[...]` 和 `env.command_manager`
- **全 GPU 操作**，用 torch 不用 numpy

### 5.2 观测函数

```python
def joint_pos_rel(env, asset_cfg=SceneEntityCfg("robot")):
    """关节位置相对默认值的偏移。"""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos
```

### 5.3 终止函数

```python
def bad_orientation(env, limit_angle: float = 1.0,
                    asset_cfg=SceneEntityCfg("robot")):
    """机器人倾斜超过阈值就 done。"""
    asset = env.scene[asset_cfg.name]
    # projected gravity 的 z 分量越接近 -1 说明越直立
    return torch.acos(-asset.data.projected_gravity_b[:, 2].clip(-1, 1)) > limit_angle
```

---

## 6. 你的轮腿机器人骨架（思路）

参考"Velocity-Anymal-C"任务，做一个"Velocity-WheelLegged-v0"：

```python
# 步骤 1：注册机器人资产
WHEEL_LEGGED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="my_robot.usd"),   # 或 URDF 转换的 USD
    init_state=ArticulationCfg.InitialStateCfg(...),
    actuators={
        "hip": ImplicitActuatorCfg(joint_names_expr=[".*hip"], ...),
        "knee": ImplicitActuatorCfg(joint_names_expr=[".*knee"], ...),
        "wheel": ImplicitActuatorCfg(joint_names_expr=[".*wheel"], ...),
    }
)

# 步骤 2：场景 + 命令 + 观测 + 奖励
@configclass
class WheelLeggedSceneCfg(InteractiveSceneCfg):
    robot = WHEEL_LEGGED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # ground、light...

@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        )
    )

# 步骤 3：每个奖励 + 权重你自己定
@configclass
class WheelLeggedRewardsCfg:
    track_lin_vel = RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel = RewardTermCfg(func=mdp.track_ang_vel_z_exp,  weight=0.5)
    base_height   = RewardTermCfg(func=my_height_reward,         weight=-0.2)
    # ... 多写几项是常态
```

> 详细的训练脚本和 cfg 拼装见**官方教程 `Isaac Lab Tasks`**，本篇主要让你**看懂结构**。

---

## 7. 训练：rsl_rl + PPO

Isaac Lab 自带 rsl_rl 训练入口：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 4096 \
    --headless
```

参数解释：
- `--task`：任务名（你的也要在 `__init__.py` 里注册）
- `--num_envs`：并行环境数
- `--headless`：不渲染（训练时关掉省时间）

训练完会输出策略检查点 `model_xxxx.pt`。

**Play（看效果）：**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 16 \
    --checkpoint /path/to/model.pt
```

---

## 8. 阅读官方代码的路径

强烈建议你**对照阅读**：

1. 找到 `isaaclab_tasks/manager_based/locomotion/velocity/`
2. 看 `anymal_c/anymal_c_env_cfg.py`（一个完整四足任务）
3. 看 `velocity_env_cfg.py`（基类，定义通用结构）
4. 看 `mdp/rewards.py`（所有奖励函数实现）

> 💡 **第一次看一定看不懂全部**。带着"奖励项怎么写"、"观测怎么定"这类**目标**读，能找到答案就够。

---

## 9. Isaac Lab 风格 vs 普通 gym Env

| 维度 | 普通 gym | Isaac Lab |
|------|---------|-----------|
| 并行 | 1 个或几个 | 几千个，GPU 上 |
| 数据 | numpy | torch.Tensor on GPU |
| 物理 | 简单或 MuJoCo | NVIDIA PhysX |
| 结构 | 一个大类 | 多个 manager + cfg |
| 奖励 | 一个函数算总 reward | 多项加权（每项一个函数） |
| 终止 | 单个 done 条件 | TerminationManager 管理多个 |
| 域随机化 | 自己写 | EventManager 配好 |

---

## 10. 写代码常用 import 速查

```python
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg, ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
    CurriculumTermCfg,
)
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
```

> 💡 这一堆 import 平时**复制**就行，不用背。

---

## 11. 三个常见误区

### 误区 1：把所有 reward 都加到 +1 范围
不一定。Isaac Lab 例子里很多负奖励权重（如 `torques=-2e-5`），是惩罚项。
**先看官方任务里同类奖励的数量级再设权重**。

### 误区 2：开太多环境
4096 是常用值，但 GPU 显存有限。**先用 64 或 256 跑通流程**，再加大。

### 误区 3：奖励写完不验证
训练前先写个测试：手动 `reset` + 一两步 `step`，**打印每个奖励项**看是不是合理范围。

```python
# debug 一个奖励
from isaaclab.envs.mdp import track_lin_vel_xy_exp
r = track_lin_vel_xy_exp(env, std=0.5)
print(r.shape, r.min(), r.max(), r.mean())
```

---

## 12. 你的实际学习路径

按这个顺序做：

1. ✅ 跑通 Isaac Lab 自带的 `Anymal-C` 速度跟踪示例
2. ✅ 阅读 `anymal_c_env_cfg.py`，把每一行注释清楚
3. ⏳ 找一个**最接近**轮腿的任务（Velocity 类）作为模板
4. ⏳ 把 URDF 转 USD，定义你的 `ArticulationCfg`
5. ⏳ 改奖励项（先抄 anymal_c 的，跑通再调）
6. ⏳ 训练 → play → 调参 → 重复

---

## 动手任务

打开你 Isaac Lab 安装目录下的任意一个 `*_env_cfg.py`，回答：
1. 它继承自什么基类？
2. 它有几个 manager 配置？分别是？
3. 它的奖励函数有哪几项？最大正奖励权重是多少？最大负奖励权重是多少？

把答案写下来对照自己未来的任务设计。

---

## 下一篇

→ [08 实战例子](08_实战例子.md)
