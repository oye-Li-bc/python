# 06 Gymnasium 接口

> Gymnasium = 强化学习环境的"通用 USB 接口"。
> Isaac Lab 的所有任务环境都遵循这个接口。

---

## 1. 为什么要有 Gymnasium

设想一下没有标准的世界：
- A 团队的迷宫环境用 `env.run(action)`
- B 团队的小车环境用 `env.update(cmd)`
- C 团队的飞行器用 `env.simulate(input)`

每换一个环境都要重写训练代码——灾难。

**Gymnasium 定义了统一接口**：所有环境都长得一样，算法可以无缝切换。

> 旧名 `gym`，OpenAI 维护；分叉后改名 `gymnasium`，社区维护。**新代码用 `gymnasium`**。

---

## 2. 一个 Env 的核心 4 个方法

```python
import gymnasium as gym

env = gym.make("CartPole-v1")          # 创建一个倒立摆环境

# 1. reset：重置环境，返回初始观测
obs, info = env.reset()

# 2. step：执行动作，返回 (新观测, 奖励, done, truncated, info)
action = env.action_space.sample()      # 随便采样一个动作
obs, reward, terminated, truncated, info = env.step(action)

# 3. render：渲染画面（可选）
env.render()

# 4. close：关闭
env.close()
```

**一个完整的 episode：**
```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

total_reward = 0
for step in range(500):
    action = env.action_space.sample()           # 随机策略
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"episode 结束于 step {step}，总奖励 {total_reward}")
        break

env.close()
```

---

## 3. observation_space 和 action_space

环境会**告诉你**：观测是什么样、动作是什么样。

```python
env = gym.make("CartPole-v1")

print(env.observation_space)
# Box(-inf, inf, (4,), float32) —— 4 维连续观测

print(env.action_space)
# Discrete(2) —— 2 个离散动作（左/右）
```

**主要类型：**

| 类型 | 含义 | 例子 |
|------|------|------|
| `Box(low, high, shape)` | 连续盒子 | 机器人关节角、速度 |
| `Discrete(n)` | 离散 n 选 1 | 离散动作（走 4 个方向） |
| `Dict(...)` | 复合 | 同时给 image + state |
| `Tuple(...)` | 多个空间组合 | 不太用 |

**轮腿机器人**：连续动作 → `Box`。

---

## 4. 五个返回值详解

```python
obs, reward, terminated, truncated, info = env.step(action)
```

| 名字 | 类型 | 含义 |
|------|------|------|
| `obs` | array | 新一帧的观测 |
| `reward` | float | 这一步的即时奖励 |
| `terminated` | bool | **任务自然结束**（成功/失败，比如车摔倒了） |
| `truncated` | bool | **时间到了被迫结束**（达到 max_steps） |
| `info` | dict | 调试用的额外信息（不参与训练） |

> 💡 **`terminated` vs `truncated`** 是 gymnasium 比老 gym 的一个进步。RL 里两者要区分对待（GAE / bootstrapping）。

---

## 5. 自己写一个最小 Env

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GoToOriginEnv(gym.Env):
    """目标：智能体从随机位置移动到原点 (0,0)。"""

    def __init__(self):
        super().__init__()
        # 观测：2D 位置
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )
        # 动作：2D 速度
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.max_steps = 200
        self.dt = 0.1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(-5, 5, size=2).astype(np.float32)
        self.steps = 0
        return self.pos.copy(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.pos += action * self.dt
        self.steps += 1

        dist = np.linalg.norm(self.pos)
        reward = -dist                          # 越近奖励越高（越小）

        terminated = dist < 0.1                  # 到达原点
        truncated = self.steps >= self.max_steps

        return self.pos.copy(), float(reward), terminated, truncated, {"dist": dist}


# 测试
env = GoToOriginEnv()
obs, _ = env.reset()
for _ in range(50):
    action = -obs / max(np.linalg.norm(obs), 1.0)   # 朝原点的方向
    obs, r, term, trunc, info = env.step(action)
    if term or trunc:
        print(f"结束: pos={obs}, dist={info['dist']:.3f}")
        break
```

> 💡 **这就是 RL 环境的全部！** Isaac Lab 把这个套子拓展成了"几千个并行环境 + GPU 加速 + 复杂物理"，但**接口本质就这 4 个方法**。

---

## 6. 矢量化环境（Vector Env）

训练 RL 时一次跑很多个环境（并行）提高样本效率。

```python
import gymnasium as gym

envs = gym.make_vec("CartPole-v1", num_envs=8)

obs, info = envs.reset()
print(obs.shape)        # (8, 4) —— 8 个环境，每个 4 维观测

actions = envs.action_space.sample()    # (8,) 8 个动作
obs, rewards, term, trunc, info = envs.step(actions)
print(rewards.shape)    # (8,)
```

**Isaac Lab 的做法**：直接在 GPU 上跑几千个环境（不是 8 个），shape 是 `(num_envs, obs_dim)`。

---

## 7. 经典任务一览（练手用）

| 任务 ID | 观测 | 动作 | 难度 |
|---------|------|------|------|
| `CartPole-v1` | 4 维连续 | 2 离散 | ⭐ |
| `MountainCar-v0` | 2 维连续 | 3 离散 | ⭐⭐ |
| `Pendulum-v1` | 3 维连续 | 1 维连续 | ⭐⭐ |
| `LunarLander-v2` | 8 维连续 | 4 离散 | ⭐⭐⭐ |
| `BipedalWalker-v3` | 24 维 | 4 连续 | ⭐⭐⭐⭐ |
| `Humanoid-v4` | 376 维 | 17 连续 | ⭐⭐⭐⭐⭐ |

```python
import gymnasium as gym
env = gym.make("Pendulum-v1", render_mode="human")
```

**强烈建议**：花 1 小时跑通 CartPole + 一个随机策略 + 渲染。看到那根杆子在你眼前晃，你对 RL 的理解会瞬间加深。

---

## 8. RL 训练流程（直观图）

```
[Agent]            [Environment]
   ↓ action               ↑
   ────────────────────→ step(a)
                          │
                          │ (obs, reward, done)
   ←──────────────────────
   ↓
[训练算法 PPO]   ← 收集很多 (s, a, r, s') 后更新策略
```

**Isaac Lab + rsl_rl 帮你做了什么？**
- Environment 部分（仿真、奖励、观测）→ Isaac Lab
- 训练算法（PPO）+ 收集数据 → rsl_rl
- 你的工作：**写好 Env 和奖励**

---

## 9. 写好奖励函数：3 个原则

1. **稠密优于稀疏**：每一步都给点正反馈，不要只在成功时给奖励
2. **不要互相矛盾**：跟踪速度的奖励 vs 节能奖励权重要平衡
3. **避免奖励 hacking**：智能体会用各种"作弊"方法刷奖励（比如旋转着走来增加移动距离）

**轮腿机器人常见奖励项**（预览，下一篇详讲 Isaac Lab）：
```python
rewards = (
    + 1.0 * tracking_linear_velocity     # 跟踪命令速度
    + 0.5 * tracking_angular_velocity
    - 0.05 * ang_vel_xy_l2               # 别乱晃
    - 2e-5 * torque_l2                   # 省电
    + 0.1 * alive                        # 活着就给奖励
    - 1.0 * fall                         # 摔倒严惩
)
```

---

## 10. 你应该跑一遍的代码

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()

for ep in range(5):
    obs, _ = env.reset()
    total = 0
    while True:
        action = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc:
            print(f"episode {ep}: 总奖励 {total}")
            break

env.close()
```

如果你没装 gymnasium：
```bash
pip install gymnasium[classic-control]
```

---

## 11. 与 Isaac Lab 的桥接

Isaac Lab 的环境（如 `Isaac-Velocity-Anymal-C-v0`）**符合 gymnasium 接口**：
```python
import gymnasium as gym
import isaaclab_tasks                 # 注册所有任务

env = gym.make("Isaac-Velocity-Rough-Anymal-C-v0", num_envs=4096)
obs, _ = env.reset()
# obs 是一个 dict，包含 'policy' 等键
```

不同点：
1. 一次开几千个 env（不是 1 个）
2. 数据全在 GPU 上（torch.Tensor，不是 numpy）
3. 复杂物理（接触、关节、惯性）

下一篇详细讲。

---

## 动手任务

把上面 `GoToOriginEnv` 改成 "GoToTarget"：
- 在 reset 时随机生成一个目标点 `target`
- 观测改为 `(智能体位置, 相对目标的向量)` 共 4 维
- reward = 负的与目标距离

参考答案（关键改动）：
```python
class GoToTargetEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-10, 10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.max_steps = 200

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(-5, 5, 2).astype(np.float32)
        self.target = self.np_random.uniform(-5, 5, 2).astype(np.float32)
        self.steps = 0
        return self._obs(), {}

    def _obs(self):
        rel = self.target - self.pos
        return np.concatenate([self.pos, rel]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        self.pos += action * 0.1
        self.steps += 1
        dist = np.linalg.norm(self.target - self.pos)
        return self._obs(), -float(dist), dist < 0.1, self.steps >= 200, {}
```

---

## 下一篇

→ [07 Isaac Lab 风格](07_Isaac_Lab风格.md)
