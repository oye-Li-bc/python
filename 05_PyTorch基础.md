# 05 PyTorch 基础

> PyTorch = NumPy + GPU + 自动求导。
> 你 90% 的时间在和 `Tensor` 打交道，**会 NumPy 就会 PyTorch**。

---

## 1. PyTorch 是什么

- **Tensor**：和 numpy 数组几乎一样的多维数组，但**能跑在 GPU 上**
- **autograd**：自动算梯度，神经网络训练靠它
- **nn 模块**：现成的层（线性层、卷积层、激活函数）
- **optim 模块**：优化器（Adam、SGD）

> 💡 Isaac Lab、rsl_rl 都构建在 PyTorch 之上。

---

## 2. 第一个 Tensor

```python
import torch

# 从 list / numpy 创建
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1, 2], [3, 4]])

# 全 0 / 全 1 / 随机
torch.zeros(3, 4)
torch.ones(2, 3)
torch.randn(3, 4)             # 标准正态分布
torch.rand(3, 4)              # 均匀分布 [0, 1)

# 等差
torch.arange(0, 10, 2)        # tensor([0, 2, 4, 6, 8])
torch.linspace(0, 1, 5)
```

**和 NumPy 几乎一对一映射：**

| NumPy | PyTorch |
|-------|---------|
| `np.array(...)` | `torch.tensor(...)` |
| `np.zeros(3,4)` | `torch.zeros(3, 4)` |
| `a.shape` | `a.shape` |
| `a.dtype` | `a.dtype` |
| `a.reshape(2,3)` | `a.reshape(2,3)` 或 `a.view(2,3)` |
| `a.T` | `a.T`（二维）/ `a.transpose(0,1)` |
| `a @ b` | `a @ b` |
| `a.sum(axis=0)` | `a.sum(dim=0)`（注意：`dim` 不是 `axis`） |
| `np.concatenate` | `torch.cat` |

---

## 3. dtype（数据类型）

```python
a = torch.tensor([1, 2, 3])           # int64（默认）
b = torch.tensor([1.0, 2.0])          # float32（默认）
c = torch.tensor([1.0], dtype=torch.float64)
d = a.float()                          # 转 float32
e = a.long()                           # 转 int64
```

**RL 里常用的：**
- `torch.float32`：网络参数、观测、奖励 → 默认就是这个
- `torch.long`：索引 / 类别标签
- `torch.bool`：mask

---

## 4. GPU：神经网络加速的关键

```python
# 检查 GPU
torch.cuda.is_available()             # True / False
device = "cuda" if torch.cuda.is_available() else "cpu"

# 把 tensor 放到 GPU
a = torch.randn(3, 4)                 # 在 CPU
a = a.to(device)                       # 移到 GPU
a = a.cuda()                           # 同上（旧写法）

# 运算结果在哪个设备，输入在哪结果就在哪
b = a * 2                              # 在 GPU

# 移回 CPU
a_cpu = a.cpu()
a_np = a.cpu().numpy()                 # 转 numpy 必须先 cpu
```

> ⚠️ **常见错误**：两个 tensor 在不同设备时不能运算。报错信息含 `Expected ... cuda ... cpu`。

---

## 5. 索引和切片（同 NumPy）

```python
a = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

a[0]                # 第一行
a[:, 1]             # 第二列
a[0:2, 1:3]         # 子矩阵

mask = a > 5
a[mask]             # tensor([6, 7, 8, 9, 10, 11])
```

---

## 6. 数学运算

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

a + b                       # 逐元素
a * b
torch.sin(a)
torch.exp(a)

a.sum()
a.mean()
a.max()                     # 返回最大值
a.argmax()                  # 返回最大值的索引

# 矩阵乘
A = torch.randn(3, 4)
B = torch.randn(4, 5)
A @ B                       # shape (3, 5)
torch.matmul(A, B)          # 同上
```

---

## 7. 自动求导（autograd）—— PyTorch 灵魂

```python
x = torch.tensor(2.0, requires_grad=True)        # 标记"需要求导"
y = x ** 2 + 3 * x + 1                            # 自动构建计算图
y.backward()                                      # 反向传播

print(x.grad)                                     # dy/dx = 2x+3 = 7
```

**机理**：每个有 `requires_grad=True` 的 tensor 参与的运算都会被记录，`.backward()` 时自动求导。

> 💡 你**不需要手动算导数**——这就是为什么 PyTorch 在深度学习里这么受欢迎。

**多元函数：**
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()           # y 是标量
y.backward()
print(x.grad)                # tensor([2., 4.])  即 dy/dx_i = 2x_i
```

---

## 8. nn 模块：搭网络

最小例子：一个线性回归
```python
import torch
import torch.nn as nn

# 一层线性变换：y = Wx + b
layer = nn.Linear(in_features=3, out_features=1)

x = torch.randn(5, 3)        # 5 个样本，每个 3 维
y = layer(x)                  # shape (5, 1)
```

**多层网络（MLP，强化学习里最常见）：**
```python
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


net = PolicyNet(obs_dim=48, act_dim=12)
obs = torch.randn(1024, 48)              # 1024 个观测
actions = net(obs)                        # 1024 个动作，shape (1024, 12)
```

要点：
- 继承 `nn.Module`
- `__init__` 里定义层
- `forward` 里写前向计算
- 调用 `net(x)` 实际上调的是 `net.forward(x)`

---

## 9. 训练流程一览（不深究）

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备模型、损失、优化器
model = PolicyNet(obs_dim=4, act_dim=2)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 2. 训练循环
for epoch in range(100):
    pred = model(x_train)                # 前向
    loss = loss_fn(pred, y_train)         # 算损失
    
    optimizer.zero_grad()                  # 清梯度（重要！）
    loss.backward()                        # 反向传播算梯度
    optimizer.step()                       # 用梯度更新参数

    if epoch % 10 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")
```

**RL 里的对应概念**：
- `model` = 策略网络 / 价值网络
- `loss` = PPO 的 surrogate loss
- `optimizer` = Adam
- 但**你大概率不用自己写训练循环**，rsl_rl 帮你写好了

---

## 10. 关键操作对照表

| 操作 | 写法 |
|------|------|
| 形状变换 | `t.reshape(...)` 或 `t.view(...)` |
| 转置 | `t.transpose(0, 1)` |
| 增加维度 | `t.unsqueeze(0)` |
| 去掉维度 | `t.squeeze()` |
| 拼接 | `torch.cat([a, b], dim=0)` |
| 堆叠（新增维度） | `torch.stack([a, b], dim=0)` |
| 广播运算 | 自动，规则同 NumPy |
| 转 NumPy | `t.cpu().numpy()` |
| 从 NumPy | `torch.from_numpy(arr)` |
| 关闭梯度 | `with torch.no_grad():` |
| 标量值 | `t.item()`（仅当 t 是单元素） |

---

## 11. 关闭梯度（推理 / 部署时）

部署策略到机器人时，**只前向不需要反向**，关掉 autograd 省内存：

```python
model.eval()                              # 切换到推理模式
with torch.no_grad():
    actions = model(obs)
```

---

## 12. 模型保存与加载

```python
# 保存
torch.save(model.state_dict(), "policy.pt")

# 加载
model = PolicyNet(obs_dim=48, act_dim=12)
model.load_state_dict(torch.load("policy.pt"))
model.eval()
```

> ⚠️ Sim2Real 部署时常见这个，记下来。

---

## 13. 完整可运行例子：拟合 sin 函数

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 准备数据：y = sin(x)
x = torch.linspace(-3.14, 3.14, 200).reshape(-1, 1)
y = torch.sin(x)

# 模型
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
)

# 损失 + 优化器
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-2)

# 训练
for epoch in range(2000):
    pred = model(x)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 200 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")

# 推理
with torch.no_grad():
    pred = model(x)
    print("最大误差:", (pred - y).abs().max().item())
```

跑这个代码，你会看到 loss 慢慢下降——你已经训练了一个神经网络。

---

## 14. 常见错误

### "RuntimeError: Expected ... but got ..."
设备不匹配。检查所有 tensor 是不是都在同一个 device：
```python
print(a.device, b.device)
```

### "RuntimeError: shape '[..]' is invalid for input of size .."
reshape / view 的 shape 不对。打 `.shape` 检查。

### "Trying to backward through the graph a second time"
对同一个 loss 调了两次 `.backward()`。每次反向后要重新前向。

### Loss 不下降
- 学习率太大 / 太小
- 没 `optimizer.zero_grad()`
- 数据没 normalize
- 模型架构有问题

---

## 15. RL 场景中的典型形状

```python
# 一个 batch 的观测
obs.shape           # (num_envs, obs_dim)，比如 (4096, 48)

# 策略输出（连续动作）
action_mean.shape   # (num_envs, act_dim)，比如 (4096, 12)
action_std.shape    # (num_envs, act_dim) 或 (act_dim,)

# 奖励
rewards.shape       # (num_envs,)，比如 (4096,)

# done flag
dones.shape         # (num_envs,)，dtype=bool 或 long
```

**记住**：RL 第一个维度永远是 batch / 环境数。

---

## 动手任务

写一段代码：
1. 创建一个形状 `(100, 3)` 的随机 tensor，模拟 100 个机器人的速度
2. 计算每个机器人的速度大小（用 `torch.norm`，`dim=1`）
3. 找出速度大于 1.0 的机器人个数
4. 把这个 tensor 移到 GPU（如果有），打印 `device`

参考答案：
```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

vels = torch.randn(100, 3)
speeds = torch.norm(vels, dim=1)         # (100,)
fast_count = (speeds > 1.0).sum().item()
print(f"速度 > 1.0 的机器人: {fast_count}")

vels = vels.to(device)
print("device:", vels.device)
```

---

## 下一篇

→ [06 Gymnasium 接口](06_Gymnasium接口.md)
