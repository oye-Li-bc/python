# 04 NumPy（科学计算核心）

> **NumPy 是 Python 整个科学计算 / AI 生态的地基**。PyTorch 的 Tensor 就是模仿它做的。
> 学会 NumPy = 学了一半 PyTorch。

---

## 1. NumPy 是什么

简单说：**让 Python 拥有像 C 一样快的数组**。

```python
# 纯 Python，慢
a = [1, 2, 3, 4, 5]
b = [x * 2 for x in a]      # 一个个乘

# NumPy，快
import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = a * 2                    # 整个数组一次性乘
```

NumPy 数组是**连续内存**的 C 数组，配合**向量化运算**，速度比纯 Python 快几十倍。

---

## 2. 创建数组

```python
import numpy as np

# 从 list 创建
a = np.array([1, 2, 3, 4])
b = np.array([[1, 2], [3, 4]])      # 二维

# 全 0 / 全 1
np.zeros(5)             # [0,0,0,0,0]
np.zeros((3, 4))        # 3 行 4 列的零矩阵
np.ones((2, 3))         # 2 行 3 列的全 1

# 等差数列
np.arange(0, 10, 2)     # [0,2,4,6,8]（类似 range）
np.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1.0]（5 个均匀分布）

# 随机
np.random.rand(3)               # 3 个 [0,1) 随机数
np.random.randn(3, 4)           # 3x4 标准正态分布
np.random.uniform(-1, 1, (5,))  # 均匀分布
```

---

## 3. 关键属性

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape         # (2, 3)：2 行 3 列
a.ndim          # 2：维度数
a.size          # 6：总元素数
a.dtype         # int64：数据类型
```

> 💡 **`shape` 是 RL / DL 里最重要的概念之一**，跟踪每一步的 shape 才不会出错。

---

## 4. 索引和切片（和 list 一样）

```python
a = np.array([10, 20, 30, 40, 50])
a[0]            # 10
a[-1]           # 50
a[1:4]          # array([20, 30, 40])
a[::2]          # array([10, 30, 50])

# 二维
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b[0, 1]         # 2（第 0 行第 1 列）
b[0]            # array([1, 2, 3])（第 0 行）
b[:, 0]         # array([1, 4, 7])（第 0 列）
b[0:2, 1:3]     # array([[2,3],[5,6]])（左上 2x2）
```

---

## 5. 布尔索引（超好用）

```python
a = np.array([1, 5, 2, 8, 3, 9])

mask = a > 3            # array([False, True, False, True, False, True])
a[mask]                 # array([5, 8, 9])

# 一步到位
a[a > 3]                # array([5, 8, 9])

# 修改满足条件的元素
a[a > 5] = 0            # 大于 5 的设为 0
# → array([1, 5, 2, 0, 3, 0])
```

**RL 场景**：找出"已结束"的环境
```python
dones = np.array([False, True, False, True, False])
reward = np.array([0.1, 0.5, -0.2, 0.3, 0.0])
reward[dones] = 0       # 已结束的环境奖励清零
```

---

## 6. 向量化运算（NumPy 的灵魂）

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

a + b           # [11, 22, 33]   逐元素加
a * b           # [10, 40, 90]   逐元素乘（不是矩阵乘！）
a ** 2          # [1, 4, 9]
np.sqrt(a)      # [1, 1.414, 1.732]
np.sin(a)
np.exp(a)
```

**和标量运算（广播）：**
```python
a + 10          # [11, 12, 13]
a * 0.5         # [0.5, 1.0, 1.5]
```

---

## 7. 广播（broadcasting）

不同 shape 的数组也能运算，前提是兼容。规则：**从右往左对齐，相同或其中一个为 1**。

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])           # shape (2, 3)
b = np.array([10, 20, 30])          # shape (3,)

a + b
# [[11, 22, 33],
#  [14, 25, 36]]   b 被广播到每一行

c = np.array([[10], [20]])          # shape (2, 1)
a + c
# [[11, 12, 13],
#  [24, 25, 26]]   c 被广播到每一列
```

> 💡 广播是 **NumPy / PyTorch 一切批量操作的基础**，必须理解。

---

## 8. 聚合函数

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.sum()             # 21
a.mean()            # 3.5
a.std()             # 标准差
a.max()             # 6
a.min()             # 1
a.argmax()          # 5（最大值的扁平索引）

# 按轴聚合
a.sum(axis=0)       # [5, 7, 9]   沿行方向求和（消掉行）
a.sum(axis=1)       # [6, 15]     沿列方向求和（消掉列）
a.mean(axis=0)      # [2.5, 3.5, 4.5]
```

**axis 怎么记？**
- `axis=0` → 沿着"行"方向走（一行行下去），结果是**列数**那么多
- `axis=1` → 沿着"列"方向走，结果是**行数**那么多
- 经验法则：`axis=k` 消掉第 k 个维度

---

## 9. 改变形状

```python
a = np.arange(12)               # [0,1,2,...,11], shape (12,)

a.reshape(3, 4)                 # 3 行 4 列
a.reshape(2, 2, 3)              # 三维
a.reshape(-1, 4)                # -1 表示"自动算"，这里变成 (3, 4)

# 增加 / 删除维度
a = np.array([1, 2, 3])         # shape (3,)
a[np.newaxis, :]                # shape (1, 3)
a[:, np.newaxis]                # shape (3, 1)
np.expand_dims(a, axis=0)       # 同 a[np.newaxis, :]
np.squeeze(np.array([[1,2,3]])) # 去掉大小为 1 的维度

# 转置
b = np.array([[1, 2, 3], [4, 5, 6]])    # (2, 3)
b.T                                       # (3, 2)
```

---

## 10. 拼接

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.concatenate([a, b])              # [1,2,3,4,5,6]

# 二维
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.concatenate([A, B], axis=0)      # 按行拼接 → 4x2
np.concatenate([A, B], axis=1)      # 按列拼接 → 2x4

# 也可以用 vstack / hstack
np.vstack([A, B])                   # 垂直堆叠
np.hstack([A, B])                   # 水平堆叠
```

---

## 11. 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A * B               # 逐元素乘
A @ B               # 矩阵乘法（推荐）
np.dot(A, B)        # 矩阵乘法（老写法）

np.linalg.inv(A)            # 逆矩阵
np.linalg.eig(A)            # 特征值分解
np.linalg.norm(A)           # 范数
```

---

## 12. 实战例子

### 例 1：归一化一组传感器数据
```python
import numpy as np

# 一段 IMU 角速度数据，shape (100, 3) → 100 个三维向量
imu_data = np.random.randn(100, 3) * 0.5

# 计算均值和标准差
mean = imu_data.mean(axis=0)        # shape (3,)
std = imu_data.std(axis=0)          # shape (3,)

# 标准化
normalized = (imu_data - mean) / (std + 1e-8)
```

### 例 2：计算 N 个 PID 控制器的输出（向量化）
```python
import numpy as np

N = 100                              # 100 个机器人并行
kp = 2.0
errors = np.random.randn(N)          # 100 个误差
outputs = kp * errors                # 一行算完！
```

### 例 3：滑动平均
```python
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
window = 3
result = np.convolve(data, np.ones(window)/window, mode='valid')
# [2., 3., 4., 5., 6.]
```

### 例 4：构造轨迹
```python
t = np.linspace(0, 1, 100)           # 100 个时间点
x = np.sin(2 * np.pi * t)            # 正弦轨迹
y = np.cos(2 * np.pi * t)
trajectory = np.stack([x, y], axis=1)    # shape (100, 2)
```

---

## 13. NumPy ↔ PyTorch（无缝转换）

```python
import numpy as np
import torch

a = np.array([1, 2, 3])
t = torch.from_numpy(a)              # numpy → tensor
b = t.numpy()                         # tensor → numpy
```

> 注意：在 CPU 上时**共享内存**（改一个另一个也变）。

---

## 14. 常见错误

### "ValueError: operands could not be broadcast together"
shape 不兼容。打印 shape 排查：
```python
print(a.shape, b.shape)
```

### "IndexError: too many indices"
索引数超过维度数。
```python
a = np.array([1, 2, 3])       # 一维
a[0, 0]                        # ❌ 一维数组用了二维索引
```

### "TypeError: unhashable type: 'numpy.ndarray'"
把 numpy 数组当 dict 键 / set 元素。NumPy 数组不能 hash。

---

## 15. RL 中的典型用法

```python
# 一个 batch 的观测：(num_envs, obs_dim)
obs = np.random.randn(4096, 48)

# 计算奖励
target_vel = np.array([1.0, 0.0, 0.0])    # 目标速度
curr_vel = obs[:, 0:3]                     # 取每个 env 的速度，(4096, 3)
err = curr_vel - target_vel                # 广播：(4096, 3)
reward = -np.linalg.norm(err, axis=1)      # 每个 env 一个奖励，(4096,)
```

> 💡 **批处理（batched）思维**：永远不用 for 循环跑 4096 个环境，而是用一个 (4096, ...) 形状的数组一次算完。

---

## 动手任务

给你 1000 个机器人的姿态四元数（shape `(1000, 4)`）和速度（shape `(1000, 3)`），完成：
1. 把每个四元数归一化（除以自己的模）
2. 计算每个机器人的速度大小（标量）
3. 找出速度大于 1.0 的机器人索引

参考答案：
```python
import numpy as np

quats = np.random.randn(1000, 4)
vels = np.random.randn(1000, 3)

# 1. 归一化四元数
norms = np.linalg.norm(quats, axis=1, keepdims=True)  # (1000, 1)
quats_norm = quats / norms                             # 广播除

# 2. 速度大小
speed = np.linalg.norm(vels, axis=1)                   # (1000,)

# 3. 速度 > 1.0 的索引
fast_idx = np.where(speed > 1.0)[0]
print(f"快车数量: {len(fast_idx)}")
```

---

## 下一篇

→ [05 PyTorch 基础](05_PyTorch基础.md)
