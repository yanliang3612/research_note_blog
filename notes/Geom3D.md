# Geom3D Paper

## SE(3)

- 旋转
- 平移


## We classify the geometric methods into three categories: 

- invariant model, 

- SE(3)-equivariant model with spherical frame basis,

- vector frame basis. 

- The invariant models only consider features that are constant w.r.t. the SE(3) group, while the two families of equivariant models can be further unified using the frame basis to capture equivariant symmetry. 

- An illustration of three categories is in Fig. 2. Building equivariant models on the frame basis provides a novel and unified view of understanding geometric models and paves the way for intriguing more ML researchers to explore scientific problems.

## Invariant Model

- 不变模型只考虑在 SE(3) 群下保持不变的特征，只考虑 0 型特征（type-0 features）

在几何深度学习，尤其是 SE(3)-等变模型中，  
**type-0 features（0 型特征）** 指的是旋转、平移下保持不变的 **标量特征（scalar features）**。



#### ✅ type-0 features 是什么？

**type-0 = scalar = 不随旋转/平移变化的量**

如果你把三维分子整体旋转、平移，这些特征的数值不会改变。例如：

- 原子间距离（distance）
- 角度（angle）
- 二面角（dihedral angle）
- 原子类别（atomic type）
- 电荷（charge）

这些量 **不依赖坐标的具体方向**，因此在 SE(3) 群（旋转与平移）作用下保持不变。

所以它们叫 **type-0**，也称 **0 阶球谐分量**。

#### 🟩 为什么叫 type-0？

在 SE(3)-等变表示论中，特征被分为不同的 **types**（也可称作 orders 或 irreps）：

| 类型    | 含义                 | 例子                          |
| ------- | -------------------- | ----------------------------- |
| type-0  | 标量，不变           | 距离、角度、原子类型          |
| type-1  | 向量，随旋转线性变换 | 坐标差向量 \(r_{ij}\)、力向量 |
| type-2+ | 高阶张量             | 球谐分量 \(Y_2, Y_3 ...\)     |

因此，type-0 的定义就是：  
**旋转矩阵 \( R \) 作用时不发生变化**

\[
R x = x
\]

---

## 👀 举个直观例子

假设两个原子之间距离 \( d = 1.2 \text{ Å} \)。

你把整个分子旋转 90°、180°、随便怎么旋转：

- 坐标会变化  
- 向量方向会变  
- 但距离 **1.2 Å 永远不变**

因此它是 **type-0**。

---

如果是向量（比如 \(r_{ij}\)），就会随旋转而变化，那就是 **type-1**。

#### 📝 总结（一句话）

**type-0 features = 不随旋转平移改变的标量特征，是几何不变量。**

---

## Invariant Model 是怎么运作的



# Invariant Model（不变模型）是怎么运作的？用三层结构讲清楚：

1. 核心思想（最直观版）
2. 数学机制（为什么不变）
3. 在 GNN/深度学习/分子建模中的实际用途（怎么用）

这是“Invariant Model vs Equivariant Model”区分的基础。

##  1. Invariant Model 的核心思想（直观版）

Invariant model = 只使用在 SE(3) 旋转和平移下保持不变的特征。

也就是说，**无论你**：

- 在 3D 空间把物体旋转 90°
- 平移到别的地方
- 翻来覆去、倒过来

输入给模型的特征值 **保持完全一样**。

模型因此完全“不知道方向”，只能看到：

- 距离多长？
- 角度多大？
- 原子是什么类型？
- 哪些原子连接？
- 拓扑结构如何？

这些都不会随旋转变化，所以叫 invariant（不变）。

------

## ⭐ 直观示例：

把水分子旋转 360°：

- O–H 距离不变
- H–O–H 角度不变
- 原子类型不变
- 键连接关系不变

➡️ 模型输入不变 ➜ 模型输出也不变 ➜ 模型本质是 invariant。

## 🔴 2. 数学机制：Invariant 特征是什么？

符合 SE(3) 不变性的特征称为 **type-0 特征**：

[
 f(Rx + t) = f(x)
 ]

典型不变特征包括：

| 类型        | 不变特征                | 是否方向相关 |
| ----------- | ----------------------- | ------------ |
| 标量 type-0 | 距离 d_ij、角度、二面角 | ❌ 不包含方向 |
| 拓扑        | 邻接矩阵、键类型        | ❌ 不包含方向 |
| 原子特征    | 元素（C/H/O/N）、电荷   | ❌ 不包含方向 |

------

## 🔵 这些都是 rotation-invariant

因为：

- **距离**：‖R(x_j - x_i)‖ = ‖x_j - x_i‖
- **角度**：没有方向与向量顺序的区分
- **二面角**：只与三个平面关系有关
- **原子类型**：与位置无关

------

## 📌 因此 invariant models 本质是：

> **只能用 type-0 特征进行提纯或 Transformer 编码。**

------

# 🔴 3. Invariant Model 是怎么做消息传递的？

使用的 GNN 一般类似：

[
 h_i^{(l+1)} = MLP(h_i^{(l)}, \sum_{j\in N(i)} \phi(h_i, h_j, d_{ij}, \theta_{ij}, ...))
 ]

其中：

- **h_i**：节点（原子）标量特征
- **φ(·)**：只依赖不变几何量
- 无方向特征
- 所有输入都为 type-0

------

## 典型方法：

- SchNet（最经典的 invariant GNN）
- DimeNet++（加角度信息）
- PhysNet, GemNet-T
- Transformer + distance encoding（如 MolFormer）

这些模型都“不知道空间方向”，只知道距离和角度等标量。

------

# 🔴 4. 为什么 Invariant Model 不包含方向？

因为 invariant model 完全忽略三维的方向向量，如：

$\vec{r}_{ij} = x_j - x_i$

- ❌ 不被直接输入
- ❌ 只能用它的长度
- ❌ 甚至不知道“H 在 O 的左边还是右边”

也就是说：

> 模型不知道三维世界是怎么旋转的，它只知道保持不变的几何关系。

➡️ 这也是无法直接预测 vector（如力）的原因。

------

# 🔴 5. Invariant Model 能输出什么？不能输出什么？

## ✔️ 能输出：标量（都不依赖方向）

例如：

- 能量
- 化学性质（HOMO、LUMO）
- 物性（溶解度、蒸气压）
- 密度、分子量
- 任何全局标量

------

## ❌ 不能输出：任何需要“方向”的量

例如：

- 分子力（force）
- 偏导方向
- 3D 矢量场
- 电场
- 力常数张量

因为 invariant model **完全没有方向信息**。

------

# 🔴 6. Invariant Model 的优缺点

## 优点：

- 模型非常简单
- 训练稳定
- 计算成本低
- 适合标量性质预测

## 缺点：

- 无法处理方向信息 ➜ 无法预测力
- 无法做到 SE(3)-等变
- 表达能力比等变模型弱

------

# 🔴 7. 图中的 invariant model 对应什么？

- Distance / Angle / Dihedral Angle → 都是 type-0
- Feature Extraction → 提取不变特征
- Representation → MLP/GNN 编码

模型**不使用**：

- 向量基底
- 球谐基底
- 局部坐标系
- 任可 SE(3) 等变结构

➡️ 因此分类为：**Invariant Representation（纯 type-0 特征模型）**

------

# 🎉 最终总结（10 秒版）

**Invariant Model = 使用距离/角度/二面角/标量等不随旋转变化的几何特征进行 GNN 或 Transformer 模型。**

它只用 type-0 特征，因此：

- 不感知方向
- 只能预测标量
- 不是等变
- 属于最早、最简单的 3D 几何模型类型

---


## Vector Frame Bias

- PaiNN、ET、DimeNet++、MACE、Allegro

### 1. Vector Frame 的核心思想（一句话）
为每个原子构建一个局部坐标系（frame），所有邻居原子的坐标都投影到这个局部坐标系里，使得特征自然保持等变性。

也就是说：

- 全局旋转 → 局部坐标系也跟着旋转

- 在局部坐标系里的特征（如向量坐标）保持一致的旋转行为

- 因此整个网络具备 SE(3)-等变性。

### 2. 如何构建一个局部坐标系（vector frame）？

通常做法：

#### (1) 选择一个中心原子 i

- 以它为局部坐标系的原点。

#### (2) 从邻居中选两个原子 j、k，定义方向向量：

\[
\mathbf{v}_1 = \frac{\mathbf{r}_{ij}}{||\mathbf{r}_{ij}||}
\]

\[
\mathbf{v}_2 = \mathbf{r}_{ik} - (\mathbf{r}_{ik} \cdot \mathbf{v}_1)\mathbf{v}_1 
\quad\text{（投影到垂直平面）}
\]
- 这一步有点说法在里面

---

#### 2.1 点乘的含义

点乘（dot product）是向量之间最基本的运算之一。  
它的结果是一个 **标量（数字）**，不是向量。

下面我为你写三个层次——数学定义、几何意义、直观例子，帮助你彻底理解。

---

##### ✅ 1. 数学定义（如何计算）

对两个向量：

\[
\mathbf{a} = (a_x, a_y, a_z), \quad 
\mathbf{b} = (b_x, b_y, b_z)
\]

它们的点乘是：

\[
\mathbf{a}\cdot\mathbf{b} = a_x b_x + a_y b_y + a_z b_z
\]

也就是说：对应坐标相乘然后相加。

---

##### ✅ 2. 几何意义（最重要）

\[
\mathbf{a}\cdot\mathbf{b} = |\mathbf{a}|\,|\mathbf{b}|\cos\theta
\]

它衡量两向量之间方向的关系：

- **同向（0°）**：点乘 > 0 —— 正数  
- **垂直（90°）**：点乘 = 0 —— 完全不相关  
- **反方向（180°）**：点乘 < 0 —— 负数  

### 👉 你可以把点乘理解为：

> **向量 a 在向量 b 方向上的“投影长度”乘以 |b|**

特别是当 **b 是单位向量（如你的 v1）** 时：

\[
\mathbf{a}\cdot\mathbf{b} = \text{向量 a 在 b 方向上的投影长度}
\]

这是点乘最重要也最常用的理解。

---

##### ✅ 3. 一个简单直观的例子

设：

\[
\mathbf{a} = (3,4), \quad \mathbf{b} = (1,0)
\]

计算点乘：

\[
\mathbf{a}\cdot\mathbf{b} = 3\cdot 1 + 4\cdot 0 = 3
\]

这表示：

- 向量 **a = (3,4)** 在 **x 方向** 的投影长度是 **3**  
- 因为 **b 是沿 x 轴的单位向量**

图示（markdown 示意）：

##### 2.2  $\mathbf{r}_{ik} - (\mathbf{r}_{ik} \cdot \mathbf{v}_1)\mathbf{v}_1$ 为啥和$v_1$ 一定正交

这是个特别好的问题！  
关键点在于：你从一个向量中减去了它在另一个方向上的全部平行分量，所以剩下来的必然是垂直的。

下面我用直觉 + 数学双重解释，让你彻底理解为什么一定正交。

##### ✅ 1. 直觉解释（最容易理解）

把 **rₖ** 分解成两部分：

1. **平行于 v₁ 的部分**
   \[
   (\mathbf{r}_{ik}\cdot\mathbf{v}_1)\mathbf{v}_1
   \]

2. **垂直于 v₁ 的部分**  
   ——就是我们要找的 **v₂**

当你用：

\[
\mathbf{r}_{ik} - (\mathbf{r}_{ik}\cdot\mathbf{v}_1)\mathbf{v}_1
\]

等于：

**从整个向量中扣掉它平行的部分 → 剩下的当然是垂直部分。**

所以正交是必然的结果。

##### ✅ 2. 数学证明（简单且优美）

目标是证明：

\[
\mathbf{v}_2 = \mathbf{r}_{ik} - (\mathbf{r}_{ik}\cdot\mathbf{v}_1)\mathbf{v}_1
\]

与 **v₁ 正交**，也就是：

\[
\mathbf{v}_2 \cdot \mathbf{v}_1 = 0
\]

计算：

\[
\mathbf{v}_2 \cdot \mathbf{v}_1
= [\mathbf{r}_{ik} - (\mathbf{r}_{ik}\cdot\mathbf{v}_1)\mathbf{v}_1]\cdot\mathbf{v}_1
\]

展开：

\[
= \mathbf{r}_{ik}\cdot\mathbf{v}_1
- (\mathbf{r}_{ik}\cdot\mathbf{v}_1)(\mathbf{v}_1\cdot\mathbf{v}_1)
\]

因为 **v₁ 是单位向量**，所以：

\[
\mathbf{v}_1\cdot\mathbf{v}_1 = 1
\]

于是：

\[
= \mathbf{r}_{ik}\cdot\mathbf{v}_1 - (\mathbf{r}_{ik}\cdot\mathbf{v}_1) = 0
\]

点乘为 0 ⟹ 垂直（正交）。

##### 🎉 3. 最简一句话总结

> 你把一个向量的平行部分完全减掉，剩下来的部分自然就是与该方向垂直的。  
> 这就是为什么 v₂ 必然与 v₁ 正交。
>
> 

---

#### (3) 归一化并构建正交坐标轴

\[
\mathbf{e}_1 = \mathbf{v}_1
\]

\[
\mathbf{e}_2 = \frac{\mathbf{v}_2}{||\mathbf{v}_2||}
\]

\[
\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2
\]

于是就得到一个局部坐标框架：

\[
F_i = (\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3)
\]

这就是一个 \(3\times 3\) 的旋转矩阵 \(R\)，表示局部坐标系。

---

##### 3.1 叉乘

这里的 “×” 叫 **叉乘（cross product）**，是向量运算中的 **向量积**。

在你给的式子里：

\[
\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2
\]

意思是：

- 用 **右手定则** 将向量 \(\mathbf{e}_1\) 和 \(\mathbf{e}_2\) 做叉乘；
- 得到的结果 \(\mathbf{e}_3\) 是 **垂直于 $\mathbf{e}_1$ 和 $\mathbf{e}_2$** 的向量；
- 其方向由右手定则确定，长度等于  

\[
|\mathbf{e}_1| \, |\mathbf{e}_2| \sin\theta
\]

其中 \(\theta\) 是二者夹角。

在标准正交基中，常见关系是：

\[
\mathbf{e}_1 \times \mathbf{e}_2 = \mathbf{e}_3,\quad
\mathbf{e}_2 \times \mathbf{e}_3 = \mathbf{e}_1,\quad
\mathbf{e}_3 \times \mathbf{e}_1 = \mathbf{e}_2
\]


##### 3.2 🖐️ 右手定则怎么用？

要判断 **a × b** 的方向：

1. **伸出右手**，让手指自然弯曲。
2. **四指从 a 指向 b**（表示从第一个向量“旋”向第二个向量）。  
   👉 *注意：要用最小角度旋过去（≤180°）。*
3. **大拇指竖起来**，它所指的方向就是  

   \[
   \mathbf{a} \times \mathbf{b}
   \]

   的方向。

##### 📌 例子

如果：

- \(\mathbf{e}_1\) 指向 x 轴正方向  
- \(\mathbf{e}_2\) 指向 y 轴正方向  

用右手将手指从 \(\mathbf{e}_1\) 弯向 \(\mathbf{e}_2\)，大拇指就指向 z 轴正方向。

因此：

\[
\mathbf{e}_1 \times \mathbf{e}_2 = \mathbf{e}_3
\]

##### 🎯 记住关键点

- 必须用 **右手** 而不是左手  
- 四指从第一个向量转向第二个  
- 大拇指给出叉乘结果方向



###  3. 🔢  如何把邻居原子的坐标投影到局部坐标系？

对每个邻居 j：

原始坐标是：

\[
\mathbf{r}_{ij}
\]

投影到 frame \( \mathbf{F}_i \) 上：

\[
\mathbf{r}_{ij}^{\text{local}} = \mathbf{F}_i^{\top} \mathbf{r}_{ij}
\]

进一步非常关键的是：

🌟 **全局旋转下，全局坐标变化，但 local 坐标不变！**  
因此模型获得的 **方向信息** 保留，但 **不依赖全局坐标系**。

### 🔄 4. 为什么 vector frames 天然是等变的？

假设全局旋转为矩阵 \( R \)：

全局坐标变为：

\[
\mathbf{r}'_{ij} = R \mathbf{r}_{ij}
\]

局部坐标轴的三帧（e1, e2, e3）也会随之变成：

\[
\mathbf{F}'_i = R \mathbf{F}_i
\]

因此：

\[
\mathbf{r}_{ij}^{\text{local}'} 
= (\mathbf{F}'_i)^{\top} \mathbf{r}'_{ij} 
= (R\mathbf{F}_i)^{\top} (R\mathbf{r}_{ij}) 
= \mathbf{F}_i^{\top} R^{\top} R \mathbf{r}_{ij} 
= \mathbf{F}_i^{\top} \mathbf{r}_{ij} 
= \mathbf{r}_{ij}^{\text{local}}
\]

📌 **局部投影值完全不变，因此整体保持 SE(3)-等变。**



### 📡 5. Vector Frames 如何用于消息传递（GNN）？

例如，给定每个邻居 j 的局部坐标：

\[
\mathbf{r}_{ij}^{\text{local}} = (x, y, z)
\]

模型可以输入：

- 标量特征（type-0）
- 向量特征（type-1）在 local frame 下的系数
- 多头注意力
- 3D 相对注意力等

消息 \( M_{i \leftarrow j} \) 也保持等变。

最后，若要恢复 global 向量：

\[
\mathbf{v}_i^{\text{global}} = \mathbf{F}_i \mathbf{v}_i^{\text{local}}
\]

### 📌 6. Vector Frames 的优缺点

### ✅ 优点

- **计算快**（比球谐简单很多）
- **数值稳定**
- **效果好**（因比 invariant 模型强很多）
- **无需高阶张量**
- 模型几乎是 plug-and-play（如 PaiNN/ET/MACE 非常易用）

### ❌ 缺点

- 需要选择 frame 的规则（比如选择临近原子 i,k → 可能有歧义）
- 对局部噪声更敏感
- 框架必须保持连续性（否则会出现跳变）



## 7. $R_{i,j }$ 不是坐标差吗，怎么$r_{i,j}^{local}$怎么又变成j的局部坐标了



## 🔍 关键点：\( r_{ij} \) 是向量，但一个向量在某个坐标系下的分量就是“局部坐标”

### ⭐ 1. \( r_{ij} \) 是坐标差向量

\[
r_{ij} = x_j - x_i
\]

它表示：

- 从原子 \( i \) 指向原子 \( j \) 的 **向量**
- 在 **全局坐标系** 下表示（例如 xyz 全局坐标）

---

### ⭐ 2. 但向量在不同坐标系里的分量其实就是“坐标坐标”

假设一个向量 \( v = (1, 1, 1) \)，你换一个旋转后的坐标系，它在新坐标系下会变成：

\[
v' = (2, 0, -1) \quad (\text{示意})
\]

➡️ **“2,0,-1” 就是 \( v \) 在新坐标系下的坐标分量。**

⚠️ 注意：

- 向量本身 *没有改变*
- 只是坐标系表示法变了

用矩阵表示：

\[
r^{\text{local}}_{ij} = F_i^\top r_{ij}
\]

表示：

📌 把 \( r_{ij} \) 在 **原子 i 的局部坐标系（frame\_i）** 里表达出来，它就是邻居 j 的局部坐标。

---

## ✨ 3. 为什么投影后就成了“邻居 j 的局部坐标”？

因为：

- \( r_{ij} = x_j - x_i \)
- 在局部坐标系里，**原点就是原子 i**

也就是说：

📌 在局部坐标系中，原子 \( i \) 的坐标是 \( (0, 0, 0) \)，  
📌 \( r_{ij} \) 的坐标分量就是原子 \( j \) 的坐标。

这点非常关键！

可以可视化一下：

---

### 🌐 全局坐标系（global）

- i 的坐标：\( x_i \)
- j 的坐标：\( x_j \)
- 坐标差：\( r_{ij} = x_j - x_i \)

---

### 🧭 局部坐标系（frame\_i）

- i 的坐标：\( (0,0,0) \)（局部原点）
- j 的坐标：\( r_{ij}^{\text{local}} \) 的分量

即：

\[
\text{坐标}_{\text{local}}(j) = r^{\text{local}}_{ij}
\]

---

## ⭐ 4. 看一个简单例子

假设原子 i 的局部坐标系是标准 xyz（为了简单）：

- \( x_i = (3, 5, 1) \)
- \( x_j = (4, 8, 0) \)

全局坐标中的向量：

\[
r_{ij} = (4, 8, 0) - (3, 5, 1) = (1, 3, -1)
\]

如果局部坐标系就是标准 xyz，则：

\[
r^{\text{local}}_{ij} = (1, 3, -1)
\]

那么：

- 在 local 是 \( (0,0,0) \)
- 在 local 的 j 是 \( (1,3,-1) \)

这就是「邻居 j 的局部坐标」。

---

## ⭐ 5. 更一般地（局部坐标系是任意旋转的）

如果局部 frame\_i 是旋转矩阵 \( R \)：

\[
r^{\text{local}}_{ij} = R^\top r_{ij}
\]

理解：

- \( r_{ij} \) 是方向向量  
- \( R^\top \) 把它“重新表达”在 **局部坐标轴** 上  
- 得到的结果就是今天的坐标系的分量

仍然成立：  
📌 **local 坐标是同一个坐标表达，不是一个新的空间点。**

---

## ⭐ 6. “向量坐标”与“点坐标”之间只差一个原点

为什么 \( r_i^{\text{local}} \) 可以表示 j 的局部坐标？

因为：

- 向量坐标 + 局部原点 = 点坐标  
- 局部原点就是原子 i：\( (0,0,0) \)

因此：

📌 邻居 j 的局部坐标 = \( r_{ij} \) 在局部坐标系下的坐标分量。

---

## ✨ 最终总结（非常关键）

\( r_{ij} \) 是「相对」的向量，而每个局部坐标系的原点是定义在原子 i 上，  
因此，\( r_{ij} \) 在局部坐标系中的表示法（分量）自然就是原子 j 在该局部坐标系下的坐标。

也就是说：

\[
x^{\text{local}}_j = r^{\text{local}}_{ij}
\]

这就是为什么“坐标差”能被称为“局部坐标”的根本原因。

---

## 8. 为啥是等变呢，推导出来的不应该是不变吗

- ✔ local 不变
-  ✔ global 等变
- ✔ 完全符合 SE(3)-等变定义

 

你指出“推导有问题”是完全正确的。  
我来仔细校对一下刚才的推导，并给出严格、无误、标准的等变推导。

你会看到：

- ✔️ 前面的推理总体是 **概念对的**（local 不变、global 等变）
- ✖️ 但我写的符号表达方式确实不够严谨，容易造成误解

现在我给你最正确、最标准的版本。

---

## 🧠 先把关键变量彻底定义清楚（很多混乱来自于符号定义不清楚）

我们定义：

- \( r_{ij} \)：全局坐标差向量
- \( F_i \)：从 local → global 的局部坐标系矩阵（即 frame\_i 的坐标轴）
- \( F_i^\top \)：从 global → local 的投影矩阵
- \( v_i^{\text{local}} \)：模型内部使用的特征（local frame 下表达）
- \( v_i^{\text{global}} = F_i v_i^{\text{local}} \)：把 local 向量恢复到 global 表达

全局旋转 \( R \)（关于原点）作用于坐标：

- 向量旋转：\( r'_{ij} = R r_{ij} \)
- frame 旋转：\( F_i' = R F_i \)

这些都属于 SE(3)/SO(3) 变换规则，没有争议。

---

## ◆ 第一步：local 投影确实“不变”

\[
r_{ij}^{\text{local}} = F_i^\top r_{ij}
\]

对旋转后的输入：

\[
r'_{ij} = R r_{ij}
\]
\[
F_i' = R F_i
\]

局部投影为：

\[
(r_{ij}^{\text{local}})' = (F_i')^\top r'_{ij} 
= (R F_i)^\top (R r_{ij})
= F_i^\top R^\top R r_{ij}
= F_i^\top r_{ij}
= r_{ij}^{\text{local}}
\]

📌 **数学无误**  
📌 **局部投影确实不变**  
📌 **这是 frame-based 方法的核心性质之一**

---

## ◆ 第二步（关键）：模型内部的向量是如何变换的？

网络内部计算得到的是：

\[
v_i^{\text{local}} = \text{some function of } r_{ij}^{\text{local}}
\]

由于输入 \( r_{ij}^{\text{local}} \) 不变，得到的 \( v_i^{\text{local}} \) 也不变：

\[
(v_i^{\text{local}})' = v_i^{\text{local}}
\]

这一点正确且重要。

---

## ◆ 第三步：最终输出如何恢复到全局？（这里是重点）

恢复公式：

\[
v_i^{\text{global}} = F_i v_i^{\text{local}}
\]

旋转后的全局输出：

\[
(v_i^{\text{global}})' 
= F_i' (v_i^{\text{local}})' 
= (R F_i)(v_i^{\text{local}})
= R(F_i v_i^{\text{local}})
= R v_i^{\text{global}}
\]

这是核心：

## 📌 全局输出随着旋转一起旋转 → **等变性成立**

✔️ 这个推导是 100% 正确的  
✔️ 没有任何额外假设  
✔️ 是标准的 SE(3)-等变表示变换法则  

---

# 🎯 最终精炼总结（100% 严格正确）

1. **投影到 local frame（局部不变）**

\[
(r_{ij}^{\text{local}})'= r_{ij}^{\text{local}}
\]

2. **网络内部 local 特征不变**

\[
(v_i^{\text{local}})' = v_i^{\text{local}}
\]

3. **frame 自身随着输入旋转**

\[
F_i' = R F_i
\]

4. **把 local 恢复到 global 得到等变输出**

\[
(v_i^{\text{global}})'
= (R F_i)(v_i^{\text{local}})
= R(F_i v_i^{\text{local}})
= R v_i^{\text{global}}
\]

⭐ **等变性严格成立。**





## Spherical Frame Basis
