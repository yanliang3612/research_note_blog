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

(1) 选择一个中心原子 i

- 以它为局部坐标系的原点。

(2) 从邻居中选两个原子 j、k，定义方向向量：

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



(3) 归一化并构建正交坐标轴

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




## Spherical Frame Basis 


