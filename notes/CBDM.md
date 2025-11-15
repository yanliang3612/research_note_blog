### Paper Reading Note of CBDM



## Classifier-Free Guidance (CFG)

In the sampling, the labelguided model estimates the noise with a linear interpolation $\hat{\epsilon} = (1 + \omega) \, \epsilon_\theta(x_t, y, t) - \omega \, \epsilon_\theta(x_t, t)$  to recover $x_{t-1}$ , which is often referred as Classifier-Free Guidance (CFG) [12].

---

## CBDM

- data distribution $q(x, y)$
- the joint distribution $p_\theta(x,y)
- true label distribution $q(y)$
- the prior label distribution $p_\theta(y)$

- the density ratio $r$  (密度比)

$r = \frac{q(x, y)}{p_\theta(x, y)} = \frac{q(x \mid y)}{p_\theta(x \mid y)} \cdot \frac{q(y)}{p_\theta(y)}$.

- $p_\theta^*(x_{t-1} \mid x_t, y)$ be the optimum trained in the case that 
  $\frac{q(y)}{p_\theta(y)}$ is correctly estimated
- $p_\theta(x_{t-1} \mid x_t, y)$ 
  be the one trained in a class-imbalanced case. 

- 模型无关部分 $\frac{q^*(x_t)}{q(x_t)}$.
- 通过进一步将 $p_\theta(x_{t-1})$ 和 $p_\theta^*(x_{t-1})$ %分解为条件概率 $p_\theta^*(x_{t-1}\mid x_{t:T}, y)$ 的期望，

- 被补偿的目标先验 $q_Y^*$ ，可以是均匀分布，也可以是true label distribution的开方分布

---

**Proposition 2.**  
For the adjusted loss  
$\mathcal{L}^{\star}_{\mathrm{DM}}=\sum_{t=1}^{T}\mathcal{L}^{\star}_{t-1}$,  
an upper bound of the target training objective to calibrate at timestep $t$
(i.e., $\mathcal{L}^{\star}_{t-1}$) can be derived as:

$$
\sum_{t\ge 1}\mathcal{L}^{\star}_{t-1}
= \sum_{t\ge 1}
D_{\mathrm{KL}}\!\left[
q(x_{t-1}\mid x_t,x_0)\; \big\|\; p_{\theta}^{\phi}(x_{t-1}\mid x_t,y)
\right]
$$

$$
\le \sum_{t\ge 1}\Bigg(
\underbrace{
D_{\mathrm{KL}}\!\left[
q(x_{t-1}\mid x_t,x_0)\; \big\|\; p_{\theta}(x_{t-1}\mid x_t,y)
\right]
}_{\text{Diffusion model loss }\mathcal{L}_{\mathrm{DM}}}
+
\underbrace{
t\,\mathbb{E}_{y'\sim q^{\star}_{y}}\!\left[
D_{\mathrm{KL}}\!\left[
p_{\theta}(x_{t-1}\mid x_t)\; \big\|\; p_{\theta}(x_{t-1}\mid x_t,y')
\right]
\right]
}_{\text{Distribution adjustment loss }\mathcal{L}_{r}}
\Bigg)
$$

---
## 两个Loss Term：CBDM 中 `L_r` 和 `L_rc` 的区别与意义

- Calculate the first regularization term



- Calculate the regularization commitment term

### 1. 两个 Loss 的形式

记
- \( a = \varepsilon_\theta(x_t^{(i)}, y^{(i)}) \)
- \( b = \varepsilon_\theta(x_t^{(i)}, y'^{(i)}) \)

忽略前面的系数 \(t\tau\)，两项分别是：

#### 正则项 \(L_r\)

\[
L_r = \left\|\,\varepsilon_\theta(x_t^{(i)}, y^{(i)}) 
      - \operatorname{sg}\big(\varepsilon_\theta(x_t^{(i)}, y'^{(i)})\big)\right\|^2
  = \|a - \operatorname{sg}(b)\|^2
\]

> 对 **\(y'\) 那一支**（即 \(b\)）做了 `stop-gradient`。

---

#### 承诺项 \(L_{rc}\)

\[
L_{rc} = \left\|\operatorname{sg}\big(\varepsilon_\theta(x_t^{(i)}, y^{(i)})\big) 
      - \varepsilon_\theta(x_t^{(i)}, y'^{(i)})\right\|^2
  = \|\operatorname{sg}(a) - b\|^2
\]

> 对 **\(y\) 那一支**（即 \(a\)）做了 `stop-gradient`。

---

### 2. 梯度流向的区别

### 对 \(L_r = \|a - \operatorname{sg}(b)\|^2\)

- 对 \(a\)（真实条件 \(y\) 分支）有梯度：
  \[
  \frac{\partial L_r}{\partial a} \propto (a - \operatorname{sg}(b))
  \]
- 对 \(b\)（伪条件 \(y'\) 分支）**没有梯度**。

👉 意味着：
> 只更新 **真实条件 \(y\)** 这一支的输出，  
> 让 \(a\) 向“冻结的” \(b\) 靠近；  
> \(b\) 在这一项里只是一个固定的 target / 老师。

---

#### 对 \(L_{rc} = \|\operatorname{sg}(a) - b\|^2\)

- 对 \(a\) 没有梯度。
- 对 \(b\) 有梯度：
  \[
  \frac{\partial L_{rc}}{\partial b} \propto (b - \operatorname{sg}(a))
  \]

👉 意味着：
> 只更新 **伪条件 \(y'\)** 这一支的输出，  
> 让 \(b\) 向“冻结的” \(a\) 靠近；  
> \(a\) 在这一项里是一个固定的 target / 老师。

---

### 3. 直觉上的意义

把同一网络在不同条件下的输出视为两“塔”：

### \(L_r\)：regularization term

- 约束 **真实条件 \(y\) 分支** 的输出不要和 \(y'\) 分支差太远。
- 因为 \(b\) 不反传梯度，\(b\) 可以看作相对稳定的“老师 / 参考”。
- 直觉：
  > 在保证基本扩散损失 \(L_{\text{DM}}\) 拟合噪声的前提下，  
  > 再轻轻把 \(y\) 分支往 \(y'\) 分支的预测拉一拉，  
  > 起到“**平滑/正则化条件依赖**”的作用。

---

#### \(L_{rc}\)：regularization **commitment** term

- 强制 **伪条件 \(y'\) 分支** 不要随便乱跑，要向真实条件 \(y\) 分支对齐。
- 类似 VQ-VAE 里的 “commitment loss”：学生必须“承诺”去贴近老师。
- 直觉：
  > 不仅要让 \(y\) 靠近 \(y'\)，  
  > 也要强迫 \(y'\) 真正学到与 \(y\) 一致的表示，  
  > 防止这个辅助分支学成一个无意义的、退化的表示。

---

#### 4. 为什么要两个一起用？

如果只用对称的 \(\|a - b\|^2\) 且两边都反传梯度：

- 为了减小这项损失，最简单的办法就是  
  **两边一起塌缩成一个与条件无关的常数输出**。

现在拆成带 `stop-gradient` 的两项 \(L_r + \gamma L_{rc}\)，再加上原始扩散损失：

\[
L_{\text{CBDM}} = L_{\text{DM}} + L_r + \gamma L_{rc}
\]

综合效果：

1. \(L_{\text{DM}}\)：保证每个条件都要预测对噪声（防止完全忽略条件）。  
2. \(L_r\)：正则真实条件分支，使 w.r.t. 条件的变化不过于剧烈。  
3. \(L_{rc}\)：约束辅助分支必须对齐真实分支，防止它乱学、退化。

> **最终：**  
> 既缓解了不同条件输出之间的差异过大问题（更平滑、更稳定），  
> 又避免了简单的条件坍缩解（输出与条件无关）。

---

