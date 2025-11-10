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

