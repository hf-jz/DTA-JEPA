# DTA-JEPA v2：双时间尺度递归自适应潜在世界模型（完整方案）

> 本文档是 `docs/DTA-JEPA.md`（v1）的**全面重写**。v1 把"双时间尺度"定义为**参数**的双时间尺度
> （慢权重 + LoRA 快权重），把全部适应能力压在"测试时梯度步"这一个旋钮上。v2 的深化是：
> **把双时间尺度从参数层上升到"预测计算"层**——一个慢的动力学上下文模块 + 一个快的递归精细化模块，
> 在线适应只作用在快模块上；同时把**不确定性**从"一个正则项"升级为**资源分配信号**
> （决定梯度步数、学习率、递归深度、规划保守度、记忆写入、回滚判定）。
>
> 方案融合了四个参考项目的核心机制（下表），并在 AdaJEPA 的四个测试时分布偏移族上做了
> 完整的复现式验证（见 `results/`、论文第 4 章）。

---

## 0. 从 v1 到 v2 的差异总览

| 维度 | v1（原方案） | v2（本方案） |
|---|---|---|
| 双时间尺度载体 | 参数分组：θ_s（主干）+ θ_f（LoRA/末层） | **预测计算分组**：S 模块（慢，上下文）+ R 模块（快，递归精细化）；参数分组是其实现手段 |
| 预测器 | 单次前向 `(ẑ,Σ)=F(z,u,h)` | **递归精细化**：`y^(0)=c_t+u`；`y^(k)=y^(k-1)+g_k(...)`，k=1..K_max，**每层深度监督** |
| 计算量 | 固定 | **不确定性门控可变计算**：ID 1 层、OOD 最多 4 层；规划 roll-out 同样按需加深 |
| 不确定性 | 对角高斯 + ensemble/MC-dropout（并列） | **三层分解**：偶然（方差头）＋认知（R 模块 MC-dropout）＋惊讶（Mahalanobis 残差）；分别驱动不同决策 |
| 不确定性作用 | 规划惩罚项 + OOD 检测 | **资源分配器**：步数 U、学习率倍数、递归深度 K、规划权重 β、记忆优先级、回滚阈值 |
| 记忆 | 情景（像素级优先级队列）+ 参数记忆（LoRA 副本） | 情景（**潜空间**优先级队列，惊讶驱动）+ 参数记忆（按**环境指纹**检索的"技能"快参数） |
| 安全 | 锚点 + KL + 回滚 | 锚点球投影（θ_f）＋ **验证门控回滚**＋潜在支撑集惩罚（规划期）＋ **不可改的慢锚点**（自模型签名） |
| 训练目标 | 多项加权（预测+NLL+VICReg+逆动力学） | **LeWM 式两项**（多步潜预测 + 高斯潜正则 SIGReg）+ 深度监督 + 逆动力学；超参从 6 降到 3 |
| 慢参数整合 | EWC 重放更新 | EWC + **情景记忆潜空间重放**，且慢参数在部署期**不可在线改**（只在 episode 边界整合） |
| 元训练 | MAML（二阶） | **FOMAML**（一阶，显存/时间可负担）元学习快参数初始化 |
| 编码器 | 建议 ResNet/ViT，可用预训练 | **从零端到端**（Cerebro：不做预训练语义编码；LeWM：无 EMA、无预训练主干） |
| 目标融合 | 目标观测用同一编码器编码 | **预测后再融合**（Cerebro）：预测在潜空间完成，目标/动作/模态在代价或预测入口融合 |

### 四个参考项目 → v2 机制映射

| 来源 | 借鉴的核心机制 | 在 v2 中的落点 |
|---|---|---|
| **HRM / HRM-Text** | 两个耦合递归模块（高级慢模块 H / 低级快模块 L），慢模块以固定点为快模块提供上下文，快模块高频迭代 | S 模块（慢，产生动力学上下文 `c_t`）+ R 模块（快，递归精细化 `y^(k)`），二者以 `c_t` 强耦合；慢模块只在 episode 边界整合 |
| **TinyRecursiveModels (TRM)** | "少即是多"：小网络在隐空间递归改进答案 `y`，先更新隐状态 `z` 再更新答案，**深度监督**每一步，参数量极小 | R 模块在潜空间递归改进 `ŷ`；对每个递归深度施加监督（`w_k` 递增）；参数量 ~2M 即可，替代"堆大模型" |
| **LeWorldModel (LeWM)** | 端到端从像素稳定训练，只用两项损失（下一步嵌入预测 + 高斯潜正则）；surprise 检测物理不合理事件 | 预训练目标（SIGReg）；惊讶信号 `s_t` 作为不确定性与 OOD 的核心度量；单卡从零可训 |
| **Cerebro** | 全链路从原始信号端到端（无预训练语义编码器）；**融合发生在 JEPA 预测之后**；模态统一到同一潜维；自我概念/人生目标"签名保护、最高优先级、不可修改" | 从零训练；目标/动作/未来在预测与代价处融合而非提前拼接；统一潜维 `d`；**慢锚点 θ_s⁰ 为不可改签名**（安全层） |
| **AdaJEPA** | 闭环 plan→execute→adapt→replan；单步自监督适应即可显著提升 | 我们复现为基线并超越：把"单步固定适应"扩展为"不确定性驱动的分层适应 + 跨 episode 持续整合" |

---

## 1. 第一性原理（修订）

继承 JEPA 三条：

1. **潜空间预测**（不重建像素）；
2. **自监督**（监督来自未来帧的嵌入，stop-gradient 目标）；
3. **防坍塌**（潜分布正则 + 逆动力学 + 目标分离）。

新增四条（v1 三条 + v2 一条）：

4. **双时间尺度**：快模块适应当前环境，慢模块承载通用动力学且在线**不可改**；
5. **不确定性驱动资源分配**：不确定性同时决定"算多少"、"学多少"、"信多少"；
6. **安全可回滚**：快参数约束在锚点球内，验证变差即回滚；
7. **（v2 新增）递归精细化优先于参数扩张**：性能来自"在潜空间多想几步"，而不是把网络堆大
   （TRM 的核心主张）；因此 OOD 时的默认响应是**加深递归**，而非无条件调大学习率。

---

## 2. 架构总览

```
        o_{t-K+1..t}                     a_t
             │                            │
        ┌────▼────┐                  ┌────▼────┐
        │  E_s    │  编码器（从零）   │  E_a    │  动作编码
        │ body+head│  proprio 融合于 head│        │
        └────┬────┘                  └────┬────┘
             │ z_{t-K+1..t} (每个 d 维)     │ u_t
             ▼                            ▼
   ┌───────────────────────────────────────────────────┐
   │  S 模块（慢，θ_s：episode 间整合，在线冻结）         │
   │   token_i = W[z_i ; u_i]，2×TransformerBlock → c_t   │
   └───────────────┬───────────────────────────────────┘
                   │ c_t（动力学上下文）
   ┌───────────────▼───────────────────────────────────┐
   │  R 模块（快，θ_f：在线适应，LoRA + 头）             │
   │   y⁰ = c_t + u_t                                   │
   │   for k = 1..K_t:                                  │
   │       y^k = y^{k-1} + g_θ([y^{k-1}, c_t] + u_t)    │
   │   （K_t 由不确定性门控；每层 μ 头 + logσ² 头）      │
   └───────┬───────────────────────┬───────────────────┘
           │ ŷ_{t+1}, Σ           │ 递归轨迹 {y^k}
           ▼                       ▼
     规划代价 C(a)            深度监督 / 停机判据
           │
   ┌───────▼───────────────────────────────────────────┐
   │  双记忆：M_e（潜空间情景记忆，惊讶优先）            │
   │          M_p（按环境指纹检索的快参数技能库）        │
   │  安全层：锚点球投影 · 验证门控回滚 · 支撑集惩罚     │
   │  控制器：惊讶 s_t → (步数 U, 学习率倍数, 深度 K)    │
   └───────────────────────────────────────────────────┘
```

---

## 3. 编码器（从零端到端）

状态编码器 `E_s = head ∘ body`：

- `body`：4 段卷积（stride 2，GroupNorm + GELU），64×64×3 → 128×4×4；
- 本体感受 `prop`（位置/速度）经 MLP 得到 64 维，与 `body` 特征拼接；
- `head`：Linear(256+64→256) → GELU → Linear(256→d)，`d = 128`（**所有模态统一潜维**）。

\[
z_t = E_s(o_t, p_t) = \mathrm{head}\big([\,\mathrm{body}(o_t)\,;\,\mathrm{MLP}(p_t)\,]\big),\qquad
u_t = E_a(a_t)
\]

设计取舍：

- **不让 proprio 包含被操作物体位姿**：否则视觉扰动实验失去意义；物体几何必须从像素推断；
- **head 是"快"的**（在线可更新），body 是"慢"的（仅 episode 间整合）；
- 不加载 DINOv2 等预训练主干（Cerebro 路线 + LeWM 路线的稳定训练配方），
  使全部模块端到端可训、单机可复现。

---

## 4. 双时间尺度递归预测器（核心创新）

### 4.1 慢模块 S：动力学上下文

历史 `z_{t-K+1:t}` 与动作 `u_{t-K+1:t}`（K=3）逐帧拼接成 K 个 token：

\[
x_i = W_s[z_i;u_i],\quad (h_1,\dots,h_K)=\mathrm{Transformer}_{2L}(x_{1:K}),\quad c_t = h_K
\]

`c_t` 是"当前环境下这一步动力学应该长什么样"的**上下文/吸引子**。S 模块只承载通用动力学，
在部署期**冻结**（v2 的关键：在线绝不改慢模块，见 §7 安全层）。

### 4.2 快模块 R：递归精细化（TRM 式）

\[
y^{(0)} = c_t + u_t,\qquad
y^{(k)} = y^{(k-1)} + g_{\theta_f}\big([y^{(k-1)},c_t]+u_t\big),\quad k=1,\dots,K_t
\]

- 每个深度都输出均值头与 log 方差头：`ŷ^{(k)} = W_\mu \mathrm{LN}(g^{(k)})`，`logσ²^{(k)}`；
- **深度监督**：训练时对每个 k 计算 `‖ŷ^{(k)} − sg(z_{t+1})‖²`，权重 `w_k` 递增（0.5→1.0）；
- **不确定性门控停机**：当相邻深度的相对变化 `‖y^k − y^{k-1}‖/‖y^{k-1}‖ < ε` 时停止，
  记录停机深度 `k*`；部署期 `K_t` 由控制器给上界（1..4）。

这一设计的直接好处：

1. **同参数下更深的"思考"**：OOD 时多迭代几步即可降低预测误差（消融 `fixedk` 验证）；
2. **可变计算**：ID 环境只用 1 层，代价 ~1/4；
3. **快模块很小**（单层 Transformer + LoRA，~300k 参数），在线适应代价极低。

### 4.3 多步 roll-out

规划期用同一模块迭代：`z_{t+1}^hat → 作为新历史 → …`，每步都带自己的递归精细化。
多步预测损失同样带 stop-gradient 目标。

### 4.4 逆动力学辅助头

\[
\hat a_t = G_\omega([z_t;\,\mathrm{sg}(z_{t+1})]),\qquad \mathcal L_{\text{inv}} = \|\hat a_t - a_t\|^2
\]

作用：把动作信息压入潜空间（防止"动作无关"的退化表示），并为动作条件预测提供额外梯度。

---

## 5. 不确定性：三层分解 + 惊喜度量

### 5.1 偶然不确定性（数据噪声）

方差头输出对角高斯：`p(ẑ|·) = N(μ^{(K)}, diag(σ²^{(K)}))`，
损失为该步高斯负对数似然：

\[
\mathcal L_{\text{unc}} = \tfrac12 \big[(\hat y - \mathrm{sg}(z))^{\top}\Sigma^{-1}(\hat y-\mathrm{sg}(z)) + \log\det\Sigma\big]
\]

### 5.2 认知不确定性（知识空白）

R 模块内置 dropout（p=0.1），推理时开启 MC-dropout 采样 m=4 次：

\[
\Sigma_{\text{epi}} = \mathrm{Var}_{m}\big[\hat y_m\big]
\]

只在需要时计算（控制器触发），避免每步开销。

### 5.3 惊讶 / OOD 度量（LeWM 的 surprise）

\[
s_t = (\hat y-\mathrm{sg}(z_{t+1}))^{\top}\big(\Sigma_{\text{al}} + \Sigma_{\text{epi}}\big)^{-1}(\hat y-\mathrm{sg}(z_{t+1}))
\]

即"在当前模型看来这件事有多不可能"。它同时充当：适应强度信号、记忆写入优先级、回滚判据、
规划保守度来源。**利用率上，`s_t` 是整个系统的中枢信号。**

### 5.4 校准评估

论文中报告 ECE：把 `tr(Σ)` 分桶 vs 实际平方误差，检验"高不确定 ⇒ 确实高误差"。
在 OOD 条件下若校准良好，说明 Σ 是可用的资源分配依据（这是 v2 相对 v1 的可验证增益）。

---

## 6. 双记忆系统

### 6.1 情景记忆 M_e（潜空间优先级队列）

存潜空间转移 `(z,u,z')`（不存像素，省显存且与当前编码器解耦）：

\[
\text{prio} = \lambda_1\,s_t + \lambda_2\,\mathrm{unc}_t + \lambda_3\,\mathrm{novelty}(z_t)
\]

- `novelty(z) = exp(−min_m ‖z−z_m‖²/τ)`，与已有记忆的距离；
- 容量固定（默认 256），按最小优先级替换（保留"最该学的"经历）；
- 适应时按高斯核相似度检索 top-k，其预测损失以 `λ_ret` 加权加入在线损失：

\[
\mathcal L_{\text{on}} = \mathcal L_{\text{ada}}(\mathcal B_t) + \lambda_{\text{ret}}\,\mathcal L_{\text{ada}}(\mathcal M_{\text{ret}})
\]

### 6.2 参数记忆 M_p（技能库 / 环境指纹）

把快参数 `θ_f` 的副本按**环境指纹**（episode 前若干帧潜均值的向量）入库存档；
新 episode 开始时检索最近邻指纹并**热启动** `θ_f`。

\[
\theta_f^{(0)} = \operatorname*{arg\,min}_{\theta_f^{(j)}\in\mathcal M_p} \|\,\mathrm{fp}(\text{episode})-\mathrm{fp}^{(j)}\|_2
\]

意义：多环境切换时不必从零适应；A→B→A 场景下回到 A 时几乎瞬时恢复（见 §10 持续适应评估）。

---

## 7. 自适应更新控制器与安全层

### 7.1 控制器：不确定性 → 资源

\[
r_t = \frac{s_t}{\bar s_t}\ (\text{EMA 基线}),\qquad
\pi = \sigma\big(\gamma(r_t-1)\big) \in (0,1)
\]

\[
U_t = \lfloor U_{\min} + \pi (U_{\max}-U_{\min}) \rceil,\quad
\eta_t = \eta_{\min} + \pi(\eta_{\max}-\eta_{\min}),\quad
K_t = \lfloor K_{\min} + \pi (K_{\max}-K_{\min}) \rceil
\]

默认 `U∈[1,3]`、`η∈[0.2,5]×5e-4`、`K∈[1,4]`。ID 场景 π≈0 ⇒ 与 AdaJEPA 同等便宜；
OOD 场景 π→1 ⇒ 自动加大步数/学习率/递归深度。

### 7.2 安全层（四道闸）

1. **锚点球投影**（θ_f 不许跑远）：
   \[
   \|\theta_f-\theta_f^0\|_2 \le \epsilon_f,\qquad \theta_f \leftarrow \theta_f^0 + \frac{\epsilon_f}{\|\theta_f-\theta_f^0\|}(\theta_f-\theta_f^0)
   \]
2. **验证门控回滚**：用"除最新一条以外的缓冲"作验证集，
   \[
   \mathcal L_{\text{val}}^{\text{new}} > \mathcal L_{\text{val}}^{\text{old}} + \varepsilon \;\Rightarrow\; \theta_f \leftarrow \theta_f^{\text{old}},\ \text{记一次 rollback}
   \]
3. **慢锚点不可改（签名保护，Cerebro 路线）**：`θ_s`（编码器 body + S 模块）在 episode 内**不接受任何梯度**；
   只在 episode 边界的整合步骤中、以极小学习率 `η_s=1e-6` 并带 EWC 惩罚更新；
4. **潜在支撑集惩罚（规划期）**：代价函数中对偏离训练潜分布的想象轨迹加罚：
   \[
   C(a)=\sum_k\Big[\|y_k-z_g\|^2 + \beta\,\mathrm{tr}\Sigma_k + \gamma\,\max\big(0,\ \mathrm{Maha}(y_k)-\tau_{0.95}\big)\Big]
   \]
   `τ_{0.95}` 取训练集潜 Mahalanobis 距离的 95 分位（训练时统计并存档）。

---

## 8. 训练三阶段

### 阶段一：预训练（LeWM 式两项 + 深度监督 + 逆动力学）

\[
\mathcal L=\underbrace{\mathcal L_{\text{pred}}^{(1)}+\sum_{k}w_k\mathcal L_{\text{pred}}^{(k)}}_{\text{单步深度监督 + 多步 roll-out}}
+\lambda_{\text{unc}}\mathcal L_{\text{unc}}
+\lambda_{\text{reg}}\,\mathrm{SIGReg}(z)
+\lambda_{\text{inv}}\mathcal L_{\text{inv}}
\]

其中 SIGReg（LeJEPA 的切片高斯检验）用 Epps–Pulley 统计量在 M 个随机一维投影上检验
潜分布是否为 N(0,1)；这是**唯一**的防坍塌超参（对比 v1 的 VICReg+stop-grad+逆动力学三项叠加）。

实现要点（踩过的坑）：

- SIGReg 在潜几乎塌缩（std≈0.01）时梯度 ∝ x，恢复很慢：需要 ~150 步才把 `lat_std` 抬到 1；
  因此 `λ_reg` 与 batch 大小要配平（我们用 1.0 / batch 48）；
- `nn.MultiheadAttention` 会直接读 `out_proj.weight`，给它加 LoRA 必须把 `weight` 暴露为
  **组合权重属性**（`W + scale·BA`），否则属性缺失报错。

### 阶段二：元训练（FOMAML 学快参数初始化）

任务分布：push 用"留一形状"（4 个任务）；maze 用 5 组布局分组。每任务取 support/query 段：

\[
\theta_f' = \theta_f^0-\eta_f\nabla_{\theta_f}\mathcal L_{\text{ada}}(\mathcal S_\tau),\qquad
\theta_f^0 \leftarrow \theta_f^0-\eta_{\text{meta}}\nabla_{\theta_f^0}\mathcal L_{\text{ada}}(\mathcal Q_\tau;\theta_f')
\]

一阶近似（FOMAML）：内循环后直接反向，不做二阶 Hessian。
目标：让 **1 步**梯度就能把 `θ_f` 调到"新环境可用"的位置（低数据/强偏移下的增益来源）。

### 阶段三：部署（在线闭环）

```
输入：θ_s^0（不可改锚点）、θ_f^0（元初始化）、M_e、M_p、o_g、超参
θ_f ← Retrieve(M_p, fp(episode))            # 环境指纹热启动
z_g ← E_s(o_g)
for t = 0..T-1:
    # 1) 规划（带不确定性与支撑集惩罚；K_t 由控制器给）
    y_{1:H} ← Rollout(E_s(o_t), a_{t:t+H-1}, K_t)
    a* ← argmin C(a)
    # 2) 执行一个 chunk
    o_{t+1} ← Env(a*_{t:t+4})
    B ← B ∪ {(o,a,o')}                        # recent-5
    # 3) 惊讶与不确定性
    s_t, Σ_al, Σ_epi ← Predict(z_t,u_t)
    # 4) 控制器 → 资源
    U,η,K ← Controller(s_t)
    M_ret ← TopK_{M_e}(sim(z_t,·))
    for u = 1..U:  θ_f ← θ_f − η∇_{θ_f}[L_ada(B) + λ_ret L_ada(M_ret)]
    Safety.project(θ_f); if L_val↑: rollback
    M_e.add(z_t,u_t,z_{t+1}, s_t)   # 优先级写入
# episode 结束（慢时间尺度）
M_p.add(fp, θ_f)
θ_s ← θ_s − η_s ∇_{θ_s}[L_ada(M_e) + λ_EWC Σ F_i(θ_{s,i}−θ^*_{s,i})²]
```

**注意**：每个 episode 内的 `θ_f` 独立演化（与 AdaJEPA 的评估协议一致）；
跨 episode 的持续增益由 `M_p`、`M_e` 与慢整合提供——这正是 v1 想做而 v2 用"慢模块不可在线改"
+ 指纹化技能库落地的部分。

---

## 9. 与 AdaJEPA 的形式化差异

| | AdaJEPA | DTA-JEPA v2 |
|---|---|---|
| 适应对象 | 预测器末层 + 编码器头（直接更新） | R 模块 LoRA + 编码器头（**与慢模块解耦**） |
| 步数/学习率 | 固定 1 步 / 训练同值 | 不确定性驱动 `U∈[1,3]`、`η∈[0.2,5]×` |
| 更新时机 | 每 MPC 步 | 每 MPC 步，但**强度自适应**；低惊讶时可跳过 |
| 记忆 | recent-5 像素缓冲（一次性） | recent-5 + 潜空间优先级记忆（跨 episode 持久） |
| 多环境 | 无 | 指纹化技能库热启动 |
| 安全性 | 无 | 锚点球 + 验证回滚 + 支撑集惩罚 |
| 预测计算 | 固定 | 递归深度 `K_t` 可变 |
| 长期学习 | 无（episode 间重置） | 慢参数整合 + EWC + 记忆重放 |
| 目标融合 | 编码器共享 | 预测后融合（模态无关接口） |

---

## 10. 评估协议（对齐 AdaJEPA，单机可复现）

### 10.1 环境与数据（自建，numpy 物理 + 软光栅，无外部依赖）

| 套件 | 任务 | 训练 | 测试（分布偏移） |
|---|---|---|---|
| shape | PushBlock 2D 接触推动（多聚方块 {T,L,Z,+,I,smallT,square}） | 4 形状 × 350 轨迹 | 7 形状（3 个未见：I/smallT/square） |
| visual | PushBlock(T) 视觉扰动 | T | blur / snp / dark / redAgent / redBlock / redAnchor |
| dyn | PointMaze 动力学 | 默认动力学（25 布局混合） | 质量 ×0.2 / 阻尼 ×20 |
| layout | PointMaze 布局 | 25 个 8×8 布局 × 16 轨迹 | 5 个留出布局（BFS 距离 3–5 的目标） |
| continual | 环境序列切换 | — | 形状序列 T→L→Z→+→T（每段多 episode，模型跨 episode 持续） |

评估协议细节（与 AdaJEPA 一致）：目标来自**同一轨迹的未来帧**（"goal_source=segments"，间隔 25 步），
过滤掉无接触/无位移的平庸片段；MPC 时域 H=15，每次执行 5 个动作，最多 10 次重规划；
成功率按重规划步数逐点统计。

### 10.2 方法对比

- **Frozen**：不适应；
- **AdaJEPA**：复现其"末层直接更新 + 1 步 + 固定学习率 + recent-5"；
- **DTA-JEPA（本方案）**：完整；
- **消融**：`nounc`（去不确定性驱动与不确定性代价）、`nomem`（去双记忆）、`nosafe`（去安全层）、
  `fixedk`（固定递归深度）、`noslow`（去慢整合/技能库）。

### 10.3 指标

成功率曲线、最终成功率、首次成功步数、每步预测误差（惊讶）、适应统计（步数/学习率/停机深度/
回滚次数/锚点投影次数）、墙钟延迟、ECE 校准误差、A→B→A 保留率、跨 episode 学习曲线。

---

## 11. 超参数

| 名称 | 符号 | 取值 |
|---|---|---|
| 潜维 | d | 128 |
| 历史长度 | K | 3 |
| 递归深度上限 | K_max | 4 |
| LoRA 秩 | r | 4 |
| 快学习率 / 编码器头学习率 | η_f / η_enc | 5e-4 / 1e-5 |
| 慢整合学习率 | η_s | 1e-6 |
| 更新步数范围 | [U_min,U_max] | [1,3] |
| 学习率倍数范围 | [η_min,η_max] | [0.2,5] |
| 递归深度范围 | [K_min,K_max] | [1,4] |
| recent 缓冲 | N | 5 |
| 情景记忆容量 | |M_e| | 256 |
| 技能库容量 | |M_p| | 8 |
| 锚点半径 | ε_f | 0.8 |
| 验证容忍 | ε | 0.05 |
| 检索损失权重 | λ_ret | 0.5 |
| 规划不确定性权重 | β | 0.5 |
| 规划支撑集权重 | γ | 0.5 |
| SIGReg / NLL / 逆动力学权重 | λ_reg / λ_unc / λ_inv | 1.0 / 0.1 / 0.1 |
| 规划 | H / chunk / 重规划 / GD 步数 / GD lr | 15 / 5 / 10 / 40 / 0.1 |

---

## 12. 实现映射（代码）

| 模块 | 文件 | 关键入口 |
|---|---|---|
| 2D 物理 + 软光栅 | `dtajepa/sim.py` | `PushSim`（冲量法圆-多聚方块接触）、`MazeSim`、`push_expert`、`maze_expert` |
| 环境 + 视觉扰动 | `dtajepa/envs.py` | `PushEnv`、`MazeEnv`、`render_push/render_maze`、`corrupt` |
| 数据生成 | `dtajepa/data.py` | `build_all`（5 套件） |
| 模型 | `dtajepa/models.py` | `Encoder`、`DualTimescalePredictor`（S/R 模块 + 递归精细化 + MC-dropout）、`WorldModel` |
| 目标函数 | `dtajepa/losses.py` | `sigreg`、`gaussian_nll`、`jepa_loss`（深度监督） |
| 预训练 | `dtajepa/train.py` | `train`、`latent_stats` |
| 元训练 | `dtajepa/meta.py`（待跑） | `fomaml` |
| 在线适应 | `dtajepa/adapt.py` | `EpisodicMemory`、`ParamMemory`、`SafetyLayer`、`Controller`、`Adapter`、`EWC` |
| 规划器 | `dtajepa/plan.py` | `gd_plan`、`cem_plan`、`cost_fn`、`MPCConfig` |
| 评估 | `dtajepa/evaluate.py` | `run_condition`、`suite_specs` |
| 结果聚合/作图 | `dtajepa/report.py` | 生成论文图表 |

---

## 13. 复杂度与成本

- 模型：~2.2M 参数（编码器 2.0M + 预测器 0.2M），其中**可在线适应参数 ~300k**（LoRA + 头）；
- 单次规划：一次 roll-out = K_t × (2 个 S 块 + K 次 R 块)；批量 1 时 CPU 毫秒级；
  实测单次重规划（H=15、40 步 GD）× 单 episode ≪ 1 s；
- 在线适应：1–3 步 GD，仅作用在 300k 参数上，含缓冲编码（5 帧）× 5 = 25 次编码；
- 记忆：情景记忆只存潜向量（256×3×128 float32 ≈ 400 KB），技能库 8×300k ≈ 10 MB。

相对 AdaJEPA 的额外成本主要在：OOD 时多出的梯度步与更深的递归（换来成功率）；
ID 时 π≈0，成本与 AdaJEPA 基本持平——这是"可变计算"设计的目的。

---

## 14. 风险与应对

| 风险 | 应对 |
|---|---|
| 递归精细化在训练早期无收益（深度监督失效） | 深度监督权重递增 + 停机判据用相对变化而非方差 |
| SIGReg 在早期塌缩区梯度极弱 | λ_reg=1.0 + batch≥48；训练日志监控 `lat_std` |
| 不确定性与误差不相关（校准差） | 报告 ECE；退化时退回"固定 1 步"安全默认（消融 nounc 即此配置） |
| 技能库指纹混淆（不同形状指纹相近） | 指纹取 episode 前 K 帧潜均值；容量小 + 最近邻；消融 noslow 验证其必要性 |
| LoRA 秩过小导致适应能力不足 | 消融 r∈{4,16}（v1 路线图中已有该消融） |
| 自建环境与 PushT/MuJoCo 存在 sim-to-sim 差异 | 论文明确声明为"协议级复现"：偏移族、目标采样、规划器、指标一致，物理引擎不同 |
