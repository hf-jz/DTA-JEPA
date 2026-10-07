
# DTA-JEPA：双时间尺度持续自适应潜在世界模型详细方案

---

## 一、总体设计哲学

### 1.1 三条第一性原理

从原始 JEPA 继承：

1. **潜在预测**：预测发生在表示空间，不重建像素；
2. **自监督**：监督信号来自数据自身；
3. **防坍塌**：表示不能退化成常数。

新增三条：

4. **双时间尺度**：快适应当前环境，慢整合长期经验；
5. **不确定性驱动**：预测不确定性指导适应强度与规划保守度；
6. **安全可回滚**：在线更新不破坏预训练锚点，异常时回滚。

### 1.2 一句话概括

> DTA-JEPA 是一个快慢双时间尺度、不确定性驱动、记忆增强、安全可回滚的持续自适应潜在世界模型。它在 episode 内快速适应当前环境，在 episode 间缓慢整合经验，在规划时惩罚不确定性，在更新时约束偏离锚点。

---

## 二、模型架构

### 2.1 编码器

**状态编码器 \(E_s\)**：

\[
z_t = E_s(o_t; \theta_s^E, \theta_f^E)
\]

- 慢参数 \(\theta_s^E\)：预训练得到，跨 episode 缓慢更新；
- 快参数 \(\theta_f^E\)：低秩适配器（LoRA）或最后阶段，episode 内快速更新。

结构建议：

- 视觉：ResNet 或 ViT，输出全局特征 + 空间特征；
- 低维状态：MLP；
- 快参数放在最后阶段或插入 LoRA。

**动作编码器 \(E_a\)**：

\[
u_t = E_a(a_t; \psi_s, \psi_f)
\]

- 动作空间连续时用 MLP；
- 离散时用 embedding table；
- 动作块时用 1D 卷积或 transformer。

### 2.2 不确定性感知预测器

\[
(\hat{z}_{t+1}, \Sigma_{t+1}) = F_\psi(z_t, u_t, h_t)
\]

- 输入：当前潜在状态 \(z_t\)、动作嵌入 \(u_t\)、历史 \(h_t\)；
- 输出：均值 \(\hat{z}_{t+1}\) 和协方差 \(\Sigma_{t+1}\)；
- 协方差参数化：对角或低秩 + 对角：

\[
\Sigma = \operatorname{diag}(\sigma^2) + LL^\top
\]

- 结构：Transformer 或 GRU，输出两个头：均值头、方差头；
- 快参数：最后 transformer block 或 LoRA。

### 2.3 逆动力学头（辅助）

\[
\hat{a}_t = G_\omega(z_t, z_{t+1})
\]

- 从相邻潜在状态预测动作；
- 增强动作条件表示；
- 帮助防止表示坍塌，提供额外自监督信号。

### 2.4 历史编码

\[
h_t = \operatorname{Enc}_h(z_{t-K:t}, u_{t-K:t-1})
\]

- K 帧历史；
- 简单方案：拼接或 GRU；
- 强方案：因果 transformer。

### 2.5 目标编码

\[
z_g = E_s(o_g)
\]

目标观测用同一状态编码器编码，共享参数。

---

## 三、双时间尺度参数化

### 3.1 参数分组

\[
\theta = \theta_s \oplus \theta_f
\]

| 参数组 | 符号 | 更新频率 | 作用 | 实现 |
|---|---|---|---|---|
| 慢参数 | \(\theta_s\) | episode 间 | 通用动力学 | 预训练主干 |
| 快参数 | \(\theta_f\) | episode 内 | 当前环境适应 | LoRA / 最后层 |

### 3.2 快参数设计

推荐 LoRA：

\[
W' = W_s + BA
\]

- \(W_s\)：慢权重，冻结；
- \(B \in \mathbb{R}^{d \times r}\)，\(A \in \mathbb{R}^{r \times k}\)，秩 \(r \ll \min(d,k)\)；
- 只更新 \(A, B\)。

好处：

- 参数量小，计算轻；
- 不破坏预训练权重；
- 可存储多个 LoRA 模块对应不同环境模式。

备选：

- 只更新最后 transformer block + LayerNorm；
- 只更新编码器最后阶段 + 预测器最后 block；
- FiLM 调制：\(\gamma, \beta\) 由快参数生成。

### 3.3 慢参数更新

episode 结束后，用重放 + 正则更新：

\[
\theta_s \leftarrow \theta_s - \eta_s \nabla_{\theta_s} \mathcal{L}_{\text{consolidate}}
\]

其中：

\[
\mathcal{L}_{\text{consolidate}} = \mathcal{L}_{\text{ada}}(\mathcal{M}_e) + \lambda_{\text{EWC}} \sum_i F_i (\theta_{s,i} - \theta_{s,i}^*)^2
\]

- \(\mathcal{M}_e\)：情景记忆；
- \(F_i\)：Fisher 信息；
- \(\theta_{s,i}^*\)：上个 episode 的慢参数。

---

## 四、不确定性建模

### 4.1 预测分布

\[
p(z_{t+1} | z_t, u_t, h_t) = \mathcal{N}(\hat{z}_{t+1}, \Sigma_{t+1})
\]

### 4.2 适应损失（高斯 NLL）

\[
\mathcal{L}_{\text{ada}} = \frac{1}{2} e^\top \Sigma^{-1} e + \frac{1}{2} \log \det \Sigma
\]

其中 \(e = \hat{z}_{t+1} - \mathrm{sg}(z_{t+1})\)。

### 4.3 不确定性分解

\[
\Sigma = \Sigma_{\text{aleatoric}} + \Sigma_{\text{epistemic}}
\]

- **偶然不确定性**：数据噪声，由方差头输出；
- **认知不确定性**：模型知识不足，用 ensemble 或 MC dropout 估计。

实用方案：

- 训练多个预测器头，取方差作为认知不确定性；
- 或单模型 + MC dropout。

### 4.4 OOD 检测

\[
\text{OOD}(z_t, u_t) = \alpha \cdot \operatorname{tr}(\Sigma_{\text{epi}}) + \beta \cdot \|\hat{z}_{t+1} - z_{t+1}\|^2 + \gamma \cdot d_{\text{latent}}(z_t, \mathcal{D}_{\text{train}})
\]

- 第一项：认知不确定性；
- 第二项：预测误差；
- 第三项：潜在空间距离训练分布。

---

## 五、记忆系统

### 5.1 情景记忆 \(\mathcal{M}_e\)

存储高价值转移：

\[
\mathcal{M}_e = \{(o_i, a_i, o_{i+1}, \text{priority}_i)\}
\]

优先级：

\[
\text{priority}_i = \alpha \cdot \text{pred\_error}_i + \beta \cdot \text{uncertainty}_i + \gamma \cdot \text{novelty}_i
\]

- 预测误差大：需要学习；
- 不确定性高：知识空白；
- 新奇度高：潜在空间远离已有记忆。

容量：固定大小，优先队列或 reservoir sampling。

### 5.2 参数记忆 \(\mathcal{M}_p\)

存储多个快参数副本：

\[
\mathcal{M}_p = \{(\theta_f^{(1)}, \text{context}^{(1)}), (\theta_f^{(2)}, \text{context}^{(2)}), \dots\}
\]

- 每个副本对应一种环境模式；
- context 可用环境指纹、任务描述、潜在统计量表示；
- 新 episode 开始时，检索最匹配的快参数初始化：

\[
\theta_f^{(0)} = \operatorname{Retrieve}(\mathcal{M}_p, \text{context})
\]

检索方式：

- 最近邻；
- 注意力加权；
- 聚类中心。

### 5.3 记忆检索

适应时，从 \(\mathcal{M}_e\) 检索相关转移：

\[
\mathcal{M}_{\text{retrieved}} = \operatorname{TopK}_{(o,a,o') \in \mathcal{M}_e} \operatorname{sim}(z_t, z_{\text{mem}})
\]

相似度：

\[
\operatorname{sim}(z_t, z_m) = \exp(-\|z_t - z_m\|^2 / \tau)
\]

适应损失：

\[
\mathcal{L}_{\text{online}} = \mathcal{L}_{\text{ada}}(\mathcal{B}_t) + \lambda_{\text{ret}} \mathcal{L}_{\text{ada}}(\mathcal{M}_{\text{retrieved}})
\]

---

## 六、自适应更新控制器

### 6.1 更新强度

\[
s = \sigma(\alpha \cdot \text{OOD} + \beta \cdot \text{unc} + \gamma \cdot \text{pred\_error})
\]

- \(s \in [0, 1]\)：更新强度；
- 映射到更新步数：

\[
U = \lfloor U_{\min} + s \cdot (U_{\max} - U_{\min}) \rfloor
\]

- 映射到学习率：

\[
\eta = \eta_{\min} + s \cdot (\eta_{\max} - \eta_{\min})
\]

### 6.2 更新层选择

| OOD 程度 | 更新范围 |
|---|---|
| 低 | 不更新 |
| 中 | 预测器最后 block + 编码器最后阶段 |
| 高 | 预测器最后两个 block + 编码器最后两个阶段 + LoRA |
| 极高 | 全部 LoRA + 最后 block |

### 6.3 更新频率

- 每步都检测 OOD；
- OOD 低于阈值：跳过更新；
- OOD 高于阈值：更新；
- 连续高 OOD：增加更新强度。

### 6.4 更新后验证

更新后，在保留验证集 \(\mathcal{B}_{\text{val}}\) 上检查：

\[
\mathcal{L}_{\text{val}}^{\text{new}} < \mathcal{L}_{\text{val}}^{\text{old}} + \epsilon
\]

若不满足，回滚：

\[
\theta_f \leftarrow \theta_f^{\text{old}}
\]

---

## 七、安全层

### 7.1 锚点约束

预训练参数 \(\theta_0\) 作为锚点：

\[
\|\theta_f - \theta_f^0\|_2 \le \epsilon_f
\]

或逐参数：

\[
|\theta_{f,i} - \theta_{f,i}^0| \le \epsilon_i
\]

### 7.2 KL 约束

\[
D_{\text{KL}}\left(p_\theta(z_{t+1}|z_t,u_t) \| p_{\theta_0}(z_{t+1}|z_t,u_t)\right) \le \delta
\]

防止预测分布偏离太远。

### 7.3 安全规划约束

规划代价加入安全项：

\[
C = \sum_{k=1}^H \alpha_k \left[ d(\hat{z}_{t+k}, z_g) + \beta \operatorname{tr}(\Sigma_{t+k}) + \gamma \mathbb{1}[\hat{z}_{t+k} \notin \mathcal{S}_{\text{safe}}] \right]
\]

- \(\mathcal{S}_{\text{safe}}\)：安全潜在区域；
- 可用训练数据潜在分布定义，如马氏距离阈值。

### 7.4 回滚机制

触发条件：

- 验证损失变差；
- 不确定性激增；
- 预测误差持续高；
- 潜在状态偏离训练流形。

回滚操作：

\[
\theta_f \leftarrow \theta_f^0
\]

并记录事件，降低后续更新强度。

---

## 八、训练流程

### 8.1 阶段一：预训练

**数据**：无奖励离线轨迹 \(\mathcal{D}_{\text{off}} = \{(o_t, a_t, o_{t+1})\}\)。

**损失**：

\[
\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{pred}} + \lambda_1 \mathcal{L}_{\text{unc}} + \lambda_2 \mathcal{L}_{\text{reg}} + \lambda_3 \mathcal{L}_{\text{inv}}
\]

各项：

- 多步潜在预测：

\[
\mathcal{L}_{\text{pred}} = \frac{1}{K}\sum_{k=1}^K \ell(\hat{z}_{t+k}, \mathrm{sg}(z_{t+k}))
\]

- 不确定性 NLL：

\[
\mathcal{L}_{\text{unc}} = \frac{1}{2} e^\top \Sigma^{-1} e + \frac{1}{2} \log \det \Sigma
\]

- 防坍塌正则（VICReg）：

\[
\mathcal{L}_{\text{reg}} = \lambda_v \mathcal{L}_{\text{var}} + \lambda_c \mathcal{L}_{\text{cov}}
\]

- 逆动力学：

\[
\mathcal{L}_{\text{inv}} = \|\hat{a}_t - a_t\|^2
\]

**防坍塌组合**：

- stop-gradient 目标分支；
- VICReg 方差-协方差正则；
- 逆动力学辅助任务。

### 8.2 阶段二：元训练

**目标**：学习快参数初始化 \(\theta_f^0\)，使其能通过少量梯度步适应新环境。

**方法**：MAML 式元学习。

对每个任务 \(\tau\)：

1. 采样 support set \(\mathcal{S}_\tau\) 和 query set \(\mathcal{Q}_\tau\)；
2. 内循环：

\[
\theta_f' = \theta_f^0 - \eta_f \nabla_{\theta_f} \mathcal{L}_{\text{ada}}(\mathcal{S}_\tau; \theta_f^0)
\]

3. 外循环：

\[
\theta_f^0 \leftarrow \theta_f^0 - \eta_{\text{meta}} \nabla_{\theta_f^0} \mathcal{L}_{\text{ada}}(\mathcal{Q}_\tau; \theta_f')
\]

**任务分布**：

- 不同形状、视觉偏移、动力学、布局；
- 每个任务是一个 episode 或短轨迹。

**慢参数**：元训练时冻结或小学习率更新。

### 8.3 阶段三：部署

每个 episode：

1. 从 \(\mathcal{M}_p\) 检索快参数初始化；
2. 执行 **plan-execute-adapt-consolidate-replan** 循环；
3. episode 结束后，整合经验到慢参数和记忆。

---

## 九、在线算法

```text
算法：DTA-JEPA 在线持续自适应

输入：
  慢参数 θ_s = (θ_s^E, ψ_s, ω_s)
  快参数初始化 θ_f^0
  目标观测 o_g
  记忆 M_e, M_p
  超参：U_min, U_max, η_min, η_max, ε_f, δ, τ

初始化：
  θ_f ← Retrieve(M_p, context)
  B ← ∅
  z_g ← E_s(o_g)

for t = 0, 1, ..., T-1:

    # ===== 1. 规划 =====
    z_t ← E_s(o_t; θ_s^E, θ_f^E)
    h_t ← Enc_h(z_{t-K:t}, u_{t-K:t-1})
    a*_{t:t+H-1} ← MPC_Plan(z_t, z_g, h_t; F_ψ, Σ)
        # 代价：d(ẑ, z_g) + β tr(Σ) + γ 安全惩罚

    # ===== 2. 执行 =====
    执行第一个动作 a_t
    观察 o_{t+1}
    z_{t+1} ← E_s(o_{t+1}; θ_s^E, θ_f^E)
    B ← B ∪ {(o_t, a_t, o_{t+1})}
    if |B| > N: 按优先级裁剪

    # ===== 3. OOD 与不确定性 =====
    (ẑ_{t+1}, Σ_{t+1}) ← F_ψ(z_t, u_t, h_t)
    e ← ẑ_{t+1} - sg(z_{t+1})
    unc ← tr(Σ_{t+1})
    ood ← α·unc + β·||e||² + γ·d_latent(z_t, D_train)

    # ===== 4. 自适应更新 =====
    if ood > threshold_ood:
        s ← σ(α·ood + β·unc + γ·||e||²)
        U ← floor(U_min + s·(U_max - U_min))
        η ← η_min + s·(η_max - η_min)

        # 检索相关记忆
        M_ret ← TopK_{m ∈ M_e} sim(z_t, z_m)

        # 更新
        for u = 1, ..., U:
            L ← L_ada(B) + λ_ret · L_ada(M_ret)
            θ_f ← θ_f - η ∇_{θ_f} L

        # 安全检查
        if ||θ_f - θ_f^0|| > ε_f or KL(p_θ || p_{θ_0}) > δ:
            θ_f ← θ_f^0   # 回滚
            log_event("rollback")

        # 验证
        if L_val(θ_f) > L_val(θ_f_old) + ε:
            θ_f ← θ_f_old   # 回滚

    # ===== 5. 重规划 =====
    # 用更新后的模型继续下一轮

# ===== 6. Episode 结束：整合 =====
# 存入高价值转移
for (o, a, o') in B:
    priority ← α·pred_error + β·uncertainty + γ·novelty
    if priority > threshold_mem:
        M_e ← M_e ∪ {(o, a, o', priority)}

# 更新慢参数
for batch in M_e:
    L_consolidate ← L_ada(batch) + λ_EWC · Σ_i F_i (θ_{s,i} - θ_{s,i}^*)²
    θ_s ← θ_s - η_s ∇_{θ_s} L_consolidate

# 存储快参数
M_p ← M_p ∪ {(θ_f, context)}
```

---

## 十、超参数建议

| 超参数 | 符号 | 建议值 | 说明 |
|---|---|---|---|
| 快学习率 | \(\eta_f\) | \(10^{-4}\) ~ \(5 \times 10^{-4}\) | 与训练一致 |
| 慢学习率 | \(\eta_s\) | \(10^{-5}\) ~ \(10^{-6}\) | 更小，防遗忘 |
| 元学习率 | \(\eta_{\text{meta}}\) | \(10^{-4}\) | MAML 外循环 |
| 更新步数范围 | \([U_{\min}, U_{\max}]\) | [1, 5] | OOD 驱动 |
| 学习率范围 | \([\eta_{\min}, \eta_{\max}]\) | [0.2×, 5×] 训练率 | OOD 驱动 |
| Buffer 大小 | \(N\) | 5 ~ 20 | recent + priority |
| 记忆容量 | \(|\mathcal{M}_e|\) | 1000 ~ 10000 | 优先队列 |
| LoRA 秩 | \(r\) | 4 ~ 16 | 快参数 |
| EWC 强度 | \(\lambda_{\text{EWC}}\) | \(10^2\) ~ \(10^4\) | 防遗忘 |
| KL 约束 | \(\delta\) | 0.01 ~ 0.1 | 安全 |
| 锚点约束 | \(\epsilon_f\) | 0.1 ~ 1.0 | 安全 |
| 不确定性权重 | \(\beta\) | 0.1 ~ 1.0 | 规划 |
| 安全权重 | \(\gamma\) | 1.0 ~ 10.0 | 规划 |
| 历史长度 | \(K\) | 3 ~ 5 | |
| 规划时域 | \(H\) | 10 ~ 30 | |
| 执行动作块 | — | 1 ~ 5 | |

---

## 十一、评估方案

### 11.1 评估维度

| 维度 | 指标 | 说明 |
|---|---|---|
| 分布内 | 成功率 | 与 AdaJEPA 对比 |
| 分布外 | 形状/视觉/动力学/布局成功率 | 四类偏移 |
| 持续适应 | 多 episode 成功率曲线 | 是否越跑越好 |
| 多环境切换 | A→B→A 保留率 | 是否遗忘 |
| 安全性 | 危险动作率、回滚次数 | 安全层有效性 |
| 计算效率 | 每步延迟、内存、总时间 | 实际部署 |
| 表示质量 | 潜在漂移、解码一致性 | 是否坍塌 |
| 数据效率 | 低数据适应增益 | 样本效率 |
| 长时域 | 远目标成功率 | 长时规划 |
| 真实世界 | 机器人/自动驾驶 | 仿真到现实 |

### 11.2 对比基线

- 冻结 JEPA；
- AdaJEPA；
- Tent / TTT；
- 在线 MBRL（TD-MPC2、Dreamer）；
- AdaWM；
- Parthasarathy et al. 训练时数据合成；
- 无不确定性消融；
- 无双时间尺度消融；
- 无记忆消融；
- 无安全消融。

### 11.3 消融实验

| 消融 | 目的 |
|---|---|
| 去掉慢参数整合 | 验证持续学习必要性 |
| 去掉快参数 | 验证在线适应必要性 |
| 去掉不确定性 | 验证不确定性建模价值 |
| 去掉情景记忆 | 验证长期记忆价值 |
| 去掉参数记忆 | 验证多环境检索价值 |
| 去掉安全层 | 验证安全约束价值 |
| 去掉元训练 | 验证初始化价值 |
| 固定更新步数 | 验证自适应控制器价值 |
| 不同 LoRA 秩 | 验证快参数容量 |
| 不同记忆容量 | 验证记忆规模 |

---

## 十二、预期收益与风险

### 12.1 预期收益

1. **跨 episode 持续提升**：不再每 episode 重置；
2. **多环境不遗忘**：慢参数 + 记忆 + EWC；
3. **OOD 更鲁棒**：不确定性驱动更新与保守规划；
4. **安全可回滚**：锚点约束 + 验证回滚；
5. **超参自适应**：OOD 驱动更新强度；
6. **低数据优势**：元训练 + 快适应；
7. **表示稳定**：VICReg + 逆动力学 + stop-grad 多重防坍塌。

### 12.2 风险与挑战

1. **复杂度高**：双时间尺度、记忆、元训练、安全层叠加；
2. **计算开销**：慢参数整合、记忆检索、元训练成本；
3. **不确定性估计不准**：可能导致错误更新或过度保守；
4. **安全约束过保守**：降低性能；
5. **理论分析难**：持续学习、在线适应、安全约束的联合分析；
6. **真实世界验证**：仿真到现实差距；
7. **超参敏感**：EWC、KL、锚点约束等需调；
8. **记忆管理**：优先级、容量、检索效率；
9. **多环境模式**：参数记忆可能爆炸；
10. **评估复杂**：需要长期、多环境、安全评估协议。

---

## 十三、实现路线图

### 阶段一：最小可行版本

- 单时间尺度 + 不确定性；
- 在 PushT / PointMaze 上验证；
- 对比 AdaJEPA。

### 阶段二：加入双时间尺度

- LoRA 快参数 + 慢参数整合；
- 多 episode 持续适应评估。

### 阶段三：加入记忆

- 情景记忆 + 参数记忆；
- 多环境切换评估。

### 阶段四：加入安全层

- 锚点约束 + 回滚；
- 安全性评估。

### 阶段五：元训练

- MAML 式快参数初始化；
- 低数据评估。

### 阶段六：真实世界

- 机器人或自动驾驶平台；
- 仿真到现实。

---

## 十四、一句话总结

DTA-JEPA 的详细方案是：

> **用双时间尺度参数分离通用动力学与当前环境适应；用不确定性感知预测器指导适应强度与规划保守度；用情景记忆与参数记忆支持跨 episode 持续学习与多环境检索；用安全锚点、KL 约束与回滚机制保证在线更新不破坏预训练表示；用元训练学习快参数初始化，使单步梯度即可适应新环境。**

它比 AdaJEPA 更接近“部署时持续学习”的愿景，但复杂度、计算开销、安全与理论分析也显著增加。建议按路线图分阶段实现，先在仿真中验证每一模块，再逐步叠加，最后走向真实世界。
