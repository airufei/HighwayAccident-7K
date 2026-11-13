# -*- coding: utf-8 -*-
"""
temporal_model — V22 增强时序 Transformer 模块

在 V21 TemporalContextModule 基础上实现四项架构创新:

  ┌──────────────────────────────────────────────────────────────────┐
  │              V22 增强时序因果推理架构                            │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  帧特征序列 [B,T,256]                                           │
  │       │                                                          │
  │       ▼                                                          │
  │  ┌──────────────┐                                                │
  │  │ 1D-CNN (L1)  │ 局部运动感知 (继承 V21)                       │
  │  └──────┬───────┘                                                │
  │         │                                                        │
  │         ▼                                                        │
  │  ┌──────────────┐                                                │
  │  │ Pos Encoding │ 可学习位置编码 (继承 V21)                     │
  │  └──────┬───────┘                                                │
  │         │                                                        │
  │         ▼                                                        │
  │  ┌──────────────────────────────────────┐                        │
  │  │ CTTA Transformer Encoder (L2-L4)     │                        │
  │  │                                       │                        │
  │  │  ┌─────────────────────────────────┐ │                        │
  │  │  │ Type-Conditional Attention      │ │  ← V22 创新点1         │
  │  │  │ Attention(Q,K,V) + B(type_id)  │ │     CTTA                │
  │  │  └─────────────────────────────────┘ │                        │
  │  │  ┌─────────────────────────────────┐ │                        │
  │  │  │ FFN + Residual                  │ │                        │
  │  │  └─────────────────────────────────┘ │                        │
  │  └──────────────────────────────────────┘                        │
  │         │                                                        │
  │         ▼                                                        │
  │  ┌──────────────┐                                                │
  │  │ CLS Token    │ → 决策头                                       │
  │  └──────────────┘                                                │
  │                                                                  │
  │  V22 创新附加:                                                   │
  │    [2] UASC: 不确定性感知标签平滑 (作用于 scene_loss)            │
  │    [3] CDCR: 跨域一致性正则化 (作用于特征对齐)                   │
  │    [4] PTCL: 渐进式类型课程掩码 (作用于 type_loss)               │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

设计原则:
  - 不修改 V21 temporal_context.py (只读引用)
  - 通过继承 TemporalContextModule 扩展功能
  - 保持与 V21 三阶段五步训练的完全兼容

依赖:
  - V21 temporal_context.py (通过 sys.path 引入)
  - torch, torch.nn
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 通过 train_config 的 sys.path 注入引入 V21 模块
import train_config  # noqa: F401 — 确保路径初始化
import temporal_context as v24_temporal
from temporal_context import (
    TemporalContextModule,
    AccidentDecisionHead,
    FrameFeaturePool,
    trunc_normal_,
    _gn,
    HAS_3D,
)

# 从 train_config 获取创新配置
from train_config import CTTA_CONFIG, UASC_CONFIG, CDCR_CONFIG, PTCL_CONFIG
from base_config import cause_active_mask  # noqa: E402  成因位开关唯一真相源 (G-303 A6)


# ============================================================
# 创新点 1: CTTA — 碰撞类型感知注意力层
# ============================================================

class CTTAAttentionLayer(nn.Module):
    """碰撞类型感知注意力层 (CTTA)

    在标准 Multi-Head Self-Attention 基础上注入碰撞类型条件偏置:

        Attention(Q,K,V,t) = softmax(QK^T/sqrt(d) + B(t)) * V

    其中 B(t) 是由碰撞类型 embedding 生成的 T×T 偏置矩阵。

    不同碰撞类型产生不同的时序注意力模式:
      - rear_end: 关注纵向跟车距离变化 (相邻帧强关联)
      - t-bone:   关注横向接近速度 (交叉帧关联)
      - head_on:  关注对向车道接近 (全局趋势)
      - sideswipe: 关注微小平行位移 (弱关联)
    """

    def __init__(self, feat_dim=256, num_heads=8, num_types=7,
                 type_embed_dim=64, seq_len=30, dropout=0.1,
                 bias_init='zero', use_type_bias=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.seq_len = seq_len
        self.num_types = num_types
        # 消融开关的功能读者（V25_ABLATE_CTTA 走这里）：置假 ⇒ 不注入 B(t)，
        # 本层退化为标准注意力。参数与 state_dict 键一律不动，好让 3b 权重照载、
        # 两个臂只差"条件化"这一件事。
        self.use_type_bias = bool(use_type_bias)

        # 标准注意力投影
        self.W_q = nn.Linear(feat_dim, feat_dim)
        self.W_k = nn.Linear(feat_dim, feat_dim)
        self.W_v = nn.Linear(feat_dim, feat_dim)

        # CTTA: 碰撞类型嵌入 (num_types+1 容纳 padding/unknown)
        self.type_embed = nn.Embedding(num_types + 1, type_embed_dim,
                                       padding_idx=num_types)
        # 偏置生成网络: type_embed_dim → T×T
        self.bias_net = nn.Sequential(
            nn.Linear(type_embed_dim, type_embed_dim * 2),
            nn.GELU(),
            nn.Linear(type_embed_dim * 2, seq_len * seq_len),
        )

        # C6修复(Critical): bias_init='zero' — 零初始化 bias_net 输出层,
        # 使训练初期 CTTA 等价于标准注意力, 避免随机偏置扰动
        if bias_init == 'zero':
            nn.init.zeros_(self.bias_net[-1].weight)
            nn.init.zeros_(self.bias_net[-1].bias)

        # 输出投影
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x, type_ids=None):
        """前向传播

        Args:
            x: [B, T, feat_dim] 输入特征序列
            type_ids: [B] 碰撞类型 ID (0-6)。None ⇒ 不注入偏置（本层即标准注意力）；
                      传 padding 索引 7 ⇒ 仍注入常量模板 B(7)，两者**不等价**
                      （评估期实传 7，见 eval_acceptance.py）

        Returns:
            [B, T, feat_dim] 类型条件注意力输出
        """
        B, T, C = x.shape

        # 标准 QKV 投影
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [B, num_heads, T, T]

        # CTTA: 注入类型条件偏置
        if self.use_type_bias and type_ids is not None:
            type_emb = self.type_embed(type_ids)          # [B, type_embed_dim]
            bias = self.bias_net(type_emb)                # [B, T*T]
            bias = bias.view(B, self.seq_len, self.seq_len)
            # 动态适配实际序列长度 (含 CLS token 后 T 可能 > seq_len)
            if T > self.seq_len:
                pad = T - self.seq_len
                bias = F.pad(bias, (0, pad, 0, pad), value=0.0)  # [B, T, T]
            else:
                bias = bias[:, :T, :T]                        # [B, T, T]
            scores = scores + bias.unsqueeze(1)           # 广播到多头

        # Softmax + Dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = torch.matmul(attn, V)                       # [B, h, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影 + 残差 + LayerNorm
        out = self.norm(x + self.dropout(self.out_proj(out)))
        return out


class CTTATransformerLayer(nn.Module):
    """CTTA Transformer Encoder Layer

    = CTTAAttentionLayer (类型条件注意力)
    + FFN (前馈网络)
    + 残差连接
    """

    def __init__(self, feat_dim=256, num_heads=8, num_types=7,
                 type_embed_dim=64, seq_len=30, dropout=0.1,
                 ff_mult=4, use_type_bias=True):
        super().__init__()
        self.ctta_attn = CTTAAttentionLayer(
            feat_dim=feat_dim, num_heads=num_heads,
            num_types=num_types, type_embed_dim=type_embed_dim,
            seq_len=seq_len, dropout=dropout,
            use_type_bias=use_type_bias,
        )
        # FFN
        ff_hidden = feat_dim * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, feat_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(feat_dim)

    def forward(self, x, type_ids=None):
        """前向传播

        Args:
            x: [B, T, feat_dim]
            type_ids: [B] 碰撞类型 ID

        Returns:
            [B, T, feat_dim]
        """
        # 类型条件自注意力
        x = self.ctta_attn(x, type_ids)

        # FFN + 残差
        x = self.ffn_norm(x + self.ffn(x))
        return x


# ============================================================
# 创新点 1: CTTATemporalContext — 完整 CTTA 时序模块
# ============================================================

class CTTATemporalContext(nn.Module):
    """CTTA 增强时序上下文模块

    在 V21 TemporalContextModule 基础上替换 Transformer 后 2 层为 CTTA 层,
    实现碰撞类型感知时序注意力。

    架构:
      L1: 1D-CNN 局部运动 (继承 V21)
      L2: 标准 Self-Attention (继承 V21, Phase 2 已收敛)
      L3-L4: CTTA 类型条件注意力 (V22 创新, Phase 3a 启动)

    渐进式训练兼容:
      Phase 2: 使用 V21 标准 TemporalContextModule (无 CTTA)
      Phase 3a: 切换到 CTTATemporalContext, 冻结 L1-L2, 训练 CTTA L3-L4
      Phase 3b: 解冻全部 CTTA 层
      Phase 3c: 全系统联合优化
    """

    def __init__(self, feat_dim=256, num_heads=8, depth=4,
                 seq_len=30, dropout=0.3, num_types=7,
                 ctta_layers=2, ctta_config=None):
        """
        Args:
            feat_dim: 特征维度
            num_heads: 注意力头数
            depth: 总 Transformer 层数
            seq_len: 序列长度
            dropout: Dropout
            num_types: 碰撞类型数
            ctta_layers: CTTA 层数 (放在最后 ctta_layers 层)
            ctta_config: CTTA 配置 dict (覆盖默认)
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.depth = depth
        self.ctta_layers = ctta_layers
        self.standard_layers = depth - ctta_layers

        cfg = ctta_config or CTTA_CONFIG
        type_embed_dim = cfg.get('type_embed_dim', 64)
        # 两个开关的功能读者（#35）：任一置假 ⇒ 类型条件偏置不进前向。
        # 只在关闭时出声，默认档一行不多打 ⇒ 现网日志与改前逐字相同。
        use_type_bias = bool(cfg.get('enable', True)) and \
            bool(cfg.get('type_conditional_bias', True))
        if not use_type_bias:
            print('[CTTA] 类型条件偏置注入已关闭 ⇒ CTTA− 对照臂（层数/参数形状不变）')

        # ── L1: 1D-CNN 局部运动 (完全继承 V21) ──
        self.local_cnn = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
            _gn(feat_dim),
            nn.GELU(),
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
        )

        # ── 位置编码 ──
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, feat_dim))
        trunc_normal_(self.pos_embed, std=.02)

        # ── CLS Token ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        trunc_normal_(self.cls_token, std=.02)

        # ── L2: 标准 Transformer 层 (继承 V21 设计) ──
        self.standard_encoder = nn.ModuleList([
            self._make_standard_layer(feat_dim, num_heads, dropout)
            for _ in range(self.standard_layers)
        ])

        # ── L3-L4: CTTA 类型条件 Transformer 层 (V22 创新) ──
        self.ctta_encoder = nn.ModuleList([
            CTTATransformerLayer(
                feat_dim=feat_dim, num_heads=num_heads,
                num_types=num_types, type_embed_dim=type_embed_dim,
                seq_len=seq_len, dropout=dropout,
                use_type_bias=use_type_bias,
            )
            for _ in range(ctta_layers)
        ])

        # ── 输出 LayerNorm ──
        self.norm = nn.LayerNorm(feat_dim)

    def _make_standard_layer(self, feat_dim, num_heads, dropout):
        """构建标准 Transformer Encoder Layer (与 V21 对齐)"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        return encoder_layer

    def forward(self, x, type_ids=None, return_frames=False):
        """前向传播

        Args:
            x: [B, T, feat_dim] 帧特征序列
            type_ids: [B] 碰撞类型 ID (Phase 3a+ 提供)
            return_frames: 是否返回逐帧输出

        Returns:
            cls_out: [B, feat_dim] CLS Token 输出
            frame_out: [B, T, feat_dim] 逐帧输出 (可选)
        """
        B, T, C = x.shape

        # 1. 1D-CNN 局部运动 + 残差
        x_cnn = self.local_cnn(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_cnn

        # 2. 位置编码
        x = x + self.pos_embed[:, :T, :]

        # 3. CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, C]

        # 4. 标准 Transformer 层 (L2)
        for layer in self.standard_encoder:
            x = layer(x)

        # 5. CTTA 层 (L3-L4) — 注入类型条件偏置
        for layer in self.ctta_encoder:
            x = layer(x, type_ids)

        # 6. 提取 CLS Token
        cls_out = self.norm(x[:, 0])
        frame_out = self.norm(x[:, 1:]) if return_frames else None

        if return_frames:
            return cls_out, frame_out
        return cls_out

    def get_progressive_params(self):
        """获取渐进式训练参数组

        返回:
            standard_params: 标准层参数 (Phase 2 已收敛, Phase 3a 可冻结)
            ctta_params: CTTA 层参数 (Phase 3a 训练目标)
            base_params: CNN + pos_embed + cls_token (通常不冻结)
        """
        standard_params = []
        for layer in self.standard_encoder:
            standard_params.extend(layer.parameters())

        ctta_params = []
        for layer in self.ctta_encoder:
            ctta_params.extend(layer.parameters())

        base_params = []
        base_params.extend(self.local_cnn.parameters())
        base_params.append(self.pos_embed)
        base_params.append(self.cls_token)
        base_params.extend(self.norm.parameters())

        return {
            'standard': standard_params,
            'ctta': ctta_params,
            'base': base_params,
        }

    def freeze_standard_layers(self):
        """Phase 3a: 冻结标准层 (L2), 仅训练 CTTA 层"""
        for p in self.get_progressive_params()['standard']:
            p.requires_grad = False
        for p in self.get_progressive_params()['base']:
            p.requires_grad = False

    def unfreeze_all(self):
        """Phase 3b/3c: 解冻全部"""
        for p in self.parameters():
            p.requires_grad = True


# ============================================================
# 创新点 2: UASC — 不确定性感知标签平滑
# ============================================================

class UASCSmoothLabel(nn.Module):
    """不确定性感知场景分类标签平滑

    根据标注者投票一致性动态生成软标签:
      - 高一致性 (≥ threshold): 硬标签 (epsilon=0)
      - 低一致性 (< threshold): 软标签 (epsilon=smoothing_low)

    输入:
        scene_cls: [B] 场景类别 (1=normal, 2=pre_crash, 3=crash)
        agreement: [B] 标注者一致性比例 [0, 1] (可选)

    输出:
        soft_label: [B, nc_cls] 平滑后的概率分布
    """

    def __init__(self, nc_cls=3, config=None):
        super().__init__()
        cfg = config or UASC_CONFIG
        self.nc_cls = nc_cls
        self.smoothing_high = cfg.get('label_smoothing_high', 0.0)
        self.smoothing_low = cfg.get('label_smoothing_low', 0.3)
        self.agreement_threshold = cfg.get('agreement_threshold', 0.8)

    def forward(self, scene_cls, agreement=None):
        """生成软标签

        Args:
            scene_cls: [B] 原始标签 (1=normal, 2=pre_crash, 3=crash)
            agreement: [B] 标注者一致性 [0,1], None 时用默认 smoothing_low

        Returns:
            [B, nc_cls] 软标签概率分布
        """
        B = scene_cls.shape[0]
        device = scene_cls.device

        # 标签映射: 1→0, 2→1, 3→2
        cls_idx = (scene_cls - 1).clamp(0, self.nc_cls - 1)

        # 确定 epsilon
        if agreement is not None:
            epsilon = torch.where(
                agreement >= self.agreement_threshold,
                torch.tensor(self.smoothing_high, device=device),
                torch.tensor(self.smoothing_low, device=device),
            )
        else:
            # 无一致性信息时使用中等平滑
            epsilon = torch.full((B,), self.smoothing_low * 0.5,
                                 device=device)

        # 标签平滑: hard * (1-eps) + uniform * eps
        one_hot = F.one_hot(cls_idx, self.nc_cls).float()  # [B, nc_cls]
        uniform = torch.ones_like(one_hot) / self.nc_cls

        eps = epsilon.unsqueeze(1)  # [B, 1]
        soft_label = one_hot * (1.0 - eps) + uniform * eps

        return soft_label


# ============================================================
# 创新点 3: CDCR — 跨域一致性正则化损失
# ============================================================

class CDCRLoss(nn.Module):
    """跨域一致性正则化损失

    对齐真实 CCTV 与 CARLA 合成数据的特征表示:

        L_cdcr = 0.5 * (KL(p_real || p_synth) + KL(p_synth || p_real))

    使用温度参数控制分布平滑度。
    """

    def __init__(self, temperature=2.0, config=None):
        super().__init__()
        cfg = config or CDCR_CONFIG
        self.temperature = cfg.get('temperature', temperature)

    def forward(self, feat_real, feat_synth):
        """计算跨域一致性损失

        Args:
            feat_real:  [B_r, feat_dim] 真实集特征
            feat_synth: [B_s, feat_dim] 合成集特征

        Returns:
            scalar: CDCR 损失值 (无有效输入时返回 None, 由调用方处理)
        """
        if feat_real is None or feat_synth is None:
            # C8修复(Critical): 返回 None 而非 CPU tensor, 由调用方安全跳过
            return None

        # 温度缩放 — 分别准备 log/prob 两种形式 (F.kl_div 要求 input=log_prob, target=prob)
        T = self.temperature
        p_real_log = F.log_softmax(feat_real / T, dim=-1)
        p_real_prob = F.softmax(feat_real / T, dim=-1)
        p_synth_log = F.log_softmax(feat_synth / T, dim=-1)
        p_synth_prob = F.softmax(feat_synth / T, dim=-1)

        # 双向 KL 散度 (batchmean)
        # W7修复(Warning): F.kl_div(log_p, q) 计算 KL(q || p), 梯度流经 log_p
        # 需要 padding 到相同 batch size
        min_b = min(feat_real.size(0), feat_synth.size(0))
        p_real_log = p_real_log[:min_b]
        p_real_prob = p_real_prob[:min_b]
        p_synth_log = p_synth_log[:min_b]
        p_synth_prob = p_synth_prob[:min_b]

        # kl_real_to_synth: KL(p_synth || p_real), 梯度使 real 逼近 synth
        kl_real_to_synth = F.kl_div(p_real_log, p_synth_prob.detach(), reduction='batchmean')
        # kl_synth_to_real: KL(p_real || p_synth), 梯度使 synth 逼近 real
        kl_synth_to_real = F.kl_div(p_synth_log, p_real_prob.detach(), reduction='batchmean')

        loss = 0.5 * (kl_real_to_synth + kl_synth_to_real) * (T ** 2)
        return loss


# ============================================================
# 创新点 4: PTCL — 渐进式类型课程掩码
# ============================================================

class PTCLTypeMask(nn.Module):
    """渐进式类型课程学习掩码

    根据 Phase 控制激活的碰撞类型:

      Phase 2:   [rear_end, side_impact]
      Phase 3a:  [rear_end, side_impact, head_on]
      Phase 3b:  [rear_end, side_impact, head_on, fixed_object]
      Phase 3c:  [rear_end, side_impact, head_on, fixed_object, other]
    """

    def __init__(self, num_types=7, config=None):
        super().__init__()
        cfg = config or PTCL_CONFIG
        self.num_types = num_types
        self.phase_masks = {
            '2':   cfg.get('phase2_active_types', [0, 1]),
            '3a':  cfg.get('phase3a_active_types', [0, 1, 4]),
            '3b':  cfg.get('phase3b_active_types', [0, 1, 4, 5]),
            '3c':  cfg.get('phase3c_active_types', [0, 1, 4, 5, 6]),
        }

    def get_mask(self, phase):
        """获取当前 Phase 的类型激活掩码

        Args:
            phase: '1', '2', '3a', '3b', '3c'

        Returns:
            [num_types] float tensor, 1.0=激活, 0.0=掩码
        """
        active = self.phase_masks.get(phase, list(range(self.num_types)))
        mask = torch.zeros(self.num_types)
        for t in active:
            if 0 <= t < self.num_types:
                mask[t] = 1.0
        return mask

    def apply_mask(self, type_logits, type_targets, phase):
        """过滤掉未激活类型的样本 — 仅保留 ground-truth 在当前 Phase 激活集中的样本

        Args:
            type_logits: [B, num_types]
            type_targets: [B] 类型标签
            phase: 当前训练阶段

        Returns:
            filtered_logits, filtered_targets
        """
        mask = self.get_mask(phase).to(type_logits.device)
        active_types = self.phase_masks.get(phase, list(range(self.num_types)))
        # 仅保留 ground-truth 在激活集中的样本 (避免 -inf 导致 CE 产生 +inf)
        valid = torch.zeros_like(type_targets, dtype=torch.bool)
        for t in active_types:
            valid |= (type_targets == t)
        if not valid.any():
            return type_logits[:0], type_targets[:0]  # 空张量, CE 对空输入返回 0
        masked_logits = type_logits[valid].masked_fill(mask == 0, float('-inf'))
        masked_targets = type_targets[valid]
        return masked_logits, masked_targets


# ============================================================
# V24TemporalModel — 完整 V22 时序模型
# ============================================================

class V24TemporalModel(nn.Module):
    """V22 完整时序因果推理模型

    = V21 V18TemporalModel + CTTA 增强 Transformer

    架构:
      YOLO26 Backbone → FrameFeaturePool → CTTATemporalContext → AccidentDecisionHead

    Phase 兼容性:
      Phase 1/2: 使用标准 forward (单帧模式, 不启用 CTTA)
      Phase 3a:  启用 CTTA, 冻结标准 Transformer 层
      Phase 3b:  解冻全部, 加入 CDCR
      Phase 3c:  全系统联合, 所有创新启用
    """

    def __init__(self, detection_model, feat_dim=256, num_heads=8,
                 depth=4, seq_len=30, dropout=0.3,
                 use_tracklet=True, max_objs=8, use_3d=False,
                 ctta_config=None, uasc_config=None,
                 cdcr_config=None, ptcl_config=None):
        super().__init__()
        self.detection_model = detection_model
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.use_3d = use_3d and HAS_3D

        # ── 帧特征池化 (继承 V21) ──
        detect_head = detection_model.model[-1]
        ch_list = [seq[0].conv.in_channels for seq in detect_head.cv2]
        self.ch_list = ch_list
        self.frame_pool = FrameFeaturePool(ch_list, feat_dim, use_3d=self.use_3d)

        # ── V22 创新: CTTA 增强时序模块 ──
        self.temporal_context = CTTATemporalContext(
            feat_dim=feat_dim,
            num_heads=num_heads,
            depth=depth,
            seq_len=seq_len,
            dropout=dropout,
            ctta_config=ctta_config,
        )

        # ── 决策头 (继承 V21) ──
        nc_cls = getattr(detection_model, 'nc_cls', 3)
        self.decision_head = AccidentDecisionHead(
            feat_dim=feat_dim, nc_cls=nc_cls,
            num_causes=9, num_types=7, dropout=dropout,
        )
        self.nc_cls = nc_cls

        # ── V22 创新模块实例 ──
        self.uasc = UASCSmoothLabel(nc_cls=nc_cls, config=uasc_config)
        self.cdcr_loss = CDCRLoss(config=cdcr_config)
        self.ptcl_mask = PTCLTypeMask(config=ptcl_config)

        # ── 当前 Phase 状态 ──
        self._current_phase = '1'

    def set_phase(self, phase, freeze_standard=True):
        """设置当前训练 Phase, 控制创新模块启用

        G-278(R1): freeze_standard=False 给"无前阶段时序权重可继承"的 3a 用 ——
        standard/base 层此时是构造时随机初始化, 冻结等于把主干永久钉在随机权重上
        (freeze_standard_layers 的自述前提 "Phase 2 已收敛" 在本管线上为假:
        P2 种子 pickle 里 temporal_context 出现 0 次)。
        """
        self._current_phase = phase
        if phase == '3a':
            if freeze_standard:
                self.temporal_context.freeze_standard_layers()
        elif phase in ('3b', '3c'):
            self.temporal_context.unfreeze_all()

    def forward(self, x, temporal_mode=False, type_ids=None, agreement=None,
                feat_real=None, feat_synth=None, **kwargs):
        """前向传播

        Args:
            x: [B, 3, H, W] 或 [B, T, 3, H, W]
            temporal_mode: 是否启用时序模式 (Phase 3)
            type_ids: [B] 碰撞类型 ID (CTTA 需要)
            agreement: [B] 标注者一致性 (UASC 需要)
            feat_real/feat_synth: 跨域特征 (CDCR 需要)

        Returns:
            Phase 1/2: V21 标准 5 元组
            Phase 3: V22 增强输出 (含创新损失项)
        """
        if not temporal_mode or x.dim() == 4:
            return self.detection_model(x)

        # Phase 3: Clip 时序模式
        B, T, C, H, W = x.shape

        # 逐帧特征提取
        x_flat = x.view(B * T, C, H, W)
        frame_features = self._extract_frame_features(x_flat)

        # 帧特征池化
        frame_vecs = self.frame_pool(frame_features)
        frame_vecs = frame_vecs.view(B, T, self.feat_dim)

        # CTTA 增强时序推理
        cls_token, frame_seq = self.temporal_context(
            frame_vecs, type_ids=type_ids, return_frames=True
        )

        # 决策头
        scene_logits, tta_pred, cause_logits, type_logits = \
            self.decision_head(cls_token)
        frame_scene_logits = self.decision_head.forward_frames(frame_seq)

        # V22 创新附加输出 (嵌入 geo_3d_dict 保持 7 元组兼容 V21)
        v24_extras = {
            'uasc_soft_label': None,
            'cdcr_loss': torch.tensor(0.0, device=cls_token.device),
            'ptcl_phase': self._current_phase,
            # C1修复(Critical): 不 detach CLS token — CDCR 需要梯度回传到模型参数
            # CDCRLoss 内部已对 target 侧 detach (知识蒸馏标准做法)
            'cls_token': cls_token,
        }

        # UASC: 传递 agreement 给 loss 计算
        # W10修复(Warning): 保留原始 agreement (clip级 [B]), loss 中再扩展到帧级
        if agreement is not None:
            v24_extras['uasc_soft_label'] = agreement  # [B] clip级一致性

        # CDCR: 跨域一致性损失
        if feat_real is not None and feat_synth is not None:
            v24_extras['cdcr_loss'] = self.cdcr_loss(feat_real, feat_synth)

        # 保持 7 元组向后兼容 V21 run_phase3_temporal 的解包逻辑
        geo_3d_dict = {'geo_3d': None, 'geo_3d_decoded': None, 'ttc_3d': None,
                       'v24_extras': v24_extras}
        return (None, scene_logits, tta_pred, cause_logits, type_logits,
                frame_scene_logits, geo_3d_dict)

    def _extract_frame_features(self, x):
        """从 YOLO backbone 提取多尺度特征 (与 V21 _forward_until_detect 对齐)"""
        model = self.detection_model.model
        y = [None] * len(model)
        features_for_detect = []

        for i, m in enumerate(model):
            if isinstance(m.f, int):
                if m.f == -1:
                    y[i] = m(x if i == 0 else y[i - 1])
                else:
                    y[i] = m(y[m.f])
            elif isinstance(m.f, (list, tuple)):
                inputs = []
                for f_idx in m.f:
                    if f_idx == -1:
                        inputs.append(x if i == 0 else y[i - 1])
                    elif 0 <= f_idx < len(y) and y[f_idx] is not None:
                        inputs.append(y[f_idx])
                y[i] = m(inputs)
            else:
                y[i] = m(x if i == 0 else y[i - 1])

            if hasattr(m, 'cv2') and hasattr(m, 'cv3'):
                if isinstance(m.f, (list, tuple)):
                    for idx in m.f:
                        # W9修复(Warning): 正确处理负索引 (如 -2, -3)
                        if idx == -1:
                            src = i - 1
                        elif idx < 0:
                            src = i + idx  # 相对索引转绝对索引
                        else:
                            src = idx
                        if 0 <= src < len(y) and y[src] is not None:
                            features_for_detect.append(y[src])
                break

        if not features_for_detect:
            for idx in range(len(y) - 3, len(y)):
                if y[idx] is not None and isinstance(y[idx], torch.Tensor):
                    if y[idx].dim() == 4:
                        features_for_detect.append(y[idx])

        if not features_for_detect or len(features_for_detect) < 3:
            raise RuntimeError(
                f"无法从 YOLO backbone 提取多尺度特征 "
                f"(期望 3 个尺度, 实际 {len(features_for_detect)} 个)。"
                f"请检查模型结构。"
            )
        return features_for_detect[-3:]


# ============================================================
# V22TemporalMultiTaskLoss — 增强损失函数
# ============================================================

class V22TemporalMultiTaskLoss(nn.Module):
    """V22 增强多任务联合损失

    在 V21 TemporalMultiTaskLoss 基础上增加四项创新损失:

    L_v22 = L_det                                     (检测, 继承 V21)
          + λ₂ × L_scene_uasc                         (UASC 增强场景损失)
          + λ₃ × L_tta                                (TTA 回归, 继承 V21)
          + λ₅ × L_type_ptcl                          (PTCL 课程类型损失)
          + λ₇ × L_ctta                               (CTTA 正则, 可选)
          + λ₉ × L_cdcr                               (CDCR 跨域一致性)
    """

    def __init__(self, nc_cls=3, num_causes=9, max_tta=60.0,
                 lambda_scene=2.0, lambda_tta=0.5,
                 lambda_type=0.5, lambda_cdcr=0.1,
                 lambda_cause=0.0,
                 cls_weight=None,
                 uasc_config=None, cdcr_config=None, ptcl_config=None):
        super().__init__()
        self.nc_cls = nc_cls
        self.max_tta = max_tta
        self.num_causes = num_causes
        self.lambda_scene = lambda_scene
        self.lambda_tta = lambda_tta
        self.lambda_type = lambda_type
        self.lambda_cdcr = lambda_cdcr
        self.lambda_cause = lambda_cause

        # V22-fix(cause-回填): 成因 pos_weight (与 V21 temporal_context 一致,
        # 来自环境变量 V24_CAUSE_POS_WEIGHT, train.py 动态计算)
        _cpw = os.environ.get('V24_CAUSE_POS_WEIGHT', '')
        try:
            _cpw_list = [float(x) for x in _cpw.split(',') if x.strip()]
        except ValueError:
            _cpw_list = []
        _cpw_list = (_cpw_list + [1.0] * num_causes)[:num_causes]
        self.register_buffer('cause_pos_weight',
                             torch.tensor(_cpw_list, dtype=torch.float32))
        # G-303(A6): base_config.py:165 声明"摘掉一个因子只做三件事: 从 loss 里屏蔽、
        # 从评测里屏蔽、从预测里压成 0"。P1/P2 有 core/losses.py:330 的 factor_mask,
        # 相位3 的 cause 分支此前没有 ⇒ 已判停的 bit4(pedestrian_highway)/bit7
        # (distracted_driving) 仍以 pos_weight=1.0 吃 BCE 梯度, 而评测面
        # (eval_acceptance.py:365) 与部署面 (video_inference.py:71) 都把这两位屏蔽/
        # 压 0 —— 训的是本项目已作废的标签。分子分母同屏蔽, 归一形式与 P1/P2 逐字一致
        # ((bce*mask).sum()/mask.sum()), 全启用时与原式数值等价。
        self.register_buffer('cause_factor_mask',
                             torch.tensor(cause_active_mask()[:num_causes],
                                          dtype=torch.float32).unsqueeze(0))  # [1, C]

        # V22 创新模块
        self.uasc = UASCSmoothLabel(nc_cls=nc_cls, config=uasc_config)
        self.cdcr_loss_fn = CDCRLoss(config=cdcr_config)
        self.ptcl = PTCLTypeMask(config=ptcl_config)

        # 基础损失函数
        # C2修复(Critical): 存储 cls_weight 供 forward 中 CE 使用
        self._cls_w = torch.tensor(cls_weight, dtype=torch.float32) if cls_weight else None
        self.scene_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=0.1, ignore_index=-100, weight=self._cls_w)
        self.tta_loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.type_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, preds, batch, det_loss=0.0, phase='3a',
                feat_real=None, feat_synth=None):
        """计算 V22 增强联合损失

        Args:
            preds: V22 七元组输出 (geo_3d_dict 内嵌 v24_extras)
            batch: 数据 batch
            det_loss: 检测损失 (标量)
            phase: 当前训练阶段
            feat_real/feat_synth: 跨域特征 (CDCR)

        Returns:
            total_loss, loss_dict
        """
        # 解包 V22 输出 (兼容 7 元组和 8 元组)
        v24_extras = {}
        if len(preds) == 7:
            (_, scene_logits, tta_pred, cause_logits, type_logits,
             frame_scene_logits, geo_3d_dict) = preds
            # Impact#6修复(Suggestion): 使用 .get() 替代 .pop() 避免突变输入 dict
            if isinstance(geo_3d_dict, dict):
                v24_extras = geo_3d_dict.get('v24_extras', {}) or {}
        elif len(preds) == 8:
            # 兼容旧版 8 元组格式
            (_, scene_logits, tta_pred, cause_logits, type_logits,
             frame_scene_logits, geo_3d_dict, v24_extras) = preds
        else:
            (_, scene_logits, tta_pred, cause_logits, type_logits,
             frame_scene_logits) = preds[:6]
            geo_3d_dict = None
        if v24_extras is None:
            v24_extras = {}

        device = scene_logits.device

        # V23-fix(uasc-nan v4): 根因防线 — 退化 clip 在 fp16 下使上游注意力输出非有限
        # (3b/3c batch299 确定性复现), 此时 UASC KL 与 kl-nan v3 回退的 CE 同为 NaN,
        # 兜底失效。将各头 logits 的非有限值替换为 0 (等效均匀分布, 损失≈ln C),
        # 保证任何 batch 的损失有限; 对退化 batch 等价于温和均匀正则, 无副作用。
        frame_scene_logits = torch.nan_to_num(
            frame_scene_logits, nan=0.0, posinf=0.0, neginf=0.0)
        scene_logits = torch.nan_to_num(
            scene_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if type_logits is not None:
            type_logits = torch.nan_to_num(
                type_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if cause_logits is not None:
            cause_logits = torch.nan_to_num(
                cause_logits, nan=0.0, posinf=0.0, neginf=0.0)

        # ── L_scene (UASC 增强) ──
        scene_loss = torch.tensor(0.0, device=device)
        if frame_scene_logits is not None and 'scene_cls_seq' in batch:
            scene_seq = batch['scene_cls_seq'].to(device, dtype=torch.long)
            valid_seq = (scene_seq >= 1) & (scene_seq <= 3)
            seq_idx = (scene_seq - 1).clamp(0, 2)
            seq_idx[~valid_seq] = -100
            B, T, C = frame_scene_logits.shape

            # UASC: 使用不确定性感知软标签替代硬标签
            _w = self._cls_w.to(device) if self._cls_w is not None else None
            agreement = v24_extras.get('uasc_soft_label')
            # V23-fix(kl-nan v5): 根因确认 — batch299 的 scene_cls_seq 全为 0
            # (ignore 帧), 展平后有效帧数为 0, kl_div(batchmean) 除以 0 → NaN。
            # 此前用 valid_seq.any() 判定 ([B] 与 [B,T] 语义不一致), 必须按
            # 展平后的有效帧数守卫。
            if agreement is not None and bool(valid_seq.any()):
                # W10修复(Critical): agreement 是 clip级 [B], 需扩展到帧级 [B,T]
                if agreement.dim() == 1:
                    agreement_expanded = agreement.unsqueeze(1).expand(-1, T)
                else:
                    agreement_expanded = agreement
                # 生成软标签 (仅对有效帧)
                valid_scene = scene_seq[valid_seq]
                valid_agreement = agreement_expanded[valid_seq]
                soft_labels = self.uasc(valid_scene, valid_agreement)
                # 用 KL 散度计算软标签损失
                # V22-fix(kl-nan v2): fp16 下 log_softmax 可输出 -inf (置信度极高),
                # kl_div 会算出 exp(log_t)*(log_t-(-inf))=inf → clamp 到有限域;
                # log_target=True 规避 0*log(0)=nan
                # V23-fix(kl-nan v3): AMP 下 soft_labels 为 fp16, 小概率分量下溢为 0,
                # .log()=-inf → kl_div(log_target) 计算 0*(-inf)=NaN (3b batch299 实测)。
                # log-target 全程 fp32 + 硬下界
                log_probs = F.log_softmax(
                    frame_scene_logits.reshape(B * T, C)[valid_seq.reshape(-1)], dim=-1)
                log_probs = log_probs.clamp(min=-20.0, max=0.0)
                _log_target = soft_labels.float().clamp_min(1e-8).log().clamp(min=-20.0)
                # #94: scene 证据档降权 — clip 权重 [B] 扩到帧级 [B,T] 后按 valid_seq 取,
                # 用逐帧加权均值替换 batchmean；权重全 1 ⇒ (kl.sum(C)).mean() 与 batchmean 数值等价。
                _kl_pf = F.kl_div(log_probs, _log_target, reduction='none',
                                  log_target=True).sum(dim=1)  # [N_valid]
                _w_pf = torch.ones_like(_kl_pf)
                # G-274(cls-weight-回归KL分支): agreement≡1.0 (train_data.py:74-75
                # "ACCIDENT 无逐 clip 数据, 使用默认值") ⇒ UASC 软标签退化成 one-hot，
                # 这条 KL 分支被恒走，而类别权重只挂在回退 CE 上（C2 修复漏了这里）
                # ⇒ 三分类 [1.0, 1.5, 5.0] 的强调在相位3 帧级通道整条失效，
                # 最稀的 pre_crash 首先被多数类淹死（尺② 预测列 e3/e5 恒 0）。
                # 按目标类加权 + 权重和归一（与 F.cross_entropy(weight=…) 同语义）；
                # 权重全 1 时与改前逐值等价。
                if _w is not None:
                    _tgt_pf = seq_idx.reshape(B * T)[valid_seq.reshape(-1)]
                    _w_pf = _w.to(_kl_pf.device).float()[_tgt_pf.clamp(0, C - 1)]
                _sw = batch.get('scene_sup_weight', None)
                if (_sw is not None and torch.is_tensor(_sw)
                        and _sw.numel() == B):
                    _sw_pf = _sw.view(B, 1).expand(B, T)[valid_seq].to(
                        _kl_pf.device).float().clamp(min=0.0)
                    _w_pf = _w_pf * _sw_pf
                scene_loss = (_kl_pf * _w_pf).sum() / (_w_pf.sum() + 1e-8)
            else:
                # 回退: 固定 label_smoothing 的标准 CE
                # C2修复(Critical): 补回 cls_weight 类别权重 (V21 已修复的 Bug 回归)
                _flat_targets = seq_idx.reshape(B * T)
                # V23-fix(ce-nan): batch299 根因 — 全帧 ignore (-100) 时
                # CE(mean) 计算 0/0 = NaN。返回挂图零损失 (梯度为 0, 无副作用)
                if not bool((_flat_targets != -100).any()):
                    scene_loss = frame_scene_logits.sum() * 0.0
                else:
                    scene_loss = F.cross_entropy(
                        frame_scene_logits.reshape(B * T, C),
                        _flat_targets,
                        weight=_w,
                        label_smoothing=0.1, ignore_index=-100,
                    )

            # V23-fix(kl-nan v3): 兜底 — UASC 路径仍产生非有限值时回退标准 CE,
            # 避免 NaN 进入总损失 (GradScaler 虽会跳过该步, 但损失统计被污染)
            if not torch.isfinite(scene_loss):
                _flat_targets = seq_idx.reshape(B * T)
                # V23-fix(ce-nan): 同上 — 全 ignore 目标时 0/0 防御
                if not bool((_flat_targets != -100).any()):
                    scene_loss = frame_scene_logits.sum() * 0.0
                else:
                    scene_loss = F.cross_entropy(
                        frame_scene_logits.reshape(B * T, C),
                        _flat_targets,
                        weight=self._cls_w.to(device) if self._cls_w is not None else None,
                        label_smoothing=0.1, ignore_index=-100,
                    )

        # C4修复(Critical): 恢复 V21 V20-fix(C1) — clip 级辅助 scene loss
        # 直接监督 AccidentDecisionHead.scene_head, 提供额外梯度信号
        if 'scene_cls' in batch:
            scene_gt = batch['scene_cls'].to(device, dtype=torch.long)
            valid_clip = (scene_gt >= 1) & (scene_gt <= 3)
            if valid_clip.any():
                clip_idx = (scene_gt - 1).clamp(0, 2)
                clip_idx[~valid_clip] = -100
                _w = self._cls_w.to(device) if self._cls_w is not None else None
                clip_loss = F.cross_entropy(
                    scene_logits, clip_idx,
                    weight=_w, label_smoothing=0.1, ignore_index=-100)
                scene_loss = scene_loss + 0.3 * clip_loss

        # ── L_tta ──
        tta_loss = torch.tensor(0.0, device=device)
        if 'tta' in batch:
            tta_gt = batch['tta'].to(device, dtype=torch.float32)
            valid = (tta_gt >= 0) & (tta_gt <= self.max_tta)
            if valid.any():
                tta_pred_scaled = torch.sigmoid(
                    tta_pred[valid].squeeze(-1)) * self.max_tta
                tta_loss = self.tta_loss_fn(tta_pred_scaled, tta_gt[valid])

        # ── L_type (PTCL 增强) ──
        type_loss = torch.tensor(0.0, device=device)
        if 'accident_type' in batch:
            type_gt = batch['accident_type'].to(device, dtype=torch.long)
            if 'scene_cls' in batch:
                scene_t = batch['scene_cls'].to(device, dtype=torch.long)
                crash_mask = (scene_t == 3)
                if crash_mask.any():
                    masked_logits, masked_targets = self.ptcl.apply_mask(
                        type_logits[crash_mask], type_gt[crash_mask], phase)
                    # C3修复(Critical): 空张量守卫 — PTCL 返回空张量时 CE 会 NaN
                    if masked_targets.numel() > 0:
                        type_loss = self.type_loss_fn(
                            masked_logits, masked_targets)

        # ── L_cause (V22-fix(cause-回填): 移植 V21 temporal_context 公式) ──
        # 仅 pre_crash(2)/crash(3) clip 参与; BCE + pos_weight; λ_cause=0 时跳过
        cause_loss = torch.tensor(0.0, device=device)
        if self.lambda_cause > 0 and 'cause' in batch and cause_logits is not None:
            cause_gt = batch['cause'].to(device, dtype=torch.float32)
            if cause_gt.dim() == 1:
                cause_gt = cause_gt.view(-1, self.num_causes)
            cause_gt = cause_gt.clamp(0.0, 1.0)
            if 'scene_cls' in batch:
                scene_t = batch['scene_cls'].to(device, dtype=torch.long)
                active = (scene_t >= 2).float().unsqueeze(1)
            else:
                active = torch.ones(cause_gt.size(0), 1, device=device)
            # V22-fix(cause-标注缺失): 无成因标注的 clip 屏蔽 (值掩码, 不改变计算图)
            if 'cause_valid' in batch:
                cv = batch['cause_valid'].to(device, dtype=torch.float32)
                active = active * cv.view(-1, 1)
            # G-303(A6): 已摘除因子 (CAUSE_DISABLED_BITS) 不进分子也不进分母,
            # 与 core/losses.py:330 同式 ⇒ active 由 [B,1] 变 [B,C]。
            active = active * self.cause_factor_mask.to(device)
            if active.sum() > 0:
                bce = F.binary_cross_entropy_with_logits(
                    cause_logits, cause_gt,
                    pos_weight=self.cause_pos_weight.unsqueeze(0).to(device),
                    reduction='none')
                cause_loss = (bce * active).sum() / (active.sum() + 1e-8)

        # 从 v24_extras 获取 CDCR 损失 (如果前向传播时已计算)
        # C8修复(Critical): CDCRLoss 可能返回 None (无有效输入时)
        cdcr_loss_raw = v24_extras.get('cdcr_loss', None)
        if feat_real is not None and feat_synth is not None:
            # 外部传入的 feat_real/feat_synth 覆盖 (用于训练循环显式传入的场景)
            cdcr_loss_raw = self.cdcr_loss_fn(feat_real, feat_synth)

        # 安全处理 None 返回值
        # V23-fix(cdcr-finite): 退化 clip 的 cls_token 可使 KL 非有限, 拒绝入总损失
        if (cdcr_loss_raw is not None and isinstance(cdcr_loss_raw, torch.Tensor)
                and torch.isfinite(cdcr_loss_raw)):
            cdcr_loss = cdcr_loss_raw
        else:
            cdcr_loss = torch.tensor(0.0, device=device)

        # ── 总损失 ──
        total_loss = (
            det_loss
            + self.lambda_scene * scene_loss
            + self.lambda_cause * cause_loss
            + self.lambda_tta * tta_loss
            + self.lambda_type * type_loss
            + self.lambda_cdcr * cdcr_loss
        )

        # C7修复(Critical): 确保 total_loss 有梯度图连接
        # 当所有子损失都是无梯度的 0 张量 (极端 batch) 时, backward 会崩溃
        if not total_loss.requires_grad:
            # 连接到模型参数以确保 grad_fn 存在
            _dummy = sum(p.sum() * 0.0 for p in self.parameters()
                         if p.requires_grad)
            total_loss = total_loss + _dummy

        # loss_dict 键名对齐 V21 epoch_losses 累加器 (兼容新旧两种键名)
        scene_val = scene_loss.item()
        type_val = type_loss.item()
        cdcr_val = cdcr_loss.item() if isinstance(cdcr_loss, torch.Tensor) else 0.0
        det_val = det_loss.item() if isinstance(det_loss, torch.Tensor) else det_loss
        loss_dict = {
            # V21 兼容键 (epoch_losses 累加器使用)
            'total': total_loss.item(),
            'scene': scene_val,
            'cause': cause_loss.item() if isinstance(cause_loss, torch.Tensor) else 0.0,
            'type': type_val,
            '3d_geo': 0.0,  # V22 Phase 3 不计算 3D 几何损失
            'ttc_3d': 0.0,  # V22 Phase 3 不计算 TTC 损失
            # V22 创新键 (用于详细日志)
            'det': det_val,
            'scene_uasc': scene_val,
            'type_ptcl': type_val,
            'cdcr': cdcr_val,
        }
        return total_loss, loss_dict


# ============================================================
# 工具函数: 从 V21 检查点加载并构建 V22 模型
# ============================================================

def build_temporal_model(detection_model, clip_len=30,
                             ctta_config=None, uasc_config=None,
                             cdcr_config=None, ptcl_config=None):
    """构建 V22 增强时序模型

    Args:
        detection_model: V21 DetectClassifyModelV18 实例 (已加载权重)
        clip_len: Clip 帧数
        ctta_config/uasc_config/cdcr_config/ptcl_config: 创新模块配置

    Returns:
        V24TemporalModel 实例
    """
    # W2修复(Warning): V24TemporalModel 不支持 tracklet (CTTA 替代了对象级建模)
    # 明确传 False 避免误导
    model = V24TemporalModel(
        detection_model=detection_model,
        feat_dim=256,
        num_heads=8,
        depth=4,
        seq_len=clip_len,
        dropout=0.5,
        use_tracklet=False,  # CTTA 已替代对象级 tracklet 建模
        max_objs=8,
        use_3d=getattr(detection_model, 'has_3d', False),
        ctta_config=ctta_config,
        uasc_config=uasc_config,
        cdcr_config=cdcr_config,
        ptcl_config=ptcl_config,
    )
    return model


def print_v24_transformer_info():
    """打印 V22 Transformer 增强信息"""
    print("=" * 70)
    print("V22 增强 Transformer 模块信息")
    print("=" * 70)
    print(f"  [1] CTTA 碰撞类型感知时序注意力")
    print(f"      type_embed_dim: {CTTA_CONFIG['type_embed_dim']}")
    print(f"      attention_heads: {CTTA_CONFIG['attention_heads']}")
    print(f"      temporal_window: {CTTA_CONFIG['temporal_window']}")
    print(f"      架构: L2 标准 + L3-L4 CTTA (2 层条件注意力)")
    print()
    print(f"  [2] UASC 不确定性感知标签平滑")
    print(f"      smoothing: high={UASC_CONFIG['label_smoothing_high']}, "
          f"low={UASC_CONFIG['label_smoothing_low']}")
    print(f"      threshold: {UASC_CONFIG['agreement_threshold']}")
    print()
    print(f"  [3] CDCR 跨域一致性正则化")
    print(f"      lambda: {CDCR_CONFIG['lambda_cdcr']}")
    print(f"      temperature: {CDCR_CONFIG['temperature']}")
    print(f"      warmup: {CDCR_CONFIG['warmup_epochs']} epochs")
    print()
    print(f"  [4] PTCL 渐进式类型课程学习")
    print(f"      Phase 2: {PTCL_CONFIG['phase2_active_types']}")
    print(f"      Phase 3a: {PTCL_CONFIG['phase3a_active_types']}")
    print(f"      Phase 3b: {PTCL_CONFIG['phase3b_active_types']}")
    print(f"      Phase 3c: {PTCL_CONFIG['phase3c_active_types']}")
    print("=" * 70)


if __name__ == '__main__':
    print_v24_transformer_info()
