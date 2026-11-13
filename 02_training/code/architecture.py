#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
architecture.py — V26 架构升级模块 (独立于 V25)

核心升级:
  [1] MultiScaleTemporalPyramid — 多尺度时序金字塔 (15/30/60帧)
  [2] OpticalFlowFusion — 光流模态融合
  [3] IndependentTypeHead — 独立类型分类头 (不依赖 CTTA)
  [4] EnhancedPooling — 多尺度注意力池化 (替代 cls_token)
  [5] DropPath + LabelSmoothing — 正则化

不修改 V25 任何文件, 通过 import 引入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. 多尺度时序金字塔 (核心架构升级)
# ============================================================
class MultiScaleTemporalPyramid(nn.Module):
    """三尺度时序编码: 短(15帧) + 中(30帧) + 长(60帧)
    解决 V25 单尺度 feat_dim=256 容量不足问题
    """
    def __init__(self, feat_dim=512, depths=(2, 3, 2), num_heads=12,
                 scales=(15, 30, 60), dropout=0.1, drop_path=0.1):
        super().__init__()
        self.scales = scales
        self.feat_dim = feat_dim

        # 三尺度 Transformer 编码器
        self.encoders = nn.ModuleList()
        for depth in depths:
            layer = nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=num_heads,
                dim_feedforward=feat_dim * 4,
                dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True,  # Pre-LN
            )
            self.encoders.append(nn.TransformerEncoder(layer, num_layers=depth))

        # 跨尺度注意力融合
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(feat_dim * 3),
            nn.Linear(feat_dim * 3, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        """
        x: [B, T, feat_dim]
        返回: [B, feat_dim] 三尺度融合特征
        """
        B, T, C = x.shape
        scale_feats = []

        for i, (encoder, scale) in enumerate(zip(self.encoders, self.scales)):
            # 自适应池化到目标尺度
            if T >= scale:
                # 下采样到 scale 长度
                indices = torch.linspace(0, T - 1, scale, device=x.device).long()
                x_scaled = x[:, indices, :]
            else:
                # 上采样 (填充)
                x_scaled = F.interpolate(
                    x.transpose(1, 2), size=scale, mode='linear', align_corners=False
                ).transpose(1, 2)

            # Transformer 编码
            feat = encoder(x_scaled)  # [B, scale, C]

            # 注意力池化 (替代简单 mean)
            attn_weights = F.softmax(feat.mean(dim=-1), dim=-1).unsqueeze(-1)
            pooled = (feat * attn_weights).sum(dim=1)  # [B, C]
            scale_feats.append(pooled)

        # 跨尺度融合
        multi_scale = torch.cat(scale_feats, dim=-1)  # [B, 3*C]
        fused = self.output_proj(multi_scale)  # [B, C]
        return self.drop_path(fused)


# ============================================================
# 2. 独立类型分类头 (不依赖 CTTA 条件化)
# ============================================================
class IndependentTypeHead(nn.Module):
    """独立类型分类头 — 消除对 CTTA type_ids 的依赖
    解决 V25 推理时 type_ids=UNKNOWN 导致的类型偏置
    """
    def __init__(self, feat_dim=512, num_types=7, hidden_dim=256,
                 dropout=0.4, label_smoothing=0.1):
        super().__init__()
        self.num_types = num_types
        self.label_smoothing = label_smoothing

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_types),
        )

        # 类型嵌入 (可选, 用于特征增强)
        self.type_embed = nn.Embedding(num_types + 1, feat_dim // 4)

    def forward(self, features, type_labels=None):
        """
        features: [B, feat_dim]
        type_labels: [B] (训练时, 用于对比损失)
        返回: type_logits [B, num_types]
        """
        logits = self.head(features)

        if type_labels is not None and self.training:
            # 对比学习辅助损失 (可选)
            type_emb = self.type_embed(type_labels)  # [B, feat_dim//4]
            # 余弦相似度对比
            features_norm = F.normalize(features[:, :features.shape[-1]//4], dim=-1)
            type_norm = F.normalize(type_emb, dim=-1)
            contrastive = -(features_norm * type_norm).sum(dim=-1).mean()
            return logits, contrastive * 0.1

        return logits

    def compute_loss(self, logits, targets):
        """带标签平滑的交叉熵"""
        return F.cross_entropy(
            logits, targets,
            label_smoothing=self.label_smoothing,
        )


# ============================================================
# 3. 光流特征提取器 (运动模态)
# ============================================================
class OpticalFlowEncoder(nn.Module):
    """轻量光流特征编码 — 提取运动模式
    追尾: 同向减速 | 侧碰: 横向侵入 | 正面: 对向接近
    """
    def __init__(self, feat_dim=256, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),  # 输入: 光流 [B, 2, H, W]
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, feat_dim)

    def forward(self, flow):
        """flow: [B, 2, H, W] → [B, feat_dim]"""
        feat = self.encoder(flow).flatten(1)
        return self.proj(feat)


# ============================================================
# 4. 多模态融合模块
# ============================================================
class MultiModalFusion(nn.Module):
    """视觉 + 光流 + 时序 三模态注意力融合"""
    def __init__(self, feat_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, visual_feat, flow_feat=None, temporal_feat=None):
        """
        visual_feat: [B, C] 主视觉特征
        flow_feat: [B, C] 光流特征 (可选)
        temporal_feat: [B, C] 时序特征 (可选)
        """
        # 构建多模态 token
        tokens = [visual_feat.unsqueeze(1)]
        if flow_feat is not None:
            tokens.append(flow_feat.unsqueeze(1))
        if temporal_feat is not None:
            tokens.append(temporal_feat.unsqueeze(1))

        x = torch.cat(tokens, dim=1)  # [B, N, C]

        # 跨模态注意力
        attn_out, _ = self.cross_attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm(x + self.ffn(x))

        # 聚合
        return x.mean(dim=1)  # [B, C]


# ============================================================
# 7. 车道线检测头 (V26 新增)
# ============================================================
class LaneDetectionHead(nn.Module):
    """车道线检测头 — 检测车道线位置/类型 + 偏离/变道判定

    用途:
      - 偏离车道检测 (lane departure)
      - 违规变道判定 (dangerous_lane_change 成因增强)
      - 车道线类型识别 (solid/dashed/double)
    """
    def __init__(self, feat_dim=512, num_lanes=4, lane_types=None,
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.num_lanes = num_lanes
        self.lane_types = lane_types or ['solid', 'dashed', 'double', 'none']
        self.num_lane_types = len(self.lane_types)

        # 车道线位置回归 (每条车道: 多项式系数 a, b, c, d → y = ax³ + bx² + cx + d)
        self.lane_line_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_lanes * 4),  # 每条车道 4 个多项式系数
        )

        # 车道线类型分类
        self.lane_type_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_lanes * self.num_lane_types),
        )

        # 偏离车道检测
        self.departure_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # 二分类: 偏离/正常
        )

        # 变道检测 (合法/违规)
        self.lane_change_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # 三分类: 不变道/合法变道/违规变道
        )

    def forward(self, features):
        """
        features: [B, feat_dim]
        返回: dict 包含车道线位置/类型/偏离/变道预测
        """
        B = features.shape[0]

        # 车道线位置 (多项式系数)
        lane_coeffs = self.lane_line_head(features).view(B, self.num_lanes, 4)

        # 车道线类型 (每条车道的类型概率)
        lane_type_logits = self.lane_type_head(features).view(
            B, self.num_lanes, self.num_lane_types)

        # 偏离车道
        departure_logit = self.departure_head(features).squeeze(-1)  # [B]

        # 变道检测
        lane_change_logits = self.lane_change_head(features)  # [B, 3]

        return {
            'lane_coeffs': lane_coeffs,           # [B, num_lanes, 4]
            'lane_type_logits': lane_type_logits, # [B, num_lanes, num_types]
            'departure_logit': departure_logit,   # [B]
            'lane_change_logits': lane_change_logits,  # [B, 3]
        }

    def compute_loss(self, outputs, targets):
        """车道线多任务损失"""
        losses = {}
        device = outputs['lane_coeffs'].device

        # 车道线位置回归 (Smooth L1)
        if 'lane_coeffs' in targets:
            losses['lane_line'] = F.smooth_l1_loss(
                outputs['lane_coeffs'], targets['lane_coeffs'])

        # 车道线类型分类
        if 'lane_type' in targets:
            losses['lane_type'] = F.cross_entropy(
                outputs['lane_type_logits'].view(-1, self.num_lane_types),
                targets['lane_type'].view(-1))

        # 偏离检测 (二分类)
        if 'departure' in targets:
            losses['departure'] = F.binary_cross_entropy_with_logits(
                outputs['departure_logit'], targets['departure'].float())

        # 变道检测 (三分类)
        if 'lane_change' in targets:
            losses['lane_change'] = F.cross_entropy(
                outputs['lane_change_logits'], targets['lane_change'])

        return losses


# ============================================================
# 8. DropPath (Stochastic Depth)
# ============================================================
class DropPath(nn.Module):
    """随机深度正则化"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / keep_prob


# ============================================================
# 6. V26 完整时序模型 (集成以上组件)
# ============================================================
class V26TemporalModel(nn.Module):
    """V26 增强时序模型 — 二分类场景 + 7类类型 + 车道线 + 9类成因"""
    def __init__(self, feat_dim=512, num_types=7, num_scenes=2,
                 num_causes=9, drop_path=0.1):
        super().__init__()

        # 多尺度时序金字塔
        self.temporal_pyramid = MultiScaleTemporalPyramid(
            feat_dim=feat_dim, depths=(2, 3, 2), num_heads=12,
            scales=(15, 30, 60), drop_path=drop_path)

        # 光流编码器 (可选)
        self.flow_encoder = OpticalFlowEncoder(feat_dim=feat_dim // 2)

        # 多模态融合
        self.fusion = MultiModalFusion(feat_dim=feat_dim)

        # 独立类型头 (7类事故类型)
        self.type_head = IndependentTypeHead(
            feat_dim=feat_dim, num_types=num_types,
            hidden_dim=256, dropout=0.4, label_smoothing=0.1)

        # 场景头: 二分类 (normal/crash) — V26 移除 pre_crash
        self.scene_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_scenes),  # 2类: normal/crash
        )

        # 成因头 (9类)
        self.cause_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_causes),
        )

        # ★ 车道线检测头 (V26 新增)
        self.lane_head = LaneDetectionHead(
            feat_dim=feat_dim, num_lanes=4,
            lane_types=['solid', 'dashed', 'double', 'none'])

    def forward(self, visual_feat, flow_feat=None, type_labels=None):
        """
        visual_feat: [B, T, C] 时序视觉特征
        flow_feat: [B, 2, H, W] 光流 (可选)
        type_labels: [B] (训练时)
        """
        # 多尺度时序编码
        temporal_feat = self.temporal_pyramid(visual_feat)  # [B, C]

        # 光流编码 (可选)
        flow_emb = None
        if flow_feat is not None:
            flow_emb = self.flow_encoder(flow_feat)  # [B, C//2]
            # 维度对齐
            flow_emb = F.pad(flow_emb, (0, temporal_feat.shape[-1] - flow_emb.shape[-1]))

        # 多模态融合
        fused = self.fusion(temporal_feat, flow_emb)  # [B, C]

        # 多头输出
        scene_logits = self.scene_head(fused)  # [B, 2] 二分类 normal/crash
        cause_logits = self.cause_head(fused)

        # ★ 车道线检测 (V26 新增)
        lane_outputs = self.lane_head(fused)

        # 独立类型头
        if type_labels is not None and self.training:
            type_logits, contrastive_loss = self.type_head(fused, type_labels)
        else:
            type_logits = self.type_head(fused)
            contrastive_loss = torch.tensor(0.0, device=fused.device)

        return {
            'scene_logits': scene_logits,
            'type_logits': type_logits,
            'cause_logits': cause_logits,
            'lane_outputs': lane_outputs,
            'contrastive_loss': contrastive_loss,
        }
