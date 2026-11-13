"""
Temporal Context Module — 1D-CNN + Transformer 四层因果推理 (Phase 3 核心)

严格对齐《模型架构与损失函数子方案》§2 设计：
  ┌─────────┐    ┌──────────────────┐    ┌──────────┐    ┌─────────────────┐
  │ 帧特征序列 │ → │ 1D-CNN 局部运动  │ → │ Position  │ → │ Transformer 4层 │
  │ [B,T,256]│    │ kernel=3 + 残差  │    │ Encoding │    │ nhead=8, GELU   │
  └─────────┘    └──────────────────┘    └──────────┘    └────────┬────────┘
                                                                    │
                                              ┌─────────────────────┘
                                              ▼
                                     ┌───────────────┐
                                     │   CLS Token   │ → 4 任务决策头
                                     │   [B, 256]    │
                                     └───────────────┘

因果推理四层次（渐进训练）：
  L1 (Phase 2): 局部运动感知 — 1D-CNN 检测急刹/变道/加速
  L2 (Phase 2): 时序模式识别 — Transformer 前 2 层 Self-Attention
  L3 (Phase 3a-3b): 因果链推理 — Transformer 后 2 层 + CLS Token
  L4 (Phase 3a-3c): 多因子归因 — 成因/类型分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# V20: 3D 感知模块 (可选导入, 缺失时退化为纯 2D)
try:
    from geometry_3d import GeometryAwareFusion, GeometryHead3D, Geometry3DLoss, TTC3DPredictionHead
    HAS_3D = True
except ImportError:
    HAS_3D = False


# ### D13修复(严重): Phase 3 零损失可见性 — cause/type 在有效样本为零时
# 静默归零, 被总 loss 掩盖。节流告警 (首次 + 每500次) 保持可见性。
_zero_loss_counters = {}


def _warn_zero_loss(name):
    c = _zero_loss_counters.get(name, 0) + 1
    _zero_loss_counters[name] = c
    if c == 1 or c % 500 == 0:
        print(f'[D13] {name} 零返回 (有效样本=0) 累计 {c} 次 — 该损失未生效, '
              f'请检查标签分布/采样配比; 若持续发生需警惕子任务静默零收敛。')


# ### V21 MODIFIED R8-01(致命, D时序模块): GroupNorm 分组数必须整除通道数 —
# FrameFeaturePool 对每个 FPN 层降维到 feat_dim//len(ch_list) (256//3=85),
# 原硬编码 nn.GroupNorm(32, 85) 在构造时即抛 ValueError (85 % 32 != 0),
# Phase 3 启动构建时必崩。统一走整除保护的工具函数 (与
# detect_classify_head.get_groupnorm 同逻辑)。
def _gn(channels, max_groups=32):
    """GroupNorm: 逐步减半 num_groups 直到整除 channels"""
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, channels)


# ── 内置 trunc_normal_ (兼容无 timm 环境) ──
try:
    from timm.models.layers import trunc_normal_
except ImportError:
    import math
    import warnings
    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.
        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.", stacklevel=2)
        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================================
# 1. TemporalContextModule — 时序因果推理核心
# ============================================================

class TemporalContextModule(nn.Module):
    """1D-CNN 局部运动 + Transformer 全局因果推理

    输入: frame_features [B, T, feat_dim] — 每帧的 YOLO 特征 (已池化)
    输出: cls_token   [B, feat_dim]        — 整段序列的因果表示

    参数:
        feat_dim  : 特征维度 (默认 256, 与 YOLO backbone 输出对齐)
        num_heads : Transformer 注意力头数
        depth     : Transformer Encoder 层数 (默认 4: L1-L4)
        seq_len   : 默认序列长度 (实际可动态适配)
        dropout   : Dropout 概率
    """

    def __init__(self, feat_dim=256, num_heads=8, depth=4,
                 seq_len=100, dropout=0.3):
        # seq_len: 位置编码最大支持长度 (默认 100 = max(30@10fps, 90@30fps))
        #          实际输入 T 可以小于此值, pos_embed 会自动截断 [:, :T, :]
        super().__init__()
        self.feat_dim = feat_dim
        self.depth = depth

        # ── L1: 局部运动特征提取 (1D-CNN k=3 + 残差) ──
        # V20-fix(A3): 使用 GroupNorm 替代 BatchNorm1d — Phase 3 batch_size=4 时
        # BN 统计量极不稳定 (仅 4 个样本估计均值/方差), GroupNorm 不依赖 batch 维度
        self.local_cnn = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, feat_dim),
            nn.GELU(),
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
        )

        # ── 可学习位置编码 ──
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, feat_dim))
        trunc_normal_(self.pos_embed, std=.02)

        # ── L2-L4: Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        # ── CLS Token (全局因果聚合) ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        trunc_normal_(self.cls_token, std=.02)

        # ── 输出 LayerNorm ──
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x, return_attention=False, return_frames=False):
        """前向传播

        Args:
            x: [B, T, feat_dim] 帧特征序列
            return_attention: 是否返回注意力权重 (用于因果可视化)
            return_frames: V20 是否返回逐帧输出 (帧级场景分类用)

        Returns:
            cls_out: [B, feat_dim] CLS Token 输出 (因果表示)
            (return_frames) frame_out: [B, T, feat_dim] 逐帧时序表示
            (可选) attn_weights: 最后一层注意力权重
        """
        B, T, C = x.shape

        # 1. 局部运动特征 + 残差连接
        x_cnn = self.local_cnn(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_cnn                     # [B, T, C]

        # 2. 注入位置编码 (支持可变长度)
        x = x + self.pos_embed[:, :T, :]  # [B, T, C]

        # 3. 拼接 CLS Token (无位置编码, 仅学习全局因果语义)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, C]

        # 4. Transformer 全局因果推理
        if return_attention:
            # 需要手动实现以获取注意力权重
            output = x
            attn_weights = []
            for layer in self.transformer_encoder.layers:
                # V20-fix(R2-2): 缓存 norm1 输出, 避免重复计算 3 次
                normed = layer.norm1(output)
                attn_out, attn_w = layer.self_attn(
                    normed, normed, normed,
                    need_weights=True, average_attn_weights=True  # R1-23: PyTorch 正确参数名 (原 average_heads 会 TypeError)
                )
                output = output + layer.dropout1(attn_out)
                # FFN
                ff_out = layer.linear2(layer.dropout(layer.activation(
                    layer.linear1(layer.norm2(output)))))
                output = output + layer.dropout2(ff_out)
                attn_weights.append(attn_w)
            x = output
        else:
            x = self.transformer_encoder(x)

        # 5. 提取 CLS Token + LayerNorm
        cls_out = self.norm(x[:, 0])       # [B, C]
        frame_out = self.norm(x[:, 1:])    # [B, T, C] V20: 逐帧输出 (去掉 CLS)

        if return_attention:
            if return_frames:
                return cls_out, frame_out, attn_weights[-1]
            return cls_out, attn_weights[-1]  # 返回最后一层注意力
        if return_frames:
            return cls_out, frame_out
        return cls_out

    def get_phase3_params(self):
        """获取 Phase 3 分层参数组 (用于渐进式训练)
        返回:
            l12_params: Transformer 前 2 层参数 (Phase 2 已收敛, Phase 3 可冻结)
            l34_params: Transformer 后 2 层参数 (Phase 3a 训练目标)
        """
        l12_params = []
        l34_params = []
        cnn_params = list(self.local_cnn.parameters())
        pos_params = [self.pos_embed]
        cls_params = [self.cls_token]
        norm_params = list(self.norm.parameters())

        for i, layer in enumerate(self.transformer_encoder.layers):
            if i < 2:
                l12_params.extend(layer.parameters())
            else:
                l34_params.extend(layer.parameters())

        return {
            'cnn': cnn_params,           # L1 (通常 Phase 2 已收敛)
            'l12': l12_params,           # L1-L2 Transformer
            'l34': l34_params,           # L3-L4 Transformer
            'pos_embed': pos_params,
            'cls_token': cls_params,
            'norm': norm_params,
        }


# ============================================================
# 2. AccidentDecisionHead — 四任务决策头
# ============================================================

class AccidentDecisionHead(nn.Module):
    """基于 CLS Token 的四任务并行决策头

    CLS Token (256维)
        │
        ├─→ 分支 A: 场景三分类       → [normal, pre_crash, crash]
        ├─→ 分支 B: TTA 回归         → 碰撞倒计时 (0 ~ max_tta 秒)
        ├─→ 分支 C: 事故成因多标签    → 16-bit 向量 (BCE)
        └─→ 分支 D: 事故类型识别      → 10-class (CE)

    参数:
        feat_dim    : 输入特征维度
        nc_cls      : 场景分类数 (默认 3)
        num_causes  : 成因位数 (默认 13)
        num_types   : 事故类型数 (默认 7)
        max_tta     : TTA 最大预测值 (秒)
        dropout     : Dropout
    """

    def __init__(self, feat_dim=256, nc_cls=3, num_causes=9,
                 num_types=7, max_tta=60.0, dropout=0.4):
        super().__init__()
        self.nc_cls = nc_cls
        self.num_causes = num_causes
        self.num_types = num_types
        self.max_tta = max_tta

        # ── 共享特征降维 ──
        self.shared_fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # ── 分支 A: 场景三分类 (V20-fix: 增大容量, 与 Phase 2 DetectClassify 场景头匹配) ──
        self.scene_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, nc_cls),
        )

        # ── 分支 B: TTA 回归 ──
        self.tta_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # ── 分支 C: 成因多标签 ──
        self.cause_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_causes),
        )

        # ── 分支 D: 事故类型 ──
        self.type_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_types),
        )

        # ── 分支 E: 帧级场景分类 (V20, V20-fix: 增大容量) ──
        # 作用于 TemporalContext 逐帧输出 [B,T,feat_dim], 学习“第几帧风险升级”
        # 监督信号密度提升 T 倍, 且天然支持检测延迟 (detection latency) 评估
        self.frame_scene_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, nc_cls),
        )

    def forward_frames(self, frame_seq):
        """V20: 帧级场景分类

        Args:
            frame_seq: [B, T, feat_dim] TemporalContext 逐帧输出

        Returns:
            [B, T, nc_cls] 逐帧场景 logits
        """
        return self.frame_scene_head(frame_seq)

    def forward(self, cls_token):
        """前向传播

        Args:
            cls_token: [B, feat_dim] TemporalContextModule 输出的因果表示

        Returns:
            scene_logits: [B, nc_cls]    场景分类 logits
            tta_pred:     [B, 1]         TTA 预测 (原始值, 训练时由 loss 做 Sigmoid×max_tta)
            cause_logits: [B, num_causes] 成因多标签 logits
            type_logits:  [B, num_types]  事故类型 logits
        """
        feat = self.shared_fc(cls_token)  # [B, 256]

        scene_logits = self.scene_head(feat)   # [B, nc_cls]
        tta_pred = self.tta_head(feat)          # [B, 1]
        cause_logits = self.cause_head(feat)    # [B, num_causes]
        type_logits = self.type_head(feat)      # [B, num_types]

        return scene_logits, tta_pred, cause_logits, type_logits


# ============================================================
# 3. FrameFeaturePool — 帧特征提取 + 池化
# ============================================================

class FrameFeaturePool(nn.Module):
    """从 YOLO backbone 多尺度特征图中提取逐帧特征向量

    输入: YOLO FPN 输出的 list of [B, Ci, Hi, Wi] (P3, P4, P5)
    输出: [B, feat_dim] 归一化帧级特征向量

    V20: 可选 3D 几何融合路径 — 将 GeometryHead3D 预测的全局 3D 统计
         通过 GeometryAwareFusion 门控融合到视觉特征中
    """

    def __init__(self, ch_list, feat_dim=256, use_3d=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.use_3d = use_3d and HAS_3D

        # 每个尺度独立投影, 然后 Concat
        # V20-fix: GroupNorm 替代 BatchNorm2d (Phase 3 batch_size=4 时 BN 不稳定)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, feat_dim // len(ch_list), kernel_size=1),
                _gn(feat_dim // len(ch_list)),
                nn.GELU(),
            ) for c in ch_list
        ])

        # 自适应池化 + 融合
        self.pool = nn.AdaptiveAvgPool2d(1)
        c_fused = (feat_dim // len(ch_list)) * len(ch_list)
        # 补齐到 feat_dim (可能有余数)
        if c_fused != feat_dim:
            self.fuse_fc = nn.Linear(c_fused, feat_dim)
        else:
            self.fuse_fc = nn.Identity()

        # V20: 3D 几何融合路径 (可选)
        if self.use_3d:
            # 从 GeometryHead3D 的 10D 全局统计融合到视觉特征
            self.geom_fusion = GeometryAwareFusion(feat_dim=feat_dim, geom_dim=10)

    def forward(self, features, geom_global=None):
        """提取帧特征

        Args:
            features: list of [B, Ci, Hi, Wi] — YOLO FPN 多尺度输出
            geom_global: V20 可选 [B, 10] 全局 3D 几何统计
                         (depth, l, w, h, sin, cos, vx, vy, vz, speed)

        Returns:
            [B, feat_dim] 归一化特征向量
        """
        pooled = []
        for proj, feat in zip(self.projections, features):
            p = proj(feat)          # [B, C_proj, H, W]
            p = self.pool(p)        # [B, C_proj, 1, 1]
            p = p.view(p.size(0), -1)
            pooled.append(p)

        fused = torch.cat(pooled, dim=1)  # [B, C_fused]
        fused = self.fuse_fc(fused)       # [B, feat_dim]
        fused = F.normalize(fused, p=2, dim=1)

        # V20: 3D 几何门控融合
        if self.use_3d and geom_global is not None:
            fused = self.geom_fusion(fused, geom_global)
            fused = F.normalize(fused, p=2, dim=1)

        return fused


# ============================================================
# 3.5 ObjectTrackletEncoder — 对象级 tracklet 编码器 (V20)
# ============================================================

class ObjectTrackletEncoder(nn.Module):
    """对象级 tracklet 时序编码器 — 显式建模 车-车 / 车-人 交互

    与 FrameFeaturePool (整帧全局池化) 互补:
      FrameFeaturePool      → 整帧语义统计 (丢失对象级空间关系, "看不清哪辆车在接近")
      ObjectTrackletEncoder → 对象级 RoI 视觉特征 + 几何关系 + 类别嵌入,
                              帧内空间 Self-Attention 显式建模对象间相对运动

    输入:
        roi_feats  [B,T,K,C_roi] RoIAlign 提取的对象视觉特征
        geom_feats [B,T,K,geom_dim] 几何特征
                   V20 (geom_dim=8):  [cx,cy,w,h,log(area),w/h,dx,dy]
                   V20 (geom_dim=15): [cx,cy,w,h,log(area),w/h,dx,dy,
                                       x3d,y3d,z3d,vx,vy,vz,speed]
        obj_types  [B,T,K]       对象类别 (0=car,1=person,2=cyclist; padding 处任意)
        obj_mask   [B,T,K]       有效对象掩码 (True=有效, False=padding)

    输出:
        frame_repr [B,T,feat_dim] 帧级对象聚合表示
    """

    def __init__(self, roi_dim=256, geom_dim=8, feat_dim=256, num_det_classes=3,
                 num_heads=4, depth=2, max_objs=8, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_objs = max_objs
        self.num_det_classes = num_det_classes

        # 三路特征投影 (相加式融合, 保持维度)
        self.roi_proj = nn.Sequential(
            nn.Linear(roi_dim, feat_dim), nn.LayerNorm(feat_dim), nn.GELU())
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, feat_dim), nn.LayerNorm(feat_dim), nn.GELU())
        # +1 为 padding 槽位 (索引 num_det_classes)
        self.type_embed = nn.Embedding(num_det_classes + 1, feat_dim)

        self.obj_fuse = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.GELU(),
            nn.Dropout(dropout))

        # 对象排序位置编码 (按面积降序排列, 第 k 大对象)
        self.obj_pos = nn.Parameter(torch.zeros(1, 1, max_objs, feat_dim))
        trunc_normal_(self.obj_pos, std=.02)

        # 帧内空间交互 Transformer (对象间注意力: 谁与谁存在风险关系)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=num_heads,
            dim_feedforward=feat_dim * 2, dropout=dropout,
            batch_first=True, activation='gelu')
        self.spatial_transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # 池化融合: masked_mean + masked_max → feat_dim
        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.LayerNorm(feat_dim), nn.GELU())

    def forward(self, roi_feats, geom_feats, obj_types, obj_mask):
        B, T, K, _ = roi_feats.shape

        # 类型索引: padding → num_det_classes 槽位
        pad_idx = self.num_det_classes
        type_idx = torch.where(
            obj_mask, obj_types.clamp(0, pad_idx - 1),
            torch.full_like(obj_types, pad_idx))

        # 三路融合: 视觉 + 几何 + 类别
        h = self.roi_proj(roi_feats) + self.geom_proj(geom_feats) \
            + self.type_embed(type_idx)
        h = self.obj_fuse(h)                       # [B,T,K,C]
        h = h + self.obj_pos[:, :, :K, :]          # 对象排序位置编码

        # 帧内空间交互: [B*T, K, C]
        h_flat = h.reshape(B * T, K, -1)
        kpm = ~obj_mask.reshape(B * T, K)          # key_padding_mask: True=忽略

        # 全 padding 帧防护 (全 mask 行会导致 softmax NaN)
        all_pad = kpm.all(dim=1)                   # [B*T]
        if all_pad.any():
            kpm = kpm.clone()
            kpm[all_pad] = False                   # 临时放开, 输出后再清零

        h_flat = self.spatial_transformer(h_flat, src_key_padding_mask=kpm)
        h = h_flat.reshape(B, T, K, -1)

        # masked pooling: mean + max
        mask_f = obj_mask.unsqueeze(-1).float()    # [B,T,K,1]
        h_masked = h * mask_f
        mean_pool = h_masked.sum(2) / mask_f.sum(2).clamp(min=1.0)   # [B,T,C]
        max_pool = h_masked.masked_fill(
            ~obj_mask.unsqueeze(-1), float('-inf')).max(2).values
        max_pool = torch.where(
            torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)

        frame_repr = self.out_proj(
            torch.cat([mean_pool, max_pool], dim=-1))                # [B,T,C]
        # 全 padding 帧输出清零
        frame_repr = frame_repr * obj_mask.any(dim=2, keepdim=True).float()
        return frame_repr


# ============================================================
# 4. V18TemporalModel — 完整时序因果推理模型
# ============================================================

class V18TemporalModel(nn.Module):
    """YOLO26 Backbone + DetectClassify + TemporalContext 完整架构

    三阶段训练模式:
      Phase 2 模式 (temporal_mode=False):
        - 逐帧处理, 仅使用 DetectClassify 头 (与 train.py 行为一致)
        - 时序模块不参与前向/梯度计算

      Phase 3 模式 (temporal_mode=True):
        - 处理 Clip 序列 [B, T, 3, H, W]
        - YOLO backbone 逐帧提取特征 → TemporalContextModule → 决策头
        - 输出: 逐帧检测 + Clip 级场景/成因/类型/TTA

    参数:
        detection_model  : DetectClassifyModelV18 实例 (已加载权重)
        feat_dim         : 时序特征维度
        num_heads        : Transformer 注意力头数
        depth            : Transformer 层数
        seq_len          : 默认序列长度
        use_tracklet     : V20 是否启用对象级 tracklet 建模 (RoIAlign + 帧内空间注意力)
        max_objs         : V20 每帧最大对象数 (按面积降序取 Top-K)
    """

    def __init__(self, detection_model, feat_dim=256, num_heads=8,
                 depth=4, seq_len=100, dropout=0.3,
                 use_tracklet=True, max_objs=8, use_3d=False):
        # seq_len: 位置编码最大支持长度, 由各数据集 clip_len 传入
        #          TUMTraf (10fps) → 30, Accid3nD (25fps) → 75, BDD100K (30fps) → 90
        super().__init__()
        self.detection_model = detection_model
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.use_tracklet = use_tracklet
        self.max_objs = max_objs
        self.use_3d = use_3d and HAS_3D

        # 获取 YOLO 多尺度通道数
        detect_head = detection_model.model[-1]
        ch_list = [seq[0].conv.in_channels for seq in detect_head.cv2]
        self.ch_list = ch_list

        # ── 模块 1: 帧特征池化 ──
        self.frame_pool = FrameFeaturePool(ch_list, feat_dim, use_3d=self.use_3d)

        # ── 模块 1.5: 对象级 tracklet 编码 (V20) ──
        if use_tracklet:
            # RoI 投影: 每个 FPN 尺度独立 1x1 conv 降到 feat_dim
            # V20-fix: GroupNorm 替代 BatchNorm2d
            self.roi_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, feat_dim, kernel_size=1),
                    _gn(feat_dim),
                    nn.GELU(),
                ) for c in ch_list])
            self.roi_out_size = 2  # RoIAlign 输出 2x2 再 GAP
            # V20: geom_dim 从 8 扩展到 15 (新增 x3d,y3d,z3d,vx,vy,vz,speed)
            geom_dim = 15 if self.use_3d else 8
            self.tracklet_encoder = ObjectTrackletEncoder(
                roi_dim=feat_dim, geom_dim=geom_dim, feat_dim=feat_dim,
                num_det_classes=3, max_objs=max_objs, dropout=dropout)
            # 全局-对象门控融合: gate = sigmoid(MLP([global; tracklet]))
            self.tracklet_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim), nn.Sigmoid())

        # ── 模块 1.6: 3D 几何提取头 (V20) ──
        if self.use_3d:
            self.geometry_head = GeometryHead3D(ch_list)
            # V20-fix(W5): 从 DetectClassify 头复制预训练 GeometryHead3D 权重,
            # 避免 Phase 3 重复实例从随机权重开始训练 (Phase 1/2 已收敛的 3D 头被浪费)
            if hasattr(detect_head, 'geometry_head') and detect_head.geometry_head is not None:
                try:
                    self.geometry_head.load_state_dict(
                        detect_head.geometry_head.state_dict())
                    print("  ✓ 从 DetectClassify 复制 GeometryHead3D 预训练权重")
                except Exception:
                    print("  ⚠ GeometryHead3D 权重复制失败, 使用随机初始化")
            self.ttc_3d_head = TTC3DPredictionHead(
                feat_dim=feat_dim, max_objs=max_objs, max_ttc=10.0)

        # ── 模块 2: 时序因果推理 ──
        self.temporal_context = TemporalContextModule(
            feat_dim=feat_dim,
            num_heads=num_heads,
            depth=depth,
            seq_len=seq_len,
            dropout=dropout,
        )

        # ── 模块 3: 四任务决策头 ──
        nc_cls = getattr(detection_model, 'nc_cls', 3)
        self.decision_head = AccidentDecisionHead(
            feat_dim=feat_dim,
            nc_cls=nc_cls,
            num_causes=9,
            num_types=7,
            dropout=dropout,
        )

        self.nc_cls = nc_cls

    def forward(self, x, temporal_mode=False,
                gt_boxes_seq=None, gt_classes_seq=None, obj_mask_seq=None,
                gt_3d_seq=None):
        """前向传播

        Args:
            x: [B, 3, H, W]          — Phase 2 模式 (单帧)
               或 [B, T, 3, H, W]     — Phase 3 模式 (Clip 序列)
            temporal_mode: 是否启用时序推理
            gt_boxes_seq:   V20 [B,T,K,4] 归一化 xywh GT 框 (训练时对象级建模)
            gt_classes_seq: V20 [B,T,K] 对象类别 (0=car,1=person,2=cyclist)
            obj_mask_seq:   V20 [B,T,K] 有效对象掩码
            gt_3d_seq:      V20 可选 dict — 3D GT 信息 (用于 3D 损失计算)

        Returns:
            Phase 2: (det_out, scene_logits, tta_pred, cause_logits, type_logits, reid_maps)
            Phase 3: (det_out, scene_logits, tta_pred, cause_logits, type_logits,
                      frame_scene_logits, geo_3d_out)  — V20 新增 3D 输出
        """
        if not temporal_mode or x.dim() == 4:
            # Phase 2: 逐帧模式 (原始 V18 行为)
            return self.detection_model(x)

        # Phase 3: Clip 时序模式 [B, T, 3, H, W]
        B, T, C, H, W = x.shape

        # ── Step 1: 逐帧提取特征 ──
        x_flat = x.view(B * T, C, H, W)
        frame_features = self._extract_frame_features(x_flat)
        # frame_features: list of [B*T, Ci, Hi, Wi]

        # ── Step 1.5: V20 3D 几何提取 ──
        geo_3d_out = None
        geom_global = None
        geom_per_obj = None
        if self.use_3d:
            geo_3d_out, geom_global, geom_per_obj = self._extract_3d_geometry(
                frame_features, B, T,
                gt_boxes_seq=gt_boxes_seq, obj_mask_seq=obj_mask_seq)

        # ── Step 2: 逐帧池化 → 帧特征序列 ──
        frame_vecs = self.frame_pool(frame_features, geom_global=geom_global)
        frame_vecs = frame_vecs.view(B, T, self.feat_dim)  # [B, T, feat_dim]

        # ── Step 2.5: 对象级 tracklet 特征融合 (V20) ──
        roi_seq = None  # V20-fix(C3): 保留对象级特征供 TTC 头使用
        if self.use_tracklet and gt_boxes_seq is not None:
            tracklet_feat, roi_seq = self._extract_tracklet_features(
                frame_features, gt_boxes_seq, gt_classes_seq, obj_mask_seq,
                B, T, (W, H), geom_per_obj=geom_per_obj)  # [B,T,feat_dim], [B,T,K,feat_dim]
            # 门控残差融合
            gate = self.tracklet_gate(
                torch.cat([frame_vecs, tracklet_feat], dim=-1))
            frame_vecs = frame_vecs + gate * tracklet_feat

        # ── Step 3: 时序因果推理 (V20: 返回逐帧输出) ──
        cls_token, frame_seq = self.temporal_context(frame_vecs, return_frames=True)

        # ── Step 4: 四任务决策 + 帧级场景分类 (V20) ──
        scene_logits, tta_pred, cause_logits, type_logits = \
            self.decision_head(cls_token)
        frame_scene_logits = self.decision_head.forward_frames(frame_seq)

        # ── Step 4.5: V20 3D TTC 预测 ──
        # V20-fix(C2+C3): 使用对象级 roi_seq (4D) 而非帧级 frame_seq (3D);
        #                   正确解包 TTC 头返回的 (ttc_pred, risk_pred) 元组
        ttc_3d_pred = None
        if self.use_3d and roi_seq is not None and geom_per_obj is not None:
            ttc_3d_pred, _ = self.ttc_3d_head(
                roi_seq, geom_per_obj.view(B, T, -1, geom_per_obj.shape[-1]),
                obj_mask_seq)  # ttc_3d_pred: [B,T]

        # ── Step 5: 检测输出 (Phase 3 不重复计算) ──
        det_out = None

        # V20-fix(C1): 解码 geo_3d 原始输出, 供损失函数直接使用
        geo_3d_decoded = None
        if geo_3d_out is not None:
            geo_3d_decoded = self.geometry_head.decode_3d(geo_3d_out)

        # V20: 7 元组输出
        return (det_out, scene_logits, tta_pred, cause_logits, type_logits,
                frame_scene_logits, {'geo_3d': geo_3d_out,
                                      'geo_3d_decoded': geo_3d_decoded,
                                      'ttc_3d': ttc_3d_pred})

    def _extract_tracklet_features(self, frame_features, boxes_seq, classes_seq,
                                   obj_mask_seq, B, T, img_wh, geom_per_obj=None):
        """V20: RoIAlign 提取对象级 RoI 特征 → tracklet 编码

        训练时使用 GT 框 (标准做法); 推理时可替换为检测预测框。

        Args:
            frame_features: list of [B*T, Ci, Hi, Wi] (P3,P4,P5)
            boxes_seq:   [B,T,K,4] 归一化 xywh
            classes_seq: [B,T,K] long
            obj_mask_seq:[B,T,K] bool
            img_wh: (W, H) 输入图像像素尺寸
            geom_per_obj: V20 可选 [B*T, K, 7] 每对象 3D 几何 (x3d,y3d,z3d,vx,vy,vz,speed)

        Returns:
            tracklet_feat: [B,T,feat_dim] 帧级对象聚合表示
            roi_seq: [B,T,K,feat_dim] 对象级 RoI 特征 (V20-fix(C3): 供 TTC3DPredictionHead 使用)
        """
        from torchvision.ops import roi_align
        W_img, H_img = img_wh
        K = boxes_seq.shape[2]
        device = boxes_seq.device
        feat_dtype = frame_features[0].dtype

        boxes_flat = boxes_seq.view(B * T, K, 4)
        classes_flat = classes_seq.view(B * T, K)
        mask_flat = obj_mask_seq.view(B * T, K)

        # 归一化 xywh → 像素 xyxy
        cx, cy, w, h = boxes_flat.unbind(-1)
        x1 = (cx - w / 2) * W_img
        y1 = (cy - h / 2) * H_img
        x2 = (cx + w / 2) * W_img
        y2 = (cy + h / 2) * H_img
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)   # [N,K,4]

        # FPN 层级分配 (Detectron 规则): k = clamp(floor(4 + log2(sqrt(w*h)/224)), 3, 5)
        w_px = (x2 - x1).clamp(min=1.0)
        h_px = (y2 - y1).clamp(min=1.0)
        levels = torch.floor(4 + torch.log2(torch.sqrt(w_px * h_px) / 224))
        levels = levels.clamp(3, 5).long() - 3               # [N,K] ∈ {0,1,2}

        # 按 FPN 层级分组 RoIAlign
        roi_feats = torch.zeros(B * T, K, self.feat_dim,
                                device=device, dtype=feat_dtype)
        strides = [8, 16, 32]
        for lvl in range(3):
            sel = (levels == lvl) & mask_flat                # [N,K]
            if not sel.any():
                continue
            proj_feat = self.roi_projs[lvl](frame_features[lvl])  # [N,C_roi,Hf,Wf]
            n_idx, k_idx = sel.nonzero(as_tuple=True)
            rois = torch.cat([
                n_idx.float().unsqueeze(1).to(feat_dtype),
                boxes_xyxy[n_idx, k_idx].to(feat_dtype)], dim=1)  # [M,5]
            pooled = roi_align(proj_feat, rois,
                               output_size=(self.roi_out_size, self.roi_out_size),
                               spatial_scale=1.0 / strides[lvl],
                               aligned=True)                   # [M,C_roi,2,2]
            pooled = pooled.mean(dim=(2, 3))                   # [M,C_roi] GAP
            roi_feats.index_put_((n_idx, k_idx), pooled)

        # 几何特征: V20 8D [cx,cy,w,h,log(area),w/h,dx,dy]
        area = (w * h).clamp(min=1e-6)
        geom_2d = torch.stack([
            cx, cy, w, h, torch.log(area + 1e-6),
            (w / h.clamp(min=1e-6)).clamp(0.01, 100.0),
            cx - 0.5, cy - 0.5], dim=-1)                      # [B*T,K,8]

        # V20: 扩展到 15D (2D 8D + 3D 7D)
        if geom_per_obj is not None and self.use_3d:
            # geom_per_obj: [B*T, K, 7] = (x3d, y3d, z3d, vx, vy, vz, speed)
            geom = torch.cat([geom_2d, geom_per_obj], dim=-1)  # [B*T,K,15]
        elif self.use_3d:
            # 3D 启用但无几何数据: 补零
            pad_3d = torch.zeros(*geom_2d.shape[:-1], 7,
                                 device=geom_2d.device, dtype=geom_2d.dtype)
            geom = torch.cat([geom_2d, pad_3d], dim=-1)        # [B*T,K,15]
        else:
            geom = geom_2d                                      # [B*T,K,8]

        geom = geom.view(B, T, K, -1)

        roi_seq = roi_feats.view(B, T, K, self.feat_dim)
        tracklet_feat = self.tracklet_encoder(
            roi_seq, geom, classes_seq, obj_mask_seq)         # [B,T,feat_dim]
        # V20-fix(A2): L2 归一化 — frame_vecs 经过 FrameFeaturePool 的 F.normalize,
        # tracklet_feat 必须同尺度, 否则门控融合时 tracklet 幅值主导 (gate*tracklet >> frame_vecs)
        tracklet_feat = F.normalize(tracklet_feat, p=2, dim=-1)
        # V20-fix(C3): 同时返回 roi_seq 供 TTC3DPredictionHead 使用
        return tracklet_feat, roi_seq

    def _extract_3d_geometry(self, frame_features, B, T,
                             gt_boxes_seq=None, obj_mask_seq=None):
        """V20: 从 FPN 特征提取 3D 几何信息

        Args:
            frame_features: list of [B*T, Ci, Hi, Wi] (P3, P4, P5)
            B, T: batch 和 时序维度
            gt_boxes_seq: 可选 [B,T,K,4] 用于提取对象级 3D
            obj_mask_seq: 可选 [B,T,K] 有效对象掩码

        Returns:
            geo_3d_raw: list of [B*T, A*10, Hi, Wi] — GeometryHead3D 原始输出 (用于损失)
            geom_global: [B*T, 10] 全局 3D 统计 (depth,l,w,h,sin,cos,vx,vy,vz,speed)
            geom_per_obj: [B*T, K, 7] 每对象 3D (x3d,y3d,z3d,vx,vy,vz,speed) 或 None
        """
        # 通过 GeometryHead3D 回归 3D 属性
        geo_3d_raw = self.geometry_head(frame_features)  # list of [B*T, A*10, Hi, Wi]

        # 解码为结构化字典
        decoded = self.geometry_head.decode_3d(geo_3d_raw)

        # 全局 3D 统计: 对 P3 尺度做全局平均池化 (P3 分辨率最高, 信息最富)
        p3 = decoded[0]  # dict with depth[B*T,A,H,W,1], dims[B*T,A,H,W,3], ...
        # 对 anchor 和空间维度取均值 → [B*T, D]
        depth_global = p3['depth'].mean(dim=(1, 2, 3))       # [B*T, 1]
        dims_global = p3['dims'].mean(dim=(1, 2, 3))         # [B*T, 3]
        orient_global = p3['orient'].mean(dim=(1, 2, 3))     # [B*T, 2]
        orient_global = F.normalize(orient_global, dim=-1)
        vel_global = p3['vel'].mean(dim=(1, 2, 3))           # [B*T, 3]
        speed_global = vel_global.norm(dim=-1, keepdim=True)  # [B*T, 1]
        geom_global = torch.cat([
            depth_global, dims_global, orient_global,
            vel_global, speed_global], dim=-1)                # [B*T, 10]

        # 对象级 3D: 使用 GT 框中心位置采样 3D 属性
        geom_per_obj = None
        if gt_boxes_seq is not None and obj_mask_seq is not None:
            K = gt_boxes_seq.shape[2]
            device = gt_boxes_seq.device
            boxes_flat = gt_boxes_seq.view(B * T, K, 4)  # [B*T, K, 4] normalized cx,cy,w,h
            mask_flat = obj_mask_seq.view(B * T, K)       # [B*T, K]

            # 使用 P3 尺度 (stride=8) 采样对象中心位置的 3D 属性
            _, A, H, W, _ = p3['depth'].shape
            cx_norm = boxes_flat[..., 0]  # [B*T, K]
            cy_norm = boxes_flat[..., 1]

            # 归一化坐标 → 特征图坐标
            gx = (cx_norm * W).long().clamp(0, W - 1)  # [B*T, K]
            gy = (cy_norm * H).long().clamp(0, H - 1)

            # 对 anchor 维度取均值 → [B*T, H, W, D]
            depth_map = p3['depth'].mean(dim=1)     # [B*T, H, W, 1]
            vel_map = p3['vel'].mean(dim=1)         # [B*T, H, W, 3]

            # 采样对象中心位置
            bt_idx = torch.arange(B * T, device=device).unsqueeze(1).expand(-1, K)
            obj_depth = depth_map[bt_idx, gy, gx, 0]        # [B*T, K]
            obj_vel = vel_map[bt_idx, gy, gx, :]            # [B*T, K, 3]
            obj_speed = obj_vel.norm(dim=-1)                 # [B*T, K]

            # 3D 位置: 使用深度 + 2D 中心估算 (cx*depth, cy*depth, depth)
            obj_x3d = cx_norm * obj_depth
            obj_y3d = cy_norm * obj_depth
            obj_z3d = obj_depth

            # 组装: [B*T, K, 7] = (x3d, y3d, z3d, vx, vy, vz, speed)
            geom_per_obj = torch.stack([
                obj_x3d, obj_y3d, obj_z3d,
                obj_vel[..., 0], obj_vel[..., 1], obj_vel[..., 2],
                obj_speed], dim=-1)                          # [B*T, K, 7]

            # 无效对象置零
            geom_per_obj = geom_per_obj * mask_flat.unsqueeze(-1).float()

        return geo_3d_raw, geom_global, geom_per_obj

    def _extract_frame_features(self, x):
        """从 YOLO backbone 提取多尺度特征 (不经过 Detect 头)

        Args:
            x: [N, 3, H, W]

        Returns:
            list of [N, Ci, Hi, Wi] — P3, P4, P5 特征图
        """
        # #12 修复: 删除无效的逐层遍历循环 (循环体全是 pass, 未收集任何特征),
        # 直接委托给 _forward_until_detect, 它已正确实现逐层推理 + 特征收集
        return self._forward_until_detect(x)

    def _forward_until_detect(self, x):
        """逐层前向直到 Detect 层之前, 收集 P3/P4/P5 特征图

        V20-fix: 修复两个致命 BUG:
          1. f=-1 的层接收原始输入 x 而非前一层输出 y[i-1]
          2. Concat 层 m.f=[-1, 4] 中 -1 被 `if f_idx >= 0` 过滤掉
        """
        model = self.detection_model.model
        y = [None] * len(model)
        features_for_detect = []

        for i, m in enumerate(model):
            if isinstance(m.f, int):
                if m.f == -1:
                    # V20-fix: f=-1 表示取前一层输出 (i=0 时取原始输入)
                    y[i] = m(x if i == 0 else y[i - 1])
                else:
                    y[i] = m(y[m.f])
            elif isinstance(m.f, (list, tuple)):
                # V20-fix: 正确处理 -1 索引 (前一层输出)
                inputs = []
                for f_idx in m.f:
                    if f_idx == -1:
                        inputs.append(x if i == 0 else y[i - 1])
                    elif 0 <= f_idx < len(y) and y[f_idx] is not None:
                        inputs.append(y[f_idx])
                y[i] = m(inputs)
            else:
                y[i] = m(x if i == 0 else y[i - 1])

            # 收集多尺度特征: Detect 层的输入特征
            if hasattr(m, 'cv2') and hasattr(m, 'cv3'):
                if isinstance(m.f, (list, tuple)):
                    for idx in m.f:
                        src = (i - 1) if idx == -1 else idx
                        if 0 <= src < len(y) and y[src] is not None:
                            features_for_detect.append(y[src])
                break

        if not features_for_detect:
            # 备用方案: 使用最后几层中间特征
            for idx in range(len(y) - 3, len(y)):
                if y[idx] is not None and isinstance(y[idx], torch.Tensor):
                    if y[idx].dim() == 4:
                        features_for_detect.append(y[idx])
            if len(features_for_detect) < 3:
                raise RuntimeError(
                    "无法从 YOLO backbone 提取多尺度特征。"
                    "请检查模型结构或使用 V18TemporalModel.extract_features 的 fallback 模式。"
                )

        return features_for_detect[-3:]  # 取最后 3 个尺度: P3, P4, P5

    # ── Phase 3 分层参数冻结/解冻 ──

    def freeze_for_phase3a(self):
        """Phase 3a 模式: 冻结 Backbone + Transformer L1-L2"""
        for p in self.detection_model.model.parameters():
            p.requires_grad = False
        for p in self.frame_pool.parameters():
            p.requires_grad = False
        for p in self.temporal_context.local_cnn.parameters():
            p.requires_grad = False
        self.temporal_context.pos_embed.requires_grad = False

        # 冻结 Transformer 前 2 层
        for i in range(2):
            for p in self.temporal_context.transformer_encoder.layers[i].parameters():
                p.requires_grad = False

    def unfreeze_for_phase3b(self):
        """Phase 3b 模式: 解冻 Transformer 全层 + Backbone 最后 2 stage"""
        # 解冻 Transformer 全层
        for p in self.temporal_context.transformer_encoder.parameters():
            p.requires_grad = True
        for p in self.temporal_context.local_cnn.parameters():
            p.requires_grad = True
        self.temporal_context.pos_embed.requires_grad = True

        # 解冻 Backbone 最后 2 stage
        backbone = self.detection_model.model
        # YOLO backbone 最后 2 层 (通常索引 -4, -3 对应 P4, P5 的 Conv 层)
        total = len(backbone)
        for i in range(max(0, total - 4), total):
            m = backbone[i]
            if hasattr(m, 'parameters'):
                for p in m.parameters():
                    p.requires_grad = True

    def unfreeze_all(self):
        """Phase 3c 模式: 解冻所有层"""
        for p in self.parameters():
            p.requires_grad = True


# ============================================================
# 5. 时序损失函数 (Phase 3)
# ============================================================

class TemporalMultiTaskLoss(nn.Module):
    """Phase 3 多任务联合损失 (V20: 含 3D 损失)

    L_total = L_det + λ₂·L_scene + λ₃·L_tta + λ₄·L_cause + λ₅·L_type
              + λ₆·L_3d_geo + λ₇·L_ttc_3d

    参数:
        nc_cls        : 场景分类数
        num_causes    : 成因位数
        cls_weight    : 场景分类类别权重 [normal, pre_crash, crash]
        max_tta       : TTA 最大值
        lambda_scene  : λ₂ 场景损失权重
        lambda_tta    : λ₃ TTA 损失权重
        lambda_cause  : λ₄ 成因损失权重
        lambda_type   : λ₅ 类型损失权重
        lambda_3d     : λ₆ 3D 几何损失权重 (V20)
        lambda_ttc_3d : λ₇ 3D TTC 损失权重 (V20)
    """

    def __init__(self, nc_cls=3, num_causes=9, cls_weight=None,
                 max_tta=60.0, lambda_scene=2.0, lambda_tta=0.0,
                 lambda_cause=0.8, lambda_type=0.3,
                 lambda_3d=0.3, lambda_ttc_3d=0.5):
        super().__init__()
        self.max_tta = max_tta
        self.lambda_scene = lambda_scene
        self.lambda_tta = lambda_tta
        self.lambda_cause = lambda_cause
        self.lambda_type = lambda_type
        self.lambda_3d = lambda_3d
        self.lambda_ttc_3d = lambda_ttc_3d

        if cls_weight is not None:
            cls_weight_tensor = torch.tensor(cls_weight, dtype=torch.float32)
        else:
            # V21-fix: 与 Phase 2 CLASS_WEIGHTS 对齐 [1.0, 1.2, 1.1] (单一真相源 DEFAULT_CLASS_WEIGHTS)
            cls_weight_tensor = torch.tensor([1.0, 1.2, 1.1], dtype=torch.float32)
        # V20-fix(W4): 使用 register_buffer 替代裸 tensor, .to(device) 自动迁移
        self.register_buffer('cls_weight', cls_weight_tensor)
        # V21-fix: focal gamma per class — 与 Phase 2 AsymmetricFocalLoss 对齐
        # V24: 恢复对称 gamma=1.0 — 根因 (real normal 数据缺失) 已修复
        self.register_buffer('scene_gamma',
                             torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

        # V20-fix(W4): 沿用 Phase 2 的 focal CE (含 label smoothing), 而非 plain CE
        # 避免阶段切换时 loss 量纲突变导致优化震荡
        # V21: label smoothing 0.05→0.1 提升泛化 (改进方案§4.3.2)
        self.scene_loss_fn = nn.CrossEntropyLoss(
            weight=self.cls_weight, label_smoothing=0.1,
            ignore_index=-100)
        self.tta_loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.num_causes = num_causes
        # #4 修复: cause loss 加 pos_weight 处理类别不均衡 (行人 bit5 权重最高)
        # R1-01: 原 6 元素表 [:16] 切片仍为 6 维, 与 16 类 cause_logits 广播失败必崩;
        # 复用 base_config.CAUSE_POS_WEIGHT (16 元素, train.py 动态覆盖) 并按 num_causes 补齐
        try:
            from base_config import CAUSE_POS_WEIGHT as _CPW
        except ImportError:
            _CPW = [1.0, 1.5, 1.5, 2.0, 1.5, 5.0]
        cause_pos_weight = (list(_CPW) + [1.0] * num_causes)[:num_causes]
        self.register_buffer('cause_pos_weight',
                             torch.tensor(cause_pos_weight, dtype=torch.float32))
        self.type_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # 修复: type 标签含 -1 (无标注), 忽略否则负索引回绕污染
        # V20: 3D 损失函数
        if HAS_3D:
            self.geo_3d_loss_fn = Geometry3DLoss()
        else:
            self.geo_3d_loss_fn = None
        self.ttc_3d_loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, preds, batch, det_loss=0.0):
        """计算联合损失

        Args:
            preds: V20 七元组 (det_out, scene_logits, tta_pred, cause_logits,
                   type_logits, frame_scene_logits, geo_3d_dict)
                   向后兼容六元组 (无 geo_3d_dict) 和五元组
            batch: 数据 batch, 包含 'scene_cls', 'scene_cls_seq'(V20 逐帧),
                   'cause', 'accident_type', 'tta', 'ttc_3d_gt'(V20) 等
            det_loss: 来自 YOLO 的检测损失 (标量)

        Returns:
            total_loss: 标量
            loss_dict: {scene, tta, cause, type, 3d_geo, ttc_3d, total}
        """
        # V20: 解包 7 元组, 向后兼容 6/5 元组
        frame_scene_logits = None
        geo_3d_dict = None
        if len(preds) == 7:
            (_, scene_logits, tta_pred, cause_logits, type_logits,
             frame_scene_logits, geo_3d_dict) = preds
        elif len(preds) == 6:
            _, scene_logits, tta_pred, cause_logits, type_logits, frame_scene_logits = preds
        else:
            _, scene_logits, tta_pred, cause_logits, type_logits = preds
        device = scene_logits.device

        # ── L_scene: 场景分类 (V20: 帧级 + clip 级双路监督) ──
        # V20-fix(C1): 追加 clip 级辅助 scene loss, 避免 scene_head 零梯度
        scene_loss = torch.tensor(0.0, device=device)
        if frame_scene_logits is not None and 'scene_cls_seq' in batch:
            # V20 帧级监督 (主): [B,T,nc_cls] vs [B,T] — 监督信号密度提升 T 倍
            scene_seq = self._to_tensor(batch['scene_cls_seq'], device, dtype=torch.long)
            # 标签映射: 1(normal)→0, 2(pre_crash)→1, 3(crash)→2; 越界→-100
            valid_seq = (scene_seq >= 1) & (scene_seq <= 3)
            seq_idx = (scene_seq - 1).clamp(0, 2)  # 1→0, 2→1, 3→2
            seq_idx[~valid_seq] = -100             # ignore/越界 → PyTorch 标准忽略值
            B, T, C = frame_scene_logits.shape
            scene_loss = self._focal_ce(
                frame_scene_logits.reshape(B * T, C), seq_idx.reshape(B * T))
            # V20-fix(C1): clip 级辅助 scene loss — 训练 AccidentDecisionHead.scene_head
            if 'scene_cls' in batch:
                scene_gt = self._to_tensor(batch['scene_cls'], device, dtype=torch.long)
                valid_clip = (scene_gt >= 1) & (scene_gt <= 3)
                clip_idx = (scene_gt - 1).clamp(0, 2)
                clip_idx[~valid_clip] = -100
                clip_loss = self._focal_ce(scene_logits, clip_idx)
                scene_loss = scene_loss + 0.3 * clip_loss  # 辅助权重
        elif 'scene_cls' in batch:
            # 回退: 仅 clip 级监督 (Phase 2 兼容)
            scene_gt = self._to_tensor(batch['scene_cls'], device, dtype=torch.long)
            valid_clip = (scene_gt >= 1) & (scene_gt <= 3)
            scene_cls_idx = (scene_gt - 1).clamp(0, 2)
            scene_cls_idx[~valid_clip] = -100
            scene_loss = self._focal_ce(scene_logits, scene_cls_idx)

        # ── L_tta: TTA 回归 ──
        if 'tta' in batch:
            tta_gt = self._to_tensor(batch['tta'], device, dtype=torch.float32)
            valid_mask = (tta_gt >= 0) & (tta_gt <= self.max_tta)
            if valid_mask.any():
                # 修复: tta_head 输出原始无界值, 需 Sigmoid×max_tta 映射到 [0, max_tta] 秒
                # (与 tta_head 文档约定一致: "训练时由 loss 做 Sigmoid×max_tta")
                tta_pred_scaled = torch.sigmoid(tta_pred[valid_mask].squeeze(-1)) * self.max_tta
                tta_loss = self.tta_loss_fn(tta_pred_scaled, tta_gt[valid_mask])
            else:
                tta_loss = torch.tensor(0.0, device=device)
        else:
            tta_loss = torch.tensor(0.0, device=device)

        # ── L_cause: 成因多标签 (加 pos_weight + scene_mask 过滤) ──
        if 'cause' in batch:
            cause_gt = self._to_tensor(batch['cause'], device, dtype=torch.float32)
            if cause_gt.dim() == 1:
                cause_gt = cause_gt.view(-1, self.num_causes)
            cause_gt = cause_gt.clamp(0.0, 1.0)  # 防御: 强制 0/1
            # #5 修复: scene_mask — 仅 pre_crash(2)/crash(3) 参与成因计算
            if 'scene_cls' in batch:
                scene_t = self._to_tensor(batch['scene_cls'], device, dtype=torch.long)
                active = (scene_t >= 2).float().unsqueeze(1)  # [B,1] V20: pre_crash=2, crash=3
            else:
                active = torch.ones(cause_gt.size(0), 1, device=device)
            # #4 修复: BCEWithLogitsLoss + pos_weight (只作用于正样本)
            # V25-fix(loss-spike-v2): 焦点调制 + 硬裁剪 —
            #   BCE 对困难样本呈指数爆炸 (log(1/p) 当 p→0), 裁剪治标不治本
            #   焦点调制: bce * (1-pt)^gamma 自动降低易样本权重, 对离群值鲁棒
            #   gamma=2.0 是 RetinaNet 最优值, 对困难样本梯度贡献 <2x 易样本
            if active.sum() == 0:
                _warn_zero_loss('TemporalCauseLoss(no_active_scene)')
                cause_loss = torch.tensor(0.0, device=device)
            else:
                bce = F.binary_cross_entropy_with_logits(
                    cause_logits, cause_gt,
                    pos_weight=self.cause_pos_weight.unsqueeze(0).to(device),
                    reduction='none')
                # V25-fix: 焦点调制 — 对困难样本梯度贡献 <2x (vs BCE 的 10-100x)
                # V25-ablation: B3 消融开关 (gamma=0 禁用 focal)
                import os as _os
                _focal_gamma = 0.0 if _os.environ.get('V25_ABLATE_FOCAL', '0') == '1' else 2.0
                with torch.no_grad():
                    pt = torch.exp(-bce)  # 正确类概率
                    focal_weight = (1 - pt) ** _focal_gamma
                bce = bce * focal_weight
                # V25-fix(loss-spike-final): 移除 loss 裁剪 — 裁剪阻断困难样本梯度
                cause_loss = (bce * active).sum() / (active.sum() * self.num_causes + 1e-8)
        else:
            _warn_zero_loss('TemporalCauseLoss(missing_cause_key)')
            cause_loss = torch.tensor(0.0, device=device)

        # ── L_type: 事故类型 (V20-fix(W5): 与 Phase 2 对齐, 仅 crash 场景计算) ──
        if 'accident_type' in batch:
            type_gt = self._to_tensor(batch['accident_type'], device, dtype=torch.long)
            if 'scene_cls' in batch:
                # V20-fix(W5): 仅 crash(3) 场景参与类型损失, 防止 pre_crash 脏标签污染
                scene_t = self._to_tensor(batch['scene_cls'], device, dtype=torch.long)
                crash_mask = (scene_t == 3)
                if crash_mask.any():
                    type_loss = self.type_loss_fn(
                        type_logits[crash_mask], type_gt[crash_mask])
                else:
                    _warn_zero_loss('TemporalTypeLoss(no_crash)')
                    type_loss = torch.tensor(0.0, device=device)
            else:
                type_loss = self.type_loss_fn(type_logits, type_gt)
        else:
            _warn_zero_loss('TemporalTypeLoss(missing_type_key)')
            type_loss = torch.tensor(0.0, device=device)

        # ── V20: L_3d_geo — 3D 几何损失 ──
        loss_3d_geo = torch.tensor(0.0, device=device)
        if (geo_3d_dict is not None and geo_3d_dict.get('geo_3d_decoded') is not None
                and self.geo_3d_loss_fn is not None and 'gt_3d' in batch):
            # V20-fix(C1): 使用 geo_3d_decoded 和 batch['gt_3d'] 计算实际损失
            # 注: Phase 3 ClipDataset 默认不含 gt_3d, 此时 loss_3d_geo=0 (3D 在 Phase 1/2 训练)
            try:
                gt_3d = self._to_tensor(batch['gt_3d'], device, dtype=torch.float32)
                if gt_3d.dim() == 1:
                    gt_3d = gt_3d.view(-1, 10)
                # gt_3d: [N, 10] = depth, l, w, h, sin, cos, vx, vy, vz, speed
                gt_dict = {
                    'depth': gt_3d[:, 0:1],
                    'dims': gt_3d[:, 1:4],
                    'orient': gt_3d[:, 4:6],
                    'vel': gt_3d[:, 6:9],
                    'speed': gt_3d[:, 9:10],
                }
                # V20-fix(F2): 使用 compute_loss_simple (2 参数) 而非 forward (5 参数)
                loss_3d_geo = self.geo_3d_loss_fn.compute_loss_simple(
                    geo_3d_dict['geo_3d_decoded'], gt_dict)
            except Exception:
                loss_3d_geo = torch.tensor(0.0, device=device)

        # ── V20: L_ttc_3d — 3D TTC 损失 ──
        loss_ttc_3d = torch.tensor(0.0, device=device)
        if (geo_3d_dict is not None and geo_3d_dict.get('ttc_3d') is not None
                and 'ttc_3d_gt' in batch):
            ttc_3d_pred = geo_3d_dict['ttc_3d']  # [B, T]
            ttc_3d_gt = self._to_tensor(batch['ttc_3d_gt'], device, dtype=torch.float32)
            if ttc_3d_gt.dim() == 1:
                ttc_3d_gt = ttc_3d_gt.unsqueeze(-1)  # [B, 1]
            # V20-fix: 对齐 pred [B,T] 与 gt [B,1] 形状
            if ttc_3d_gt.shape != ttc_3d_pred.shape:
                ttc_3d_gt = ttc_3d_gt.expand_as(ttc_3d_pred)
            valid_ttc = (ttc_3d_gt >= 0) & (ttc_3d_gt <= 10.0)
            if valid_ttc.any():
                loss_ttc_3d = self.ttc_3d_loss_fn(
                    ttc_3d_pred[valid_ttc],
                    ttc_3d_gt[valid_ttc])

        # ── 总损失 ──
        # V25-fix(loss-spike-final): 移除 loss 裁剪 — 裁剪阻断困难样本梯度信号,
        #   导致 Frame Acc 86%→57% 崩溃。正确方案: 焦点调制 + 梯度裁剪 (不裁剪 loss 值)
        total_loss = (
            det_loss
            + self.lambda_scene * scene_loss
            + self.lambda_tta * tta_loss
            + self.lambda_cause * cause_loss
            + self.lambda_type * type_loss
            + self.lambda_3d * loss_3d_geo
            + self.lambda_ttc_3d * loss_ttc_3d
        )

        loss_dict = {
            'det': det_loss.item() if isinstance(det_loss, torch.Tensor) else det_loss,
            'scene': scene_loss.item(),
            'tta': tta_loss.item(),
            'cause': cause_loss.item(),
            'type': type_loss.item(),
            '3d_geo': loss_3d_geo.item(),
            'ttc_3d': loss_ttc_3d.item(),
            'total': total_loss.item(),
        }
        return total_loss, loss_dict

    @staticmethod
    def _to_tensor(val, device, dtype=torch.float32):
        if isinstance(val, torch.Tensor):
            return val.to(device, dtype=dtype)
        if isinstance(val, list):
            return torch.tensor(val, device=device, dtype=dtype)
        return torch.tensor([val], device=device, dtype=dtype)

    def _focal_ce(self, logits, targets):
        """V20-fix(W4): Focal CrossEntropy — 与 Phase 2 AsymmetricFocalLoss 保持一致

        对 (1-pt)^gamma 调制 + per-class alpha 权重 + label_smoothing,
        避免阶段切换时 loss 量纲突变。targets=-100 的样本自动忽略。
        """
        valid = targets != -100
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        logits_v = logits[valid]
        targets_v = targets[valid]
        # CE with label smoothing (与 Phase 2 一致; V21: 0.05→0.1)
        ce = F.cross_entropy(
            logits_v, targets_v,
            weight=self.cls_weight.to(logits_v.device),
            label_smoothing=0.1, reduction='none')
        # Focal modulation: (1-pt)^gamma
        # V25-ablation: B3 消融开关 (gamma=0 禁用 focal)
        import os as _os
        pt = torch.exp(-ce)
        gamma = self.scene_gamma.to(logits_v.device)[targets_v]
        if _os.environ.get('V25_ABLATE_FOCAL', '0') == '1':
            gamma = torch.zeros_like(gamma)
        focal_loss = ((1 - pt) ** gamma) * ce
        return focal_loss.mean()

    # V20-fix(W4): 已删除 to() 覆写 — register_buffer 自动跟随 .to(device) 迁移,
    #              scene_loss_fn 作为子模块同样自动迁移


# ============================================================
# 6. 检测延迟指标 (V20)
# ============================================================

def compute_detection_latency(frame_scene_logits, scene_seq_gt, fps=10.0,
                              confirm_frames=2):
    """V20: 计算事故预警检测延迟 (Detection Latency)

    定义:
        t_gt  : GT 中首个 pre_crash/crash (标签>=2) 的帧索引
        t_pred: 预测中首个风险帧 — 预测为 pre_crash/crash (cls idx>=1)
                且后续 confirm_frames 帧内保持风险预测 (防抖确认)
        latency = (t_pred - t_gt) / fps  (秒; 负数=提前预警, 正数=检测延迟)

    Args:
        frame_scene_logits: [B,T,nc_cls] 逐帧场景 logits
        scene_seq_gt:       [B,T] 逐帧场景标签 (0=ignore,1=normal,2=pre_crash,3=crash)
        fps:                采样帧率 (TUMTraf=10)
        confirm_frames:     防抖确认帧数 (连续 N 帧风险才确认)

    Returns:
        (mean_latency, n_valid, early_rate):
            mean_latency: 有效 clip 的平均延迟 (秒)
            n_valid:      有效 clip 数 (GT 含风险帧)
            early_rate:   提前预警率 (latency <= 0 的 clip 比例)
    """
    preds = frame_scene_logits.argmax(dim=-1)          # [B,T] in {0,1,2}
    B, T = preds.shape
    latencies = []
    n_early = 0
    for b in range(B):
        gt_risk = (scene_seq_gt[b] >= 2).nonzero(as_tuple=True)[0]
        if len(gt_risk) == 0:
            continue                                   # 纯 normal clip, 不计入
        t_gt = gt_risk[0].item()
        risk_pred = (preds[b] >= 1)                    # pre_crash(1) 或 crash(2)
        t_pred = None
        for t in range(T - confirm_frames + 1):
            if risk_pred[t] and risk_pred[t:t + confirm_frames].all():
                t_pred = t
                break
        if t_pred is None:
            t_pred = T                                 # 漏报: 记为 clip 全长 (最大延迟)
        lat = (t_pred - t_gt) / fps
        latencies.append(lat)
        if lat <= 0:
            n_early += 1
    if not latencies:
        return 0.0, 0, 0.0
    mean_lat = sum(latencies) / len(latencies)
    return mean_lat, len(latencies), n_early / len(latencies)
