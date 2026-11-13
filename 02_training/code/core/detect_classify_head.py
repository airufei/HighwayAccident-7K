"""
DetectClassify 共享模块 — YOLO26 检测+场景分类+成因+类型+3D 多头架构 (V20)

V20 核心重构:
  1. 移除 TTA 回归头 — 数据集中无 TTA 标签, loss 始终为 0, 完全死亡
  2. 移除 ReID 特征头 — 从未被训练或消费
  3. 场景分类头重新设计 — 使用 P5 特征 + 残差连接 + 更大容量
  4. 成因/类型头保留但不再共享池化特征 — 使用独立的特征提取路径
  5. 3D 几何回归头 (GeometryHead3D) — 从 FPN 多尺度特征回归 depth/dims/orient/vel
     可选启用 (has_3d=True), 无 3D 标签时输出不参与损失
  6. V26-G247 车道线分割头 (lane_head) — P3 稠密多标签 7 通道
     (divider/edge/dashed/solid/double/white/yellow), 消费 lane_labels 折线掩码

输出: (det_out, scene_logits, None, cause_logits, type_logits, None, geo_3d, lane_logits)
       基础 8 元组 (索引 7 = lane_logits); V20-fix(C6): has_3d=False 时 geo_3d=None,
       元组长度不变。V25 对比学习附加项 (vis_proj, cls_feat) 接在索引 8, 9。
       注意: 所有调用方必须按索引解包 (det/scene/cause/type/geo_3d/lane 位置固定)

本模块是 train.py 的唯一 DetectClassify 定义来源。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules import head as head_module

try:                                 # 本模块既可能被 `import detect_classify_head`
    import base_config               # (core/ 在 sys.path 上) 引入, 也可能被
except ModuleNotFoundError:          # `from core import detect_classify_head` 引入
    from core import base_config

# V20: 3D 感知模块 (可选导入)
try:
    from geometry_3d import GeometryHead3D
    HAS_3D = True
except ImportError:
    HAS_3D = False

# V25: 对比学习模块 (可选导入)
try:
    from contrastive import VisualProjectionHead
    HAS_CONTRASTIVE = True
except ImportError:
    HAS_CONTRASTIVE = False


# ============================================================
# GroupNorm 工具函数
# ============================================================

def get_groupnorm(channels):
    """GroupNorm: 确保 num_groups 能被 channels 整除"""
    num_groups = 32
    while channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels)


# ============================================================
# DetectClassify — 检测+场景+成因+类型+3D 多头架构 (V20)
# ============================================================

class DetectClassify(Detect):
    """V20: 检测头 + 场景分类 + 成因多标签 + 事故类型 + 3D 几何

    V20 架构改进:
      - 移除 TTA 头 (无标签数据, loss=0, 完全死亡)
      - 移除 ReID 头 (从未训练)
      - 场景分类: 使用 P5 特征 (最强语义) + 残差连接 + 更大容量
      - 成因/类型: 使用三尺度拼接特征 (保留空间信息)
      - 3D 几何回归头 (has_3d=True): per-pixel depth + dims + orient + vel + uncertainty

    输出 (V26 八元组, 始终返回):
        - det_out       : 检测预测 (bbox + cls)
        - scene_logits  : 场景分类 logits, shape [B, nc_cls]
        - None          : TTA 位置 (已移除, 保持索引兼容)
        - cause_logits  : 成因多标签, shape [B, num_causes]
        - type_logits   : 事故类型, shape [B, num_types]
        - None          : ReID 位置 (已移除, 保持索引兼容)
        - geo_3d        : 3D 几何输出 (list of [B, A*10, Hi, Wi]) 或 None
        - lane_logits   : 车道线多标签 logits, shape [B, 7, H/8, W/8] (V26-G247)
        注意: 所有调用方必须按索引解包 (V20-fix(C6) + V26-G247)
    """

    NUM_CAUSES = 9     # 16-bit 成因编码
    NUM_TYPES = 7      # 10 类事故类型

    def __init__(self, nc=80, nc_cls=3, ch=(),
                 num_causes=None, num_types=None, has_3d=False,
                 has_contrastive=False):
        super().__init__(nc=nc, ch=ch)
        self.nc_cls = nc_cls
        self.num_causes = num_causes or self.NUM_CAUSES
        # V26: 因子开关 —— 位下标与 scene_labels 位掩码永久绑定, 这里只把
        # base_config.CAUSE_DISABLED_BITS 的输出压成 -30 (sigmoid≈1e-13), 使其不可能
        # 出现在预测里, 同时该输出行不再收到梯度。用普通属性而非 register_buffer:
        # 避免给 state_dict 增加键而破坏旧 checkpoint 的加载。
        self.cause_disabled = [i for i in sorted(base_config.CAUSE_DISABLED_BITS)
                               if i < self.num_causes]
        self.num_types = num_types or self.NUM_TYPES
        self.has_3d = has_3d and HAS_3D
        self.has_contrastive = has_contrastive and HAS_CONTRASTIVE
        # V20-fix(W2): ch 为空时报错, 而非 IndexError
        if len(ch) == 0:
            raise ValueError("DetectClassify 需要非空 ch (FPN 通道数列表)")
        c_total = sum(ch)
        c_p5 = ch[-1]  # P5 通道数 (最强语义特征)

        # --- V22: 场景分类头 (P5 双池化 + 空间注意力 + 残差连接) ---
        # V22-fix: crash 是局部小事件 (中心区域碰撞), 原 AdaptiveAvgPool2d(1)
        # 全局平均池化将局部 crash 信号稀释, 导致 crash_Recall 仅 23%。
        # 改进: avg+max 双池化 (avg 保留全局语义, max 保留最强局部激活 = crash 区域),
        # 并增加 CBAM 空间注意力让模型聚焦于异常区域。
        self.cls_avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_maxpool = nn.AdaptiveMaxPool2d(1)
        # 空间注意力: 让模型聚焦 crash 发生的局部区域
        self.cls_spatial_attn = nn.Sequential(
            nn.Conv2d(c_p5, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        self.cls_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * c_p5, 1024),
            get_groupnorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),  # V25-ep4: 0.2→0.1, 更强梯度流过以学习crash特征
        )
        # 残差块: 1024 → 1024
        self.cls_res = nn.Sequential(
            nn.Linear(1024, 1024),
            get_groupnorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            get_groupnorm(1024),
        )
        self.cls_out = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            get_groupnorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, nc_cls),
        )

        # --- V20: 成因多标签头 (使用三尺度拼接特征) ---
        self.cause_pool = nn.AdaptiveAvgPool2d(1)
        self.cause_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_total, 512),
            get_groupnorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            get_groupnorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, self.num_causes),
        )

        # --- V20: 事故类型分类头 (使用三尺度拼接特征) ---
        self.type_pool = nn.AdaptiveAvgPool2d(1)
        self.type_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_total, 512),
            get_groupnorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            get_groupnorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, self.num_types),
        )

        # --- V26-G247: 车道线多标签分割头 (P3, stride 8) ---
        # 折线消费: lane_labels 折线 → semantic_mask 类值 → 7 通道 BCE。
        # P3 兼顾定位精度与算力 (640 输入 → 80x80 输出)。
        self.lane_head = nn.Sequential(
            nn.Conv2d(ch[0], 128, 3, padding=1),
            get_groupnorm(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            get_groupnorm(64),
            nn.GELU(),
            nn.Conv2d(64, base_config.NUM_LANE_CHANNELS, 1),
        )

        # --- V20: 3D 几何回归头 (可选) ---
        if self.has_3d:
            self.geometry_head = GeometryHead3D(ch_list=list(ch))

        # --- V25: 对比学习视觉投影头 (可选) ---
        if self.has_contrastive:
            self.vis_proj_head = VisualProjectionHead(visual_dim=1024, proj_dim=256)

    # ---- forward ----
    def forward(self, x):
        """V20 训练/推理统一返回七元组 (始终 7 元组, V20-fix(C6))"""
        # V22: 场景分类使用 P5 + 空间注意力 + avg/max 双池化
        p5_feat = x[-1]  # P5: [B, C_p5, H, W]
        # 空间注意力: 聚焦 crash 等局部异常区域
        attn = self.cls_spatial_attn(p5_feat)       # [B, 1, H, W]
        p5_weighted = p5_feat * attn                # 注意力加权
        # 双池化: avg 保留全局语义, max 保留最强局部激活 (crash 信号不被稀释)
        p5_avg = self.cls_avgpool(p5_weighted)      # [B, C_p5, 1, 1]
        p5_max = self.cls_maxpool(p5_weighted)      # [B, C_p5, 1, 1]
        p5_pooled = torch.cat([p5_avg, p5_max], dim=1)  # [B, 2*C_p5, 1, 1]
        cls_feat = self.cls_fc(p5_pooled)           # [B, 1024]
        cls_feat = cls_feat + self.cls_res(cls_feat)  # 残差连接
        scene_logits = self.cls_out(cls_feat)  # [B, nc_cls]

        # V20: 成因/类型使用三尺度拼接 (保留多尺度空间信息)
        multi_pooled = [self.cause_pool(feat) for feat in x]
        multi_feat = torch.cat(multi_pooled, dim=1)  # [B, c_total, 1, 1]
        cause_logits = self.cause_fc(multi_feat)   # [B, num_causes]
        if self.cause_disabled:                    # V26: 已摘除因子恒定 -30, 不参与预测
            cause_logits = cause_logits.clone()
            cause_logits[:, self.cause_disabled] = -30.0
        type_logits = self.type_fc(multi_feat)     # [B, num_types]

        # V26-G247: 车道线多标签分割 (P3, stride 8)
        lane_logits = self.lane_head(x[0])         # [B, 7, H/8, W/8]

        # 检测头
        det_out = Detect.forward(self, x)

        # V20: 3D 几何回归 (可选)
        geo_3d = None
        if self.has_3d:
            geo_3d = self.geometry_head(x)  # list of [B, A*10, Hi, Wi]

        # V25: 对比学习视觉投影 (可选)
        vis_proj = None
        if self.has_contrastive:
            vis_proj = self.vis_proj_head(cls_feat.detach())  # [B, 256]

        # V20 八元组 (V26-G247: lane_logits 固定索引 7) + V25: vis_proj (末尾附加)
        result = (det_out, scene_logits, None, cause_logits, type_logits, None,
                  geo_3d, lane_logits)
        if vis_proj is not None:
            result = result + (vis_proj, cls_feat.detach())
        # V26-G254: scene 表征辅助通道 — 监督对比要 embedding (未 detach, 梯度要进场景干路),
        # R-Drop 要同前向第二次 dropout 视图。无新参数 ⇒ 旧 ckpt 严格加载不受影响;
        # 仅训练态追加, eval/导出元组形状不变。dict 尾部元素不占固定索引 (PFRPN 同款约定)。
        if self.training:
            result = result + ({'supcon_feat': cls_feat,
                                'scene_view2': self.cls_out(cls_feat)},)
        return result


# ---- 注册入口 (由 train.py 按需调用, 避免隐式全局污染) ----
def register_to_ultralytics():
    """将 DetectClassify 注册到 ultralytics.nn.modules.head (按需调用)"""
    if not hasattr(head_module, 'DetectClassify'):
        head_module.DetectClassify = DetectClassify


# 保持向后兼容: 直接 import 时自动注册
register_to_ultralytics()
