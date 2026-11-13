#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_config.py — V26 泛化能力专项提升 (不影响 V25)

V26 核心问题: Type Acc 98.2%(训练) vs 28.6%(推理) — 泛化能力不足

三大根因:
  [1] 评估口径不一致 — 训练仅 crash 帧, 推理全帧
  [2] 域差距 — 训练(高速监控) ≠ 测试(YouTube行车记录仪)
  [3] 过拟合 — mosaic=0.0, 增强不足

V26 六项改进:
  [1] 无监督域适应 (一致性正则 + 自训练)
  [2] 强数据增强 (Mosaic + Mixup + CopyPaste + 强HSV)
  [3] 独立类型头 (不依赖 CTTA 条件化)
  [4] 标签平滑 + Dropout
  [5] 类型感知重采样
  [6] 早停 + 验证指标对齐
"""

import os

# ============================================================
# V26 核心定义变更 (vs V25)
# ============================================================

# ── 场景分类: 二分类 (normal/crash) ──
# V25: 3类 (normal/pre_crash/crash) → V26: 2类 (normal/crash)
# 移除 pre_crash: 人也难判定, 二分类更实用且推理更稳定
V26_SCENE_CLASSES = ['normal', 'crash']
V26_NUM_SCENES = 2

# ── 事故类型: 7 类 (保持不变) ──
V26_TYPE_NAMES = ['rear_end', 'side_impact', 'rollover', 'pedestrian_hit',
                  'head_on', 'fixed_object', 'other']
V26_NUM_TYPES = 7

# ── 新增: 车道线识别 ──
# 用途: 偏离车道检测 + 违规变道判定 (增强 dangerous_lane_change 成因)
V26_LANE_DETECTION = {
    'enable': True,
    'num_lanes': 4,
    'lane_types': ['solid', 'dashed', 'double', 'none'],
    'output': {
        'lane_lines': True,       # 车道线位置 (多项式拟合)
        'lane_departure': True,   # 偏离车道检测
        'lane_change': True,      # 变道检测 (合法/违规)
        'lane_type': True,        # 车道线类型识别
    },
    'lane_change_rules': {
        'min_interval': 3.0,      # 最小变道间隔 (秒)
        'turn_signal_required': True,
        'check_blind_spot': True,
    },
}

# ── 成因: 9 类 (保持不变) ──
V26_CAUSE_NAMES = ['tailgating', 'speeding', 'dangerous_lane_change',
                   'standing_shoulder', 'pedestrian_highway',
                   'heavy_rain', 'heavy_snow', 'distracted_driving',
                   'low_visibility']

# ============================================================
# V26 通过环境变量覆盖 V25 配置, 不修改 V25 文件
# ============================================================

# ── 1. 强数据增强 (根因3: 过拟合) ──
V26_AUGMENT = {
    'mosaic': 1.0,              # V25: 0.0 → V26: 1.0 (核心增强!)
    'mixup': 0.3,               # V25: 0.15 → V26: 0.3
    'copy_paste': 0.1,          # V25: 0.0 → V26: 0.1
    'hsv_h': 0.02,              # 色调增强
    'hsv_s': 0.9,               # 饱和度增强 (V25: 0.8)
    'hsv_v': 0.6,               # 亮度增强 (V25: 0.5)
    'degrees': 5.0,
    'translate': 0.15,          # V25: 0.1 → V26: 0.15
    'scale': 0.6,               # V25: 0.5 → V26: 0.6
    'shear': 0.3,               # V25: 0.5 → V26: 0.3 (减少)
    'fliplr': 0.5,
    'erasing': 0.4,             # V25: 0.3 → V26: 0.4
    'close_mosaic': 10,         # 最后10 epoch 关闭 Mosaic (接近真实分布)
}

# ── 2. 无监督域适应 (根因2: 域差距) ──
V26_DOMAIN_ADAPT = {
    'enable': True,
    'consistency_weight': 0.3,  # 一致性正则权重
    'consistency_temp': 0.07,   # 温度参数
    'ema_decay': 0.999,         # EMA 教师模型
    'unlabeled_ratio': 0.3,     # 无标注数据比例 (从测试域采样)
    'self_train_threshold': 0.8, # 自训练置信度阈值
}

# ── 3. 独立类型头 (根因1: CTTA 条件化依赖) ──
V26_TYPE_HEAD = {
    'independent': True,         # 不依赖 CTTA type_ids
    'hidden_dim': 256,
    'dropout': 0.4,              # V25: 0.3 → V26: 0.4 (更强正则)
    'label_smoothing': 0.1,      # 标签平滑
    'loss_weight': 1.5,          # 类型损失权重 (V25: 1.0)
}

# ── 4. 类型重采样 (根因: 类别不平衡) ──
V26_RESAMPLE = {
    'enable': True,
    'minority_classes': [0, 1],  # rear_end, side_impact
    'oversample_factor': 2.5,
    'type_aware_augment': True,
}

# ── 5. 泛化保护机制 ──
V26_GENERALIZATION = {
    'weight_decay': 0.2,          # V25: 0.15 → V26: 0.2
    'label_smoothing': 0.1,
    'early_stopping_patience': 5,
    'dropout': 0.4,               # 全局 Dropout
    'drop_path': 0.1,             # DropPath (Transformer)
    'grad_clip': 0.5,
}

# ── 6. 评估口径统一 (根因1) ──
V26_EVAL = {
    'unified_metric': True,        # 训练/推理统一评估口径
    'eval_all_frames': True,       # 全帧评估 (不仅 crash 帧)
    'cross_domain_test': True,     # 跨域测试
}

# ── V26 训练配置 ──
V26_TRAIN = {
    'epochs': 20,
    'lr': 0.0003,                  # V25: 0.0005 → V26: 0.0003 (更小, 防过拟合)
    'batch_size': 48,
    'warmup_epochs': 5,
    'scheduler': 'cosine',
}

print("=" * 60)
print("V26 泛化能力专项提升 (不影响 V25)")
print("=" * 60)
print(f"  [1] Mosaic增强: {V26_AUGMENT['mosaic']} (V25: 0.0)")
print(f"  [2] 域适应: {V26_DOMAIN_ADAPT['enable']}")
print(f"  [3] 独立类型头: {V26_TYPE_HEAD['independent']}")
print(f"  [4] 类型重采样: {V26_RESAMPLE['enable']}")
print(f"  [5] Weight Decay: {V26_GENERALIZATION['weight_decay']} (V25: 0.15)")
print(f"  [6] 评估口径统一: {V26_EVAL['unified_metric']}")
print(f"  学习率: {V26_TRAIN['lr']} (V25: 0.0005)")
print(f"  Epochs: {V26_TRAIN['epochs']}")
print("=" * 60)
