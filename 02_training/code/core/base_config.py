# -*- coding: utf-8 -*-
"""
base_config — V21 训练全局配置 / 常量 / 工具函数 (从 train.py 拆分)

包含:
  - 环境初始化 (PROJECT_ROOT / cwd / PYTHONPATH / 文件日志) — 其他 v24_* 模块均依赖本模块
  - DDP 共享导入 (dist / DDP / DistributedSampler / _has_dist)
  - 多源数据集配置 DATASET_CONFIGS
  - 成因/类型常量: CAUSE_NAMES / TYPE_NAMES / CAUSE_POS_WEIGHT / TYPE_CLASS_WEIGHT
  - calc_dynamic_cause_pos_weight / MemoryManager
  - 工具函数: load_balance_report / get_progressive_weights /
             adaptive_weight_update / get_phase3_loss_weights

注意: CAUSE_POS_WEIGHT 会在 train.main() 中被动态计算后更新,
     其他模块请通过 `import base_config` 后引用 `base_config.CAUSE_POS_WEIGHT`
     (不要 from-import 值拷贝, 否则拿不到更新后的权重)。
"""

import os
import re

_V26_DATA_ROOT = os.environ.get('V26_DATA_ROOT', 'data')
import sys
import gc
import json
import math
import logging

import torch

from ultralytics.utils import LOGGER

# ── DDP (供各模块共享) ──
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    _has_dist = True
except ImportError:
    dist = None
    DDP = None
    DistributedSampler = None
    _has_dist = False

# ── 环境初始化 ──
# R1-21 警告: 导入本模块会产生以下副作用 (其他 v24_* 模块依赖这些行为):
#   1. sys.path 插入 SCRIPT_DIR
#   2. os.chdir(PROJECT_ROOT) — 改变工作目录!
#   3. 修改 PYTHONPATH 环境变量
#   4. 创建 logs/ 目录并添加文件日志 handler
# 独立工具脚本若依赖自身 cwd, 请在导入前保存 os.getcwd() 并在导入后恢复。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# V20-fix(P1): 项目根目录 = scripts/ 的父目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# DDP 修复: 让 ultralytics 生成的临时脚本能通过 `from train import ...` 找到自定义类
# V20-fix(P1): cwd 设为 PROJECT_ROOT (而非 SCRIPT_DIR), 使相对路径 config/ runs/ logs/ 正确解析
os.chdir(PROJECT_ROOT)
_pp = os.environ.get('PYTHONPATH', '')
if SCRIPT_DIR not in _pp.split(os.pathsep):
    os.environ['PYTHONPATH'] = SCRIPT_DIR + (os.pathsep + _pp if _pp else '')

# V20-fix(P4): 创建 logs/ 目录并添加文件日志
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
_file_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'train_v21.log'), encoding='utf-8')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
LOGGER.addHandler(_file_handler)


# ============================================================
# 0. 多源数据集配置
# ============================================================

DATASET_CONFIGS = {
    # ── 本地数据集根目录（由 V26_DATA_ROOT 指定）下的数据源 ──
    # unified_v21: video_v21 + video_v24_batch2 经 merge_datasets.py 合并去均衡后
    # 的最终训练集 (默认训练数据源); 源视频统一按 10fps 抽帧转换
    'unified-v21': {
        'data_root': 'datasets/unified_v21',
        'fps': 10, 'clip_len': 30, 'clip_stride': 10,
        'has_3d': False, 'scene': 'video_mixed', 'has_accident': True,
        'phase': 'core',
    },
    # video_v21 / video_v24_batch2: 合并前的分批源数据 (保留供单批验证/增量训练)
    'video-v21': {
        'data_root': os.path.join(_V26_DATA_ROOT, 'video_v21'),
        'fps': 10, 'clip_len': 30, 'clip_stride': 10,
        'has_3d': False, 'scene': 'video_mixed', 'has_accident': True,
        'phase': 'augment',
    },
    'video-v21-batch2': {
        'data_root': os.path.join(_V26_DATA_ROOT, 'video_v24_batch2'),
        'fps': 10, 'clip_len': 30, 'clip_stride': 10,
        'has_3d': False, 'scene': 'video_mixed', 'has_accident': True,
        'phase': 'augment',
    },
    'tumtraf-r1': {
        'data_root': os.path.join(_V26_DATA_ROOT, 'tumtraf_a9'),
        'fps': 10, 'clip_len': 30, 'clip_stride': 10,
        'has_3d': True, 'scene': 'highway', 'has_accident': True,
        'cameras': 4, 'phase': 'core',
    },
    'tumtraf-r4': {
        'data_root': os.path.join(_V26_DATA_ROOT, 'tumtraf_a9'),
        'fps': 10, 'clip_len': 30, 'clip_stride': 10,
        'has_3d': True, 'scene': 'highway_v2x', 'has_accident': False,
        'has_near_miss': True, 'cameras': 5, 'phase': 'core',
    },
    'tumtraf-r0r2': {
        'data_root': os.path.join(_V26_DATA_ROOT, 'tumtraf_a9'),
        'fps': 10, 'clip_len': 30, 'clip_stride': 10,
        'has_3d': True, 'scene': 'highway_intersection', 'has_accident': False,
        'phase': 'pretrain',
    },
    # a3d: 历史别名, 已重指向 unified_v21 (与 'unified-v21' 等价, 保留向后兼容);
    # V21-fix: fps 25→10 / clip_len 75→30 — unified_v21 源视频按 10fps 转换,
    # 原 25fps 参数会使 Phase 3 时序 Clip 的时间窗口计算错误
    'a3d': {
        'data_root': os.path.join(_V26_DATA_ROOT, 'unified_v21'),
        'fps': 10, 'clip_len': 30, 'clip_stride': 10,
        'has_3d': False, 'scene': 'cctv_highway', 'has_accident': True,
        'phase': 'augment',
    },
    'nuscenes': {
        'data_root': os.path.join(_V26_DATA_ROOT, 'dataset/nuscenes/converted'),
        'fps': 12, 'clip_len': 36, 'clip_stride': 12,
        'has_3d': True, 'scene': 'urban', 'has_accident': False,
        'phase': 'pretrain',
    },
    'cadp': {
        'data_root': os.path.join(_V26_DATA_ROOT, 'dataset/cadp/converted'),
        'fps': 25, 'clip_len': 75, 'clip_stride': 25,
        'has_3d': False, 'scene': 'cctv_mixed', 'has_accident': True,
        'phase': 'augment',
    },
    'bdd100k': {
        'data_root': os.path.join(_V26_DATA_ROOT, 'dataset/bdd100k/converted'),
        'fps': 30, 'clip_len': 90, 'clip_stride': 30,
        'has_3d': False, 'scene': 'dashcam', 'has_accident': False,
        'phase': 'weather',
    },
}

# 成因标签名称 (13-bit 编码, V24: 删除3个零样本成因 fatigued/drunk/brake_failure)
CAUSE_NAMES = ['tailgating', 'speeding', 'dangerous_lane_change',
               'standing_shoulder', 'pedestrian_highway',
               'heavy_rain', 'heavy_snow', 'distracted_driving',
               'low_visibility']
NUM_CAUSES = 9

# ── 因子启用开关 (V26) ─────────────────────────────────────────────
# 位下标与 35.4 万条 scene_labels 第 2 行的位掩码**永久绑定**, 任何情况下不重编号、
# 不删除 CAUSE_NAMES 条目 —— 摘掉一个因子只做三件事: 从 loss 里屏蔽、从评测里屏蔽、
# 从预测里压成 0 (logit=-30 → sigmoid≈1e-13)。权重矩阵形状保持 [256, 9] 不变,
# 因此旧 checkpoint 仍可加载, 已标注数据无需重标。
#
# 摘除依据 (V26 crash 帧 train/val 实测):
#   4 pedestrian_highway  96/4   与类型 pedestrian_hit 语义重叠, val 仅 4 帧无法评测
#   7 distracted_driving  488/50 画面不可观测: CAUSE_PROMPT 自身写明"通常不可见 → 多为 false"
# 暂留观察 (等稀有类补采出数再定): 3 standing_shoulder 166/5, 6 heavy_snow 404/9
CAUSE_DISABLED_BITS = {4, 7}


def cause_active_mask():
    """长度 NUM_CAUSES 的 0/1 列表: 1=该因子参与训练/评测/预测, 0=已摘除."""
    return [0 if i in CAUSE_DISABLED_BITS else 1 for i in range(NUM_CAUSES)]


def apply_cause_mask(weights):
    """已摘除位的 pos_weight 置 1.0 (不参与 loss, 权重无意义, 别让 20.0 上限误导读数)."""
    return [1.0 if i in CAUSE_DISABLED_BITS else w for i, w in enumerate(weights)]


# 事故类型名称 (7 类, V24: 合并 fire/pileup/vehicle_fall → other)
TYPE_NAMES = ['rear_end', 'side_impact', 'rollover', 'pedestrian_hit',
              'head_on', 'fixed_object', 'other']
NUM_TYPES = 7

# ── 车道线 (V26-G247): 多标签车道线分割, 消费 lane_labels/*.json 的折线 ──
# 编码: 语义掩码 uint8 [H,W], 0=背景, 1..16 = 1 + role*8 + type*2 + color,
# 255=ignore。帧级 "主导类型=虚线" 全库零方差 (16,574 帧 100% dashed), 帧级
# 类型头无监督信号 ⇒ 改吃折线几何: 每条线栅格化进掩码, 解码成 7 个多标签通道。
LANE_ROLE_NAMES = ['divider', 'edge']
LANE_TYPE_NAMES = ['dashed', 'solid', 'double']
LANE_COLOR_NAMES = ['white', 'yellow']
LANE_CHANNELS = ['divider', 'edge', 'dashed', 'solid', 'double', 'white', 'yellow']
NUM_LANE_CHANNELS = 7
LANE_IGNORE = 255

# 车道线 BCE pos_weight (7 通道): 实测 2026-10-19, train 全量 14,336 帧 lane_labels,
# 640x640 栅格 → 80x80 P3 块级 (与损失同口径) 正块占比
#   divider 1.57% / edge 0.45% / dashed 1.45% / solid 0% / double 0%
#   white 1.89% / yellow 0.12%
# neg/pos 截断 [1,20] (与 CAUSE_POS_WEIGHT 同约定) → 全通道顶格 20.0。
# solid/double 当前零正例 (库里无实线/双线), 权重无副作用; 扩标落地后自动起作用。
LANE_POS_WEIGHT = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]


def lane_class_encode(role, ltype, color):
    """(role, ltype, color) → 掩码类值 1..16; ltype=None(未知) 落 type=3 档"""
    r = LANE_ROLE_NAMES.index(role)
    t = LANE_TYPE_NAMES.index(ltype) if ltype in LANE_TYPE_NAMES else 3
    c = LANE_COLOR_NAMES.index(color)
    return 1 + r * 8 + t * 2 + c


def lane_class_decode_table():
    """(target[256,7], valid[256,7]) float32 查找表: 掩码类值 → 7 通道监督

    规则: 0=背景全负; 1..16 按位展开; 255=ignore 全零有效;
    type 通道 (dashed/solid/double) 在未知类型 (type 档=3) 像素上也置无效
    —— 未知类型的线仍监督 role/color, 但不猜类型。
    """
    import numpy as np
    target = np.zeros((256, NUM_LANE_CHANNELS), dtype=np.float32)
    valid = np.zeros((256, NUM_LANE_CHANNELS), dtype=np.float32)
    for v in range(256):
        if v == LANE_IGNORE:
            continue                    # 全零有效 → 7 通道全 ignore
        if v == 0:
            valid[v, :] = 1.0           # 背景: 7 通道全负监督
            continue
        if v > 16:
            continue                    # 非法值按 ignore 处理
        k = v - 1
        role, t, color = k // 8, (k % 8) // 2, k % 2
        target[v, role] = 1.0
        valid[v, 0:2] = 1.0             # role 两通道都监督 (另一通道=负)
        if t < 3:
            target[v, 2 + t] = 1.0
            valid[v, 2:5] = 1.0         # 未知类型 (t=3): type 三通道不监督
        target[v, 5 + color] = 1.0
        valid[v, 5:7] = 1.0             # color 两通道都监督
    return target, valid

# ### B7修复(严重): video_id 提取单一真相源 —
# 历史上 naive rsplit("_", 1) 在 9 处脚本平行实现, 对 Accid3nD 式 stem
# ({scene}_{cam}_{秒级时间戳10位}_{纳秒}) 仅剔除纳秒 → 按"秒"分组碎片化,
# 造成 cause 标注键碎片化 / 跨 split 划分泄漏 / 扩窗静默无效。
# 统一优先用 ^(.+?)_(\d{10})_ 提取摄像头级 video_id, 回退 rsplit。
# 注意: expand_pre_crash.py 依赖连续整数帧号语义, 保留独立实现 (配碎片化告警)。
_VIDEO_ID_RE = re.compile(r"^(.+?)_(\d{10})_")


def extract_video_id(stem):
    """提取视频级分组 ID (摄像头级) — 全部数据管线脚本的唯一真相源

    vidXX_NNNNNN       → vidXX (回退分支)
    Accid3nD 式 stem   → {scene}_{cam} (正则分支, 防按秒碎片化)
    """
    m = _VIDEO_ID_RE.match(stem)
    if m:
        return m.group(1)
    return stem.rsplit("_", 1)[0] if "_" in stem else stem

# 成因 pos_weight: 罕见因子 (行人) 权重更高
# V21: 支持动态计算 pos_weight = total_samples / (positive_samples * num_classes)
# V24: 更新为 VLM 标注+均衡后数据驱动结果 (neg/pos 截断[1,20]); train.py 会动态覆盖
CAUSE_POS_WEIGHT = [1.55, 9.01, 1.87, 20.0, 20.0, 2.07, 20.0, 16.44, 3.72]  # V24: 删除3个零样本

# 场景分类损失权重 λ₂ (V21: 从1.5降至0.5, 降低场景损失主导性, 文档5.2.3)
# V22: ACCIDENT 无 bbox 标注, det_loss≈0, scene_loss 是唯一有效损失, 提升至 1.0
# G-249: p2-3 观测到 crash↔normal 边界持续向 normal 漂（crash R 8 面连降、成因 microR 破 57）,
# 按监控预案把场景损失权重放大 ×1.6。注意: 训练路径真正生效值是
# config/accident_detect.yaml 的 scene_cls_weight (detection.py get_model 经 self.data.get 覆盖此处),
# 该键 0.5→0.8; 本默认值对齐到 0.8 防两套真相源 (D15 教训)
DEFAULT_SCENE_CLS_WEIGHT = 0.8

# V26-G254: scene 表征辅助损失权重 (监督对比 + R-Drop 一致性)。
# 同 λ_scene: 真生效值 = config/accident_detect.yaml 的 scene_supcon_weight /
# scene_consist_weight (losses 经 model.data.get 覆盖此处); 本默认值对齐防两套真相源。
DEFAULT_SCENE_SUPCON_WEIGHT = 0.1
DEFAULT_SCENE_CONSIST_WEIGHT = 0.1

# ### D15修复(严重): 场景类别损失权重单一真相源 —
# 原 losses.v18MultiTaskLoss.CLASS_WEIGHTS=[1.5,2.5,0.8] 与 train.py 默认值
# [1.0,1.2,1.1] 不一致。运行时 train.py 经 V21_CLASS_WEIGHTS env 覆盖 criterion,
# 但覆盖前的初始值/env 缺失时仍用 losses 旧值 → 两套权重实际生效条件混乱。
# 统一为 train.py 侧设计值: [1.0,1.2,1.1] + Sampler [0.34,0.33,0.33] = ~1.0x 均衡 (1:1:1 数据均衡 + 温和补偿)
# V22修复(W1): 曾改为 [1.0,3.0,5.0] "应对71.8%normal偏差", 但 Sampler 已将采样均衡为 1:1:1,
# 叠加高权重 = 双重过补偿 → val 76.6% 帧被预测为 pre_crash, crash 召回 5.3% (与 V20 54x 过补偿同模式)。
# 恢复设计值; 如需补偿少数类, 应在无均衡采样器的前提下按逆频率计算。
# V25: crash 权重 1.1→3.0→5.0 — crash Recall 仅6.6%, EMA 会将3.0拖至1.36,
# 需更高初始值配合慢EMA(0.8/0.2)保持有效权重; pre_crash 1.5不变
DEFAULT_CLASS_WEIGHTS = [1.0, 1.5, 5.0]  # [normal, pre_crash, crash]

# 事故类型 class_weight (逆频率)
# V25-fix(type-bias): 适度重平衡类别权重 —
# head_on(idx=4) 降权 1.5→0.8 (不过低, 防止回避正确预测),
# rear_end(0)/side_impact(1) 提权 1.0/1.2→1.8/2.0 (温和提升)
# 总权重和从 11.0→11.6, 变化 <6%, 不会引发损失量级爆炸
# 2026-10-22 反 side_impact 过预测: 带权 CE 的类权重=argmax 先验旋钮,
# p2-2 实测 side_impact w2.0 预测/真值 2.87x (P 32.8%, FP 一路 5106→5852) 挤掉
# rear_end 召回 73.8→60.2 单调下滑; side 2.0→1.2 压到 rear 之下, head_on 0.8→1.2
# /other 1.0→1.2 解欠预测, rollover 2.0→1.8 对齐; 总权重和 11.9→11.5 不炸损失量级
TYPE_CLASS_WEIGHT = [2.88, 1.2, 1.8, 2.5, 1.2, 1.8, 1.2]  # [rear,side,rollover,ped,head_on,fixed,other]  # V26-G259: rear 1.8→2.88 (e10 决策点, ×1.6 类型侧处置)

# ### V21 MODIFIED R7-07(严重, C损失层/D3): 验证阶段标志位 —
# ultralytics 仅 rank0 执行验证, 且验证用 EMA 模型的独立 criterion 实例。
# 若动态 CLASS_WEIGHTS 块内的 dist.all_reduce 在验证 (仅 rank0) 时触发,
# rank0 单方参与 all_reduce → NCCL watchdog 超时死锁。V18Trainer 注册
# on_val_start/on_val_end 回调置位, losses 根据此位跳过验证时的同步。
IN_VALIDATION = False


def calc_dynamic_cause_pos_weight(cause_label_dir, num_causes=9):
    """V21: 从训练集统计动态计算 cause pos_weight

    公式: weight_c = total_samples / (positive_samples_c * num_classes)
    使 cause_loss 稳定在 0.25~0.35 (当前 0.15~0.18 偏低)

    Args:
        cause_label_dir: 场景标签目录 (含 factor_bits)
        num_causes: 成因类别数 (默认16)

    Returns:
        list: [w_0, w_1, ..., w_15] 每个因子的 pos_weight
    """
    positive_counts = [0] * num_causes
    total_samples = 0

    if not os.path.exists(cause_label_dir):
        return apply_cause_mask(CAUSE_POS_WEIGHT)

    for fname in os.listdir(cause_label_dir):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(cause_label_dir, fname)
        try:
            with open(fpath) as f:
                lines = f.read().strip().split('\n')
            if len(lines) < 2:
                continue
            factor_bits = int(lines[1])
            total_samples += 1
            for bit_idx in range(num_causes):
                if factor_bits & (1 << bit_idx):
                    positive_counts[bit_idx] += 1
        except (ValueError, IndexError):
            continue

    if total_samples == 0:
        return apply_cause_mask(CAUSE_POS_WEIGHT)

    weights = []
    for c in range(num_causes):
        pos = max(positive_counts[c], 1)
        w = total_samples / (pos * num_causes)
        # 限制在 [1.0, 10.0] 避免极端值
        weights.append(round(min(max(w, 1.0), 10.0), 2))
    return apply_cause_mask(weights)


# ============================================================
# 1. 动态显存管理
# ============================================================

class MemoryManager:
    """GPU 显存管理器"""

    @staticmethod
    def check_memory(device_id=0):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            free = total - reserved
            return {
                'allocated': allocated, 'reserved': reserved,
                'total': total, 'free': free,
                'usage_pct': 100 * reserved / total
            }
        return None

    @staticmethod
    def clear_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ============================================================
# 12. 工具函数
# ============================================================

def load_balance_report(data_root):
    """从均衡报告加载推荐权重

    V21-fix: 兼容多个报告文件名 (V20 仅读 balance_report.json, 但
    rebalance_dataset.py 实际输出 rebalance_report.json):
      1. balance_report.json      — 含 recommended_weights / recommended_sampler_ratios
      2. rebalance_report.json    — 仅类别计数, 由逆频率公式推算 CLASS_WEIGHTS
    """
    candidates = ['balance_report.json', 'rebalance_report.json', 'merge_report.json']
    for fname in candidates:
        report_path = os.path.join(data_root, 'meta', fname)
        if not os.path.exists(report_path):
            continue
        try:
            with open(report_path) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        train_info = report.get('splits', {}).get('train', {})
        weights = train_info.get('recommended_weights', None)
        ratios = train_info.get('recommended_sampler_ratios', None)
        if weights or ratios:
            LOGGER.info(f'均衡报告: {fname} (使用 recommended 字段)')
            return weights, ratios
        # V21-fix: 无 recommended 字段时由类别计数推算逆频率权重 (均值归一化 + [0.3, 3.0] 截断)
        counts = [train_info.get(k, 0) for k in ('normal', 'pre_crash', 'crash')]
        total = sum(counts)
        if total > 0 and all(c > 0 for c in counts):
            inv = [total / (3 * c) for c in counts]
            mean_w = sum(inv) / 3
            weights = [round(min(max(w / mean_w, 0.3), 3.0), 3) for w in inv]
            LOGGER.info(f'均衡报告: {fname} 逆频率推算 CLASS_WEIGHTS={weights}')
            return weights, None
    LOGGER.warning("[V21] No balance report under " + str(data_root) + "/meta; using default CLASS_WEIGHTS")
    return None, None  # R2-01: 兆底返回, 避免调用方解包 None 崩溃


def get_progressive_weights(base_weights, epoch_in_phase):
    """V20-fix: Phase 3c 渐进权重衰减 — 逐步降低 pre_crash 过采样, 回归均衡

    注: 1:1:1 均衡后 base_weights 接近 [1.0,1.0,1.0], 渐进衰减效果有限,
    保留此函数仅为向后兼容和未来非均衡场景使用。
    """
    if epoch_in_phase < 3:
        return base_weights
    decay = min(0.4, (epoch_in_phase - 2) * 0.05)
    # normal: 逐渐增加 (0.5 → 0.7)
    w_normal = base_weights[0] * min(1.5, 1.0 + decay)
    # pre_crash: 逐渐减少 (3.0 → 2.1)
    w_pre = base_weights[1] * max(0.6, 1.0 - decay * 0.5)
    # crash: 保持不变
    w_crash = base_weights[2]
    return [round(w_normal, 2), round(w_pre, 2), round(w_crash, 2)]


def adaptive_weight_update(confusion_matrix, current_weights):
    """V24: 基于验证混淆矩阵自适应调整类别权重

    问题修复: V22 公式 1/sqrt(recall+eps) + 均值归一化 产生的权重值
    始终低于 min_weights 下界, 导致8次更新结果完全相同 [1.0, 3.0, 7.0],
    自适应机制形同虚设。

    V24 新公式: w_c = scale * (1 - recall_c) / (recall_c + eps)
    - 不做均值归一化 (归一化是原公式失效的根因)
    - 不用 sqrt (过度压缩差异)
    - scale 控制权重幅度, eps 防止除零
    - 低 recall 类获得高权重, 高 recall 类获得低权重
    - min_weights 作为安全下界, 不再是唯一生效值

    V25 FIX: 防止自适应振荡 — epoch 1 crash_R=51.1% 时 normal/pre_crash 权重
    被推到 ~10x, 导致 epoch 2 crash_R 暴跌到 0.2%. 修复:
    - scale 6.0→3.0 (降低灵敏度)
    - 新权重与当前权重做 50/50 EMA (平滑过渡)
    - 单次变化不超过2x (防止极端跳跃)
    - max_weight 15.0→10.0
    """
    scale = 3.0    # V25 FIX: 6.0→3.0, 防止权重振荡 (epoch1→2 crash从51%→0.2%)
    epsilon = 0.05  # 防止除零
    recalls = []
    for c in range(3):
        total = sum(confusion_matrix[c])
        recall = confusion_matrix[c][c] / total if total > 0 else 0.01
        recalls.append(max(recall, 0.01))

    # w = scale * (1 - recall) / (recall + eps)
    raw_weights = [round(scale * (1.0 - r) / (r + epsilon), 3) for r in recalls]
    # 安全下界: 保证每个类至少有基本权重
    min_weights = [1.0, 1.5, 5.0]
    max_weight = 10.0
    raw_clamped = [round(min(max(n, m), max_weight), 3) for n, m in zip(raw_weights, min_weights)]

    # V25 FIX: EMA smoothing — 与当前权重做50/50混合, 防止单次更新幅度过大
    # epoch1: current=[1,1.5,5.0] raw=[9.46,10.12,5.24] → 会直接跳到[9.46,10.12,5.24]
    # 加EMA后: [5.23,5.81,5.12] — 更温和
    blended = [round(0.5 * c + 0.5 * r, 3) for c, r in zip(current_weights, raw_clamped)]

    # V25 FIX: 单次变化不超过2x (防止振荡)
    result = []
    for i in range(3):
        lo = current_weights[i] / 2.0
        hi = current_weights[i] * 2.0
        val = max(lo, min(hi, blended[i]))
        val = max(min_weights[i], min(max_weight, val))
        result.append(round(val, 3))

    return result


def get_phase3_loss_weights(sub_phase):
    """V20: Phase 3 各子阶段损失权重 (移除 TTA, 提高场景/成因权重)

    Phase 3a (成因预训练): 重点训练成因+类型
    Phase 3b (因果链微调): 均衡权重
    Phase 3c (全系统联合): 降低成因/类型防过拟合, 保持场景高权重

    V25-fix(type-bias): 温和提升 λ_type 3c:0.5→1.0 —
    类型 head_on 偏置需适度增强学习信号, 但不过度挤占 scene/cause
    """
    configs = {
        '3a': {'tta': 0.0, 'cause': 2.0, 'type': 1.2},
        '3b': {'tta': 0.0, 'cause': 1.5, 'type': 1.0},
        '3c': {'tta': 0.0, 'cause': 1.0, 'type': 1.0},
    }
    return configs.get(sub_phase, configs['3a'])
