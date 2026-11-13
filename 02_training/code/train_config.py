# -*- coding: utf-8 -*-
"""
train_config — V22 训练配置 (ACCIDENT 数据集专用)

设计原则:
  - 不修改 V21 任何文件
  - 通过 sys.path 引入 V21 scripts/ 目录, 复用全部 V21 核心模块
    (dataset / losses / detection / phase3_legacy)
  - 在 V22 中独立管理: ACCIDENT 数据源注册 + 训练参数 + 质量门禁适配

V22 相对 V21 的增量:
  1. DATASET_CONFIGS 新增 3 个 ACCIDENT 数据源 (real / synthetic / both)
  2. 数据质量门禁放宽 cause_coverage (ACCIDENT 无 cause 标注, 覆盖率=0)
  3. 默认数据源切换为 accident-both
  4. ACCIDENT 碰撞类型映射注册 (5类 → V21 TYPE_NAMES 10类)

依赖目录结构:
  v22_training_package/          ← V22 根目录 (PROJECT_ROOT_V22)
  ├── scripts/
  │   ├── train_config.py          ← 本文件
  │   ├── convert_accident.py    ← ACCIDENT 转换脚本
  │   └── train.py               ← V22 训练入口
  ├── config/                    ← V22 训练参数 YAML
  ├── datasets/                  ← 转换后数据存放
  └── ...
  v21_training_package/scripts/  ← V21 核心模块 (只读引用)
"""

import os
import sys

# ============================================================
# 1. 路径初始化 — 定位 V21 核心模块
# ============================================================

# V22 scripts 目录
V24_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# V22 项目根目录
PROJECT_ROOT_V22 = os.path.dirname(V24_SCRIPT_DIR)

# V21 scripts 目录 (V24 本地化: 已拷贝到 scripts/core/)
V24_BASE_DIR = os.path.join(V24_SCRIPT_DIR, 'core')

if not os.path.isdir(V24_BASE_DIR):
    raise FileNotFoundError(
        f"V21 scripts 目录不存在: {V24_BASE_DIR}\n"
        f"V22 依赖 V21 核心模块 (dataset / losses / detection / phase3_legacy), "
        f"请确保 v26_training_package 内的 core 在同一父目录下。"
    )

# 将 V21 scripts 加入 sys.path (必须在 import v21_* 之前)
if V24_BASE_DIR not in sys.path:
    sys.path.insert(0, V24_BASE_DIR)

# V22 scripts 也加入 sys.path (供 DDP 子进程 import train)
if V24_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, V24_SCRIPT_DIR)

# 设置 PYTHONPATH 环境变量 (ultralytics DDP 子进程需要)
_pp = os.environ.get('PYTHONPATH', '')
for _d in (V24_BASE_DIR, V24_SCRIPT_DIR):
    if _d not in _pp.split(os.pathsep):
        os.environ['PYTHONPATH'] = _d + (os.pathsep + _pp if _pp else '')
        _pp = os.environ['PYTHONPATH']

# 切换工作目录到 V22 PROJECT_ROOT (ultralytics 相对路径解析需要)
# C9修复(Critical): 仅在直接执行 (非被 import) 时 chdir
# 避免作为模块导入时产生副作用
_v24_cwd_applied = False

def ensure_cwd_v22():
    """显式调用切换工作目录到 V22 项目根

    train.py main() 或其他入口点应在启动时调用此函数。
    避免在 import train_config 时产生 cwd 副作用。
    """
    global _v24_cwd_applied
    os.chdir(PROJECT_ROOT_V22)
    _v24_cwd_applied = True


# ============================================================
# 2. 从 V21 继承全部配置与模块符号
# ============================================================

# 导入 V21 配置模块 (执行 V21 的环境初始化: chdir / PYTHONPATH / 文件日志)
# 注意: base_config 导入时会 os.chdir(PROJECT_ROOT), 需要在导入后恢复到 V22
_saved_cwd = os.getcwd()
import base_config  # noqa: E402
os.chdir(_saved_cwd)  # 恢复到 V22 工作目录

# Re-export V21 全部公共符号 (供 V22 train.py 使用)
from base_config import (  # noqa: E402, F401
    SCRIPT_DIR,
    dist, DDP, DistributedSampler, _has_dist,
    CAUSE_NAMES, TYPE_NAMES,
    CAUSE_POS_WEIGHT,
    DEFAULT_SCENE_CLS_WEIGHT, DEFAULT_CLASS_WEIGHTS, TYPE_CLASS_WEIGHT,
    calc_dynamic_cause_pos_weight, MemoryManager,
    load_balance_report, get_progressive_weights,
    adaptive_weight_update, get_phase3_loss_weights,
    extract_video_id,
)

# 日志目录重定向到 V22
import copy
import logging  # noqa: E402
LOGS_DIR_V22 = os.path.join(PROJECT_ROOT_V22, 'logs')
os.makedirs(LOGS_DIR_V22, exist_ok=True)

# V22 专属日志 handler
from ultralytics.utils import LOGGER as _UL_LOGGER  # noqa: E402
# W5修复(Warning): 去重检查 — 避免 LOGGER 同时拥有 V21 和 V22 两个文件 handler
_v24_handler_exists = any(
    isinstance(h, logging.FileHandler) and
    'train_v22.log' in getattr(h, 'baseFilename', '')
    for h in _UL_LOGGER.handlers
)
if not _v24_handler_exists:
    _v24_file_handler = logging.FileHandler(
        os.path.join(LOGS_DIR_V22, 'train_v22.log'), encoding='utf-8'
    )
    _v24_file_handler.setLevel(logging.INFO)
    _v24_file_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    )
    _UL_LOGGER.addHandler(_v24_file_handler)


# ============================================================
# 2.5 V22-fix: ultralytics 14 字段检测标签兼容解析
# ============================================================
# V21/V22 检测标签为 14 字段扩展格式:
#   cls x y w h | depth l w h | sin cos | in_acc ttc scene_cls
# 当前 ultralytics 版本的 verify_image_label 把字段数 >6 的行误判为
# 分割多边形, 将类之后的 13 个数值 reshape(-1, 2) → 13 为奇数抛错 →
# 全部合成集 bbox 被当作 corrupt 丢弃, val.cache 无有效标签, mAP=0。
# 此补丁仅拦截"所有行均为 14 字段"的标签文件, 取前 5 列 (cls+xywh)
# 走标准检测校验; 其余格式行为与原函数完全一致。不修改 V21/数据文件。

import numpy as _np  # noqa: E402
import ultralytics.data.dataset as _ul_dataset  # noqa: E402
import ultralytics.data.utils as _ul_data_utils  # noqa: E402
from ultralytics.data.utils import check_image as _ul_check_image  # noqa: E402

_V22_LABEL_NUM_FIELDS = 14
_orig_verify_image_label = _ul_dataset.verify_image_label


def _v24_verify_image_label(args):
    """兼容 V21/V22 14 字段扩展检测标签的 verify_image_label 包装。"""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
    if keypoint or not os.path.isfile(lb_file):
        return _orig_verify_image_label(args)
    try:
        with open(lb_file, encoding='utf-8') as f:
            rows = [x.split() for x in f.read().strip().splitlines() if len(x)]
    except OSError:
        return _orig_verify_image_label(args)
    if not rows or not all(len(x) == _V22_LABEL_NUM_FIELDS for x in rows):
        return _orig_verify_image_label(args)

    # ── 14 字段扩展格式: 取前 5 列, 校验逻辑与 ultralytics 原函数一致 ──
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        msg, shape = _ul_check_image(im_file)
        msg = f"{prefix}{msg}" if msg else ""
        nf = 1
        lb = _np.array([x[:5] for x in rows], dtype=_np.float32)
        if nl := len(lb):
            assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
            points = lb[:, 1:]
            assert points.max() <= 1.01, (
                f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
            )
            assert lb.min() >= -0.01, f"negative class labels or coordinate {lb[lb < -0.01]}"
            max_cls = 0 if single_cls else lb[:, 0].max()
            assert max_cls < num_cls, (
                f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                f"Possible class labels are 0-{num_cls - 1}"
            )
            _, i = _np.unique(lb, axis=0, return_index=True)
            if len(i) < nl:
                lb = lb[i]
        else:
            ne = 1
            lb = _np.zeros((0, 5), dtype=_np.float32)
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


# dataset.py 在模块级绑定了 verify_image_label (verify_args 返回它),
# 必须替换 dataset 模块命名空间中的引用才会生效; utils 侧一并替换。
_ul_dataset.verify_image_label = _v24_verify_image_label
_ul_data_utils.verify_image_label = _v24_verify_image_label


# ============================================================
# 3. V22 新增: ACCIDENT 数据源注册
# ============================================================

# 复制 V21 的 DATASET_CONFIGS 并追加 ACCIDENT 数据源
# (使用 deepcopy 避免修改 V21 原始配置的内部 dict 引用)
DATASET_CONFIGS = copy.deepcopy(base_config.DATASET_CONFIGS)

# ACCIDENT 数据源 — 3 种粒度
# convert_accident.py 转换后输出到 datasets/accident_v22/
_ACCIDENT_DATA_ROOT = os.path.join(PROJECT_ROOT_V22, 'datasets', 'accident_v26')  # V26: 独立数据集

DATASET_CONFIGS['accident-both'] = {
    'data_root': _ACCIDENT_DATA_ROOT,
    'fps': 10, 'clip_len': 30, 'clip_stride': 10,
    'has_3d': False,
    'scene': 'cctv_mixed',
    'has_accident': True,
    'annotation_quality': 'human',   # 3-5 名标注者人工投票
    'source': 'real+synthetic',
    'phase': 'core',
}
DATASET_CONFIGS['accident-real'] = {
    'data_root': os.path.join(_ACCIDENT_DATA_ROOT, 'real_only'),
    'fps': 10, 'clip_len': 30, 'clip_stride': 10,
    'has_3d': False,
    'scene': 'cctv_roadside',
    'has_accident': True,
    'annotation_quality': 'human',
    'source': 'real',
    'phase': 'core',
}
DATASET_CONFIGS['accident-synth'] = {
    'data_root': os.path.join(_ACCIDENT_DATA_ROOT, 'synth_only'),
    'fps': 10, 'clip_len': 30, 'clip_stride': 10,
    'has_3d': False,
    'scene': 'carla_simulated',
    'has_accident': True,
    'annotation_quality': 'human',   # 合成集同样人工标注
    'source': 'synthetic',
    'phase': 'core',
}

# V22-fix: 域分层平衡数据集 (train/val 均 ~53%真实/47%合成, 消除域不匹配)
_BALANCED_DATA_ROOT = os.path.join(PROJECT_ROOT_V22, 'datasets', 'accident_v24_balanced')
DATASET_CONFIGS['accident-balanced'] = {
    'data_root': _BALANCED_DATA_ROOT,
    'fps': 10, 'clip_len': 30, 'clip_stride': 10,
    'has_3d': False,
    'scene': 'cctv_mixed',
    'has_accident': True,
    'annotation_quality': 'human',
    'source': 'real+synthetic',
    'phase': 'core',
}

# V22 默认数据源
DEFAULT_DATASET = 'accident-both'


# ============================================================
# 4. V22 ACCIDENT 碰撞类型映射
# ============================================================

# V24 TYPE_NAMES (7类) = ['rear_end', 'side_impact', 'rollover', 'pedestrian_hit',
#                         'head_on', 'fixed_object', 'other']
ACCIDENT_TYPE_MAP = {
    'rear-end':  0,   # → rear_end
    't-bone':    1,   # → side_impact
    'head-on':   4,   # → head_on (V24: 7-class index)
    'sideswipe': 6,   # → other (V24: merged fire/pileup/vehicle_fall→other)
    'single':    5,   # → fixed_object (单车事故撞固定物)
}


# ============================================================
# 5. V22 数据质量门禁适配
# ============================================================

# V21 门禁: MIN_QUALITY_SCORE=60, MIN_CAUSE_COVERAGE=0.30
# V22: ACCIDENT 无 cause 标注 → 放宽 cause_coverage 至 0 (仅 ACCIDENT 数据源)
# V24: VLM 重标注后 ACCIDENT 已有 16-bit 成因 (scene_labels 第2行, 覆盖率 99%+),
#      清空放宽名单, 所有数据源统一执行 0.30 门禁
MIN_QUALITY_SCORE_V22 = 60.0
MIN_CAUSE_COVERAGE_V22 = 0.0   # 仅真正无成因标注的数据源走此放宽值

# 标记当前数据源是否需要放宽 cause 门禁
# V24: ACCIDENT 系列已含 VLM 成因标注, 从名单移除
_CAUSE_RELAXED_DATASETS = set()


def is_cause_relaxed(dataset_name):
    """检查数据源是否需要放宽 cause 覆盖率门禁"""
    return dataset_name in _CAUSE_RELAXED_DATASETS


# ============================================================
# 6. V22 训练超参数 (ACCIDENT 数据集优化)
# ============================================================

# ACCIDENT 数据特点:
#   - 2,027 真实 + 2,211 合成 = 4,238 clip, 抽帧后约 420K 帧
#   - 三分类分布: normal 67%, pre_crash 23%, crash 10%
#   - 5 类碰撞均匀分布 (15-27%)
#   - 合成集有 bbox, 真实集无 bbox (需 YOLO 预训练模型检测)

# ============================================================
# 6.1 V22 架构创新: 三阶段渐进式训练 + 四项创新模块
# ============================================================
#
# V22 在 V21 三阶段五步架构基础上的创新:
#
#   创新点1: 碰撞类型感知时序注意力 (CTTA)
#     - Transformer 注意力权重受碰撞类型条件控制
#     - 不同碰撞类型有不同的时序模式 (追尾需跟车距离序列, 侧撞需横向接近)
#     - 实现: type_embed → cross-attention → temporal attention bias
#
#   创新点2: 不确定性感知场景分类 (UASC)
#     - ACCIDENT 提供标注者投票分布, 用于标签平滑
#     - 高一致性行为硬标签, 低一致性行为软标签
#     - 实现: scene_loss = FocalLoss(scene_pred, soft_label)
#
#   创新点3: 跨域一致性正则化 (CDCR)
#     - 真实集与合成集的特征表示对齐
#     - 减少合成集训练带来的域偏移
#     - 实现: L_cdc = KL(softmax(f_real) || softmax(f_synth))
#
#   创新点4: 渐进式类型课程学习 (PTCL)
#     - 按碰撞类型难度排序: rear-end (易) → side → head-on → sideswipe (难)
#     - Phase 2 先训练易类, Phase 3 逐步引入难类
#     - 实现: type_loss_mask 按课程进度控制激活类型
#

# ── 创新点1: CTTA 配置 ──
CTTA_CONFIG = {
    'enable': True,               # 启用碰撞类型感知时序注意力
    'type_embed_dim': 64,         # 碰撞类型 embedding 维度
    'attention_heads': 8,         # Transformer 注意力头数 (复用 V21)
    'type_conditional_bias': True, # 类型条件偏置注入 attention
    'temporal_window': 30,        # 时序窗口 (30 帧 = 3s @ 10fps)
}

# ── 创新点2: UASC 配置 ──
UASC_CONFIG = {
    'enable': True,
    'label_smoothing_high': 0.0,  # 高一致性投票: 无平滑 (硬标签)
    'label_smoothing_low': 0.3,   # 低一致性投票: 强平滑 (软标签)
    'agreement_threshold': 0.8,   # 高一致性阈值 (3/5=0.6, 4/5=0.8, 5/5=1.0)
    'soft_label_method': 'kl',  # W6修复(Warning): 标签平滑方法: kl=KL散度 (与实现一致)
}

# ── 创新点3: CDCR 配置 ──
CDCR_CONFIG = {
    'enable': True,
    'lambda_cdcr': 0.1,           # 跨域一致性损失权重 (实验调优范围 0.05-0.3)
    'temperature': 2.0,           # KL 散度温度参数
    'real_batch_ratio': 0.5,      # 每批中真实集样本比例
    'warmup_epochs': 10,          # 前 10 epoch 不启用 (等检测头稳定)
}

# ── 创新点4: PTCL 配置 ──
# 碰撞类型课程难度排序 (由易到难)
# rear-end: 追尾, 视觉特征明显 (前车急速变大), 最易学习
# t-bone: 侧面碰撞, 横向运动模式明显
# head-on: 正面碰撞, 对向车道接近
# single: 单车事故, 场景多样
# sideswipe: 擦碰, 视觉变化微妙, 最难
PTCL_CONFIG = {
    'enable': True,
    'type_difficulty_order': [0, 1, 4, 5, 6],  # rear_end → side → head_on → fixed → other (V24: 7类)
    'phase2_active_types': [0, 1],              # Phase 2 仅训练易类 (rear_end, side_impact)
    'phase3a_active_types': [0, 1, 4],          # Phase 3a 加入 head_on
    'phase3b_active_types': [0, 1, 4, 5],       # Phase 3b 加入 fixed_object
    'phase3c_active_types': [0, 1, 4, 5, 6],    # Phase 3c 全部 5 类
}

# ── G-280: 类型计量的三档分桶 (纯函数, 无 torch; 尺①/尺② 共用一份实现) ──
# 动因: apply_mask (temporal_model.py:568-589) 只在**损失侧**按 GT 是否在课程内过滤样本,
# 而 type 头输出宽度恒为 NUM_TYPES=7 ⇒ argmax ∈ [0,6]。旧分母 (eval_acceptance.py:311 /
# phase3_train.py:892) 只按 "该 clip 有 crash 帧" 计数, 于是两类不可评分样本被算成错:
#   无槽 (GT=-1 或 >=7)   → 永不可能对, 构造性必错
#   有槽但课程外          → 从未拿到梯度, 可达却无训练 (3a 三轮实测 560 条只对 1 条)
# 尺② 宇宙实测: 分母 2,323 = 课程内 1,420 + 课程外有槽 560 + 无槽 343
# ⇒ 裸 type_acc 被 1,420/2,323=61.1% 的经验天花板压着, 真跌幅 4.44pp 被稀释成 3.70pp。
TYPE_SCORE_SLOTS = 7                            # = base_config.NUM_TYPES = len(TYPE_NAMES)


def type_score_bucket(gt, active_types=None, n_slots=TYPE_SCORE_SLOTS):
    """把一条 crash 样本的 GT 归入三档计量桶, 返回桶名 (str)。

    'cv'           可评分: 有输出槽 **且** 在本相位课程内 (apply_mask 给它算过 CE)
    'unslotable'   无输出槽 (GT<0 或 >=n_slots) ⇒ 计进分母就是构造性必错
    'labeled_out'  有槽但课程外 ⇒ 本相位从未给它梯度, 可达却无训练

    active_types=None (相位未知 / 非时序 checkpoint) ⇒ 不做课程过滤, 只剔无槽样本。
    """
    g = int(gt)
    if g < 0 or g >= n_slots:
        return 'unslotable'
    if active_types is None or g in set(active_types):
        return 'cv'
    return 'labeled_out'

# ── V25-ablation: 环境变量消融开关 ──
# 设置 V25_ABLATE_<MODULE>=1 可关闭对应模块 (论文消融实验用)
# 示例: V25_ABLATE_CTTA=1 V25_ABLATE_PTCL=1 python train.py --phase 3c ...
import os as _os
if _os.environ.get('V25_ABLATE_CTTA', '0') == '1':
    CTTA_CONFIG['enable'] = False
    CTTA_CONFIG['type_conditional_bias'] = False
    print('[Ablation] CTTA 已禁用')
if _os.environ.get('V25_ABLATE_UASC', '0') == '1':
    UASC_CONFIG['enable'] = False
    print('[Ablation] UASC 已禁用')
if _os.environ.get('V25_ABLATE_CDCR', '0') == '1':
    # 这两个键零读者 ⇒ 单看这里是死开关 (真接线在本文件 V22_INNOVATION_LOSS_WEIGHTS 之后)
    CDCR_CONFIG['enable'] = False
    CDCR_CONFIG['lambda_cdcr'] = 0.0
    print('[Ablation] CDCR 已禁用')
if _os.environ.get('V25_ABLATE_PTCL', '0') == '1':
    PTCL_CONFIG['enable'] = False
    # 全部类型立即激活 (无课程)
    PTCL_CONFIG['phase2_active_types'] = [0, 1, 2, 3, 4, 5, 6]
    PTCL_CONFIG['phase3a_active_types'] = [0, 1, 2, 3, 4, 5, 6]
    PTCL_CONFIG['phase3b_active_types'] = [0, 1, 2, 3, 4, 5, 6]
    PTCL_CONFIG['phase3c_active_types'] = [0, 1, 2, 3, 4, 5, 6]
    print('[Ablation] PTCL 已禁用 (全部类型立即激活)')
if _os.environ.get('V25_TYPE_EMBED_DIM'):
    _dim = int(_os.environ['V25_TYPE_EMBED_DIM'])
    CTTA_CONFIG['type_embed_dim'] = _dim
    print(f'[Ablation] type_embed_dim={_dim}')
if _os.environ.get('V25_ABLATE_FOCAL', '0') == '1':
    # B3 消融: 禁用所有 focal modulation (gamma=0)
    ABLATE_FOCAL = True
    print('[Ablation] Focal modulation 已禁用 (gamma=0)')
else:
    ABLATE_FOCAL = False

# ── V22 Phase 参数 (融合创新策略) ──
# Phase 1: 空间 3D 感知预热 (继承 V21)
#   冻结 backbone 前 10 层, 训练检测头 + 场景分类头
#   V22 创新: 引入 UASC 不确定性感知标签
# Phase 2: 时序因果推理 + 类型课程 (V22 创新)
#   解冻全量, 联合训练检测+场景+类型 (渐进式类型课程)
#   V22 创新: PTCL 仅训练 rear_end + side_impact
# Phase 3a: 类型扩展 + CTTA 时序注意力 (V22 创新)
#   加入 head_on 类型, 启用碰撞类型感知时序注意力
# Phase 3b: 跨域对齐 + 全类型 (V22 创新)
#   加入 fixed_object, 启用 CDCR 跨域一致性正则
# Phase 3c: 全系统联合优化
#   全部 5 类碰撞, CDCR + CTTA + UASC 全部启用
V22_PHASE_EPOCHS = {
    '1':  40,   # Phase 1: 空间 3D 感知预热
    '2':  60,   # Phase 2: 时序因果推理 + 类型课程 (rear_end + side_impact)
    '3a': 15,   # Phase 3a: 类型扩展 (+ head_on) + CTTA 启用
    '3b':  8,   # Phase 3b: 跨域对齐 (+ fixed_object) + CDCR 启用 (3b val_loss ep3最低→缩短)
    '3c': 10,   # Phase 3c: 全系统联合 (全部 5 类 + 所有创新模块) (3c ep1已过拟合→缩短)
}

V22_PHASE_LR = {
    '1':  0.005,   # Phase 1: 高 LR 快速收敛检测头
    '2':  0.0005,  # Phase 2: 微调 LR (V25: 0.001→0.0005, 防止灾难性遗忘)
    '3a': 0.0005,  # Phase 3a: 微调 LR, CTTA 模块初始化
    '3b': 0.0003,  # Phase 3b: 更低 LR, CDCR 需稳定收敛
    '3c': 0.0001,  # Phase 3c: 极小 LR, 全系统精细微调
}

# V22-fix(HW): GPU 安全 batch 上限 — 超过 48 容易 GPU 崩溃 (用户约束)
# train.py 会在解析 batch_size 后 clamp 到此上限
V22_MAX_BATCH = 48

V22_PHASE_BATCH = {
    '1':  48,   # Phase 1: 与 P2-P3c 统一 batch 48
    '2':  48,   # Phase 2: 大 batch 加速收敛 (原 64 → clamp 至 GPU 上限 48)
    '3a': 48,   # Phase 3a: 中等 batch (CTTA 增加 GPU 开销)
    '3b': 48,   # Phase 3b: 中等 batch (CDCR 需双域 batch)
    '3c': 48,   # Phase 3c: 中等 batch
}

# ── V22 创新损失权重配置 ──
# 基类损失继承 V21: L_det + λ₂L_scene + λ₄L_cause + λ₅L_type + λ₆L_3d
# V22 新增: λ_uasc (修改 L_scene 计算方式) + λ_cdcr (跨域一致性)
# C6修复(Critical)+W9修复(Warning): 修正 Phase 1/2 配置 — V22 创新仅在 Phase 3 生效
# 移除 ctta 权重 (CTTA 无独立损失项, 其效果通过修改注意力偏置实现)
V22_INNOVATION_LOSS_WEIGHTS = {
    # Phase:  (λ_uasc, λ_cdcr)
    '1':  (0.0,  0.0),   # Phase 1: 标准 V21 训练 (无 V22 创新)
    '2':  (0.0,  0.0),   # Phase 2: 标准 V21 训练 (无 V22 创新)
    '3a': (0.3,  0.0),   # Phase 3a: UASC 标签平滑 + CTTA 启动
    '3b': (0.3,  0.1),   # Phase 3b: UASC + CDCR + CTTA 解冻
    '3c': (0.3,  0.1),   # Phase 3c: 全部创新模块联合优化
}

# G-323(#273・E3): V25_ABLATE_CDCR 的真接线在这里。文件上方改 CDCR_CONFIG 两键的那段是零读者
# (E3 普查尺判 dead), λ_cdcr 的真实来源是本字典第二元: train.py:559 读 → args._innov_weights
# → phase3_train.py:302 cdcr_lambda → :999-1000 `is_cdcr_active = ... and cdcr_lambda > 0`
# 决定整条 CDCR 路径走不走 → temporal_model.py:1081 `+ self.lambda_cdcr * cdcr_loss`。
# env 未设或 =0 时本块零执行 ⇒ 默认盘面与改前逐字节等价 (五探针见本轮 EDIT 凭据)。
if _os.environ.get('V25_ABLATE_CDCR', '0') == '1':
    for _ph in ('3b', '3c'):
        _uw, _cw = V22_INNOVATION_LOSS_WEIGHTS[_ph]
        V22_INNOVATION_LOSS_WEIGHTS[_ph] = (_uw, 0.0)
    print('[Ablation] CDCR 真 λ 通道归零: V22_INNOVATION_LOSS_WEIGHTS[3b/3c][1]=0.0'
          ' ⇒ is_cdcr_active 恒 False')

# V22 默认实验名
DEFAULT_EXPERIMENT_NAME = 'accident_detect_v24'

# V22 默认 YAML 配置
DEFAULT_YAML = 'config/accident_detect.yaml'


# ============================================================
# 8. V22 Transformer 增强配置
# ============================================================
#
# V22 在 V21 TemporalContextModule 基础上的 Transformer 增强:
#
# V21 现有架构 (temporal_context.py):
#   L1: 1D-CNN (局部运动感知)
#   L2: 标准 Self-Attention (时序模式识别)
#   L3-L4: 标准 Self-Attention (因果链推理)
#
# V22 增强 (temporal_model.py):
#   L1: 1D-CNN (继承 V21)
#   L2: 标准 Self-Attention (继承 V21, Phase 2 已收敛)
#   L3-L4: CTTA 碰撞类型感知注意力 (V22 创新)
#
# CTTA 替换 L3-L4 的标准 Self-Attention:
#   原始: Attention(Q,K,V) = softmax(QK^T/sqrt(d)) * V
#   CTTA: Attention(Q,K,V,t) = softmax(QK^T/sqrt(d) + B(t)) * V
#   B(t) = bias_net(type_embed(t))
#

# Transformer 增强架构参数
TRANSFORMER_ENHANCE_CONFIG = {
    'enable_ctta': True,              # 启用 CTTA 增强
    'feat_dim': 256,                  # 特征维度 (与 V21 对齐)
    'num_heads': 8,                   # 注意力头数 (与 V21 对齐)
    'depth': 4,                       # Transformer 总层数
    'standard_layers': 2,             # 标准层 (L1-L2, 继承 V21)
    'ctta_layers': 2,                 # CTTA 层 (L3-L4, V22 创新)
    'dropout': 0.5,                   # Dropout (与 V21 Phase 3 一致)
    'use_tracklet': True,             # 对象级 tracklet (继承 V20)
    'max_objs': 8,                    # 每帧最大对象数
    # 渐进式冻结策略
    'freeze_standard_in_3a': True,    # Phase 3a 冻结 L1-L2
    'unfreeze_all_in_3b': True,       # Phase 3b 解冻全部
    'unfreeze_all_in_3c': True,       # Phase 3c 全部可训练
}

# CTTA 偏置生成网络配置
CTTA_BIAS_CONFIG = {
    'type_embed_dim': 64,             # 类型嵌入维度
    'bias_hidden_mult': 2,            # 偏置网络隐藏层倍数
    'bias_init': 'zero',              # 偏置初始化 (零初始化保证训练初期等价于标准注意力)
    'temperature_scaling': True,      # 温度缩放
}

# V22 模型参数量估算 (YOLO26-m 基准)
V22_MODEL_PARAMS_ESTIMATE = {
    'yolo26m_backbone': '20M',        # YOLO26-m backbone
    'frame_pool': '0.5M',             # 帧特征池化
    'standard_transformer_l2': '2.4M', # 标准 Transformer L1-L2 (继承 V21)
    'ctta_transformer_l3l4': '2.6M',   # CTTA Transformer L3-L4 (V22 增加 ~0.2M)
    'decision_heads': '1.5M',         # 四任务决策头
    'total_estimate': '~27M',         # 总参数量估算
    'ctta_overhead': '+0.2M (0.8%)',  # CTTA 额外开销
}


# ============================================================
# 9. V25 PF-RPN 配置 (Proposal-Free Region Proposal Network)
# ============================================================

# V25 在 V24 基础上新增 PF-RPN 并行检测头:
#   - 无类别约束的 anchor-free 候选区域生成器
#   - 能自动发现画面中所有潜在物体 (散落物、非常规车辆、遮挡目标)
#   - 与 DetectClassify 并行运行, 通过 IoU 融合补充 3 类检测头的盲区
#   - 参数量 ~5.4M, 显存开销 ~1.1GB (在 6GB 余量内)

V25_PFRPN_ENABLED = True  # 默认启用, 可通过 --no-pfrpn 关闭

# V25 Phase 参数 (与 V24 共享, 仅 P1/P2 启用 PF-RPN)
V25_PHASE_EPOCHS = {
    '1':  40,   # Phase 1: 空间感知预热 + PF-RPN (可训练)
    '2':  60,   # Phase 2: 联合训练, PF-RPN 权重减半
    '3a': 15,   # Phase 3a: 类型扩展 + CTTA, PF-RPN 冻结
    '3b':  8,   # Phase 3b: 跨域对齐 + CDCR, PF-RPN 冻结
    '3c': 10,   # Phase 3c: 全系统联合优化, PF-RPN 冻结
}

V25_PHASE_LR = {
    '1':  0.005,   # Phase 1: 高 LR 快速收敛检测头
    '2':  0.0005,  # Phase 2: 微调 LR (V25: 0.001→0.0005, 防止灾难性遗忘)
    '3a': 0.0005,  # Phase 3a: 微调 LR, CTTA 模块初始化
    '3b': 0.0003,  # Phase 3b: 更低 LR, CDCR 需稳定收敛
    '3c': 0.0001,  # Phase 3c: 极小 LR, 全系统精细微调
}

V25_PHASE_BATCH = {
    '1':  48,   # Phase 1: batch 48
    '2':  48,   # Phase 2: batch 48
    '3a': 48,   # Phase 3a: batch 48
    '3b': 48,   # Phase 3b: batch 48
    '3c': 48,   # Phase 3c: batch 48
}

# V25 创新损失权重配置
# Phase 1/2: 启用 PF-RPN (obj=1.0, box=2.0, cen=0.5, 前5ep warmup 0.5×)
# Phase 3a/3b/3c: 冻结 PF-RPN, 继承 V24 创新模块
V25_INNOVATION_LOSS_WEIGHTS = {
    # Phase:  (λ_uasc, λ_cdcr)
    '1':  (0.0,  0.0),   # Phase 1: 标准训练 + PF-RPN
    '2':  (0.0,  0.0),   # Phase 2: 标准训练 + PF-RPN (权重减半)
    '3a': (0.3,  0.0),   # Phase 3a: UASC 标签平滑 + CTTA 启动 (PF-RPN 冻结)
    '3b': (0.3,  0.1),   # Phase 3b: UASC + CDCR + CTTA 解冻
    '3c': (0.3,  0.1),   # Phase 3c: 全部创新模块联合优化
}

# V25 PF-RPN 参数量估算 (YOLO26-m 基准)
V25_MODEL_PARAMS_ESTIMATE = {
    'yolo26m_backbone': '20M',        # YOLO26-m backbone
    'frame_pool': '0.5M',             # 帧特征池化
    'standard_transformer_l2': '2.4M', # 标准 Transformer L1-L2
    'ctta_transformer_l3l4': '2.6M',   # CTTA Transformer L3-L4
    'decision_heads': '1.5M',         # 四任务决策头
    'pfrpn': '5.4M',                  # PF-RPN 新增参数 (~5.4M)
    'total_estimate': '~32.4M',       # 总参数量估算
    'pfrpn_overhead': '+5.4M (16.7%)', # PF-RPN 额外开销
}


# ============================================================
# 7. 便捷信息输出
# ============================================================

def print_train_config():
    """打印 V25 配置摘要 (包含 PF-RPN)"""
    print("=" * 70)
    print("V25 ACCIDENT 训练配置 — YOLO26-m + PF-RPN + 四项架构创新")
    print("=" * 70)
    print(f"  V22 PROJECT_ROOT:   {PROJECT_ROOT_V22}")
    print(f"  V21 SCRIPT_DIR:     {V24_BASE_DIR}")
    print(f"  ACCIDENT 数据根:    {_ACCIDENT_DATA_ROOT}")
    print(f"  默认数据源:          {DEFAULT_DATASET}")
    print(f"  默认实验名:          {DEFAULT_EXPERIMENT_NAME}")
    print(f"  日志目录:            {LOGS_DIR_V22}")
    print(f"  cause 门禁放宽:      {sorted(_CAUSE_RELAXED_DATASETS)}")
    print(f"  ACCIDENT 类型映射:   {ACCIDENT_TYPE_MAP}")
    print()
    print("  ── V25 PF-RPN (Proposal-Free RPN) ──")
    print(f"  启用状态:            {'ON' if V25_PFRPN_ENABLED else 'OFF'}")
    print(f"  参数量:              {V25_MODEL_PARAMS_ESTIMATE['pfrpn']}")
    print(f"  架构:                anchor-free, 类别无关, P3/P4/P5 三尺度")
    print(f"  损失:                obj=1.0, box=2.0, cen=0.5 (前5ep warmup 0.5×)")
    print()
    print("  ── V22 四项架构创新 ──")
    print(f"  [1] CTTA 碰撞类型感知时序注意力: {'ON' if CTTA_CONFIG['enable'] else 'OFF'}")
    print(f"      type_embed_dim={CTTA_CONFIG['type_embed_dim']}, "
          f"heads={CTTA_CONFIG['attention_heads']}, "
          f"window={CTTA_CONFIG['temporal_window']}")
    print(f"  [2] UASC 不确定性感知场景分类:   {'ON' if UASC_CONFIG['enable'] else 'OFF'}")
    print(f"      smoothing=[{UASC_CONFIG['label_smoothing_high']},"
          f"{UASC_CONFIG['label_smoothing_low']}], "
          f"threshold={UASC_CONFIG['agreement_threshold']}")
    print(f"  [3] CDCR 跨域一致性正则化:       {'ON' if CDCR_CONFIG['enable'] else 'OFF'}")
    print(f"      lambda={CDCR_CONFIG['lambda_cdcr']}, "
          f"T={CDCR_CONFIG['temperature']}, "
          f"warmup={CDCR_CONFIG['warmup_epochs']}ep")
    print(f"  [4] PTCL 渐进式类型课程学习:     {'ON' if PTCL_CONFIG['enable'] else 'OFF'}")
    print(f"      难度排序: {PTCL_CONFIG['type_difficulty_order']}")
    print(f"      P2: {PTCL_CONFIG['phase2_active_types']} → "
          f"P3a: {PTCL_CONFIG['phase3a_active_types']} → "
          f"P3c: {PTCL_CONFIG['phase3c_active_types']}")
    print()
    print("  ── V25 五阶段训练策略 (PF-RPN P1/P2 启用, P3 冻结) ──")
    for ph in ['1', '2', '3a', '3b', '3c']:
        ep = V25_PHASE_EPOCHS[ph]
        lr = V25_PHASE_LR[ph]
        bs = V25_PHASE_BATCH[ph]
        iw = V25_INNOVATION_LOSS_WEIGHTS[ph]
        pfrpn = 'ON' if ph in ('1', '2') else 'OFF'
        print(f"  Phase {ph:2s}: epochs={ep:3d}, lr={lr:.4f}, batch={bs:2d}, "
              f"loss=(uasc={iw[0]:.1f}, cdcr={iw[1]:.1f}), PF-RPN={pfrpn}")
    print("=" * 70)


if __name__ == '__main__':
    print_train_config()
