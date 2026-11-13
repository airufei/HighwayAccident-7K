# -*- coding: utf-8 -*-
"""
V22 训练入口 — YOLO26-m + AccidentFormer 三阶段渐进式训练 (ACCIDENT 数据集)

V22 继承 V21 三阶段五步架构，在此基础上提出四项架构创新:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                 V22 三阶段五步渐进式训练 + 四项创新                      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  Phase 1 (40ep): 空间 3D 感知预热 + 成因单帧判别                       │
  │    继承 V21: 冻结 backbone 前10层, 训练检测头 + 场景分类头             │
  │    V24: VLM 重标注后 scene_labels 第2行含 16-bit 成因,                 │
  │         cause loss 从 Phase 1 即启用 (早期单帧成因判别)                │
  │    [注] V22 创新模块在 Phase 3 时序循环中生效                         │
  │                                                                       │
  │  Phase 2 (60ep): 时序因果推理 + 成因强化                               │
  │    继承 V21: 解冻全量, 联合训练检测+场景+时序                         │
  │    [注] V22 创新模块在 Phase 3 时序循环中生效                         │
  │                                                                       │
  │  Phase 3a (15ep): 类型扩展 + 时序注意力 + 成因深度判定                  │
  │    V22 增强 Phase 3: 冻结 backbone, 训练 CTTA + 决策头                 │
  │    V24: 时序上下文语义理解深化成因判定 (λ_cause=2.0)                   │
  │    [创新1] CTTA: 碰撞类型感知时序注意力 (type_embed→attention bias)    │
  │    [创新2] UASC: 不确定性感知标签平滑                                  │
  │    [创新4] PTCL: 加入 head_on 类型                                     │
  │                                                                       │
  │  Phase 3b (15ep): 跨域对齐 + 全类型                                    │
  │    V22 增强 Phase 3: 解冻 Transformer 全层                             │
  │    [创新3] CDCR: 跨域一致性正则化 (KL(real||synth))                   │
  │    [创新4] PTCL: 加入 fixed_object 类型                               │
  │                                                                       │
  │  Phase 3c (20ep): 全系统联合优化                                       │
  │    V22 增强 Phase 3: 全部参数解冻, 联合优化                           │
  │    [全部创新] CTTA + UASC + CDCR + PTCL (5类碰撞全部激活)             │
  └─────────────────────────────────────────────────────────────────────────┘

核心模块复用 V21 (不修改):
  losses / dataset / detection / phase3_legacy

用法:
    cd v26_training_package
    python scripts/train.py --phase 1 --dataset accident-both
    python scripts/train.py --phase 2 --dataset accident-both
    python scripts/train.py --phase 3a --dataset accident-both
    python scripts/train.py --phase 3b --dataset accident-both
    python scripts/train.py --phase 3c --dataset accident-both

ACCIDENT 数据转换 (首次使用前必须执行):
    python scripts/convert_accident.py \
        -i /path/to/source_archive \
        -o datasets/accident_v24

标签验证 (转换后):
    python scripts/validate_conversion.py \
        -i datasets/accident_v24 \
        --full-check
"""

import os
import sys
import json
import glob
import argparse

# ── V22 配置 (链式导入 V21 核心模块) ──
import train_config
from train_config import (  # noqa: F401
    PROJECT_ROOT_V22, V24_BASE_DIR, LOGS_DIR_V22,
    DATASET_CONFIGS, DEFAULT_DATASET, DEFAULT_EXPERIMENT_NAME, DEFAULT_YAML,
    V22_PHASE_EPOCHS, V22_PHASE_LR, V22_PHASE_BATCH,
    ACCIDENT_TYPE_MAP, is_cause_relaxed,
    MIN_QUALITY_SCORE_V22, MIN_CAUSE_COVERAGE_V22,
    CTTA_CONFIG, UASC_CONFIG, CDCR_CONFIG, PTCL_CONFIG,
    V22_INNOVATION_LOSS_WEIGHTS,
    TRANSFORMER_ENHANCE_CONFIG, CTTA_BIAS_CONFIG,
    print_train_config,
)

# V22 增强 Transformer 模块
try:
    from temporal_model import (
        V24TemporalModel, V22TemporalMultiTaskLoss,
        CTTATemporalContext, CTTAAttentionLayer,
        UASCSmoothLabel, CDCRLoss, PTCLTypeMask,
        build_temporal_model, print_v24_transformer_info,
    )
    _has_temporal_model = True
except ImportError as _e:
    _has_temporal_model = False
    print(f"WARN: temporal_model 不可用: {_e}")

# V22 专用 Phase 3 训练循环
try:
    from phase3_train import run_phase3_v24_temporal
    _has_phase3_train = True
except ImportError as _e:
    _has_phase3_train = False
    print(f"WARN: phase3_train 不可用: {_e}, 回退到 V21 Phase 3")

# 从 V21 复用的核心模块 (通过 train_config 的 sys.path 注入)
import base_config
from base_config import (  # noqa: F401
    dist, DDP, DistributedSampler, _has_dist,
    CAUSE_NAMES, TYPE_NAMES,
    DEFAULT_CLASS_WEIGHTS, TYPE_CLASS_WEIGHT,
    calc_dynamic_cause_pos_weight, MemoryManager,
    load_balance_report, get_progressive_weights,
    adaptive_weight_update, get_phase3_loss_weights,
)
from losses import (  # noqa: F401
    AsymmetricFocalLoss, CauseMultiLabelLoss, AccidentTypeLoss, v18MultiTaskLoss,
)
from dataset import (  # noqa: F401
    SceneClassBalancedSampler, MultiDatasetAccident, ClipDataset,
)
from phase3_legacy import (  # noqa: F401
    _has_temporal, run_phase3_temporal,
    V18TemporalModel, TemporalMultiTaskLoss, TemporalContextModule,
    compute_detection_latency,
)
from detection import (  # noqa: F401
    DetectClassifyModelV18, V18Validator, V18Trainer,
    _update_sampler_epoch, _v24_dynamic_box_weight,
    _v24_scene_aware_augmentation, _v24_log_scene_dist,
    _v18_eval_scene_classification,
)

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    from ultralytics.utils.torch_utils import unwrap_model as de_parallel


# ============================================================
# DDP __module__ 注入 (V21 + V22 类)
# ============================================================
import __main__ as _main_module
_ddp_classes = [
    MemoryManager, AsymmetricFocalLoss, CauseMultiLabelLoss, AccidentTypeLoss,
    SceneClassBalancedSampler, MultiDatasetAccident, v18MultiTaskLoss,
    DetectClassifyModelV18, V18Validator, V18Trainer, ClipDataset,
]
# 注入 V22 类 (DDP 子进程 pickle 反序列化需要)
# W4修复(Warning): 仅注入 V22 新增类 — V21 类已在 V21 train.py 中注入
_v24_ddp_classes = []
if _has_temporal_model:
    _v24_ddp_classes.extend([
        V24TemporalModel, V22TemporalMultiTaskLoss,
        CTTATemporalContext, UASCSmoothLabel, CDCRLoss, PTCLTypeMask,
    ])
try:
    from train_data import V24ClipDataset
    _v24_ddp_classes.append(V24ClipDataset)
except ImportError:
    pass
# 注入 V22 监控类
try:
    from monitor import TrainingMonitor
    _v24_ddp_classes.append(TrainingMonitor)
except ImportError:
    pass
for _cls in _v24_ddp_classes:
    _cls.__module__ = 'train'
    setattr(_main_module, _cls.__name__, _cls)


# ============================================================
# V22 数据质量门禁 — 放宽 cause_coverage
# ============================================================

def _data_quality_gate_v22(data_root, args):
    """V22 数据质量门禁

    与 V21 差异:
      - cause_coverage 门禁对 ACCIDENT 数据源放宽至 0 (无 cause 标注)
      - 质量报告分值仍要求 >= 60
    """
    if getattr(args, 'skip_quality_gate', False):
        print("WARNING: --skip-quality-gate 已启用, 跳过数据质量门禁 (不推荐)")
        return

    # ── 门禁 1: scene_labels 存在性 ──
    scene_dir = os.path.join(data_root, 'scene_labels', 'train')
    if not os.path.isdir(scene_dir):
        print(f"ERROR: scene_labels 目录不存在: {scene_dir}")
        print(f"  请先运行 ACCIDENT 转换脚本:")
        print(f"  python scripts/convert_accident.py -i <archive路径> -o {data_root}")
        sys.exit(1)
    n_scene = len([f for f in os.listdir(scene_dir) if f.endswith('.txt')])
    if n_scene == 0:
        print(f"ERROR: scene_labels/train 无 .txt 标签文件: {scene_dir}")
        sys.exit(1)
    print(f"  V22 数据门禁: scene_labels/train 标签文件 {n_scene} 个 ✓")

    # ── 门禁 2: 质量报告 ──
    meta_dir = os.path.join(data_root, 'meta')
    report_path = None
    for fname in ('quality_report.json', 'data_quality_report.json'):
        p = os.path.join(meta_dir, fname)
        if os.path.exists(p):
            report_path = p
            break
    if report_path is None:
        print(f"ERROR: 未找到数据质量报告: {meta_dir}/quality_report.json")
        print(f"  convert_accident.py 会自动生成该报告, 请确认转换已完成。")
        print(f"  确需跳过请加 --skip-quality-gate (不推荐)")
        sys.exit(1)
    try:
        with open(report_path, encoding='utf-8') as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: 质量报告读取失败: {e}")
        sys.exit(1)

    score = report.get('overall_score', report.get('quality_score', 0))
    if score < MIN_QUALITY_SCORE_V22:
        print(f"ERROR: 数据质量门禁未通过: score={score} < {MIN_QUALITY_SCORE_V22}")
        print(f"  完整报告: {report_path}")
        sys.exit(1)
    print(f"  V22 数据门禁: 质量报告通过 (score={score}) ✓")

    # ── 门禁 3: cause 覆盖率 ──
    # V24: ACCIDENT 经 VLM 重标注后已有成因标签, 恢复 0.30 门禁
    # (实测 train 覆盖率 99.2%); is_cause_relaxed 仅对真正无成因的数据源放宽
    cause_threshold = MIN_CAUSE_COVERAGE_V22 if is_cause_relaxed(args.dataset) else 0.30

    pc_total = 0
    cause_covered = 0
    for fname in os.listdir(scene_dir):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(scene_dir, fname)
        try:
            with open(fpath) as f:
                lines = f.read().strip().split('\n')
                if len(lines) < 1:
                    continue
                scene_cls = int(lines[0])
                if scene_cls in (2, 3):
                    pc_total += 1
                    if len(lines) >= 2:
                        try:
                            int(lines[1].strip())
                            # G-238 用户裁决 A（2026-10-16）：覆盖率=成因判定在场（第 2 行已写），
                            # 不是正掩码密度——有据负例（判过=0）是合法负例（标准 §2.1 两态语义）。
                            # 门槛 30% 不变；修后实测 35.61%（31,557/88,625）。
                            cause_covered += 1
                        except ValueError:
                            pass
        except (OSError, ValueError):
            continue

    if pc_total > 0:
        cause_rate = cause_covered / pc_total
        if cause_rate < cause_threshold:
            print(f"ERROR: cause 覆盖率门禁未通过: {cause_covered}/{pc_total} "
                  f"({cause_rate:.1%}) < {cause_threshold:.0%}")
            if is_cause_relaxed(args.dataset):
                print(f"  (ACCIDENT 数据源已放宽至 0%, 此错误不应出现 — 请检查数据转换)")
            sys.exit(1)
        if is_cause_relaxed(args.dataset):
            print(f"  V22 数据门禁: cause 覆盖率 {cause_covered}/{pc_total} "
                  f"({cause_rate:.1%}) [ACCIDENT 放宽, 允许 0%] ✓")
        else:
            print(f"  V22 数据门禁: cause 覆盖率 {cause_covered}/{pc_total} "
                  f"({cause_rate:.1%}) ≥ {cause_threshold:.0%} ✓")
    else:
        print(f"  WARNING: scene_labels/train 中无 pre_crash/crash 帧")


# ============================================================
# V23-fix(cause-坍缩): 数据驱动 cause pos_weight
# ============================================================

def _calc_cause_pos_weight(data_root, num_causes=9, max_weight=20.0):
    """按 scene_labels/train 实际位频率计算 pos_weight = neg/pos (截断 [1, 20])。

    根因: BCE + sigmoid>0.5 判定下, 位的平衡点 sigmoid* = w·q/(1-q+w·q)。
    旧固化权重对低频率位 (如 bit6, q≈11%, w=2 → sigmoid*≈0.20) 永远够不到
    0.5 阈值 → 无论训练多好, 该位 Recall 结构性为 0 (unified_v21 验证视频
    恰好只有 bit6 → Cause Recall 恒 0%)。neg/pos 权重使各位平衡点回到 0.5。
    """
    scene_dir = os.path.join(str(data_root), 'scene_labels', 'train')
    pos = [0] * num_causes
    n_annot = 0
    if os.path.isdir(scene_dir):
        for fname in os.listdir(scene_dir):
            if not fname.endswith('.txt'):
                continue
            try:
                with open(os.path.join(scene_dir, fname)) as f:
                    lines = f.read().split()
                if len(lines) < 2 or int(lines[0]) < 2:
                    continue
                raw = lines[1]
                if ',' in raw:
                    bits = [int(b) for b in raw.split(',')][:num_causes]
                    bits += [0] * (num_causes - len(bits))
                else:
                    m = int(raw)
                    bits = [(m >> i) & 1 for i in range(num_causes)]
            except (OSError, ValueError):
                continue
            n_annot += 1
            for i, b in enumerate(bits):
                if b:
                    pos[i] += 1
    if n_annot == 0:
        return None
    weights = []
    for p in pos:
        if p <= 0:
            weights.append(1.0)
        else:
            weights.append(round(min(max_weight,
                                     max(1.0, (n_annot - p) / p)), 2))
    return base_config.apply_cause_mask(weights)


# ============================================================
# 链式检查点定位
# ============================================================

def _resolve_chain_checkpoint(args, run_name, weight_name):
    """V26-G258: 链式种子解析提纯 (可单测)。返回 (路径, 是否显式钉死)。

    --chain-from 显式钉种子优先 (调用方对钉死路径缺失直接退); 否则按运行名
    解析。按名解析走 _resolve_run_weight 的 -N 变体 mtime 启发式 —— 它挑的是
    "最近改过的变体", 不是"该交棒的那个 run", 只作未钉死时的兜底。
    """
    if getattr(args, 'chain_from', None):
        return args.chain_from, True
    return _resolve_run_weight(args.project, run_name, weight_name), False


def _resolve_run_weight(project, run_name, weight='best.pt'):
    """定位某次训练产出的权重文件 (默认 weights/best.pt)。

    ultralytics 的 save_dir 并不总是 `{project}/{run_name}`: detect 任务会在
    project 下嵌套 task 子目录 (实测为 `{project}/detect/runs/{run_name}`)。
    因此不能硬编码单一路径。先按已知候选布局快速查找, 未命中再在 project
    下递归搜索。找不到时返回首选候选路径 (供上层打印报错)。
    """
    candidates = [
        os.path.join(project, run_name, 'weights', weight),
        os.path.join(project, 'detect', run_name, 'weights', weight),
        os.path.join(project, 'detect', 'runs', run_name, 'weights', weight),
        os.path.join(project, 'runs', run_name, 'weights', weight),
        # 时序检查点 (best_temporal.pt) 直接放在 run 目录下
        os.path.join(project, run_name, weight),
        os.path.join(project, 'detect', 'runs', run_name, weight),
    ]
    for c in candidates:
        if os.path.exists(c):
            # V23-A: ultralytics resume 会写入递增后缀目录 (如 v23_s8_p2-3);
            # 链式取种须优先最新变体, 否则会拿到续训前的旧 best.pt
            _d = os.path.dirname(os.path.dirname(c))
            _base, _parent = os.path.basename(_d), os.path.dirname(_d)
            _var = [v for pat in (os.path.join(_parent, _base + '-*', 'weights', weight),
                                  os.path.join(_parent, _base + '-*', weight))
                    for v in glob.glob(pat)]
            _var = [v for v in _var if os.path.getmtime(v) > os.path.getmtime(c)]
            if _var:
                return max(_var, key=os.path.getmtime)
            return c
    hits = []
    for pat in (os.path.join(project, '**', run_name, 'weights', weight),
                os.path.join(project, '**', run_name, weight)):
        hits.extend(glob.glob(pat, recursive=True))
    if hits:
        # 取最近修改的一个, 避免命中同名历史目录
        return max(hits, key=os.path.getmtime)
    return candidates[0]


# ============================================================
# 训练入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='V22 ACCIDENT 训练 — YOLO26-m + AccidentFormer 三阶段渐进式 (CTTA/UASC/CDCR/PTCL)')
    parser.add_argument('--phase', type=str, required=True,
                        choices=['1', '2', '3a', '3b', '3c'],
                        help='训练阶段: 1=检测预热, 2=联合训练, '
                             '3a=成因预训练, 3b=因果链微调, 3c=全系统优化')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        choices=list(DATASET_CONFIGS.keys()),
                        help=f'数据集名称 (默认 {DEFAULT_DATASET})')
    parser.add_argument('--resume', nargs='?', const='auto', type=str, default=None,
                        help='断点续训: 裸 --resume 自动定位, 或 --resume <ckpt路径>')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖默认 epoch 数')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小 (默认按Phase自动选择)')
    parser.add_argument('--name', type=str, default=DEFAULT_EXPERIMENT_NAME,
                        help=f'实验名称 (默认 {DEFAULT_EXPERIMENT_NAME})')
    parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
    parser.add_argument('--data-root', type=str, default=None,
                        help='数据集根目录 (覆盖默认)')
    parser.add_argument('--yaml', type=str, default=DEFAULT_YAML,
                        help=f'数据配置 YAML (默认 {DEFAULT_YAML})')
    parser.add_argument('--backbone', type=str, default='m',
                        choices=['m', 'l', 'x'],
                        help='Backbone 规模: m(20M)/l(45M)/x(68M)')
    parser.add_argument('--model-yaml', type=str, default=None,
                        help='模型配置 YAML (默认根据 --backbone 自动选择)')
    parser.add_argument('--pretrained', type=str,
                        default='weights/yolo26m.pt',
                        help='预训练权重')
    parser.add_argument('--device', type=str, default='0,1', help='GPU 设备')
    parser.add_argument('--project', type=str, default='runs', help='实验输出目录')
    parser.add_argument('--workers', type=int, default=16, help='DataLoader 工作进程')
    parser.add_argument('--lr0', type=float, default=None, help='覆盖学习率')
    parser.add_argument('--no-balanced-sampler', action='store_true',
                        help='禁用 SceneClassBalancedSampler')
    parser.add_argument('--sampler-ratios', type=str, default=None,
                        help='采样器配比, e.g. "0.2,0.5,0.3"')
    parser.add_argument('--clip-len', type=int, default=None,
                        help='Phase 3 时序 Clip 帧数')
    parser.add_argument('--allow-fallback', action='store_true',
                        help='允许链式检查点缺失时回退')
    parser.add_argument('--chain-from', type=str, default=None,
                        help='V26-G247: 显式指定链式加载的权重路径 (覆盖按运行名推导的检查点)')
    parser.add_argument('--skip-quality-gate', action='store_true',
                        help='跳过数据质量门禁 (不推荐)')
    parser.add_argument('--seed', type=int, default=0, help='全局随机种子')
    parser.add_argument('--monitor-interval', type=int, default=3,
                        help='检查点评估周期 (epoch), 默认 3')
    parser.add_argument('--no-auto-stop', action='store_true',
                        help='禁用自动停止机制')
    parser.add_argument('--pfrpn', action='store_true', default=False,
                        help='V25: 启用 PF-RPN (Proposal-Free Region Proposal Network)')
    parser.add_argument('--no-pfrpn', action='store_true', default=False,
                        help='V25: 禁用 PF-RPN (覆盖默认启用)')
    args = parser.parse_args()

    # V25: PF-RPN 控制
    if args.no_pfrpn:
        args.pfrpn = False
    os.environ['V25_PFRPN'] = '1' if args.pfrpn else '0'

    # 打印 V22 配置
    print_train_config()

    # 数据集配置 — 必须在 ensure_cwd_v22() 之前解析, 因为 os.chdir 会改变相对路径基准
    ds_config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS[DEFAULT_DATASET])
    data_root = args.data_root or ds_config['data_root']
    if data_root and not os.path.isabs(data_root):
        data_root = os.path.abspath(data_root)

    # C9修复(Critical): 显式调用 cwd 切换 (而非 import 副作用)
    import train_config
    train_config.ensure_cwd_v22()

    # V24: ACCIDENT 已含 VLM 标注的 16-bit 成因 (scene_labels 第2行),
    # Phase 3a 不再切换 unified-v21, 全阶段直接在 ACCIDENT 上监督成因。

    # V22 数据质量门禁
    _data_quality_gate_v22(data_root, args)

    # 动态计算 cause pos_weight
    # V23: 固化 V22 实验验证的按位权重 (天气位按频率降权防过预测,
    # 其余位无监督信号取中性 1.0); 仍允许环境变量覆盖
    # V24: 回退默认更新为 VLM 标注+均衡后数据驱动结果 (neg/pos, 截断[1,20])
    _V23_DEFAULT_CAUSE_POS_WEIGHT = [1.55, 9.01, 1.87, 20.0, 20.0, 20.0, 2.07,
                                     20.0, 1.0, 1.0, 16.44, 3.8, 1.0, 20.0, 3.72, 1.0]
    _cpw_override = os.environ.get('V24_CAUSE_POS_WEIGHT', '')
    _cpw_vals = [float(x) for x in _cpw_override.split(',') if x.strip()]
    # G-261: 闸从 ==16 放宽到 (9, 16)。根因: 9 值注入被本闸静默吞掉回退数据
    # 驱动重算 (修法无处生效), 16 值过闸但 detection.py 回读闸要求 len==9
    # (回读失真)。temporal_model [:9] 截断对 9 值恒等, 全链一致。
    if len(_cpw_vals) in (9, 16):
        dynamic_cause_weights = _cpw_vals
        print(f"  V23 CAUSE_POS_WEIGHT (环境覆盖): {dynamic_cause_weights}")
    else:
        # V23-fix(cause-坍缩): 按当前数据实际位频率计算 (neg/pos, 截断 [1,20]),
        # 使各位 BCE 平衡点回到 sigmoid=0.5; 无标注数据回退固化默认
        _data_w = _calc_cause_pos_weight(data_root)
        if _data_w is not None:
            dynamic_cause_weights = _data_w
            print(f"  V23 CAUSE_POS_WEIGHT (数据驱动): {dynamic_cause_weights}")
        else:
            dynamic_cause_weights = list(_V23_DEFAULT_CAUSE_POS_WEIGHT)
            print(f"  V23 CAUSE_POS_WEIGHT (固化默认, 数据无成因标注): "
                  f"{dynamic_cause_weights}")
    # W6修复: 通过环境变量传递给 DDP 子进程
    os.environ['V24_CAUSE_POS_WEIGHT'] = ','.join(str(x) for x in dynamic_cause_weights)

    print("=" * 60)
    print(f"V22 数据集: {args.dataset} ({ds_config.get('scene', 'unknown')})")
    print(f"数据根目录: {data_root}")
    print(f"数据源: {ds_config.get('source', 'unknown')}")
    print(f"标注质量: {ds_config.get('annotation_quality', 'unknown')}")
    print(f"实验名: {args.name}")
    print("=" * 60)

    # 加载均衡报告
    auto_weights, auto_ratios = load_balance_report(data_root)
    if auto_weights:
        print(f"参考: balance_report 逆频率权重 {auto_weights} (仅监控用)")
    if auto_ratios:
        print(f"从 balance_report.json 加载 Sampler 配比: {auto_ratios}")

    if auto_ratios:
        sampler_ratios = auto_ratios
    elif args.sampler_ratios:
        sampler_ratios = [float(x) for x in args.sampler_ratios.split(',')]
    else:
        sampler_ratios = [0.34, 0.33, 0.33]
    if not isinstance(sampler_ratios, (list, tuple)) or len(sampler_ratios) != 3:
        print(f"⚠️ sampler_ratios 长度异常, 回退默认 [0.34, 0.33, 0.33]")
        sampler_ratios = [0.34, 0.33, 0.33]
    class_weights = list(DEFAULT_CLASS_WEIGHTS)

    print(f"Sampler 配比: {sampler_ratios}")
    print(f"CLASS_WEIGHTS: {class_weights}")

    # Backbone 自动选择
    # V23: 默认使用 fixed 版 yaml (预训练迁移缺陷修复后的验证版本)
    if args.model_yaml is None:
        args.model_yaml = f'config/yolo26{args.backbone}_detect_cls_fixed.yaml'
    if args.pretrained == 'weights/yolo26m.pt' and args.backbone != 'm':
        args.pretrained = f'weights/yolo26{args.backbone}.pt'
    # DDP 修复: core.base_config 导入时 chdir 到 scripts/, ultralytics _temp_*.py
    # 子进程里相对路径全部失锚 (实测崩于 check_file('config/accident_detect.yaml'))
    # ⇒ 传给 trainer 的路径一律绝对化, 与历史成功 run 的 args.yaml 同形
    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def _abspath(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(_pkg_root, p))
    args.yaml = _abspath(args.yaml)
    args.model_yaml = _abspath(args.model_yaml)
    args.pretrained = _abspath(args.pretrained)
    args.project = _abspath(args.project)
    print(f"  V22 Backbone: YOLO26-{args.backbone} | model_yaml: {args.model_yaml}")
    print(f"  预训练权重: {args.pretrained}")

    if not os.path.exists(args.pretrained):
        print(f"ERROR: 预训练权重不存在: {args.pretrained}")
        sys.exit(1)

    # V22 Phase 参数 + 创新损失权重
    epochs = args.epochs or V22_PHASE_EPOCHS[args.phase]
    # V22-fix(epochs-回写): run_phase3_v24_temporal 读 args.epochs 且回退硬编码 15,
    # 不回写会使 3c 按 15 epoch 运行 (配置为 20); 3a/3b 因配置恰为 15 而被掩盖
    args.epochs = epochs
    lr0 = args.lr0 or V22_PHASE_LR[args.phase]
    args.lr0 = lr0

    if args.batch_size is None:
        args.batch_size = V22_PHASE_BATCH.get(args.phase, 32)
    # V22-fix(HW): batch 超过 GPU 安全上限会崩溃 → clamp
    if args.batch_size > train_config.V22_MAX_BATCH:
        print(f"  ⚠️ batch_size {args.batch_size} 超过 GPU 安全上限 "
              f"{train_config.V22_MAX_BATCH}, 已自动降为 {train_config.V22_MAX_BATCH}")
        args.batch_size = train_config.V22_MAX_BATCH
    print(f"  V22 batch_size (Phase {args.phase}): {args.batch_size}")

    # V22 创新损失权重 (传递给训练环境变量)
    innov_weights = V22_INNOVATION_LOSS_WEIGHTS[args.phase]
    print(f"  V22 创新损失 (Phase {args.phase}): "
          f"UASC={innov_weights[0]:.1f}, CDCR={innov_weights[1]:.1f}")

    base_overrides = dict(
        data=args.yaml,
        epochs=epochs,
        imgsz=640,
        batch=args.batch_size,
        device=args.device,
        project=args.project,
        workers=args.workers,
        patience=args.patience,
        save=True,
        save_period=5,
        amp=True,
        seed=args.seed,
        deterministic=True,
        optimizer='SGD',
        cos_lr=True,
        # V22-fix(io-根因): cache='disk' 让 ultralytics 把每张图另存为未压缩 .npy
        # (720x1056x3=2.27MB, jpg 仅 60KB, 37x 读放大), 已在慢速 /data 卷堆积 754GB,
        # 读吞吐被压到 7.5MB/s、GPU 利用率仅 54%。改直读 jpg (配合本地盘)。
        # cache='ram' 不可行: 原分辨率缓存 ~306G/进程, DDP 双卡共 612G > 376G 物理内存。
        cache=False,
        conf=0.10, iou=0.45,
        hsv_h=0.015, hsv_s=0.8, hsv_v=0.5,
        degrees=5.0, translate=0.1, scale=0.5,
        shear=0.5, perspective=0.001,
        flipud=0.0, fliplr=0.5,
        mosaic=0.0,
        mixup=0.15,
        copy_paste=0.0, erasing=0.3,
        # G-262: ultralytics close_mosaic 默认 10 在 20-epoch run 里等于后半程全关
        # (epoch==epochs-10 起 mosaic/mixup/copy_paste 全清零并重建管线, trainer.py:456
        # + dataset.close_mosaic), p2-16 e11 全 train 损失同降台阶+漂移加速即此因。
        # 按 10% 尾窗收敛 (20 轮→2 轮): 保留 mixup=0.15 的正则到训练后期, 只在收尾清场。
        close_mosaic=max(2, round(epochs * 0.1)),
    )

    # V22-fix: 检测损失仅在"完全无 bbox"的数据源上归零 (accident-real 纯真实集)。
    # 历史背景: 标签解析 bug 曾把全部 14 字段 bbox 丢弃, 当时观察到"检测损失在空标签上
    # 主导总损失" (架构审计 Defect3) 并归零了所有 accident 数据源的检测权重。
    # bug 修复后合成集有 ~140 万有效 bbox; v8DetectionLoss 对空标签帧不产生梯度,
    # 混合训练安全。检测头必须训练: 它是目标检测基础, 且 Phase 2/3 时序模块的
    # tracklet 输入依赖检测输出。
    if args.dataset == 'accident-real':
        base_overrides.update(box=0.0, cls=0.0, dfl=0.0)
        print(f"  [V22] {args.dataset}: 无 bbox 标注, 检测损失权重归零, 仅训练场景分类")

    phase_suffix = f'_p{args.phase}'

    if args.phase == '1':
        print(f"\nV22 Phase 1: 空间 3D 感知预热 ({epochs} epochs)")
        print(f"  [注] UASC/CTTA/CDCR/PTCL 创新模块在 Phase 3 时序循环中生效")
        phase1_patience = min(args.patience, 15)
        overrides = dict(base_overrides)
        overrides.update(
            model=args.model_yaml,
            name=args.name + phase_suffix,
            lr0=lr0,
            warmup_epochs=2.0,
            pretrained=args.pretrained,
            patience=phase1_patience,
            freeze=list(range(10)),
        )
        print(f"  V22 early_stopping patience={phase1_patience}")

    elif args.phase == '2':
        # V26-G247: --chain-from 显式指定续训种子 (如 e10 best.pt), 否则按运行名链式取种
        if args.chain_from:
            p1_best = args.chain_from
            if not os.path.exists(p1_best):
                print(f"ERROR: --chain-from 权重不存在: {p1_best}")
                sys.exit(1)
            print(f"  链式加载 --chain-from 权重: {p1_best}")
        else:
            p1_best = _resolve_run_weight(args.project, f'{args.name}_p1', 'best.pt')
            if not os.path.exists(p1_best):
                print(f"ERROR: Phase 1 产物不存在: {p1_best}")
                sys.exit(1)
            print(f"  链式加载 Phase 1 权重: {p1_best}")
        print(f"\nV22 Phase 2: 时序因果推理 ({epochs} epochs)")
        print(f"  [注] PTCL 类型课程在 Phase 3 时序循环中生效")
        overrides = dict(base_overrides)
        overrides.update(
            model=p1_best,
            name=args.name + phase_suffix,
            lr0=lr0,
            warmup_epochs=2.0,
            warmup_momentum=0.8,
            # V23-A(patience): V22 原配方为 20 (p2_args.yaml 快照); V22 中期的
            # 12 上限在 V23 重训中导致 ep23 过早早停、弱种子拖累 3a-3c 终版指标。
            # 恢复默认 20, 允许环境变量覆盖。
            patience=int(os.environ.get('V22_P2_PATIENCE', '20') or 20),
        )

    elif args.phase in ('3a', '3b', '3c'):
        phase_chain = {
            '3a': (f'{args.name}_p2', 'best.pt'),
            '3b': (f'{args.name}_p3a', 'best_temporal.pt'),
            '3c': (f'{args.name}_p3b', 'best_temporal.pt'),
        }
        _run_name, _weight_name = phase_chain[args.phase]
        chain_checkpoint, _pinned = _resolve_chain_checkpoint(
            args, _run_name, _weight_name)
        if not os.path.exists(chain_checkpoint):
            if _pinned:
                # 显式钉死的种子缺失即退: 不走 mtime 启发式、不走回退, 防静默错种子
                print(f"ERROR: --chain-from 权重不存在: {chain_checkpoint}")
                sys.exit(1)
            print(f"ERROR: 链式检查点不存在: {chain_checkpoint}")
            print(f"  Phase {args.phase} 依赖上一阶段产物, 请先完成前置训练。")
            fallback = _resolve_run_weight(args.project, f'{args.name}_p2', 'best.pt')
            if args.allow_fallback and args.phase != '3a' and os.path.exists(fallback):
                print(f"  --allow-fallback 已启用, 回退使用 Phase 2 产物: {fallback}")
                chain_checkpoint = fallback
            else:
                sys.exit(1)

        sub_names = {
            '3a': '类型扩展 + CTTA 时序注意力',
            '3b': '跨域对齐 + CDCR 一致性正则',
            '3c': '全系统联合优化 (CTTA+UASC+CDCR+PTCL)',
        }
        print(f"\nV22 Phase {args.phase}: {sub_names[args.phase]} ({epochs} epochs)")
        # V22 创新模块状态提示
        if args.phase == '3a':
            active_types = PTCL_CONFIG['phase3a_active_types']
            print(f"  [创新] CTTA 碰撞类型感知时序注意力启动 "
                  f"(type_embed_dim={CTTA_CONFIG['type_embed_dim']})")
            print(f"  [创新] PTCL 类型扩展: {active_types} "
                  f"({[TYPE_NAMES[t] for t in active_types]})")
        elif args.phase == '3b':
            active_types = PTCL_CONFIG['phase3b_active_types']
            print(f"  [创新] CDCR 跨域一致性正则化启动 "
                  f"(lambda={CDCR_CONFIG['lambda_cdcr']}, T={CDCR_CONFIG['temperature']})")
            print(f"  [创新] PTCL 类型扩展: {active_types} "
                  f"({[TYPE_NAMES[t] for t in active_types]})")
        elif args.phase == '3c':
            active_types = PTCL_CONFIG['phase3c_active_types']
            print(f"  [创新] 全部 5 类碰撞激活: {active_types}")
            print(f"  [创新] 所有创新模块启用: CTTA+UASC+CDCR+PTCL")
        print(f"  lr0={lr0}")
        print(f"  加载权重: {chain_checkpoint}")

        if _has_temporal:
            print("  OK 时序因果推理模式已启用 (V18TemporalModel)")
        else:
            print("  WARN temporal_context.py 不可用, 回退到单帧模式")

        # V22 增强 Transformer 状态
        if _has_temporal_model and _has_phase3_train:
            print("  OK V22 增强 Transformer + Phase 3 循环已加载")
            print("  OK 创新模块将在训练中生效: CTTA + UASC + CDCR + PTCL")
            print_v24_transformer_info()
        elif _has_temporal_model:
            print("  WARN phase3_train 未加载, V22 模块可用但回退到 V21 训练循环")
        else:
            print("  WARN temporal_model 未加载, 使用 V21 标准 Transformer")

        overrides = dict(base_overrides)
        overrides.update(
            model=chain_checkpoint,
            name=args.name + phase_suffix,
            lr0=lr0,
            warmup_epochs=2.0,
            mosaic=0.0,
            degrees=2.0,
            scale=0.3,
        )

    if args.resume:
        if not (args.phase in ('3a', '3b', '3c') and _has_temporal):
            overrides['resume'] = True if args.resume == 'auto' else args.resume
        overrides['exist_ok'] = True
        # V22-fix(resume-致命): ultralytics 8.4 setup_model 会用 args.pretrained (str 时)
        # 覆盖 resume 已加载的检查点权重 (日志 "Transferred 150/791") → 模型被重置为
        # COCO 初始权重, 检测指标瞬间归零、损失翻倍。续训必须禁用 pretrained。
        overrides['pretrained'] = False

    # 验收标准提醒
    if args.phase == '3a':
        print("  验收目标: 成因 Recall ≥ 65% (ACCIDENT 无 cause, 此阶段可选)")
    elif args.phase == '3b':
        print("  验收目标: 成因 Recall ≥ 80% (ACCIDENT 无 cause, 此阶段可选)")
    elif args.phase == '3c':
        print("  验收目标: pre_crash Recall ≥ 60%, FPR < 15%")

    print(f"\n开始 V22 训练...")

    # 传递 V22 环境变量 (DDP 子进程需要)
    os.environ['V21_PHASE'] = args.phase
    os.environ['V21_DATASET_NAME'] = args.dataset
    os.environ['V21_DATA_ROOT'] = str(data_root)
    os.environ['V21_SAMPLER_RATIOS'] = ('' if args.no_balanced_sampler
        else ','.join(str(x) for x in sampler_ratios))
    os.environ['V21_CLASS_WEIGHTS'] = ','.join(str(x) for x in class_weights)
    os.environ['V24_CAUSE_POS_WEIGHT'] = ','.join(str(x) for x in dynamic_cause_weights)
    os.environ['V22_EXPERIMENT'] = 'true'
    # V25: save_dir for adaptive weight sidecar (persists across resume)
    # D15修复: 去掉 detect/runs 双前缀 (真实 run 目录= project/<name>_p<phase>[/自动-N])
    _run_name = args.name + f'_p{args.phase}'
    os.environ['V25_SAVE_DIR'] = os.path.join(args.project, _run_name)

    # DDP 修复: 增加 NCCL 超时 (默认30秒太短, 大 batch 训练时 rank 间可能不同步)
    os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')  # 禁用 InfiniBand, 使用 socket 通信更稳定

    # V22 训练监控: Phase 1/2 使用 ultralytics 内置验证
    # Phase 3 使用 V22 专用监控器 (在 phase3_train.py 中集成)
    if args.phase in ('1', '2'):
        from monitor import V22_MONITOR_CONFIG, get_system_resources, format_resources
        phase_key = f'phase{args.phase}'
        phase_thresholds = V22_MONITOR_CONFIG.get(phase_key, {})
        if phase_thresholds:
            print(f"  V22 监控: Phase {args.phase} 验收标准: {phase_thresholds}")
        print(f"  V22 资源: {format_resources(get_system_resources())}")

    # 训练执行
    if args.phase in ('3a', '3b', '3c') and _has_temporal:
        trainer = V18Trainer(
            overrides=overrides,
            sampler_ratios=None if args.no_balanced_sampler else sampler_ratios,
            data_root=data_root,
            dataset_name=args.dataset,
        )
        trainer._v24_phase = args.phase
        if args.clip_len is not None:
            effective_clip_len = args.clip_len
        else:
            ds_clip_len = ds_config.get('clip_len', 30)
            # C7修复(Critical): 使用配置值而非硬编码截断为 8
            effective_clip_len = ds_clip_len
            if ds_clip_len > 16:
                print(f"  V22 警告: clip_len={ds_clip_len} 较长, 如 GPU 显存不足请用 --clip-len 16")
        print(f"  V22 clip_len: {effective_clip_len}")

        # V22: 优先使用 V22 专用 Phase 3 循环 (CTTA+UASC+CDCR+PTCL 真正生效)
        if _has_phase3_train and _has_temporal_model:
            print("  V22: 使用增强 Phase 3 训练循环 (CTTA + UASC + CDCR + PTCL)")
            # W1修复(Warning): 将 V22_INNOVATION_LOSS_WEIGHTS 传入训练循环
            args._innov_weights = innov_weights
            # G-303(A4): 相位3 的 batch/workers 在这里**硬编码**, 与 args.batch/args.workers
            # (P1/P2 的 48/16) 无关 —— 相位3 的 batch 单位是 clip: 4 clip × 16 帧 = 64 帧/step,
            # 再乘 accum=4 ⇒ 256 帧/update。直接套 args 会同时换掉显存与梯度量级 (配方级,
            # 不在本次开机必修内)。历史三场巡检把 "batch 48" 报错, 是因为日志只印
            # effective_batch=16 而没印真值。下面只把真值落到开机面, 数值一字未改。
            _p3_batch, _p3_workers = 4, 4
            print(f"  V22 相位3 装载参数 (train.py 硬编码, args 不生效): batch={_p3_batch} clip "
                  f"= {_p3_batch}×{effective_clip_len} 帧 = {_p3_batch*effective_clip_len} 帧/step, "
                  f"accum=4 ⇒ {_p3_batch*4} clip/update | num_workers={_p3_workers} "
                  f"(args.batch={getattr(args, 'batch', None)}, "
                  f"args.workers={getattr(args, 'workers', None)})")
            run_phase3_v24_temporal(
                trainer=trainer,
                args=args,
                clip_len=effective_clip_len,
                batch_size=_p3_batch,
                num_workers=_p3_workers,
                class_weights=class_weights,
            )
        else:
            # V25-fix(type-leak): V21 回退循环 use_tracklet=True, 会把 GT
            # boxes/classes 经 RoIAlign 融进预测特征 (确认级泄漏), 禁止静默回退
            raise RuntimeError(
                "V22 Phase 3 训练循环 (phase3_train) 不可用, 拒绝回退到 "
                "含 GT-tracklet 泄漏的 V21 循环。请修复导入错误后重试。")
    else:
        # Phase 1/2: 使用 ultralytics V18Trainer
        # V22 监控集成: 通过回调机制在 ultralytics 训练后执行检查
        trainer = V18Trainer(
            overrides=overrides,
            sampler_ratios=None if args.no_balanced_sampler else sampler_ratios,
            data_root=data_root,
            dataset_name=args.dataset,
        )
        trainer._v24_phase = args.phase

        # V26-G350: 车道线头 only 微调档（用户明令新增车道线能力，CULane 补训走这里）。
        # lane loss 只在检测侧 criterion（Phase 1/2），Phase 3 检测支路冻结 ⇒ 车道线
        # 补训必须 Phase-2 regime；只放开 model.23.lane_head ⇒ 时序头特征面逐字节不动。
        if args.phase == '2' and os.environ.get('V26_LANE_HEAD_ONLY') == '1':
            from lane_finetune import install_lane_head_only
            install_lane_head_only(trainer)
            print('  [G-350] V26_LANE_HEAD_ONLY=1 ⇒ 仅 lane_head 可训（护栏见 lane_finetune.py）')

        # V22-fix(resume-致命v2): ultralytics check_resume 用 ckpt 内保存的 args 整体
        # 替换 trainer.args, 且只重放白名单键 (imgsz/batch/workers/freeze...),
        # pretrained 不在白名单 → 上面设置的 pretrained=False 被丢弃, setup_model
        # 重新迁移 yolo26m.pt 覆盖 resume 权重 (日志特征: resume 时仍出现
        # "Transferred 636/689" 且 loss 回跳到初始水平)。构造后强制清零。
        if args.resume:
            trainer.args.pretrained = False

        # V22: 事故焦点可视化 — 验证期强制筛选事故帧, 生成 GT/Pred 对比网格图
        # (GT 类别 + 场景标签双重筛选; 异常静默忽略, 不影响训练流程)
        try:
            from accident_focus_vis import AccidentFocusVisualizer
            AccidentFocusVisualizer(
                target_classes=(),          # ACCIDENT 数据集无事故 bbox 类别
                scene_filter=(2, 3),        # pre_crash / crash
                data_root=data_root,
            ).attach(trainer)
            print('  V22: accident_focus_vis 已挂载 (验证期事故帧 GT/Pred 网格图)')
        except Exception as e:
            print(f'  V22 警告: accident_focus_vis 挂载失败 (不影响训练): {e}')

        if args.phase == '1':
            print('  V22: Phase 1 使用 V21 V18Trainer (创新模块在 Phase 3 时序循环中生效)')
        elif args.phase == '2':
            print('  V22: Phase 2 使用 V21 V18Trainer (创新模块在 Phase 3 时序循环中生效)')

        trainer.train()

        # ── V22 Phase 1/2 训练后验收检查 ──
        try:
            from monitor import V22_MONITOR_CONFIG, get_system_resources
            phase_key = f'phase{args.phase}'
            pc = V22_MONITOR_CONFIG.get(phase_key, {})
            if pc:
                # 读取 ultralytics 训练结果（G-257: 必须解析实际 run 目录——
                # 固定拼 name+_p2 会命中首个 run 的旧 results.csv，假绿）
                results_dir = str(getattr(trainer, 'save_dir', '') or
                                  getattr(getattr(trainer, 'trainer', None),
                                          'save_dir', '') or '')
                if not results_dir or not os.path.isfile(
                        os.path.join(results_dir, 'results.csv')):
                    import glob as _glob
                    _base = os.path.join(args.project, f'{args.name}{phase_suffix}')
                    _cands = [d for d in _glob.glob(_base + '*')
                              if os.path.isfile(os.path.join(d, 'results.csv'))]
                    if _cands:
                        results_dir = max(
                            _cands,
                            key=lambda d: os.path.getmtime(
                                os.path.join(d, 'results.csv')))
                results_csv = os.path.join(results_dir, 'results.csv')
                print(f"  验收读取: {results_csv}")
                if os.path.exists(results_csv):
                    import csv
                    with open(results_csv) as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        last = rows[-1]
                        final_loss = (float(last.get('train/box_loss', 0)) +
                                      float(last.get('train/cls_loss', 0)))
                        final_map50 = float(last.get('metrics/mAP50(B)', 0))
                        final_prec = float(last.get('metrics/precision(B)', 0))
                        final_rec = float(last.get('metrics/recall(B)', 0))

                        print(f"\n  ── Phase {args.phase} 最终指标 ──")
                        print(f"  mAP50={final_map50:.4f} Precision={final_prec:.4f} "
                              f"Recall={final_rec:.4f} Loss={final_loss:.4f}")

                        # 验收标准检查
                        failures = []
                        if 'val_loss_max' in pc and final_loss > pc['val_loss_max']:
                            failures.append(
                                f"检测损失 {final_loss:.3f} > {pc['val_loss_max']}")
                        if final_map50 > 0 and final_map50 < 0.3:
                            failures.append(
                                f"mAP50 {final_map50:.1%} < 30% (Phase {args.phase} 基准)")

                        if failures:
                            print(f"  ⚠️ Phase {args.phase} 验收未达标:")
                            for fail in failures:
                                print(f"    - {fail}")
                            print(f"  建议: 调整学习率/数据增强后继续训练")
                        else:
                            print(f"  ✅ Phase {args.phase} 验收通过")

                        # 目标达标检查 (mAP50>50%, P>60%, R>60%)
                        if args.phase == '2':
                            print(f"\n  ── 最终目标检查 ──")
                            print(f"  mAP50 > 50%:   {'PASS' if final_map50 > 0.50 else 'NEED MORE TRAINING'} ({final_map50:.2%})")
                            print(f"  Precision > 60%: {'PASS' if final_prec > 0.60 else 'NEED MORE TRAINING'} ({final_prec:.2%})")
                            print(f"  Recall > 60%:    {'PASS' if final_rec > 0.60 else 'NEED MORE TRAINING'} ({final_rec:.2%})")
        except Exception as _e:
            print(f"  V22 监控: Phase {args.phase} 验收检查失败 {_e}")

        # 打印最终资源
        try:
            from monitor import get_system_resources, format_resources
            print(f"  最终资源: {format_resources(get_system_resources())}")
        except Exception:
            pass

    print(f"\nV22 训练完成! 输出目录: {args.project}/{args.name}{phase_suffix}")


if __name__ == '__main__':
    main()
