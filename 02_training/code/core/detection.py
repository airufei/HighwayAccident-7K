# -*- coding: utf-8 -*-
"""
detection — V21 模型 / 验证器 / 训练器 / 回调 (从 train.py 拆分)

包含:
  - DetectClassifyModelV18   YOLO26 + DetectClassify 头 (6元组输出)
  - V18Validator             精简验证器 (记录 trainer 反向引用)
  - V18Trainer               多数据集 + 四任务 + Phase 3a/3b/3c 训练器
  - 回调: _update_sampler_epoch / _v24_dynamic_box_weight /
          _v24_scene_aware_augmentation / _v24_log_scene_dist /
          _v18_eval_scene_classification
"""

import os
import re
import sys
import json
from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

# V25: PF-RPN
from core.pfrpn import PFRPN, PFRPNLoss

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import RANK, DEFAULT_CFG, colorstr, LOGGER
try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    from ultralytics.utils.torch_utils import unwrap_model as de_parallel
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data.build import build_dataloader, InfiniteDataLoader
from ultralytics.utils.plotting import plot_images, plot_results

import base_config
from base_config import (
    dist, _has_dist, DATASET_CONFIGS, DEFAULT_SCENE_CLS_WEIGHT,
    adaptive_weight_update, get_phase3_loss_weights,
    CAUSE_NAMES, TYPE_NAMES, LANE_CHANNELS,
)
from losses import v18MultiTaskLoss, _d13_epoch_mark, _d13_epoch_check, lane_target_valid
from dataset import SceneClassBalancedSampler, MultiDatasetAccident


def _env_floats(name):
    # R1-03: parse comma-separated floats from env; None if unset/invalid
    raw = os.environ.get(name, '')
    if not raw:
        return None
    try:
        return [float(x) for x in raw.split(',')]
    except ValueError:
        return None

try:
    from detect_classify_head import DetectClassify, get_groupnorm, HAS_3D
except ImportError:
    print("ERROR: 未找到 detect_classify_head.py 模块")
    sys.exit(1)

# V25: 对比学习文本编码器 (可选)
try:
    from contrastive import SimpleTextEncoder, TextProjectionHead, SemanticTokenizer
    HAS_CONTRASTIVE = True
except ImportError:
    HAS_CONTRASTIVE = False


# ============================================================
# 8. 自定义模型 (YOLO26 + DetectClassify V18)
# ============================================================

class DetectClassifyModelV18(DetectionModel):
    """YOLO26 + DetectClassify V18 (6元组输出)"""

    def __init__(self, cfg="yolo26m_detect_cls.yaml", ch=3, nc=None, verbose=True):
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        # V20-fix: 使用 deepcopy 避免 pop 操作污染原始 yaml (多次构建模型时会丢失 nc_cls 等 key)
        yaml_work = deepcopy(self.yaml) if isinstance(self.yaml, dict) else self.yaml
        nc_cls = yaml_work.pop("nc_cls", 3)
        num_causes = yaml_work.pop("num_causes", 13)
        num_types = yaml_work.pop("num_types", 7)
        has_3d = yaml_work.pop("has_3d", False)  # V20-fix(C7): 提取 has_3d, 否则 3D 头永远不会被创建
        has_contrastive = yaml_work.pop("has_contrastive", False)  # V25: 对比学习

        # 为 ultralytics 构造临时 yaml (Detect 部分)
        yaml_for_build = deepcopy(yaml_work)
        for section in ("backbone", "head"):
            if section in yaml_for_build:
                for layer in yaml_for_build[section]:
                    if layer[2] == "DetectClassify":
                        layer[2] = "Detect"
                        if len(layer[3]) > 1:
                            layer[3] = layer[3][:1]
        # NOTE: super().__init__ → _initialize_yolo_model 会用 yaml_for_build 覆盖 self.yaml,
        # 丢失自定义 key (num_causes, nc_cls 等)。先保存完整 yaml, super 之后恢复。
        _full_yaml = self.yaml if isinstance(self.yaml, dict) else None
        super().__init__(yaml_for_build, ch=ch, nc=nc, verbose=verbose)
        if _full_yaml is not None:
            self.yaml = _full_yaml

        # 替换最后一个 Detect 为 DetectClassify
        detect_idx = None
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], Detect):
                detect_idx = i
                break
        if detect_idx is not None:
            old_detect = self.model[detect_idx]
            ch_list = [seq[0].conv.in_channels for seq in old_detect.cv2]
            new_head = DetectClassify(
                nc=old_detect.nc, nc_cls=nc_cls, ch=ch_list,
                num_causes=num_causes, num_types=num_types,
                has_3d=has_3d,  # V20-fix(C7): 传递 has_3d, 启用 3D 几何回归头
                has_contrastive=has_contrastive,  # V25: 对比学习
            )
            new_head.cv2 = old_detect.cv2
            new_head.cv3 = old_detect.cv3
            new_head.dfl = old_detect.dfl
            new_head.stride = old_detect.stride
            for attr in ('no', 'nl', 'reg_max', 'nc', 'type',
                         'end2end', 'format', 'max_det', 'deploy',
                         'rotary', 'export', 'iou', 'i', 'f', 'np', 'save'):
                if hasattr(old_detect, attr):
                    setattr(new_head, attr, getattr(old_detect, attr))
            new_head.inplace = self.yaml.get("inplace", True)
            self.model[detect_idx] = new_head
            self.stride = new_head.stride
        self.nc_cls = nc_cls
        self.has_3d = has_3d and HAS_3D  # V20-fix(C7): 模型级属性, 供 Phase 3 检测是否启用 3D
        self.has_contrastive = has_contrastive  # V25: 对比学习

        # V25: PF-RPN — 与 DetectClassify 并行的无类别候选区域生成器
        self.pfrpn_enabled = True
        if detect_idx is not None:
            self._detect_idx = detect_idx
            self.pfrpn = PFRPN(ch_list=ch_list, hidden_dim=256, strides=[8, 16, 32])
        else:
            self._detect_idx = None
            self.pfrpn = None

        # V25: 对比学习文本编码器 (端到端训练)
        self.text_encoder = None
        self.text_proj_head = None
        if has_contrastive and HAS_CONTRASTIVE:
            self.text_encoder = SimpleTextEncoder(
                vocab_size=4096, embed_dim=256, hidden_dim=256, num_layers=2
            )
            self.text_proj_head = TextProjectionHead(
                text_dim=512, visual_dim=1024, proj_dim=256
            )

    def _extract_fpn_features(self, x):
        """提取 FPN 输出 (P3/P4/P5), 在 DetectClassify 前截取."""
        model = self.model
        y = [None] * len(model)
        for i, m in enumerate(model):
            if i == getattr(self, '_detect_idx', -1):
                break  # 在 DetectClassify 之前停止
            if hasattr(m, 'f') and isinstance(m.f, int):
                fi = m.f
                # V25-fix: fi==-1 表示取前一层输出 (YOLO 惯例), 非原始输入
                if fi == -1:
                    y[i] = m(x if i == 0 else y[i - 1])
                else:
                    y[i] = m(y[max(fi, 0)])
            elif hasattr(m, 'f') and isinstance(m.f, (list, tuple)):
                inputs = []
                for fi in m.f:
                    if fi == -1:
                        inputs.append(x if i == 0 else y[i - 1])
                    elif 0 <= fi < len(y) and y[fi] is not None:
                        inputs.append(y[fi])
                    else:
                        inputs.append(x if i == 0 else y[i - 1])
                y[i] = m(inputs)
            else:
                y[i] = m(x if i == 0 else y[i - 1])

        detect_layer = model[getattr(self, '_detect_idx', len(model) - 1)]
        if hasattr(detect_layer, 'f') and isinstance(detect_layer.f, (list, tuple)):
            feats = []
            for fi in detect_layer.f:
                if 0 <= fi < len(y) and y[fi] is not None:
                    feats.append(y[fi])
            if len(feats) == 3:
                return feats

        # fallback: 运行完整 forward 并用 hook 捕获
        return None

    def init_criterion(self):
        w = getattr(self, 'scene_cls_weight', DEFAULT_SCENE_CLS_WEIGHT)
        # R1-03: apply V21 env state here (runs in DDP children too);
        # old post-hoc injection in train.py was a lazy-criterion no-op.
        cause_pw = _env_floats('V24_CAUSE_POS_WEIGHT')
        if cause_pw and len(cause_pw) == 9:
            base_config.CAUSE_POS_WEIGHT = cause_pw
        criterion = v18MultiTaskLoss(self, cls_weight=w)
        class_weights = _env_floats('V21_CLASS_WEIGHTS')
        if class_weights and len(class_weights) == 3:
            criterion.update_class_weights(class_weights)
        # V25 FIX: load adaptive weights from sidecar file on resume —
        # ultralytics strips criterion from checkpoints, so env var has defaults.
        # D15修复: 优先 trainer 真值 save_dir (env 旧拼法 detect/runs 双前缀且不含自动 -N 后缀)
        _save_dir = str(getattr(self, 'save_dir', '') or '') or os.environ.get('V25_SAVE_DIR', '')
        if _save_dir:
            try:
                import json as _json
                _sw_path = os.path.join(_save_dir, 'adaptive_class_weights.json')
                if os.path.isfile(_sw_path):
                    with open(_sw_path) as f:
                        _sw = _json.load(f)
                    _aw = _sw.get('class_weights')
                    if _aw and len(_aw) == 3:
                        criterion.update_class_weights(_aw)
                        if RANK in (-1, 0):
                            LOGGER.info(f'[V25] Loaded adaptive class weights from sidecar: {_aw}')
            except Exception as _e:
                LOGGER.warning(f'[V25] Failed to load adaptive weights sidecar: {_e}')
        phase = os.environ.get('V21_PHASE', '')
        # V24: VLM 重标注后 ACCIDENT 已有 cause 标签 (scene_labels 第2行 16-bit),
        # 成因监督从 Phase 1 即启用 (早期单帧判别), Phase 3 通过时序上下文深度判定。
        # 原 V22 H2 修复 (ACCIDENT 无 cause → 全阶段关闭 cause loss) 已过时删除。
        if phase == '1':
            # Phase 1: 检测预热 + scene/cause 单帧监督; 3D/type 头未训练
            criterion.update_loss_weights(tta=0.0, cause=1.0, type_w=0.0)
            criterion.lambda_3d = 0.0
            if RANK in (-1, 0):
                LOGGER.info('[V24] Phase 1: cause loss enabled (VLM标注), 3D/type losses disabled')
        elif phase == '2':
            # Phase 2: 融合阶段; 3D 头仍未训练→保持关闭; type 头给 0.15 小权重预热。
            # 为什么预热: 类型头若一路随机到 Phase 3a, 只有 15 轮窗口要从零学会
            # 7 类判别, 稀有类 (rollover/pedestrian_hit) 极易欠拟合。0.15 ≈ 3a(1.2)
            # 的 1/8, 只把头部从随机初始化拉到可判别区间, 不挤占 scene/cause 梯度。
            # 噪声安全: AccidentTypeLoss 内部掩码到 scene==3 且过滤 -1/越界 (losses.py:328-343),
            # 故 S05 的"非 crash 帧也带类型"面不会进入损失。
            # G-323(#12/E3): 论文撰写方案 §6-E3 要 λ_type∈{0,0.15} 两臂消融 ⇒ 这里开一条
            #   **只在 V26_E3_TYPE_PREWARM 被显式设置时**才改写的通道; 不设时行为与历史逐字相同。
            _e3_tw_raw = str(os.environ.get('V26_E3_TYPE_PREWARM', '') or '').strip()
            _e3_tw = 0.15
            if _e3_tw_raw:
                try:
                    _e3_tw = float(_e3_tw_raw)
                except ValueError:
                    raise SystemExit(f'[G-323 E3] V26_E3_TYPE_PREWARM={_e3_tw_raw!r} 不是数 ⇒ 中止'
                                     f'(宁可不开跑, 不许静默回退到 0.15 冒充"已消融")')
                if not (0.0 <= _e3_tw <= 1.0):
                    raise SystemExit(f'[G-323 E3] V26_E3_TYPE_PREWARM={_e3_tw} 越界 [0,1] ⇒ 中止')
            criterion.update_loss_weights(tta=0.0, cause=1.0, type_w=_e3_tw)
            criterion.lambda_3d = 0.0
            # 后置闸: 权重必须真落到 criterion 上 (config 有键≠有接线; 只信 update 的返回值不算证据)
            if abs(float(getattr(criterion, 'type_weight', -1.0)) - _e3_tw) > 1e-9:
                raise SystemExit(f'[G-323 E3] λ_type 过滤器没有牙齿: 声明 {_e3_tw} '
                                 f'实得 {getattr(criterion, "type_weight", None)} ⇒ 中止')
            if _e3_tw_raw and RANK in (-1, 0):
                LOGGER.info(f'[G-323 E3] λ_type 预热通道生效: 声明值={_e3_tw_raw} 落到 criterion='
                            f'{criterion.type_weight:.6g} (在册默认 0.15 已被显式覆盖; '
                            f'本臂与默认臂的唯一差别就是这一项)')
            if RANK in (-1, 0):
                # G-323: 这行原来把 0.15 写死在文案里 ⇒ 通道覆盖后横幅会谎报在册值(谎值形态)。
                #   改读 criterion 实际权重, 默认路径的字节输出与历史逐字相同 (0.15 → "λ=0.15")。
                LOGGER.info(f'[V24] Phase 2: cause λ=1.0, type λ={criterion.type_weight:g} '
                            f'({"预热" if criterion.type_weight > 0 else "未预热/λ_type=0 臂"}), 3D loss disabled, '
                            f'lane λ={criterion.lane_weight:.2f} (V26-G247)')
        elif phase in ('3a', '3b', '3c'):
            criterion.update_loss_weights(**get_phase3_loss_weights(phase))
        return criterion

    def forward(self, x, *args, **kwargs):
        """V25: 训练时同时运行 DetectClassify 和 PFRPN."""
        # ultralytics trainer 调用 model(batch) 传入 dict; 需走 loss 路径
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)

        preds = self.predict(x, *args, **kwargs)

        # PFRPN 分支 (仅训练且启用时)
        # V25 FIX: detach FPN features to prevent double backward through backbone
        if (self.training and getattr(self, 'pfrpn_enabled', False)
                and self.pfrpn is not None):
            fpn_feats = self._extract_fpn_features(x)
            if fpn_feats is not None:
                fpn_feats = [f.detach() for f in fpn_feats]
                pfrpn_out = self.pfrpn(fpn_feats)
                return (preds, pfrpn_out)

        return preds

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])

        # V25: 拆包 PFRPN 输出
        pfrpn_preds = None
        if isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[1], dict):
            preds, pfrpn_preds = preds

        if isinstance(preds, (tuple, list)):
            # 将 PFRPN 预测追加到 preds 末尾 (criterion 从末尾搜索 dict)
            if pfrpn_preds is not None:
                preds = list(preds) + [pfrpn_preds]
            total, items = self.criterion(preds, batch)
            # Ultralytics ≥8.4 expects loss_items as a dict, not a tensor.
            if isinstance(items, torch.Tensor) and not isinstance(items, dict):
                names = getattr(self, 'loss_names',
                                ('box_loss', 'cls_loss', 'dfl_loss',
                                 'scene_loss', 'cause_loss', 'type_loss', 'lane_loss',
                                 '3d_loss',
                                 'pfrpn_obj', 'pfrpn_box', 'pfrpn_cen',
                                 'contrastive_loss',
                                 'supcon_loss', 'consist_loss'))
                items = dict(zip(names, items.detach()))
            return total, items
        device = preds.device
        return torch.zeros(1, device=device), torch.zeros(7, device=device)


# ============================================================
# 9. 精简验证器
# ============================================================

class V18Validator(DetectionValidator):

    def __call__(self, trainer=None, model=None):
        # 修复 val scene eval: on_val_end 回调入参是 validator (ultralytics 在
        # validator.run_callbacks("on_val_end") 中传 self), 而 validator.__call__ 只把
        # trainer 当局部变量, 不存反向引用 → 回调无法取到 trainer.model/test_loader。
        # 此处记录 trainer 引用, 供 _v18_eval_scene_classification 恢复真实 trainer。
        if trainer is not None:
            self._trainer_ref = trainer
        self._scene_fitness_val = None
        res = super().__call__(trainer, model)
        # ### #252接线真修: V22 的 fitness 覆盖只写到 trainer.fitness/metrics,
        # 但 trainer.validate() 的选择值取自返回 dict pop('fitness'), 而该 dict
        # 在 on_val_end 之前已快照 ⇒ 覆盖从未到达 best.pt 选择通道 (best.pt 恒=最新轮)。
        # 覆盖必须落在返回 dict 的 'fitness' 键上, 才会经 trainer.py:873 进 self.fitness。
        if isinstance(res, dict) and self._scene_fitness_val is not None:
            res['fitness'] = self._scene_fitness_val
        return res

    def postprocess(self, preds):
        if isinstance(preds, (tuple, list)) and len(preds) >= 1:
            preds = preds[0]
        return super().postprocess(preds)


# ============================================================
# 10. V18 训练器
# ============================================================

def _v26_restore_optim_state(optimizer, ckpt_path, scaler=None):
    """G-260 根治件: 链式续接种子内的优化器动量/AMP 状态。

    重启瞬态根因: 链式加载只读权重, 种子里的动量缓冲全被丢弃, LR 又从
    warmup 爬回峰值, 头 1-2 轮把已收敛判界打散 4-9pp (p2-14 e3 71.9 →
    p2-15 e1 67.3 → e2 62.8)。本函数把种子内 optimizer/scaler 状态读回
    新优化器 (同一模型架构下参数分组天然一致)。
    任何失败只告警并保留全新优化器, 不阻塞训练。
    返回 True=已续接 / False=回退全新。
    """
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f'[G-260] 种子读取失败, 优化器保持全新: {e}')
        return False
    opt_state = ckpt.get('optimizer') if isinstance(ckpt, dict) else None
    if not opt_state:
        print('[G-260] 种子内无 optimizer 状态, 优化器保持全新')
        return False
    try:
        # G-262: load_state_dict 整组覆写 param_groups, 连 lr/initial_lr 一起被种子值
        # 抢占。ultralytics 8.4.112 每步 lr = initial_lr * lf(epoch) (trainer.py:480),
        # 种子里 initial_lr=0.0005 是 V22_PHASE_LR['2'] 的遗留基准 ⇒ 实跑 LR 变成
        # --lr0 (2e-4) 计划值的 2.5 倍 (p2-16 results.csv lr/pg0 峰值 4.96e-4 实证),
        # 高 LR 微调已收敛种子把 crash/pre 判界当噪声慢慢磨掉。动量缓冲照旧续接
        # (G-260 本意), 但 LR 基准必须回到本次构建值。
        _fresh = [(g.get('lr'), g.get('initial_lr')) for g in optimizer.param_groups]
        optimizer.load_state_dict(opt_state)
        _seed_lr = [g.get('initial_lr') for g in optimizer.param_groups]
        for _g, (_lr, _init) in zip(optimizer.param_groups, _fresh):
            _g['lr'] = _lr
            _g['initial_lr'] = _init if _init is not None else _lr
        print(f'[G-262] LR 基准归位: 种子 initial_lr={_seed_lr[0]} → {_fresh[0][0]} (动量缓冲仍续接)')
    except Exception as e:
        print(f'[G-260] 优化器状态不匹配, 保持全新: {e}')
        return False
    n_buf = len(getattr(optimizer, 'state', {}))
    if scaler is not None and isinstance(ckpt.get('scaler'), dict):
        try:
            scaler.load_state_dict(ckpt['scaler'])
        except Exception:
            pass  # scaler 状态读不回只影响 AMP 缩放节奏, 不影响正确性
    print(f'[G-260] 链式续接优化器状态: {n_buf} 组动量缓冲已读回, 源={ckpt_path}')
    return True


class V18Trainer(DetectionTrainer):
    """V18 训练器: 多数据集 + 四任务 + Phase 3a/3b/3c"""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None,
                 sampler_ratios=None, data_root=None, dataset_name=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect"
        # R1-03: DDP children rebuild Trainer from overrides only; recover
        # custom ctor args from V21_* env vars set by the main process.
        if sampler_ratios is None:
            sampler_ratios = _env_floats('V21_SAMPLER_RATIOS')
        if data_root is None:
            data_root = os.environ.get('V21_DATA_ROOT') or None
        if dataset_name is None:
            dataset_name = os.environ.get('V21_DATASET_NAME') or None
        self._sampler_ratios = sampler_ratios  # None 表示禁用均衡采样
        self._data_root = data_root
        self._dataset_name = dataset_name
        self._v24_phase = os.environ.get('V21_PHASE') or None  # R1-03: DDP env preset

        # 数据集配置
        self._ds_config = DATASET_CONFIGS.get(dataset_name, {}) if dataset_name else {}

        super().__init__(cfg, overrides, _callbacks)
        # ### V21 MODIFIED R7-07(严重, C损失层/D3): 验证期间置位 IN_VALIDATION,
        # 使 losses 的动态 CLASS_WEIGHTS all_reduce 不在仅-rank0 验证时触发 (防 NCCL 死锁)。
        self.add_callback("on_val_start", _v24_set_in_validation)
        self.add_callback("on_train_epoch_end", _v24_log_scene_dist)
        self.add_callback("on_train_epoch_end", _d13_zero_loss_epoch_check)  # D13增强: 跨epoch零损失检测
        self.add_callback("on_train_epoch_start", _update_sampler_epoch)
        self.add_callback("on_train_epoch_start", _v24_dynamic_box_weight)  # V21: 动态定位损失权重
        self.add_callback("on_train_epoch_start", _v24_scene_aware_augmentation)  # V21: 场景感知增强+困难样本挖掘
        self.add_callback("on_val_end", _v18_eval_scene_classification)
        # V21: 事故可视化自动采集 (论文展示) — 按 accident_detect.yaml accident_viz 配置节
        try:
            from accident_viz import collect_accident_visuals
            self.add_callback("on_val_end", collect_accident_visuals)
        except ImportError:
            LOGGER.warning('[V21] accident_viz.py 不可用, 跳过事故可视化采集回调')
        # 清位回调必须最后注册: 确保上面的 on_val_end 回调 (可能调用 criterion) 均处于验证态内
        self.add_callback("on_val_end", _v24_clear_in_validation)
        self._p3_epoch_counter = 0
        # V25: PF-RPN epoch tracking (控制 warmup)
        self.add_callback("on_train_epoch_start", _v25_pfrpn_epoch_update)

    def build_optimizer(self, model, *args, **kwargs):
        # G-260 根治件: 链式续接种子内优化器动量/AMP 状态 (V26_CHAIN_OPTIM_STATE
        # 由 run_train.py 从 --chain-from 透传)。防重启瞬态: 动量清零+LR 回峰
        # 两因叠加, 头 1-2 轮打散已收敛判界 4-9pp。失败自动回退全新优化器。
        optimizer = super().build_optimizer(model, *args, **kwargs)
        _chain = os.environ.get('V26_CHAIN_OPTIM_STATE', '')
        if _chain:
            _v26_restore_optim_state(optimizer, _chain, getattr(self, 'scaler', None))
        return optimizer

    def build_dataset(self, img_path, mode="train", batch=None, augment=None, fraction=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        data_root = self._data_root or self.data.get("path", "")

        # 场景/成因/类型标签目录
        scene_dir = os.path.join(data_root, self.data.get("scene_labels_dir", "scene_labels"), mode)
        cause_dir = os.path.join(data_root, self.data.get("cause_labels_dir", "cause_labels"), mode)
        type_dir = os.path.join(data_root, self.data.get("type_labels_dir", "type_labels"), mode)
        # V26-G247: 车道线折线标注目录 (lane_labels/{stem}.json)
        lane_dir = os.path.join(data_root, self.data.get("lane_labels_dir", "lane_labels"), mode)

        dataset_id = self._ds_config.get('scene', 'unknown') if self._dataset_name else 'unknown'

        # V25: 语义对比学习 — 加载语义标注文件和分词器
        semantic_file = None
        tokenizer = None
        if self.data.get('has_contrastive', False) and HAS_CONTRASTIVE:
            sem_cfg = self.data.get('semantic_labels_dir', '')
            if sem_cfg:
                # 配置是模板: meta/v25_semantic_clips.jsonl → meta/v25_semantic_clips_train.jsonl
                base, ext = os.path.splitext(sem_cfg)
                semantic_file = os.path.join(data_root, f"{base}_{mode}{ext}")
                if not os.path.exists(semantic_file):
                    semantic_file = None  # 文件不存在时静默跳过

            # 分词器: 优先使用 V25 均衡后的词表, 回退到旧版
            tokenizer_path = os.path.join(data_root, 'meta', 'v25_vlm_tokenizer.json')
            if not os.path.exists(tokenizer_path):
                tokenizer_path = os.path.join(data_root, 'meta', 'semantic_tokenizer.json')
            tokenizer = SemanticTokenizer(vocab_size=4096, max_len=128)
            if os.path.exists(tokenizer_path):
                tokenizer.load(tokenizer_path)
            elif semantic_file and os.path.exists(semantic_file):
                # 从标注文件构建词表
                import json as _json
                texts = []
                with open(semantic_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = _json.loads(line)
                        desc = r.get('vlm', {}).get('description', '')
                        if desc:
                            texts.append(desc)
                if texts:
                    tokenizer.build_vocab(texts)
                    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
                    tokenizer.save(tokenizer_path)

        # V25: 加载帧级采样权重 (accident_type × risk_level 均衡)
        frame_weights = None
        weights_file = os.path.join(data_root, 'meta', f'v25_vlm_frame_weights_{mode}.jsonl')
        if os.path.exists(weights_file):
            import json as _json
            frame_weights = {}
            with open(weights_file) as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    r = _json.loads(line)
                    frame_weights[i] = r.get('weight', 1.0)
            LOGGER.info(f'[V25] 加载帧级采样权重: {weights_file} ({len(frame_weights)} 帧)')
        self._frame_weights = frame_weights

        # V21-fix: augment 可显式覆盖 (Phase 3 需要 mode='train' 但 augment=False)
        use_augment = augment if augment is not None else (mode == "train")
        # V21-fix: fraction 可显式覆盖 (Phase 3 Clip 构建始终使用全量数据)
        use_fraction = fraction if fraction is not None else (self.args.fraction if mode == "train" else 1.0)

        return MultiDatasetAccident(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=use_augment, hyp=self.args, rect=(mode == "val"),
            cache=self.args.cache or False, single_cls=self.args.single_cls,
            stride=gs, pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "), task="detect", data=self.data,
            classes=self.args.classes,
            fraction=use_fraction,
            scene_labels_dir=scene_dir,
            cause_labels_dir=cause_dir,
            type_labels_dir=type_dir,
            lane_labels_dir=lane_dir,
            dataset_id=dataset_id,
            semantic_labels_file=semantic_file,
            tokenizer=tokenizer,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not '{mode}'."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        # V21: workers 上限 = min(16, CPU核心数-2), 减少I/O瓶颈
        world_size = dist.get_world_size() if _has_dist and dist.is_initialized() else 1
        workers = min(workers, max(1, (os.cpu_count() or 4) // max(world_size, 1)))

        if mode == "train" and shuffle and self._sampler_ratios is not None:
            batch_size = min(batch_size, len(dataset))
            sampler = SceneClassBalancedSampler(
                dataset, batch_size, target_ratios=self._sampler_ratios,
                seed=self.args.seed or 0,
                frame_weights=getattr(self, '_frame_weights', None)
            )
            sampler.set_epoch(self.start_epoch or 0)
            # ### V21 MODIFIED R5-01(严重, C采样层): 保存 sampler 引用 —
            # _v24_scene_aware_augmentation 通过 trainer._v18_sampler 调整困难类
            # 采样权重, 原代码从未赋值该属性 → 整段困难样本提升逻辑是死代码
            self._v18_sampler = sampler
            world_size = dist.get_world_size() if _has_dist and dist.is_initialized() else 1
            nw = min((os.cpu_count() or 4) // max(world_size, 1), workers)
            generator = torch.Generator()
            generator.manual_seed(6148914691236517205 + (RANK if RANK != -1 else 0))
            return InfiniteDataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=False,
                num_workers=nw, sampler=sampler,
                pin_memory=True, collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=None, generator=generator,
                persistent_workers=(nw > 0),  # V21: 避免每 epoch 重新初始化 worker
            )
        # R1-22: 回退路径 — 仅在 val 模式或禁用均衡采样时到达;
        # DDP 分片由 build_dataloader 内部的 DistributedSampler 处理
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # V24-fix(resume): checkpoint 模型的 yaml 被 DetectionModel.__init__ 剥离了
        # num_causes 等自定义 key → 从 data yaml 补回, 避免 default=13 导致 size mismatch
        if isinstance(cfg, dict):
            cfg = dict(cfg)  # shallow copy to avoid mutating the original
            for key in ("num_causes", "num_types", "nc_cls", "has_3d", "has_contrastive"):
                if key not in cfg and key in self.data:
                    cfg[key] = self.data[key]
        model = DetectClassifyModelV18(
            cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        model.nc_cls = self.data.get("nc_cls", 3)
        model.scene_cls_weight = self.data.get("scene_cls_weight", DEFAULT_SCENE_CLS_WEIGHT)
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        device = self.device

        for key in ["scene_cls", "accident_type"]:
            if key in batch:
                batch[key] = self._to_device_tensor(batch[key], device, torch.long)

        if "cause" in batch:
            cause = batch["cause"]
            if isinstance(cause, (list, tuple)):
                tensors = []
                for ct in cause:
                    if isinstance(ct, torch.Tensor):
                        tensors.append(ct.view(1, -1) if ct.dim() == 1 else ct)
                    else:
                        tensors.append(torch.zeros(1, 9, dtype=torch.float32))
                cause = torch.cat(tensors, dim=0)
            elif isinstance(cause, torch.Tensor):
                if cause.dim() == 1:
                    cause = cause.view(1, -1)
            batch["cause"] = cause.to(device, non_blocking=True) if isinstance(cause, torch.Tensor) else cause

        if "tta" in batch:
            batch["tta"] = self._to_device_tensor(batch["tta"], device, torch.float32)

        # V20-fix(C9): 3D 标签 device 迁移
        if "gt_3d" in batch and isinstance(batch["gt_3d"], torch.Tensor):
            batch["gt_3d"] = batch["gt_3d"].to(device, non_blocking=True)
        if "gt_3d_mask" in batch and isinstance(batch["gt_3d_mask"], torch.Tensor):
            batch["gt_3d_mask"] = batch["gt_3d_mask"].to(device, non_blocking=True)
        if "has_3d" in batch:
            has_3d = batch["has_3d"]
            if isinstance(has_3d, (list, tuple)):
                batch["has_3d"] = torch.tensor(has_3d, dtype=torch.bool, device=device)
            elif isinstance(has_3d, torch.Tensor):
                batch["has_3d"] = has_3d.bool().to(device, non_blocking=True)

        # V26-G247: 车道线掩码 device 迁移 ([B,H,W] uint8, 损失侧解码)
        if "semantic_mask" in batch and isinstance(batch["semantic_mask"], torch.Tensor):
            batch["semantic_mask"] = batch["semantic_mask"].to(device, non_blocking=True)

        return batch

    @staticmethod
    def _to_device_tensor(val, device, dtype):
        if isinstance(val, (list, tuple)):
            tensors = []
            for t in val:
                tensors.append(t.view(-1) if isinstance(t, torch.Tensor)
                               else torch.tensor([t], dtype=torch.long))
            return torch.cat(tensors).to(device, non_blocking=True)
        elif isinstance(val, torch.Tensor):
            return val.view(-1).to(device, non_blocking=True)
        return val

    def get_validator(self):
        base = ("box_loss", "cls_loss", "dfl_loss",
                "scene_loss", "cause_loss", "type_loss", "lane_loss", "3d_loss")
        if getattr(self, 'pfrpn_enabled', False) and self.pfrpn is not None:
            base = base + ("pfrpn_obj", "pfrpn_box", "pfrpn_cen")
        if getattr(self, 'has_contrastive', False):
            base = base + ("contrastive_loss",)
        self.loss_names = base
        return V18Validator(
            self.test_loader, save_dir=self.save_dir,
            args=copy(self.args), _callbacks=self.callbacks,
        )

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch", "GPU_mem", *self.loss_names, "Instances", "Size",
        )

    def plot_training_samples(self, batch, ni):
        try:
            labels = {
                "im_file": batch.get("im_file", []),
                "cls": batch.get("cls", torch.tensor([])),
                "bboxes": batch.get("bboxes", torch.tensor([])),
                "batch_idx": batch.get("batch_idx", torch.tensor([])),
            }
            plot_images(
                labels=labels, images=batch["img"],
                paths=batch.get("im_file"),
                fname=str(self.save_dir / f"train_batch{ni}.jpg"),
                on_plot=self.on_plot,
            )
        except Exception:
            pass

    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)


# ============================================================
# 11. 回调函数
# ============================================================

def _update_sampler_epoch(trainer):
    if hasattr(trainer.train_loader, 'sampler') and hasattr(trainer.train_loader.sampler, 'set_epoch'):
        trainer.train_loader.sampler.set_epoch(trainer.epoch)


def _v25_pfrpn_epoch_update(trainer):
    """V25: 更新 PFRPN epoch (控制 warmup 权重)"""
    os.environ['V25_PFRPN_EPOCH'] = str(trainer.epoch)


def _v24_dynamic_box_weight(trainer):
    """V21: 动态提升定位损失权重 — 从 0.8 逐步升至 2.0

    强化短时碰撞事件敏感度。在训练总 epoch 的前 20% 保持基础值,
    之后线性递增至 2.0。
    """
    try:
        model = de_parallel(trainer.model)
        crit = getattr(model, 'criterion', None)
        if crit is None:
            return
        # V21-fix: box 权重超参在 criterion 自身的 hyp 上 (v8DetectionLoss.__init__
        # 设 self.hyp = model.args), 原代码取 crit.model (模型对象) 是错误引用
        hyp = getattr(crit, 'hyp', None)
        if hyp is None:
            return
        total_epochs = trainer.epochs if hasattr(trainer, 'epochs') else 50
        current_epoch = trainer.epoch if hasattr(trainer, 'epoch') else 0
        progress = min(1.0, current_epoch / max(total_epochs * 0.2, 1))
        # 从 0.8 线性升至 2.0
        dynamic_box_w = 0.8 + progress * 1.2
        # ultralytics 默认 box=7.5, 这里按比例缩放
        # ### V21 MODIFIED R5-04(一般, C损失层): 上限 18.75→12.0 —
        # 原 7.5×2.0/0.8=18.75 使 det 损失膨胀 2.5x, 叠加 λ_scene=0.5 后
        # 场景/成因梯度信号占比被过度稀释 (经验估计: ultralytics 推荐
        # box 区间 5~10, 超 12 后多任务头收敛变慢)
        new_box = min(7.5 * dynamic_box_w / 0.8, 12.0)

        # V22-fix(val-box-jump): 验证走 EMA 模型, 其 criterion.hyp 在首次 val 时
        # 捕获 deepcopy 出来的旧 args (box=7.5), 之后永不更新 → 全新运行 val/box_loss
        # 恒按 7.5 报告, resume 后 EMA 重建却拿到 box=12 → 跳变精确 1.6x
        # (P1 ep36、P2 ep15 两次复现)。此处同步训练侧与 EMA 侧两个 criterion。
        def _set_box(hyp_obj):
            if hyp_obj is None:
                return False
            if isinstance(hyp_obj, dict):
                hyp_obj['box'] = new_box
                return True
            if hasattr(hyp_obj, 'box'):
                hyp_obj.box = new_box
                return True
            return False

        _set_box(hyp)
        ema = getattr(trainer, 'ema', None)
        if ema is not None:
            ema_crit = getattr(ema.ema, 'criterion', None)
            if ema_crit is not None:
                _set_box(getattr(ema_crit, 'hyp', None))
        if current_epoch % 10 == 0:
            LOGGER.info(f'  [V21] Dynamic box weight: {dynamic_box_w:.2f} '
                        f'(hyp box={new_box:.2f})')
    except Exception as e:
        _ep = getattr(trainer, 'epoch', 0)
        if _ep % 20 == 0:
            LOGGER.debug(f'[V21] box weight schedule skipped: {e}')


# V21: 困难样本权重表 — 用于主动学习筛选
_v21_hard_sample_weights = {}  # {sample_path: weight}


def _v24_set_in_validation(trainer):
    """### V21 MODIFIED R7-07(严重, C损失层/D3): 验证开始时置位,
    阻止 losses 动态权重块在仅-rank0 验证时发起 all_reduce 导致 NCCL 死锁。"""
    base_config.IN_VALIDATION = True


def _v24_clear_in_validation(trainer):
    """### V21 MODIFIED R7-07(严重, C损失层/D3): 验证结束时复位。"""
    base_config.IN_VALIDATION = False


def _v24_scene_aware_augmentation(trainer):
    """V21: 场景感知动态增强 — 根据当前 Phase 和场景分布调整增强强度

    - Phase 1: 基础增强 (聚焦检测特征学习)
    - Phase 2+: 困难样本定向增强 (pre_crash 时间切片 + 弯道透视变换)
    - 自动提升 mAP<0.3 样本的采样权重
    """
    try:
        current_epoch = trainer.epoch if hasattr(trainer, 'epoch') else 0
        phase = getattr(trainer, '_v24_phase', '1')

        # Phase 2+ 启用更强增强
        if phase != '1' and current_epoch % 5 == 0:
            model = de_parallel(trainer.model)
            # 检查最近的验证结果, 识别困难类别
            metrics = getattr(trainer, 'metrics', {})
            if metrics:
                # 提取各类别 mAP, 标记困难样本
                fitness = metrics.get('fitness', 0)
                if fitness < 0.3:
                    LOGGER.info(f'  [V21] Hard sample mining active: fitness={fitness:.3f} < 0.3')
                    # ### V21 MODIFIED R5-03(一般, D10): 删除无效的 erasing 调整 —
                    # ultralytics 的 erasing 仅用于 classify 任务增强管线,
                    # detect 任务不读取; 且 transforms 已在 dataset 构建时绑定,
                    # 事后修改 trainer.args.erasing 不会重建管线。
                    # 原日志"Increased erasing"会误导监控者以为增强已生效。

        # 动态调整 sampler 权重: 对 recall 低的类别提升采样率
        # ### V21 MODIFIED R7-05(严重, C采样层/D3): 原块在 %5 守卫之外, 每 epoch 触发;
        # 且 _last_confusion 仅 rank0 有值 → 各 rank 计算的 boost 不一致 → sampler
        # target_ratios DDP 发散 (两卡采样分布不同, 训练不可复现)。现在:
        # (1) 与增强守卫同周期 (phase!=1 且每 5 epoch), 配合幂等 update_weights;
        # (2) 仅 rank0 从混淆矩阵算 boost, broadcast 到所有 rank 后再统一 update。
        sampler = getattr(trainer, '_v18_sampler', None)
        if sampler is not None and hasattr(sampler, 'update_weights') \
                and phase != '1' and current_epoch % 5 == 0:
            boost = None
            if RANK in (-1, 0):
                confusion = getattr(trainer, '_last_confusion', None)
                if confusion is not None:
                    recalls = []
                    for c in range(3):
                        total = sum(confusion[c]) if c < len(confusion) else 0
                        recall = confusion[c][c] / total if total > 0 else 0
                        recalls.append(recall)
                    # 困难类别 (recall < 0.3) 提升 1.5x 采样权重
                    boost = [1.5 if r < 0.3 else 1.0 for r in recalls]
                    if current_epoch % 10 == 0:
                        LOGGER.info(f'  [V21] Sampler boost factors: {boost} (recalls={recalls})')
            # DDP: broadcast rank0 的 boost 到所有 rank, 消除 target_ratios 发散
            # V25 FIX: 用 tensor broadcast 替代 broadcast_object_list —
            # broadcast_object_list 使用 pickle+多次NCCL op, 在 DDP forward 内调用时
            # 与 DDP backward 的梯度 all_reduce 产生 SeqNum 竞争, 导致4次死锁。
            # 直接 tensor broadcast 是单次 NCCL op, 更可靠。
            if _has_dist and dist.is_initialized():
                import torch
                boost_tensor = torch.tensor(boost if boost is not None else [1.0, 1.0, 1.0],
                                            dtype=torch.float32)
                if RANK not in (-1, 0):
                    boost_tensor = torch.zeros(3, dtype=torch.float32)
                # 移到模型所在 GPU, NCCL 要求 CUDA tensor
                _dev = next(trainer.model.parameters()).device
                boost_tensor = boost_tensor.to(_dev)
                dist.broadcast(boost_tensor, src=0)
                boost = boost_tensor.cpu().tolist()
            if boost is not None:
                sampler.update_weights(boost)

    except Exception as e:
        LOGGER.debug(f'[V21] scene-aware augmentation skipped: {e}')


def _v24_log_scene_dist(trainer):
    """每 epoch 结束后打印场景分类分布"""
    try:
        model = de_parallel(trainer.model)
        crit = getattr(model, 'criterion', None)
        if crit is None or not hasattr(crit, '_step_count') or crit._step_count == 0:
            return

        device = next(trainer.model.parameters()).device
        pred = crit._pred_counts.float().to(device)
        gt = crit._gt_counts.float().to(device)

        if _has_dist and dist.is_initialized():
            dist.all_reduce(pred, op=dist.ReduceOp.SUM)
            dist.all_reduce(gt, op=dist.ReduceOp.SUM)

        total_pred = pred.sum()
        total_gt = gt.sum()
        if total_pred == 0 or total_gt == 0:
            return

        names = ['normal', 'pre_crash', 'crash']
        parts = []
        for i, name in enumerate(names):
            p_pct = 100 * pred[i] / total_pred
            g_pct = 100 * gt[i] / total_gt
            parts.append(f"{name}: pred={p_pct:.1f}% gt={g_pct:.1f}%")
        msg = f"  [Train] Scene cls dist: {' | '.join(parts)}"
        print(msg)
        LOGGER.info(msg)
        crit._pred_counts.zero_()
        crit._gt_counts.zero_()
        crit._step_count = 0
    except Exception as e:
        LOGGER.warning(f'[Train] Scene cls dist logging: {type(e).__name__}: {e}')


# V22-fix(scene-1): 验证集场景先验缓存 {data_root: tensor[3]}
_VAL_SCENE_PRIOR_CACHE = {}


def _val_scene_prior(trainer):
    """统计验证集有效帧 (scene=1/2/3) 的场景先验, 返回 float64 tensor[3]。

    用于均衡采样下的评估先验校正: 训练侧先验≈sampler_ratios (≈1:1:1),
    验证侧为自然分布 (normal 占比 ~70%), 直接 argmax 会系统性偏向少数类。
    """
    root = getattr(trainer, '_data_root', None)
    if not root:
        data = getattr(trainer, 'data', None)
        root = data.get('path') if isinstance(data, dict) else None
    if not root:
        return None
    if root in _VAL_SCENE_PRIOR_CACHE:
        return _VAL_SCENE_PRIOR_CACHE[root]
    d = os.path.join(root, 'scene_labels', 'val')
    if not os.path.isdir(d):
        return None
    counts = torch.zeros(4, dtype=torch.float64)
    for fn in os.listdir(d):
        if not fn.endswith('.txt'):
            continue
        try:
            with open(os.path.join(d, fn)) as f:
                c = int(f.readline().strip())
            if 1 <= c <= 3:
                counts[c] += 1
        except Exception:
            pass
    if counts[1:].sum() <= 0:
        return None
    p = counts[1:] / counts[1:].sum()
    _VAL_SCENE_PRIOR_CACHE[root] = p
    return p


def _d13_zero_loss_epoch_check(trainer):
    """### D13增强(严重, 审查提示词): 跨 epoch 零损失持续检测回调 —
    每个 epoch 结束时调用, 检查各子任务损失是否连续多个 epoch 静默零收敛。
    连续 3 个 epoch 零损失即触发 CRITICAL WARNING。
    """
    try:
        _d13_epoch_mark(len(trainer.train_loader))  # V22-fix(D13): 传入真实步数供占比判定
        _d13_epoch_check()  # 检查连续零损失 epoch 数
    except Exception as e:
        LOGGER.debug(f'[D13] epoch check skipped: {e}')


def _prefetch_batches(loader, maxsize=3):
    """单线程预取器: 让「解码+装配」与「GPU 前向」重叠, 不起子进程。

    保住 V22-fix(deadlock) 的语义 (num_workers=0 ⇒ 无 worker 进程、无 fork 挂起路径);
    只有一个后台线程触碰 loader 迭代器 ⇒ 无并发 next() 竞态; 输出顺序与直接迭代一致。
    异常从主线程原样抛出, 不许被预取线程吞掉。
    """
    import queue as _queue
    import threading as _threading
    q = _queue.Queue(maxsize=maxsize)
    _DONE = object()

    def _pump():
        try:
            for b in loader:
                q.put(b)
        except BaseException as e:
            q.put(e)
        finally:
            q.put(_DONE)

    _threading.Thread(target=_pump, daemon=True).start()
    while True:
        item = q.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _lane_eval_accum(out, batch, device, lane_tp, lane_fp, lane_fn, lane_frames):
    """V26-G350: G247 lane 累计块抽出共用 (P3 块级、sigmoid>0.5、仅有效块，口径一字未动)。

    动因: 原块位于 `valid_mask.sum()==0 → continue` 之后，纯 lane 验证集
    (scene 全 ignore，如 CULane) 每批都在 forward 前被跳过 ⇒ lane_frames 恒 0、
    [Val] Lane 永远出不了数。前移到 forward 后统一调用，scene 有效批行为不变。
    """
    if len(out) > 7 and isinstance(out[7], torch.Tensor) and out[7].dim() == 4:
        sem_mask = batch.get("semantic_mask", None)
        if isinstance(sem_mask, torch.Tensor) and sem_mask.dim() == 3:
            l_tgt, l_val = lane_target_valid(sem_mask.to(device), out[7].shape[-2:])
            pred_pos = out[7].sigmoid() > 0.5
            tgt_pos = l_tgt > 0.5
            v = l_val > 0
            lane_tp = lane_tp + (pred_pos & tgt_pos & v).sum(dim=(0, 2, 3)).cpu()
            lane_fp = lane_fp + (pred_pos & ~tgt_pos & v).sum(dim=(0, 2, 3)).cpu()
            lane_fn = lane_fn + (~pred_pos & tgt_pos & v).sum(dim=(0, 2, 3)).cpu()
            lane_frames += int((l_val.sum(dim=(1, 2, 3)) > 0).sum().item())
    return lane_tp, lane_fp, lane_fn, lane_frames


def _lane_only_val_report(trainer, validator, lane_tp, lane_fp, lane_fn, lane_frames):
    """V26-G350: lane-only 验证读数 (scene_total=0 的 run, 如 CULane 车道线微调)。

    与完整路径 lane 块同口径 (P3 块级、sigmoid>0.5、仅有效块、同一累计器)；
    写 cause_type_metrics.csv (lane 行) + [Val] Lane(only) 横幅 +
    fitness=lane macro F1 —— 走 #252 通道 (validator._scene_fitness_val →
    返回 dict 的 'fitness' 键)，否则纯 lane run 的 fitness 恒 0、best.pt 冻在 epoch1。
    """
    rows, f1s, names, prs = [], [], [], []
    for i, nm in enumerate(LANE_CHANNELS):
        tp, fp, fn = int(lane_tp[i]), int(lane_fp[i]), int(lane_fn[i])
        if tp + fp + fn == 0:
            continue
        p_i = 100 * tp / max(tp + fp, 1)
        r_i = 100 * tp / max(tp + fn, 1)
        rows.append(f'lane,{nm},{tp + fn},{tp + fp},{tp},{fp},{fn},{p_i:.2f},{r_i:.2f}')
        f1s.append(2 * p_i * r_i / max(p_i + r_i, 1e-9))
        names.append(nm)
        prs.append((p_i, r_i))
    sub_path = os.path.join(str(trainer.save_dir), 'cause_type_metrics.csv')
    with open(sub_path, 'w') as f:
        f.write('task,name,gt_positive,predicted_positive,tp,fp,fn,precision,recall\n')
        f.write('\n'.join(rows) + '\n')
        f.write(f'# lane_frames(有有效监督块),{lane_frames}\n')
        f.write('# note,G-350 lane-only val (scene_total=0)\n')
    macro_f1 = (sum(f1s) / len(f1s) / 100.0) if f1s else 0.0
    msg = ("  [Val] Lane(only): "
           + " ".join(f"{nm} P={p:.1f}%/R={r:.1f}%" for nm, (p, r) in zip(names, prs))
           + f" (块级 n={lane_frames}) macroF1={100 * macro_f1:.1f}%")
    print(msg)
    LOGGER.info(msg)
    validator._scene_fitness_val = macro_f1
    if isinstance(trainer.metrics, dict):
        trainer.metrics['fitness'] = macro_f1
    return macro_f1


def _v18_eval_scene_classification(validator):
    """V20: 验证结束后评估场景分类, 输出混淆矩阵 + 事故检测专用指标

    新增指标:
      - Pre-crash Detection Rate (PDR): pre_crash 被识别为"非正常"的比例
      - False Alarm Rate (FAR): normal 被误报为 pre_crash/crash 的比例
      - 场景转换准确率: normal→pre_crash→crash 转换点检测准确率
    """
    try:
        trainer = getattr(validator, '_trainer_ref', None)
        if trainer is None:
            return  # validator 无 trainer 反向引用 (如独立验证), 跳过

        # D17修复(DDP死锁): IN_VALIDATION flag bug 已修复 (losses.py:476)。
        # NCCL_TIMEOUT=1800 (30分钟), scene_cls 评估约5分钟, 不会超时。
        # rank 1+ 在 dist.reduce 等待 rank 0 完成评估后一起同步。
        _is_ddp = dist.is_initialized() if _has_dist else False
        if _is_ddp:
            LOGGER.info("  [Val] Scene cls: DDP 模式, rank 0 执行全量评估 (rank 1+ 在 dist.reduce 等待)")

        # 用 EMA 模型 (validator 验证时用的就是 EMA)
        model = de_parallel(trainer.ema.ema or trainer.model)
        if not hasattr(model, 'nc_cls'):
            return

        val_loader = trainer.test_loader
        if val_loader is None:
            return

        # 清空 CUDA 缓存, 释放验证阶段占用的显存
        torch.cuda.empty_cache()

        # V22修复: DDP 模式下 DistributedSampler 将数据分片, rank0 只看到50%数据。
        # scene_cls callback 仅在 rank0 运行, 需要完整评估所有样本。
        # 创建使用 SequentialSampler 的 DataLoader, 确保评估完整性。
        _is_ddp = dist.is_initialized() if _has_dist else False
        if _is_ddp:
            from torch.utils.data import DataLoader, SequentialSampler
            # V22-fix(deadlock): s7 ep17 该评估在 val workers 下死锁 25+ 分钟
            # (rank0 卡死, rank1+ 永等 dist.reduce)。num_workers=0 主进程同步迭代,
            # 评估慢一点但彻底消除 worker 挂起路径。
            eval_loader = DataLoader(
                val_loader.dataset,
                batch_size=val_loader.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=val_loader.collate_fn,
                pin_memory=False,
                drop_last=False,
            )
            LOGGER.info("  [Val] Scene cls: DDP 模式, SequentialSampler + num_workers=0 评估完整数据集")
        else:
            eval_loader = val_loader

        model.eval()
        try:
            device = next(model.parameters()).device
            # V22-fix(scene-1): 均衡采样先验错配校正 — 训练侧先验≈sampler_ratios
            # (≈1:1:1), 验证侧自然分布 (normal ~70%)。
            # V22-fix(scene-2): 全量校正幅度(~1.6)大于早期模型置信边距(~1.0),
            # 会把判对样本也翻成 normal (全量评估退化为全 normal)。因此校正结果
            # 仅作参考口径累计, 主口径恢复为 raw logits argmax (与 s4 可比)。
            prior_corr = None
            try:
                ratios = getattr(trainer, '_sampler_ratios', None) or _env_floats('V21_SAMPLER_RATIOS')
                if ratios and len(ratios) == 3 and sum(ratios) > 0:
                    p_train = torch.tensor(ratios, dtype=torch.float64)
                    p_train = p_train / p_train.sum()
                    p_val = _val_scene_prior(trainer)
                    if p_val is not None:
                        prior_corr = (torch.log(p_val) - torch.log(p_train)).to(device).float()
                        LOGGER.info(f'  [Val] Scene cls: 先验校正 p_train=[{p_train[0]:.3f},{p_train[1]:.3f},{p_train[2]:.3f}] '
                                    f'p_val=[{p_val[0]:.3f},{p_val[1]:.3f},{p_val[2]:.3f}] shift={prior_corr.tolist()}')
            except Exception as e:
                LOGGER.debug(f'[Val] Scene cls: 先验校正不可用: {e}')
            scene_correct = 0
            corr_correct = 0  # V22-fix(scene-2): 先验校正口径仅作参考
            scene_total = 0
            scene_loss_sum = 0.0  # V21: 无加权 CE 损失累计
            scene_loss_count = 0
            confusion = [[0] * 3 for _ in range(3)]  # 3x3: normal/pre_crash/crash
            # V22-fix(scene-3): 按域拆分混淆矩阵 — 真实 CCTV (real_*) vs 合成切块,
            # 紧盯目标①在真实域的表现 (此前真实域失效被合成域指标掩盖)
            confusion_real = [[0] * 3 for _ in range(3)]
            confusion_syn = [[0] * 3 for _ in range(3)]
            # ### D16修复(致命): cause/type 子任务独立指标累计器 —
            # 原验证仅评估 scene_cls 混淆矩阵 + 检测 mAP, cause/type 头的
            # 静默零收敛 (标签全零 BCE 收敛到平凡解) 完全不可见。
            _eval_phase = getattr(trainer, '_v24_phase', '1')
            # V24: cause 从 Phase 1 起训练/评估; type 头 Phase 1 仍未训练
            _eval_cause = True
            _eval_type = _eval_phase != '1'
            cause_tp = torch.zeros(len(CAUSE_NAMES), dtype=torch.long)
            cause_fp = torch.zeros(len(CAUSE_NAMES), dtype=torch.long)
            cause_fn = torch.zeros(len(CAUSE_NAMES), dtype=torch.long)
            cause_active_samples = 0   # pre_crash/crash 帧数 (cause 有效样本)
            cause_pred_pos_total = 0   # 预测正例总数 (平凡解检测)
            type_confusion = [[0] * len(TYPE_NAMES) for _ in range(len(TYPE_NAMES))]
            type_total = 0             # crash 帧且 type 标签有效样本数
            # V26-G247: 车道线分割指标累计 (7 通道, P3 块级, 与训练损失同口径)
            lane_tp = torch.zeros(len(LANE_CHANNELS), dtype=torch.long)
            lane_fp = torch.zeros(len(LANE_CHANNELS), dtype=torch.long)
            lane_fn = torch.zeros(len(LANE_CHANNELS), dtype=torch.long)
            lane_frames = 0            # 有有效监督块的帧数

            with torch.no_grad():
                # G-244: 线程预取重叠装载与前向 (num_workers=0 不变, 顺序不变)
                for batch in _prefetch_batches(eval_loader):
                    batch = trainer.preprocess_batch(batch)
                    imgs = batch.get("img", None)
                    scene_targets = batch.get("scene_cls", None)
                    if imgs is None or scene_targets is None:
                        continue

                    if isinstance(scene_targets, (list, tuple)):
                        scene_targets = torch.cat([
                            t.view(-1) if isinstance(t, torch.Tensor)
                            else torch.tensor([t], dtype=torch.long)
                            for t in scene_targets
                        ])
                    if isinstance(scene_targets, torch.Tensor):
                        scene_targets = scene_targets.to(device).long().view(-1)

                    # V20: 跳过 ignore_index=0 的样本
                    valid_mask = (scene_targets != 0)
                    # V22-fix(tile-mask): 2x2 瓦片场景标签与内容解耦, 从场景指标中
                    # 排除 (与部署口径一致: 全帧判定)
                    im_files = batch.get("im_file", None)
                    if im_files is not None and len(im_files) == len(valid_mask):
                        is_tile = torch.tensor(
                            [re.search(r'_t\d{2}\.[A-Za-z]+$', os.path.basename(str(f))) is not None
                             for f in im_files],
                            dtype=torch.bool, device=device)
                        valid_mask = valid_mask & ~is_tile
                    out = model(imgs)
                    if not isinstance(out, (tuple, list)) or len(out) < 2:
                        continue
                    # G-350: lane 累计必须在"scene 全 ignore"批也执行 ⇒ forward 上提、
                    # lane 块抽成共用函数；scene 部分降级为 valid_mask 跳过 (事故 val 每批
                    # 都有有效样本，行为逐字不变)
                    lane_tp, lane_fp, lane_fn, lane_frames = _lane_eval_accum(
                        out, batch, device, lane_tp, lane_fp, lane_fn, lane_frames)
                    if valid_mask.sum() == 0:
                        continue
                    scene_logits = out[1]
                    # V22-fix(scene-2): 主口径 = raw logits argmax
                    pred_cls_idx = scene_logits.argmax(dim=1)  # [B] in {0,1,2}
                    
                    # V20: 将标签映射到分类索引: 1->0, 2->1, 3->2
                    gt_cls_idx = (scene_targets - 1).clamp(0, 2)  # [B] in {0,1,2}
                    
                    # 仅对有效样本计算混淆矩阵
                    # V22-fix(scene-3): 按域拆分 (real_* = 真实 CCTV)
                    im_files = batch.get("im_file", None)
                    is_real = None
                    if im_files is not None and len(im_files) == len(gt_cls_idx):
                        is_real = torch.tensor(
                            [os.path.basename(str(f)).startswith('real_') for f in im_files],
                            dtype=torch.bool, device=device)
                    for c in range(3):
                        gt_mask = (gt_cls_idx == c) & valid_mask
                        for p in range(3):
                            n_cp = ((pred_cls_idx == p) & gt_mask).sum().item()
                            confusion[c][p] += n_cp
                            if is_real is not None:
                                rp_mask = (pred_cls_idx == p) & gt_mask & is_real
                                n_r = rp_mask.sum().item()
                                confusion_real[c][p] += n_r
                                confusion_syn[c][p] += n_cp - n_r
                    scene_correct += ((pred_cls_idx == gt_cls_idx) & valid_mask).sum().item()
                    scene_total += valid_mask.sum().item()
                    # V22-fix(scene-2): 校正口径仅累计 Acc 作参考, 不进混淆矩阵
                    if prior_corr is not None:
                        corr_pred = (scene_logits + prior_corr).argmax(dim=1)
                        corr_correct += ((corr_pred == gt_cls_idx) & valid_mask).sum().item()

                    # V21: 累计无加权 CE 损失 (用于监控, 不受 CLASS_WEIGHTS 影响)
                    valid_logits = scene_logits[valid_mask]
                    valid_gt = gt_cls_idx[valid_mask]
                    if valid_logits.numel() > 0:
                        scene_loss_sum += F.cross_entropy(valid_logits, valid_gt, reduction='sum').item()
                        scene_loss_count += valid_mask.sum().item()

                    # ### D16修复(致命): cause/type 子任务独立指标 (与训练损失同口径) —
                    # cause 仅统计 pre_crash/crash 帧 (对齐训练 scene_mask);
                    # type 仅统计 crash 帧且标签 >=0 (对齐 AccidentTypeLoss)。
                    if isinstance(out, (tuple, list)):
                        scene_raw = scene_targets  # 0=ignore 1=normal 2=pre_crash 3=crash
                        # ── cause 多标签 (16-bit, sigmoid>0.5) ──
                        if _eval_cause and len(out) >= 4 and out[3] is not None:
                            cause_t = batch.get("cause", None)
                            if cause_t is not None:
                                if isinstance(cause_t, (list, tuple)):
                                    cause_t = torch.cat([
                                        (t.view(1, -1) if isinstance(t, torch.Tensor) and t.dim() == 1
                                         else t if isinstance(t, torch.Tensor)
                                         else torch.tensor([t], dtype=torch.float32))
                                        for t in cause_t
                                    ], dim=0)
                                elif isinstance(cause_t, torch.Tensor) and cause_t.dim() == 1:
                                    cause_t = cause_t.view(1, -1)
                                if isinstance(cause_t, torch.Tensor) and cause_t.size(-1) == len(CAUSE_NAMES):
                                    cause_t = cause_t.to(device).float()
                                    active = (scene_raw >= 2)  # pre_crash/crash 帧
                                    # V24: 无成因标注帧 (cause_valid=0) 不计入指标
                                    cv_e = batch.get("cause_valid", None)
                                    if cv_e is not None:
                                        if isinstance(cv_e, (list, tuple)):
                                            cv_e = torch.tensor(
                                                [int(t) if not isinstance(t, torch.Tensor) else int(t.item())
                                                 for t in cv_e],
                                                dtype=torch.long, device=device)
                                        elif isinstance(cv_e, torch.Tensor):
                                            cv_e = cv_e.to(device).long().view(-1)
                                        if cv_e.numel() == active.numel():
                                            active = active & (cv_e > 0)
                                    if active.any():
                                        cause_active_samples += active.sum().item()
                                        probs = torch.sigmoid(out[3][active])
                                        pred_pos = probs > 0.5
                                        tgt_pos = cause_t[active] > 0.5
                                        cause_tp += (pred_pos & tgt_pos).sum(dim=0).cpu()
                                        cause_fp += (pred_pos & ~tgt_pos).sum(dim=0).cpu()
                                        cause_fn += (~pred_pos & tgt_pos).sum(dim=0).cpu()
                                        cause_pred_pos_total += pred_pos.sum().item()
                        # ── type 单标签 (10 类, 仅 crash 帧) ──
                        if _eval_type and len(out) >= 5 and out[4] is not None:
                            type_t = batch.get("accident_type", None)
                            if type_t is not None:
                                if isinstance(type_t, (list, tuple)):
                                    type_t = torch.cat([
                                        t.view(-1) if isinstance(t, torch.Tensor)
                                        else torch.tensor([t], dtype=torch.long)
                                        for t in type_t
                                    ])
                                if isinstance(type_t, torch.Tensor):
                                    type_t = type_t.to(device).long().view(-1)
                                    crash_valid = (scene_raw == 3) & (type_t >= 0) & (type_t < len(TYPE_NAMES))
                                    if crash_valid.any():
                                        type_pred = out[4][crash_valid].argmax(dim=1)
                                        type_gt = type_t[crash_valid]
                                        for g, p in zip(type_gt.tolist(), type_pred.tolist()):
                                            type_confusion[g][p] += 1
                                        type_total += crash_valid.sum().item()
                        # ── V26-G247: 车道线多标签分割 (P3 块级, 与训练损失同口径) ──
                        # G-350: 此累计已前移到 forward 后统一走 _lane_eval_accum
                        # (全 scene-ignore 批也计)，此处不再重复累计。

            if scene_total == 0:
                # G-350: 纯 lane 验证集 (CULane 微调 run) 无 scene 监督样本 ⇒ scene 块整段
                # 早退。但 lane 块不能跟着退：①[Val] Lane 没读数，巡检看不到车道线；
                # ②fitness 覆盖不发生 → ultralytics 默认 mAP 恒 0 → best.pt 停在 epoch 1。
                if lane_frames > 0:
                    _lane_only_val_report(trainer, validator, lane_tp, lane_fp,
                                          lane_fn, lane_frames)
                return

            # 修复: 此 ultralytics 版本验证仅在 rank 0 执行
            # (trainer.py: `if RANK in {-1,0}: self.validate()`), on_val_end 只在 rank 0 触发,
            # 原 dist.all_reduce 会让 rank 0 等待其他 rank → DDP 死锁。
            # rank 0 已处理完整验证集, 混淆矩阵已完整, 无需 all_reduce。

            acc = 100 * scene_correct / scene_total
            names = ['normal', 'pre_crash', 'crash']
            
            # V20: 计算事故检测专用指标
            # PDR (Pre-crash Detection Rate): pre_crash 被识别为"非正常"(pre_crash或crash)的比例
            pdr_tp = confusion[1][1] + confusion[1][2]  # pre_crash 预测为 pre_crash 或 crash
            pdr_total = sum(confusion[1])  # pre_crash 总数
            pdr = 100 * pdr_tp / pdr_total if pdr_total > 0 else 0
            
            # FAR (False Alarm Rate): normal 被误报为 pre_crash/crash 的比例
            far_fp = confusion[0][1] + confusion[0][2]  # normal 预测为 pre_crash 或 crash
            far_total = sum(confusion[0])  # normal 总数
            far = 100 * far_fp / far_total if far_total > 0 else 0
            
            # 场景转换准确率: pre_crash 被正确识别为 pre_crash (不是 crash 也不是 normal)
            pre_recall = 100 * confusion[1][1] / sum(confusion[1]) if sum(confusion[1]) > 0 else 0
            pre_prec = 100 * confusion[1][1] / sum(confusion[r][1] for r in range(3)) if sum(confusion[r][1] for r in range(3)) > 0 else 0
            
            # V21: 计算无加权验证损失
            val_scene_loss = scene_loss_sum / max(scene_loss_count, 1)

            parts = [f"Acc={acc:.1f}%", f"CE={val_scene_loss:.4f}"]
            for c in range(3):
                tp = confusion[c][c]
                fp = sum(confusion[r][c] for r in range(3)) - tp
                fn = sum(confusion[c]) - tp
                rec = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
                prec = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
                parts.append(f"{names[c]}: R={rec:.1f}% P={prec:.1f}%")
            
            # V20: 添加事故检测专用指标
            parts.append(f"PDR={pdr:.1f}% FAR={far:.1f}% preR={pre_recall:.1f}% preP={pre_prec:.1f}%")
            
            msg = f"  [Val] Scene cls: {' | '.join(parts)}"
            print(msg)
            LOGGER.info(msg)

            # V22-fix(scene-3): 分域 Acc — 真实 CCTV vs 合成切块 (紧盯目标①真实域)
            def _dom_acc(cm):
                tot = sum(sum(r) for r in cm)
                if tot == 0:
                    return None
                tp = sum(cm[i][i] for i in range(3))
                return 100 * tp / tot, tot
            r_res = _dom_acc(confusion_real)
            s_res = _dom_acc(confusion_syn)
            dom_parts = []
            if r_res is not None:
                dom_parts.append(f"real Acc={r_res[0]:.1f}% (n={r_res[1]})")
                r_rec = [100 * confusion_real[i][i] / sum(confusion_real[i])
                         if sum(confusion_real[i]) > 0 else 0.0 for i in range(3)]
                dom_parts.append(f"real R(n/p/c)={r_rec[0]:.1f}/{r_rec[1]:.1f}/{r_rec[2]:.1f}")
            if s_res is not None:
                dom_parts.append(f"syn Acc={s_res[0]:.1f}% (n={s_res[1]})")
            if dom_parts:
                dom_msg = f"  [Val] Scene cls by domain: {' | '.join(dom_parts)}"
                print(dom_msg)
                LOGGER.info(dom_msg)
            # V22-fix(scene-2): 参考口径 (先验校正) 单独打印, 不干扰主指标
            if prior_corr is not None and scene_total > 0:
                ref_msg = f"  [Val] Scene cls (prior-corrected ref): Acc={100 * corr_correct / scene_total:.1f}%"
                print(ref_msg)
                LOGGER.info(ref_msg)

            cm_path = trainer.save_dir / 'scene_confusion.csv'
            with open(cm_path, 'w') as f:
                f.write('gt\\pred,normal,pre_crash,crash\n')
                for r in range(3):
                    f.write(f'{names[r]},{confusion[r][0]},{confusion[r][1]},{confusion[r][2]}\n')
                # V20: 追加事故检测专用指标
                f.write(f'\n# Metrics\n')
                f.write(f'Accuracy,{acc:.2f}%\n')
                f.write(f'PDR,{pdr:.2f}%\n')
                f.write(f'FAR,{far:.2f}%\n')
                f.write(f'pre_crash_Recall,{pre_recall:.2f}%\n')
                f.write(f'pre_crash_Precision,{pre_prec:.2f}%\n')
                # V21: 新增 crash/normal 召回率和均衡度
                crash_recall = 100 * confusion[2][2] / sum(confusion[2]) if sum(confusion[2]) > 0 else 0
                normal_recall = 100 * confusion[0][0] / sum(confusion[0]) if sum(confusion[0]) > 0 else 0
                recall_diff = abs(pre_recall - crash_recall)
                f.write(f'crash_Recall,{crash_recall:.2f}%\n')
                f.write(f'normal_Recall,{normal_recall:.2f}%\n')
                f.write(f'recall_imbalance,{recall_diff:.2f}%\n')
                f.write(f'val_CE_loss_raw,{val_scene_loss:.4f}\n')
            LOGGER.info(f'  Scene confusion matrix saved to {cm_path}')

            # ### V22修复(致命, checkpoint 选择失效): ACCIDENT 无 bbox 标签,
            # ultralytics 默认 fitness = 1.0*mAP50-95 恒为 0 → best.pt 选择完全失效
            # (best.pt 始终保存最近/最过拟合的 epoch, 而非场景分类最优)。
            # 覆盖 fitness 为场景分类加权 macro-recall: crash 权重 0.5 (用户核心目标),
            # pre_crash 0.3, normal 0.2。范围 [0,1], 1.0 = 三类召回均完美。
            scene_fitness = (0.5 * crash_recall + 0.3 * pre_recall + 0.2 * normal_recall) / 100.0
            trainer.fitness = scene_fitness
            if isinstance(trainer.metrics, dict):
                trainer.metrics['fitness'] = scene_fitness
            validator._scene_fitness_val = scene_fitness  # #252: 交给 __call__ 注入返回 dict
            LOGGER.info(f'  [Val] Scene fitness override: {scene_fitness:.4f} '
                        f'(crash_R={crash_recall:.1f}% pre_R={pre_recall:.1f}% norm_R={normal_recall:.1f}%)')

            # ### D16修复(致命): cause/type 子任务有效性报告 —
            # 输出各因子 micro P/R + 平凡解检测 (预测正例率过低 → 疑似零收敛)
            if (_eval_cause and cause_active_samples > 0) or (_eval_type and type_total > 0) \
                    or lane_frames > 0:
                sub_path = trainer.save_dir / 'cause_type_metrics.csv'
                with open(sub_path, 'w') as f:
                    f.write('task,name,gt_positive,predicted_positive,tp,fp,fn,precision,recall\n')
                    if cause_active_samples > 0:
                        cause_micro_tp = int(cause_tp.sum())
                        cause_micro_fp = int(cause_fp.sum())
                        cause_micro_fn = int(cause_fn.sum())
                        c_prec = 100 * cause_micro_tp / max(cause_micro_tp + cause_micro_fp, 1)
                        c_rec = 100 * cause_micro_tp / max(cause_micro_tp + cause_micro_fn, 1)
                        pos_label_total = int(cause_tp.sum() + cause_fn.sum())
                        pred_rate = 100 * cause_pred_pos_total / max(cause_active_samples * len(CAUSE_NAMES), 1)
                        for i, nm in enumerate(CAUSE_NAMES):
                            tp, fp, fn = int(cause_tp[i]), int(cause_fp[i]), int(cause_fn[i])
                            if tp + fp + fn == 0:
                                continue  # 验证集无该因子样本且无预测, 跳过
                            p_i = 100 * tp / max(tp + fp, 1)
                            r_i = 100 * tp / max(tp + fn, 1)
                            f.write(f'cause,{nm},{tp + fn},{tp + fp},{tp},{fp},{fn},{p_i:.2f},{r_i:.2f}\n')
                        f.write(f'cause,MICRO_ALL,{pos_label_total},{cause_pred_pos_total},'
                                f'{cause_micro_tp},{cause_micro_fp},{cause_micro_fn},{c_prec:.2f},{c_rec:.2f}\n')
                        f.write(f'# cause_active_samples,{cause_active_samples}\n')
                        f.write(f'# positive_prediction_rate,{pred_rate:.2f}%\n')
                        sub_msg = (f"  [Val] Cause: microP={c_prec:.1f}% microR={c_rec:.1f}% "
                                   f"active={cause_active_samples} pred_pos_rate={pred_rate:.2f}%")
                        # 平凡解检测: 有正例标签但模型几乎不预测正例 → 疑似零收敛
                        if pos_label_total > 0 and cause_pred_pos_total < pos_label_total * 0.01:
                            sub_msg += " ⚠ 疑似平凡解: 模型几乎不预测成因, 请检查 cause 标签分布/损失生效性!"
                        print(sub_msg)
                        LOGGER.info(sub_msg)
                    if type_total > 0:
                        type_correct = sum(type_confusion[i][i] for i in range(len(TYPE_NAMES)))
                        t_acc = 100 * type_correct / type_total
                        for i, nm in enumerate(TYPE_NAMES):
                            gt_i = sum(type_confusion[i])
                            if gt_i == 0:
                                continue
                            r_i = 100 * type_confusion[i][i] / gt_i
                            f.write(f'type,{nm},{gt_i},{sum(type_confusion[r][i] for r in range(len(TYPE_NAMES)))},'
                                    f'{type_confusion[i][i]},{sum(type_confusion[r][i] for r in range(len(TYPE_NAMES))) - type_confusion[i][i]},'
                                    f'{gt_i - type_confusion[i][i]},0,{r_i:.2f}\n')
                        f.write(f'# type_total(crash帧且标签有效),{type_total}\n')
                        sub_msg2 = f"  [Val] Type: Acc={t_acc:.1f}% (crash帧 n={type_total})"
                        print(sub_msg2)
                        LOGGER.info(sub_msg2)
                    if lane_frames > 0:
                        # V26-G247: 车道线逐通道 P/R (P3 块级, sigmoid>0.5, 仅有效块)
                        lane_names, lane_ps, lane_rs = [], [], []
                        for i, nm in enumerate(LANE_CHANNELS):
                            tp, fp, fn = int(lane_tp[i]), int(lane_fp[i]), int(lane_fn[i])
                            if tp + fp + fn == 0:
                                continue  # 验证集无该通道监督且无预测, 跳过
                            p_i = 100 * tp / max(tp + fp, 1)
                            r_i = 100 * tp / max(tp + fn, 1)
                            f.write(f'lane,{nm},{tp + fn},{tp + fp},{tp},{fp},{fn},{p_i:.2f},{r_i:.2f}\n')
                            lane_names.append(nm)
                            lane_ps.append(p_i)
                            lane_rs.append(r_i)
                        f.write(f'# lane_frames(有有效监督块),{lane_frames}\n')
                        if lane_names:
                            sub_msg3 = ("  [Val] Lane: "
                                        + " ".join(f"{nm} P={p:.1f}%/R={r:.1f}%"
                                                   for nm, p, r in zip(lane_names, lane_ps, lane_rs))
                                        + f" (块级 n={lane_frames})")
                            print(sub_msg3)
                            LOGGER.info(sub_msg3)
                LOGGER.info(f'  Cause/Type subtask metrics saved to {sub_path}')
            # V21: 13项核心指标看板
            phase = getattr(trainer, '_v24_phase', '1')
            current_epoch = getattr(trainer, 'epoch', 0)
            map50 = 0.0
            map5095 = 0.0
            # V22-fix(dashboard): on_val_end 时刻 trainer.metrics 尚未赋值
            # (ultralytics 在 validate() 返回后才写 trainer.metrics) → 看板 mAP 恒 0。
            # validator.metrics 是本次验证的 DetMetrics 对象, 优先从它读取。
            vmetrics = getattr(validator, 'metrics', None)
            if hasattr(vmetrics, 'box'):
                map50 = float(getattr(vmetrics.box, 'map50', 0.0) or 0.0)
                map5095 = float(getattr(vmetrics.box, 'map', 0.0) or 0.0)
            else:
                tta_metrics = getattr(trainer, 'metrics', {})
                if isinstance(tta_metrics, dict):
                    inner = tta_metrics.get('metrics', tta_metrics)
                    if isinstance(inner, dict):
                        map50 = inner.get('metrics/mAP50(B)', inner.get('mAP50(B)', 0))
                        map5095 = inner.get('metrics/mAP50-95(B)', inner.get('mAP50-95(B)', 0))
            crit = getattr(model, 'criterion', None)
            cur_weights = crit.CLASS_WEIGHTS if crit else 'N/A'
            # ### V21 MODIFIED R5-05(一般, D9): 看板字段断链 — monitor_dashboard.py
            # 读取 mAP50/mAP50-95/CLASS_WEIGHTS/CAUSE_POS_WEIGHT 四字段,
            # 原 CSV 从未写入 → 看板第 9-12 项永远显示 0/N/A。追加写入。
            try:
                with open(cm_path, 'a') as f:
                    f.write(f'mAP50,{map50*100:.2f}%\n')
                    f.write(f'mAP50-95,{map5095*100:.2f}%\n')
                    # dashboard 用 naive split(',') 解析, list 内逗号换成分号
                    f.write(f"CLASS_WEIGHTS,{str(cur_weights).replace(',', ';')}\n")
                    f.write(f"CAUSE_POS_WEIGHT,{str(base_config.CAUSE_POS_WEIGHT).replace(',', ';')}\n")
            except OSError:
                pass
            dashboard = (
                f"\n  {'='*50}\n"
                f"  V21 核心指标看板 (Epoch {current_epoch+1}, Phase {phase})\n"
                f"  {'='*50}\n"
                f"  1. Accuracy:         {acc:.1f}%\n"
                f"  2. CE Loss(raw):     {val_scene_loss:.4f}\n"
                f"  3. normal Recall:    {normal_recall:.1f}%\n"
                f"  4. pre_crash Recall: {pre_recall:.1f}% (target>=50%)\n"
                f"  5. crash Recall:     {crash_recall:.1f}% (target>=75%)\n"
                f"  6. Recall Diff:      {recall_diff:.1f}% (target<=15%)\n"
                f"  7. PDR:              {pdr:.1f}%\n"
                f"  8. FAR:              {far:.1f}%\n"
                f"  9. mAP50:            {map50*100:.2f}% (target>=12.5%)\n"
                f" 10. mAP50-95:         {map5095*100:.2f}%\n"
                f" 11. CLASS_WEIGHTS:    {cur_weights}\n"
                f" 12. CAUSE_POS_WEIGHT: {base_config.CAUSE_POS_WEIGHT}\n"
                f" 13. Phase:            {phase}\n"
                f"  {'='*50}"
            )
            print(dashboard)
            LOGGER.info(dashboard)

            # V21: 标签-模型联动验证 — 混淆矩阵均衡性校验
            balance_warnings = []
            if normal_recall < 30:
                balance_warnings.append(f'normal Recall {normal_recall:.1f}% < 30% (标签可能存在偏差)')
            if crash_recall < 30:
                balance_warnings.append(f'crash Recall {crash_recall:.1f}% < 30% (标签可能存在偏差)')
            if recall_diff > 15:
                balance_warnings.append(f'Recall Diff {recall_diff:.1f}% > 15% (场景分类不均衡)')
            if far > 15:
                balance_warnings.append(f'FAR {far:.1f}% > 15% (误报率过高, 检查阈值设置)')
            if balance_warnings:
                warn_msg = f'  [V21 联动验证] ⚠️ 发现 {len(balance_warnings)} 项问题:\n' + '\n'.join(f'    - {w}' for w in balance_warnings)
                print(warn_msg)
                LOGGER.warning(warn_msg)
                # 保存联动验证报告
                linkage_path = trainer.save_dir / 'label_model_linkage.json'
                linkage_report = {
                    'epoch': current_epoch + 1,
                    'phase': phase,
                    'issues': balance_warnings,
                    'metrics': {
                        'normal_recall': normal_recall,
                        'pre_crash_recall': pre_recall,
                        'crash_recall': crash_recall,
                        'recall_diff': recall_diff,
                        'far': far,
                        'map50': map50 * 100,
                    },
                    'recommendation': '建议检查标签边界 + 调整 CLASS_WEIGHTS + 复核 TTC 阈值',
                }
                with open(linkage_path, 'w') as f:
                    json.dump(linkage_report, f, indent=2, ensure_ascii=False)
            else:
                ok_msg = f'  [V21 联动验证] ✓ 混淆矩阵均衡性检查通过 (Recall Diff={recall_diff:.1f}% ≤ 15%, FAR={far:.1f}%)'
                print(ok_msg)
                LOGGER.info(ok_msg)

            # 保存混淆矩阵引用供场景感知增强回调使用
            trainer._last_confusion = confusion

            # V21: 自适应权重调度 (Phase 2+)
            # V25 FIX: every epoch (was every 5) — crash Recall=6.6% needs faster adaptation
            # The EMA every-100-steps update drags crash weight from 3.0→1.36 within one epoch,
            # so 5-epoch intervals are too slow. Running every epoch lets the adaptive system
            # (scale=6.0, max=15.0) push crash weight back up before the next epoch starts.
            if phase != '1' and (current_epoch + 1) % 1 == 0:
                crit = getattr(model, 'criterion', None)
                if crit is not None and hasattr(crit, 'update_class_weights'):
                    new_w = adaptive_weight_update(confusion, crit.CLASS_WEIGHTS)
                    crit.update_class_weights(new_w)
                    # R1-07: 同步更新训练模型 criterion
                    train_crit = getattr(de_parallel(trainer.model), 'criterion', None)
                    if train_crit is not None and hasattr(train_crit, 'update_class_weights'):
                        train_crit.update_class_weights(new_w)
                    # V25 FIX: skip in-training EMA for rest of this epoch —
                    # the EMA (every 100 steps) pulls weights toward raw_w,
                    # undoing the adaptive update within the same epoch.
                    # Both ranks run this independently (same confusion matrix
                    # from SequentialSampler), no DDP broadcast needed.
                    crit._skip_ema_blend = True
                    if train_crit is not None:
                        train_crit._skip_ema_blend = True
                    LOGGER.info(f'  [V21] Adaptive weight update @ epoch {current_epoch+1}: {new_w}')
                    # V25 FIX: persist adaptive weights to sidecar file —
                    # ultralytics strips criterion from checkpoints (ema.criterion=None),
                    # so weights are always lost on resume. The sidecar survives restarts.
                    # D15修复: 优先 trainer 真值 save_dir, env 只作兜底
                    _save_dir = str(getattr(trainer, 'save_dir', '') or '') or os.environ.get('V25_SAVE_DIR', '')
                    if _save_dir and RANK in (-1, 0):
                        try:
                            import json as _json
                            _sw_path = os.path.join(_save_dir, 'adaptive_class_weights.json')
                            with open(_sw_path, 'w') as f:
                                _json.dump({'class_weights': new_w, 'epoch': current_epoch + 1}, f)
                        except Exception as _e:
                            LOGGER.warning(f'  [V21] Failed to save adaptive weights: {_e}')
        finally:
            model.eval()  # R2-09: EMA 模型应始终保持 eval 模式, 避免 BN running stats 被污染
            # 释放场景评估占用的显存
            torch.cuda.empty_cache()
    except Exception as e:
        LOGGER.warning(f'[Val] Scene cls eval: {type(e).__name__}: {e}')
