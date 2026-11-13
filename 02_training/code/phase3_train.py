# -*- coding: utf-8 -*-
"""
phase3_train — V22 Phase 3 时序因果推理训练循环

在 V21 run_phase3_temporal 基础上:
  - 使用 V24TemporalModel (CTTA 增强 Transformer)
  - 使用 V22TemporalMultiTaskLoss (UASC + PTCL + CDCR)
  - 传递 type_ids 给 CTTA 注意力偏置
  - 设置 phase 控制 PTCL 课程掩码
  - loss_dict 键名兼容 V21 epoch_losses 累加器

设计原则:
  - 复用 V21 的 DDP/数据集/优化器/验证逻辑
  - 仅替换模型构建 + 损失计算 + 前向传播调用
  - 不修改 V21 任何文件
"""

import os
import sys
import math
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch

from ultralytics.utils import LOGGER
try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    from ultralytics.utils.torch_utils import unwrap_model as de_parallel

from base_config import (
    dist, DDP, DistributedSampler,
    DEFAULT_SCENE_CLS_WEIGHT, get_phase3_loss_weights, NUM_TYPES,
    CAUSE_DISABLED_BITS,
)
from dataset import ClipDataset

# V21 时序模块 (回退用)
from phase3_legacy import _has_temporal, compute_detection_latency

# V22 增强模块
from temporal_model import (
    V24TemporalModel, V22TemporalMultiTaskLoss,
    build_temporal_model, CTTA_CONFIG, UASC_CONFIG, CDCR_CONFIG, PTCL_CONFIG,
)

# V22 数据管道扩展 (source_domain + agreement)
from train_data import wrap_clip_dataset

# G-280(type-instrument): 三档计量分桶的唯一实现 (尺② eval_acceptance.py 同源 import)
from train_config import type_score_bucket

# V22 训练监控与诊断系统
from monitor import (
    TrainingMonitor, V22_MONITOR_CONFIG,
    save_training_state, load_training_state, find_resume_checkpoint,
    get_system_resources, format_resources,
)

# V25-fix(type-leak): 训练时 CTTA 的 type 输入原本直接喂 GT accident_type,
# 而 [Val] Type Acc 又用同一标签判定 → 模型只需复制输入, 指标恒为 ~100%。
# 训练中以该概率将 crash clip 的 type 输入替换为 padding(7), 迫使 type head
# 从视觉特征学习; 验证侧一律使用 padding (无泄漏 pass)。
V25_TYPE_INPUT_DROPOUT = float(os.environ.get('V25_TYPE_INPUT_DROPOUT', '0.5'))


def is_no_decay_param(name, p):
    """G-282: 时序/决策侧的 norm 增益、所有 bias、pos_embed、cls_token 不吃 weight_decay。
    实测: 解冻后这些 1-D 张量在 wd=0.15 下每轮乘性缩水 (temporal_context.norm.weight
    15.34→8.45), 而时序层被冻住的历史轮里同一个张量逐字停在出厂 16.000 —— 见 §16-10。
    范围刻意排除 detection_model(P2 交付的那份骨干): 它的优化行为逐字不变 ⇒ 变更面
    只覆盖出事的时序侧, mAP/检测面仍可与历史直接对账。"""
    if 'detection_model' in name:
        return False
    if p.ndim <= 1:
        return True
    return name.endswith('pos_embed') or name.endswith('cls_token')


def is_r1_exposed_param(name):
    """G-286: 取"R1 解冻才第一次进可训练集"的那批时序参数 (standard+base)。

    名字口径与 temporal_model.CTTATemporalContext.freeze_standard_layers() 逐字对齐
    (= get_progressive_params()['standard'] + ['base']):
      standard_encoder.* / local_cnn.* / pos_embed / cls_token / norm.*
    刻意**不按 ndim 取**: 按 ndim 会 (a) 漏掉 local_cnn 的 4-D 卷积权重,
    (b) 误伤 frame_pool/decision_head 的 1-D 偏置 —— 后者在冻结年代同样是可训练的,
    不是本轮新暴露的变量。证据见 §16-15-8 的同 λ 控制对照 (temporal 可训练 2.4M→4.4M)。
    """
    if name in ('temporal_context.pos_embed', 'temporal_context.cls_token'):
        return True
    return name.startswith(('temporal_context.standard_encoder.',
                            'temporal_context.local_cnn.',
                            'temporal_context.norm.'))


def run_phase3_v24_temporal(trainer, args, clip_len=16, batch_size=4,
                            num_workers=4, class_weights=None):
    """V22 Phase 3 时序因果推理训练循环 (CTTA + UASC + CDCR + PTCL)

    与 V21 run_phase3_temporal 的核心差异:
      1. 模型: V24TemporalModel (CTTA 增强) 替代 V18TemporalModel
      2. 损失: V22TemporalMultiTaskLoss (UASC + PTCL + CDCR) 替代 TemporalMultiTaskLoss
      3. 前向: 传递 type_ids 给 CTTA 注意力偏置
      4. 冻结: set_phase() 控制 PTCL 课程和 CTTA 冻结/解冻

    Args:
        trainer: V18Trainer 实例
        args: 命令行参数
        clip_len: Clip 帧数
        batch_size: Clip batch size per GPU
        num_workers: DataLoader workers
        class_weights: 类别权重
    """
    # ── DDP 初始化 ──
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_ddp = world_size > 1

    # 全局种子 (与 V21 对齐)
    import random as _seed_random
    _seed = getattr(args, 'seed', 0) or 0
    _seed_random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    try:
        import numpy as _np
        _np.random.seed(_seed)
    except ImportError:
        pass
    if rank == 0:
        print(f"  V22: 全局种子 seed={_seed}")

    _own_pg = False
    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
            _own_pg = True
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        if rank == 0:
            print(f"  V22 DDP: world_size={world_size}, rank={rank}")
    else:
        device = torch.device(f'cuda:{args.device.split(",")[0]}'
                              if torch.cuda.is_available() else 'cpu')

    # G-304(F7): 致命中止的统一出口 —— 先摘进程组再退。
    #   根因: 此前 6 处闸 (数据集不变量 / val 去增强 / 宇宙对齐 / 留出集为空 / ...)
    #   都是裸 sys.exit(2) ⇒ 只有报错的那个 rank 退出, 对端 rank 还卡在下一个
    #   collective 上等 NCCL watchdog (默认 30min) ⇒ 期间两张卡的显存与 /dev/shm
    #   都不释放, 正是 #245 记过的那类"DDP 孤儿占卡"的直接成因。
    #   monitor 早停路径 (:1610 broadcast 停手) 本来就有对齐处理, 这里补齐"故障路径"。
    def _fatal(_msg, _code=2):
        try:
            if dist is not None and dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as _e:
            print(f"  [FATAL rank{rank}] destroy_process_group 异常 (忽略, 继续退出): {_e}",
                  flush=True)
        print(f"  [FATAL rank{rank}] {_msg}", flush=True)
        sys.exit(_code)

    # ── 构建 V22 增强时序模型 ──
    if not isinstance(trainer.model, torch.nn.Module):
        if rank == 0:
            print(f"  V22: 构建检测模型...")
        trainer.setup_model()
        trainer.model = trainer.model.to(device)
    det_model = de_parallel(trainer.model)

    # V22: 使用 build_temporal_model 构建增强模型
    temporal_model = build_temporal_model(
        detection_model=det_model,
        clip_len=clip_len,
        ctta_config=CTTA_CONFIG,
        uasc_config=UASC_CONFIG,
        cdcr_config=CDCR_CONFIG,
        ptcl_config=PTCL_CONFIG,
    ).to(device)

    if rank == 0:
        n_params = sum(p.numel() for p in temporal_model.parameters())
        n_trainable = sum(p.numel() for p in temporal_model.parameters() if p.requires_grad)
        print(f"  V22 模型: 总参数 {n_params/1e6:.1f}M, 可训练 {n_trainable/1e6:.1f}M")

    # ── 加载前一阶段权重 ──
    prev_phase_map = {'3b': '3a', '3c': '3b'}
    prev_phase = prev_phase_map.get(args.phase)
    if prev_phase:
        allow_fallback = getattr(args, 'allow_fallback', False)
        prev_ckpt = Path(args.project) / f'{args.name}_p{prev_phase}' / 'best_temporal.pt'
        if not prev_ckpt.exists():
            # 回退到 weights/best.pt (V21 产物)
            prev_ckpt = Path(args.project) / f'{args.name}_p{prev_phase}' / 'weights' / 'best.pt'
        if prev_ckpt.exists():
            try:
                state = torch.load(prev_ckpt, map_location=device, weights_only=False)
                model_state = state.get('model_state_dict', state.get('model', state))
                own_state = temporal_model.state_dict()
                loaded = {k: v for k, v in model_state.items()
                          if k in own_state and own_state[k].shape == v.shape}
                match_rate = len(loaded) / max(len(own_state), 1)
                if match_rate < 0.8 and not allow_fallback:
                    _fatal(f"V22 检查点匹配率过低: {match_rate:.1%}", 1)
                own_state.update(loaded)
                temporal_model.load_state_dict(own_state, strict=False)
                if rank == 0:
                    print(f"  V22: 加载前阶段 ({prev_phase}) 权重: "
                          f"{len(loaded)}/{len(own_state)} keys ({match_rate:.1%})")
            except Exception as e:
                if allow_fallback:
                    if rank == 0:
                        print(f"  V22 WARN: 加载失败: {e}, --allow-fallback")
                else:
                    _fatal(f"V22 加载权重失败: {e}", 1)
        elif not allow_fallback:
            _fatal(f"V22 前阶段检查点不存在: {prev_ckpt}", 1)

    # ── V22: 设置 Phase 控制创新模块 ──
    # G-278(R1): 3a 在 prev_phase_map 里没有前阶段 ⇒ 时序层是构造时随机初始化;
    # 原实现无条件 freeze_standard_layers() 把 standard_encoder/local_cnn/pos_embed/
    # cls_token/norm 钉死在随机权重上 (e3→e8 这 34 个张量逐位不变, 而 forward 每步都过)。
    # 只有确实继承了前阶段时序权重 (3b/3c) 才允许冻结。
    _freeze_std = bool(prev_phase)
    # G-303(#271b): 检测支路的冻结**不能**挂在 _freeze_std 上。教训＝一个谓词管两个
    # 对象: _freeze_std 的取舍理由是"时序 standard 层可能是随机初始化" (G-278(R1)),
    # 与检测支路无关; 而相位3 三个子阶段的检测支路**一律**来自 P1/P2 已训权重
    # (3a 链自 --chain-from 的 P2 件, 3b/3c 链自上一子阶段的 best_temporal.pt)。
    # 实测反证: 3a 的横幅 "冻结标准层=False" ⇒ 挂在它上面的三件套在 3a 全程不生效,
    # 而 §16-37 的 mAP 0.63122→0.00284 塌方恰恰发生在 3a。故此处独立成谓词,
    # 并由下面的 num_batches_tracked>0 在体闸证明"确实继承了已训检测支路"。
    _freeze_det = True
    temporal_model.set_phase(args.phase, freeze_standard=_freeze_std)
    if rank == 0:
        _tr = sum(p.numel() for p in temporal_model.parameters() if p.requires_grad)
        print(f"  V22 Phase {args.phase}: 创新模块状态已设置 "
              f"(冻结标准层={_freeze_std}, 可训练 {_tr/1e6:.1f}M, "
              f"检测支路冻结={_freeze_det})")

    # ── G-303(#271): 检测支路真冻结 (三件套之一 requires_grad) ──
    # 根因: freeze_standard_layers() 冻的是 CTTATemporalContext 内的
    # standard_encoder/local_cnn/pos_embed/cls_token/norm (temporal_model.py:362/390),
    # 而 detection_model 是 V24TemporalModel 的兄弟属性 (:617) ⇒ "冻结标准层=True"
    # 横幅从未覆盖检测支路; set_phase('3b'/'3c') 还额外 unfreeze_all() 反冻。
    # 实测代价: 582/762 张量漂移 (BN buffer 318 + BN 仿射 176 + conv 88 + cls 54),
    # 标准 mAP50 0.63122→0.00013, 只换 BN 运行统计即可归 0 (G-302 干预测试)。
    # A/B (runs/ab_v3_g303) 显示三个头绑死在漂移特征上 ⇒ 零重训路径不成立。
    _bn_base_sum, _bn_base_n = 0, 0
    if _freeze_det:
        _det_mod = de_parallel(temporal_model).detection_model

        def _det_bn_now():
            """检测支路 BN 计数器总和 (单次同步读数, 在体闸的被测标量)。"""
            return int(torch.stack(_bn_bufs).sum()) if _bn_bufs else -1

        _det_np = sum(1 for _ in _det_mod.parameters())
        _det_ne = sum(p.numel() for p in _det_mod.parameters())
        _det_mod.requires_grad_(False)
        _det_mod.eval()
        _still = [n for n, p in temporal_model.named_parameters()
                  if 'detection_model' in n and p.requires_grad]
        if _still:
            _fatal(f"[G-303 det-freeze] 检测支路仍有 {len(_still)} 只张量 "
                   f"requires_grad=True (示例 {_still[:3]}) ⇒ 冻结未生效, 中止")
        # 句柄一次取齐: 逐只 .item() 等于每步 318 次设备同步, 5659 batch/epoch 不可接受
        _bn_bufs = [m.num_batches_tracked for m in _det_mod.modules()
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        _bn_base_n = len(_bn_bufs)
        _bn_base_sum = _det_bn_now()
        if _bn_base_n == 0:
            print("  [G-303 det-freeze] FATAL: 检测支路找不到任何 BN 计数器张量 ⇒ "
                  "闸无被测对象 (历史值 318), 中止")
            _fatal("G-303 det-freeze: 检测支路无 BN 计数器张量, 冻结闸无被测对象")
        # 反向读法说明 (G-303 271b 复盘): 基线 sum **可以合法为 0** —— 本项目保存路径
        # 不写 num_batches_tracked (G-302 已记: "ckpt 的计数器 0 是保存路径不写, 不是
        # 没训过的证据"; 实测 P2-18 weights/best.pt 的 106 只 nbt 全 0)。闸判的是
        # "变化量", 与绝对值无关 ⇒ 这里只登记, 不设 fail。
        if rank == 0 and _bn_base_sum == 0:
            print("  [G-303 det-freeze] INFO: 入口 BN 计数器基线 sum=0 "
                  "(与 P2-18 种子一致, 系保存路径不写计数器) ⇒ 闸改判 Δ 而非绝对值")
        if rank == 0:
            _tr2 = sum(p.numel() for p in temporal_model.parameters() if p.requires_grad)
            print(f"  [G-303 det-freeze] 检测支路已冻结: {_det_np} 张量 / "
                  f"{_det_ne/1e6:.1f}M, BN 计数器基线 sum={_bn_base_sum} "
                  f"(nbn={_bn_base_n}) —— 本 run 全程该 sum 不得变化; "
                  f"冻结后可训练 {_tr2/1e6:.1f}M")

    # ── DDP 包装 ──
    if is_ddp:
        temporal_model = DDP(temporal_model, device_ids=[local_rank],
                             output_device=local_rank,
                             find_unused_parameters=True)

    # ── V22 增强损失函数 ──
    eff_cls_weight = class_weights if class_weights is not None else [1.0, 1.2, 1.1]
    phase_lw = get_phase3_loss_weights(args.phase)
    # C6修复: 从 V22_INNOVATION_LOSS_WEIGHTS 获取逐 Phase 创新损失权重
    # 新格式: (λ_uasc, λ_cdcr) — 二元组
    innov_w = getattr(args, '_innov_weights', None) or (0.0, 0.0)
    cdcr_lambda = innov_w[1] if args.phase in ('3b', '3c') else 0.0
    # V24: ACCIDENT 已含 VLM 标注的 16-bit 成因 (scene_labels 第2行), 且成因自
    # Phase 1 起即监督; Phase 3 通过时序上下文 (CTTA 等) 深化成因判定。
    # 历史教训 (保留备查): cause_lambda=0 会在后续阶段灾难性遗忘成因头
    # (3b 实测 92.1%→0%), 故 3a/3b/3c 全程保持成因监督。
    cause_lambda = phase_lw['cause'] if args.phase in ('3a', '3b', '3c') else 0.0
    # G-274(λ_scene 接线): 相位3 此前恒用 DEFAULT_SCENE_CLS_WEIGHT(=0.8)，而 P1/P2 的生效值是
    # config/accident_detect.yaml 的 scene_cls_weight —— P2 整轮调参收敛到 5.243，动因正是
    # G-249 定性的"scene 边界向 normal 漂移"。⇒ 两个真相源，P2 调出来的药从没进过相位3。
    # 现按 core/detection.py:574 同一读法取值 (self.data.get)，读不到才回退默认。
    try:
        lambda_scene = float(trainer.data.get('scene_cls_weight',
                                              DEFAULT_SCENE_CLS_WEIGHT))
        _lam_src = 'trainer.data[scene_cls_weight]'
    except Exception:
        lambda_scene = float(DEFAULT_SCENE_CLS_WEIGHT)
        _lam_src = 'DEFAULT_SCENE_CLS_WEIGHT(兜底)'
    criterion = V22TemporalMultiTaskLoss(
        nc_cls=3, num_causes=9,
        lambda_scene=lambda_scene,
        lambda_tta=phase_lw.get('tta', 0.5),  # W2修复: 从 Phase 配置读取
        lambda_type=phase_lw['type'],
        lambda_cdcr=cdcr_lambda,
        lambda_cause=cause_lambda,
        cls_weight=eff_cls_weight,  # C2修复: 传入类别权重
        uasc_config=UASC_CONFIG,
        cdcr_config=CDCR_CONFIG,
        ptcl_config=PTCL_CONFIG,
    ).to(device)

    if rank == 0:
        print(f"  V22 Phase {args.phase} 损失: "
              f"scene={lambda_scene} [{_lam_src}], tta={phase_lw.get('tta', 0.5)}, "
              f"type={phase_lw['type']}, cdcr={cdcr_lambda}, cause={cause_lambda}")

    # ── 优化器 (分层学习率) ──
    base_lr = args.lr0 or 5e-4
    # G-286(暴露组降档): R1 新暴露的 standard+base 单独一档 lr=base_lr*0.05。
    # 只动 lr, 不动任何参数的 weight_decay (免衰减的那部分拆到两组里仍是 wd=0.0),
    # 也不动其余三组的 lr 数值与赛程 ⇒ 未命中参数逐步长曲线逐位相同 (复测见 §16-15-8)。
    exposed_lr_mult = 0.05
    backbone_params, temporal_params = [], []
    temporal_nodecay = []  # G-282: 时序侧 1-D 参数与 pos/cls 另组, weight_decay=0.0
    exposed_params, exposed_nodecay = [], []  # G-286: R1 暴露集 (吃 / 不吃 decay 各一列)
    for name, p in temporal_model.named_parameters():
        if not p.requires_grad:
            continue
        if 'detection_model' in name:
            backbone_params.append(p)
        elif is_no_decay_param(name, p):
            if is_r1_exposed_param(name):
                exposed_nodecay.append(p)
            else:
                temporal_nodecay.append(p)
        elif is_r1_exposed_param(name):
            exposed_params.append(p)
        else:
            temporal_params.append(p)
    n_trainable = (sum(p.numel() for p in backbone_params) + sum(p.numel() for p in temporal_params)
                   + sum(p.numel() for p in temporal_nodecay)
                   + sum(p.numel() for p in exposed_params) + sum(p.numel() for p in exposed_nodecay))
    _all_nodecay = temporal_nodecay + exposed_nodecay  # G-282 那条横幅的口径保持全量
    _n_exposed = len(exposed_params) + len(exposed_nodecay)
    _e_exposed = sum(p.numel() for p in exposed_params) + sum(p.numel() for p in exposed_nodecay)
    _e_temporal = n_trainable - sum(p.numel() for p in backbone_params)
    if rank == 0:
        print(f"  V22 可训练: {n_trainable/1e6:.1f}M "
              f"(backbone={sum(p.numel() for p in backbone_params)/1e6:.1f}M, "
              f"temporal={_e_temporal/1e6:.1f}M)")
        print(f"  [G-282] 免衰减组: {len(_all_nodecay)}张量 / "
              f"{sum(p.numel() for p in _all_nodecay)}个 ⇒ weight_decay=0.0 "
              f"(其余两组 lr 逐字不变, detection_model 整条不参与本补丁)")
        print(f"  [G-286] R1暴露组: {_n_exposed}张量 / {_e_exposed}个 "
              f"(其中免衰减 {len(exposed_nodecay)}张量) ⇒ lr=base_lr*{exposed_lr_mult:g} "
              f"⇒ 峰值 {base_lr*exposed_lr_mult:.3e}; 其余各组 lr 曲线逐字不动")

    param_groups = [
        {'params': temporal_params, 'lr': base_lr},
        {'params': temporal_nodecay, 'lr': base_lr, 'weight_decay': 0.0},
        {'params': backbone_params, 'lr': base_lr * 0.05},
        {'params': exposed_params, 'lr': base_lr * exposed_lr_mult},
        {'params': exposed_nodecay, 'lr': base_lr * exposed_lr_mult, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.15, betas=(0.9, 0.999))  # V25-fix: weight_decay 0.10→0.15 减少过拟合

    # ── 数据集 (复用 V21 逻辑) ──
    # G-303(A0): 主路径历史上**每个 run 都恒崩**, 被下面的宽 except 静默吞掉。
    # 崩点在 ultralytics/data/base.py:131
    #   self.max_buffer_length = min((self.ni, self.batch_size*8, 1000)) if self.augment else 0
    # 而 build_dataset 的 batch 形参默认 None ⇒ augment=True 时 None*8 TypeError。
    # 复现: /tmp/g303/a0_repro.py (batch=None 必崩 / batch=16 正常)。
    # 实测痕迹: 3c 日志 train_v26_dw_lane_g286_p3c.log:3518 打完"场景标签加载成功"
    # 紧跟 :3520 "V22 数据集构建失败: unsupported operand ... 回退...", 标签被扫了两遍。
    # 修法 = 主路径显式传 batch = 回退分支实际生效的那个值 (get_dataloader 的
    # batch_size 默认 16, 与训练 batch 无关, 只喂 max_buffer_length), 使两条分支产出
    # 逐属性相同的数据集 ⇒ 训练数值零变化; 分叉本身改为"带 traceback + 在体不变量闸"。
    _DS_BATCH_FOR_BUFFER = 16  # 必须等于 Trainer.get_dataloader 的 batch_size 默认值
    _ds_primary_tb = None
    try:
        data_root = trainer._data_root or trainer.data.get('path', '')
        train_subdir = trainer.data.get('train', 'images/train')
        train_img_path = (train_subdir if os.path.isabs(train_subdir)
                          else os.path.join(data_root, train_subdir))
        base_dataset = trainer.build_dataset(train_img_path, mode="train",
                                             batch=_DS_BATCH_FOR_BUFFER,
                                             augment=True, fraction=1.0)  # V25-fix: 启用数据增强减少过拟合
        if rank == 0:
            print(f"  V22 数据集: {train_img_path}")
    except Exception as e:
        _ds_primary_tb = traceback.format_exc()
        LOGGER.warning(f"V22 数据集构建失败: {e}, 回退...\n{_ds_primary_tb}")
        try:
            _dr = trainer._data_root or trainer.data.get('path', '')
            fallback_path = os.path.join(_dr, trainer.data.get('train', 'images/train'))
            _loader = trainer.get_dataloader(fallback_path, mode='train')
            base_dataset = _loader.dataset
            data_root, train_subdir = _dr, trainer.data.get('train', 'images/train')
        except Exception as e2:
            raise RuntimeError(f"V22 Phase 3 数据集构建失败: {e}\n回退也失败: {e2}") from e2

    # G-303(A0) 在体不变量闸: 不论走哪条分支, train 通道必须是"增强开 + 全量 + 非矩形"
    # 且 buffer batch 与预期同源 —— 静默换管道 (fraction/augment/rect) 从此开机即红。
    _dsi = {k: getattr(base_dataset, k, None) for k in
            ('augment', 'batch_size', 'max_buffer_length', 'fraction', 'rect')}
    _dsi['ni'] = len(base_dataset)
    _dsi_exp = {'augment': True, 'batch_size': _DS_BATCH_FOR_BUFFER,
                'max_buffer_length': min(_dsi['ni'], _DS_BATCH_FOR_BUFFER * 8, 1000),
                'fraction': 1.0, 'rect': False, 'ni': _dsi['ni']}
    if _dsi != _dsi_exp:
        print(f"  [G-303 ds-invariant] FATAL: train 数据集有效配置偏离预期\n"
              f"    期望 {_dsi_exp}\n    实得 {_dsi}")
        _fatal("G-303 ds-invariant: train 数据集有效配置偏离预期 (管道被静默换掉)")
    if rank == 0:
        print(f"  [G-303 ds-invariant] train 通道 OK: "
              f"{base_dataset.__class__.__name__} ni={_dsi['ni']} "
              f"augment={_dsi['augment']} batch_size={_dsi['batch_size']} "
              f"max_buffer_length={_dsi['max_buffer_length']} "
              f"fraction={_dsi['fraction']} rect={_dsi['rect']} | "
              f"主路径异常={'无' if _ds_primary_tb is None else '有(traceback 已打印)'}")

    # ── G-304(F1): 相位3 train 通道逐帧几何/混合增强归零 ──
    # 根因: ClipDataset.__getitem__ (core/dataset.py:756) 对 clip 的 16 帧逐帧调
    #   self.base[fi] → BaseDataset.__getitem__:397 → self.transforms(...) ⇒
    #   ultralytics 的 v8_transforms 是**每帧独立抽签**的。原 hyp (train.py:585-596)
    #   在相位3 训练通道上实际生效的是: fliplr=0.5 / degrees=5.0 / shear=0.5 /
    #   translate=0.1 / scale=0.5 / perspective=0.001 / mixup=0.15 (mosaic 已被
    #   相位3 override 成 0)。后果: 相邻帧以独立概率被镜像/旋转/错切, 15% 的帧槽
    #   被 MixUp 与 RAM buffer 里**另一段视频**的帧混合 (scene/type/cause/bbox 标签
    #   一并混叠)。时序头看到的不是时序正则, 是"目标瞬移 + 别人的事故叠进来"的矛盾
    #   监督 (core/phase3_legacy.py:272 早已注明逐帧翻转破坏时序一致性, 只是训练通道
    #   从没据此收紧; val 通道 G-303(A2) 已 augment=False)。
    # 保留: RandomHSV (h0.015/s0.8/v0.5) —— 逐帧光度扰动不移动目标、不改身份, 是
    #   正当正则; erasing 对 detect 任务本就不进管线 (core/detection.py:782-786 已注明)。
    # 注意 build_transforms 在 __init__ 内已经跑过 (data/base.py:148), **只改 hyp 数值
    #   不重建 = 零动作** ⇒ 必须换 hyp 并重建 transforms; 且用浅拷贝, 绝不改
    #   trainer.args 本体 (相位1/2 与 val 通道共用同一个 namespace)。
    #   hyp 的真身在 `trainer.args` —— core/detection.py:508 就是把它当 hyp 传进 dataset 的;
    #   **dataset 对象自己不存 hyp** (实测 core/dataset.py 里没有 self.args, 且 8.4.112 的
    #   BaseDataset 已无 set_transforms ⇒ transforms 只在 __init__ 里建一次)。
    #   2026-11-02 23:39 首跑就是踩这个：写 `base_dataset.args` ⇒ AttributeError 直接撞死发车。
    import copy as _copy
    from ultralytics.data.augment import (Compose as _ACompose, CopyPaste as _ACopyPaste,
                                          CutMix as _ACutMix, Mosaic as _AMosaic,
                                          MixUp as _AMixUp, RandomFlip as _ARandomFlip,
                                          RandomPerspective as _ARandomPersp)
    _P0_AUG = (_AMosaic, _ACopyPaste, _AMixUp, _ACutMix, _ARandomFlip)
    _AFFINE_KEYS = ('degrees', 'translate', 'scale', 'shear', 'perspective')
    _TRAIN_AUG_OFF = {'mosaic': 0.0, 'copy_paste': 0.0, 'flipud': 0.0, 'fliplr': 0.0,
                      'mixup': 0.0, 'cutmix': 0.0,
                      **{k: 0.0 for k in _AFFINE_KEYS}}
    _hyp_src = trainer.args                              # dataset 的 hyp 就是它 (detection.py:508)
    _hyp_before = {k: getattr(_hyp_src, k, '<缺键>') for k in _TRAIN_AUG_OFF}
    _hyp_tr = _copy.copy(_hyp_src)                       # 浅拷贝: 绝不动 trainer.args 本体
    for _k, _v in _TRAIN_AUG_OFF.items():
        setattr(_hyp_tr, _k, _v)
    base_dataset.transforms = base_dataset.build_transforms(hyp=_hyp_tr)

    # 在体闸: 遍历重建后的管线, 断言"会移动/混合像素"的算子全部不可触发。
    # p 门实测在 BaseMixTransform.__call__:341 (Mosaic/MixUp/CutMix/CopyPaste) 与
    # RandomFlip.__call__:1572; RandomPerspective 无 p 门, 只吃五个数值 ⇒ 分别校验。
    def _walk_transforms(_node):
        _out = []
        for _sub in (getattr(_node, 'transforms', None) or []):   # Compose 子节点
            _out.extend(_walk_transforms(_sub))
        _pre = getattr(_node, 'pre_transform', None)              # Mix* 内嵌 pre_transform
        if _pre is not None:
            _out.extend(_walk_transforms(_pre))
        _out.append(_node)
        return _out

    _flat = _walk_transforms(base_dataset.transforms)
    _gate_bad = []
    for _t in _flat:
        if isinstance(_t, _P0_AUG) and float(getattr(_t, 'p', 0.0) or 0.0) != 0.0:
            _gate_bad.append(f"{type(_t).__name__}.p={getattr(_t, 'p', None)}")
        if isinstance(_t, _ARandomPersp):
            _aff = {k: getattr(_t, k, None) for k in _AFFINE_KEYS}
            if any((v if isinstance(v, (int, float)) else 1.0) not in (0, 0.0)
                   for v in _aff.values()):
                _gate_bad.append(f"RandomPerspective.{_aff}")
    if _gate_bad:
        print(f"  [G-304 train-aug] FATAL: 训练通道仍会逐帧移动/混合像素 ⇒ 时序矛盾信号\n"
              f"    违规算子: {_gate_bad}\n"
              f"    原 hyp: {_hyp_before}\n"
              f"    ⇒ 宁可不跑, 也不要拿 16 帧各自镜像/各自 MixUp 的 clip 训 38 小时")
        _fatal("G-304 train-aug: 训练通道仍会逐帧移动/混合像素 (时序矛盾信号)")
    if rank == 0:
        _hsv = next((t for t in _flat if type(t).__name__ == 'RandomHSV'), None)
        print(f"  [G-304 train-aug] 训练通道几何/混合增强已归零 (clip 时序一致性恢复):\n"
              f"    改前 hyp={_hyp_before}\n"
              f"    管线算子={[type(t).__name__ for t in base_dataset.transforms.transforms]}\n"
              f"    RandomHSV 保留={None if _hsv is None else (_hsv.hgain, _hsv.sgain, _hsv.vgain)} "
              f"(光度扰动, 不动目标位置/身份)\n"
              f"    检测框 boxes_seq 与图像同源: transform 内 apply_instances 仍随同一 M 变换")

    ds_clip_stride = trainer._ds_config.get('clip_stride', 8) if trainer._ds_config else 8
    ds_fps = trainer._ds_config.get('fps', 10.0) if trainer._ds_config else 10.0
    _raw_clip_dataset = ClipDataset(base_dataset, clip_len=clip_len,
                                    stride=ds_clip_stride, max_objs=8)
    # V22-fix(tile-clip): tile 与原帧场景标签解耦 (s7 tile 掩码修复的结论),
    # 且每组 tile 仅 4 帧 → 填充退化为"重复帧 clip", 占全部 clip ~80%,
    # 既主导时序采样又注入错误场景监督 → 从 Phase 3 排除。
    import re as _re
    _base_labels = _raw_clip_dataset.base.labels
    _tile_idx = {i for i in range(len(_base_labels))
                 if _re.search(r'_t\d{2}\.[A-Za-z]+$',
                               os.path.basename(str(_base_labels[i]['im_file'])))}
    if _tile_idx:
        _n_clips_before = len(_raw_clip_dataset.clips)
        _raw_clip_dataset.clips = [c for c in _raw_clip_dataset.clips
                                   if not any(fi in _tile_idx for fi in c[0])]
        if rank == 0:
            print(f"  V22-fix(tile-clip): 排除 tile 退化 clip, "
                  f"{_n_clips_before} → {len(_raw_clip_dataset.clips)}")
    # G-303(A2): val 通道去增强。原实现 val_clip 与 train_clip 共用**同一个**
    # augment=True 的 base_dataset (ClipDataset.__getitem__ → self.base[fi] →
    # BaseDataset.__getitem__:397 → self.transforms(...)), 实测 hyp =
    # mixup 0.15 / fliplr 0.5 / hsv_s 0.8 / hsv_v 0.5 / degrees 2 / scale 0.3 /
    # shear 0.5 (train.py:585-596 + 相位3 overrides mosaic=0.0) ⇒ 每轮 [Val] 里约
    # 15% 帧是与 RAM buffer 里另一张训练图混合后的产物 (MixUp 从 dataset.buffer 取
    # 配对样本), 场景/类型标签随之失真 —— in-run 效果面当时不是可用仪器。
    # 修法 = 另建一份 augment=False 的同源数据集; mode 仍取 'train' ⇒ 读同一批
    # images/train + scene_labels/train, **train/val 拆分一字未动**, 只差变换开关。
    try:
        val_base_dataset = trainer.build_dataset(train_img_path, mode="train",
                                                 batch=_DS_BATCH_FOR_BUFFER,
                                                 augment=False, fraction=1.0)
    except Exception as _e:
        print(f"  [G-303 val-deaugment] FATAL: 去增强 val 通道建立失败 ({_e}) —— "
              f"宁可不跑, 也不要用被随机增强污染的效果面跑 38 小时")
        traceback.print_exc()
        _fatal("G-303 val-deaugment: 去增强 val 通道建立失败 (效果面不可用)")
    _raw_val_clip_dataset = ClipDataset(val_base_dataset, clip_len=clip_len,
                                        stride=ds_clip_stride, max_objs=8)
    if _tile_idx:
        _raw_val_clip_dataset.clips = [c for c in _raw_val_clip_dataset.clips
                                       if not any(fi in _tile_idx for fi in c[0])]
    # 宇宙对齐闸: 去增强的 val 数据集必须与 train 数据集逐 clip 同构 (同视频 id、
    # 同帧号序列、同总数), 否则 val 读数与训练留出集不同源 ⇒ 开机即中止。
    _uc_train = [(c[1], tuple(c[0])) for c in _raw_clip_dataset.clips]
    _uc_val = [(c[1], tuple(c[0])) for c in _raw_val_clip_dataset.clips]
    if _uc_train != _uc_val:
        _d0 = next((i for i, (_a, _b) in enumerate(zip(_uc_train, _uc_val))
                    if _a != _b), None)
        _fatal(f"[G-303 val-deaugment] val/train clip 宇宙不对齐 "
               f"(n={len(_uc_train)} vs {len(_uc_val)}, 首个差异位 {_d0})")
    val_clip_dataset = wrap_clip_dataset(_raw_val_clip_dataset, default_agreement=1.0)
    if rank == 0:
        print(f"  [G-303 val-deaugment] val 通道 augment=False (MixUp/Flip/HSV/"
              f"Perspective 全关), clip 宇宙与 train 侧逐位对齐 n={len(_uc_train)}; "
              f"train 通道增强一字未动")
    # V22-fix(fps): s6 为 stride-4 子集 (帧间隔 0.4s), 配置 fps=10 是源帧率;
    # 按帧号间隔估计有效 fps, 校正检测延迟统计口径。
    try:
        _diffs = []
        for _fi_list, _vid, _ in _raw_clip_dataset.clips[:200]:
            _fnos = []
            for _fi in _fi_list:
                _m = _re.search(r'_(\d+)\.[A-Za-z]+$',
                                str(_base_labels[_fi]['im_file']))
                if _m:
                    _fnos.append(int(_m.group(1)))
            _d = [b - a for a, b in zip(_fnos, _fnos[1:]) if b > a]
            if _d:
                _diffs.append(_d[len(_d) // 2])
        if _diffs:
            _eff_stride = sorted(_diffs)[len(_diffs) // 2]
            if _eff_stride > 1:
                ds_fps = ds_fps / _eff_stride
                if rank == 0:
                    print(f"  V22-fix(fps): 子采样帧间隔 {_eff_stride} → "
                          f"有效 fps={ds_fps:.2f} (延迟统计口径)")
    except Exception:
        pass
    # V22 数据管道扩展: 包装为 V24ClipDataset (增加 source_domain + agreement)
    clip_dataset = wrap_clip_dataset(_raw_clip_dataset, default_agreement=1.0)
    if rank == 0:
        print(f"  V22: ClipDataset 已包装为 V24ClipDataset (source_domain + agreement)")

    # 视频级拆分 (与 V21 对齐)
    import random as _random
    _vids = sorted({c[1] for c in clip_dataset.clips})
    _rng = _random.Random(42)
    _rng.shuffle(_vids)
    # V22-fix(val-cause): val 视频必须与训练集共享 cause 词汇, 否则
    # Cause Recall 是结构性零样本问题 (vid07: bit8/10 仅存在于该视频),
    # 3a 选型指标 (0.6*frame + 0.4*cause_recall) 失效
    try:
        import collections as _coll
        _sl_dir = Path(data_root) / 'scene_labels' / 'train'
        _vidbits = _coll.defaultdict(set)
        _bit_bad = 0
        if _sl_dir.is_dir():
            for _f in _sl_dir.iterdir():
                if not _f.name.endswith('.txt'):
                    continue
                try:
                    _lines = _f.read_text().split()
                    if len(_lines) >= 2 and int(_lines[0]) >= 2:
                        _mask = int(_lines[1])
                        _vid = _re.sub(r'_\d+\.txt$', '', _f.name)
                        for _i in range(16):
                            if (_mask >> _i) & 1:
                                _vidbits[_vid].add(_i)
                except Exception:
                    _bit_bad += 1
                    pass
        if _vidbits and len(_vids) > 1:
            def _isolated(_v):
                _others = set()
                for _u in _vids:
                    if _u != _v:
                        _others |= _vidbits.get(_u, set())
                return len(_vidbits.get(_v, set()) & _others) == 0
            # 稳定排序: 孤立 cause 视频排后, 其余保持 shuffle 顺序
            _vids = sorted(_vids, key=_isolated)
            if rank == 0:
                _iso = [v for v in _vids if _isolated(v)]
                print(f"  V22-fix(val-cause): cause 词汇检查 — "
                      f"隔离视频={_iso or '无'}, 候选顺序前3={_vids[:3]}, "
                      f"位表解析失败文件={_bit_bad}")
    except Exception as _e:
        # G-304(F11): 原来这里 LOGGER.warning("跳过") ⇒ 静默 fail-open。排序一失败,
        #   _vids 保持原序 ⇒ _vval = _vids[:10%] 抽到的视频集合整个换掉 ⇒ 留出集换人,
        #   而 val 面是所有历史轮次比较的分母。留出集变动必须是"响亮故障", 不是跳过。
        #   (实测: g303/g282/g286 三份在体日志该 warning 命中数 0 ⇒ 本闸不会误杀合法启动)
        _fatal(f"V22-fix(val-cause) cause 词汇排序失败: {_e.__class__.__name__}: {_e} "
               f"⇒ 无法保证注册过的留出集, 中止")
    # V23-fix(val-单视频): 视频数 >=5 时 val 至少取 2 个视频 —
    # 单视频验证 (9 视频×10%=1) 使 val Frame Acc 波动 ±20pp,
    # 检查点选型与停滞判定被单视频特性主导 (v23_s8 3a 实测)
    _n_vval = max(2, int(len(_vids) * 0.1)) if len(_vids) >= 5 \
        else max(1, int(len(_vids) * 0.1))
    _vval = set(_vids[:_n_vval])
    # G-304(F11): 留出集指纹落日志。val 面是跨轮比较的分母, 而它由
    #   "sorted→shuffle(42)→cause 隔离重排→取前 10%" 四步串出来 —— 任一步被改动,
    #   读数口径就换了却不自曝。指纹 (md5(sorted(_vval)) 前 12 位) 让"换了人"
    #   在开机横幅上就能对账, 不用事后反推。
    import hashlib as _hl
    _vfp = _hl.md5('\n'.join(sorted(_vval)).encode()).hexdigest()[:12]
    if rank == 0:
        print(f"  [G-304 holdout] val 视频数={len(_vval)}/{len(_vids)} "
              f"指纹={_vfp} 首个={sorted(_vval)[0]}", flush=True)

    class _VideoSubset(torch.utils.data.Dataset):
        def __init__(self, ds, vset):
            self.ds = ds
            self.idxs = [i for i, c in enumerate(ds.clips) if c[1] in vset]
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, i):
            return self.ds[self.idxs[i]]

    # G-323(#9・E4 训练域矩阵): 按【声明判据】裁剪训练视频集 —— env 未设 ⇒ 本块零执行,
    #   基线路径一字未动。判据唯一来源 paper_domain_report.frame_domain
    #   (= core/detection.py:993 basename.startswith("real_")), 不在此复制前缀串
    #   (§3 强制口径: 以"域"为轴的贡献必须先声明判据, 且与所报指标同源)。
    #   与盘上另两把尺的分叉必须写清, 否则三处读数会被当同一件事:
    #     · :692 "V22 域分布" 与 train_data.py:60-64 CDCR 的 source_domain
    #       都用 startswith('synthetic'/'synth') ⇒ realv/tum/tv 判给 real;
    #     · 本闸用声明判据 ⇒ realv/tum/tv 判给 syn (train 侧实测 10,054 clip 分歧)。
    _e4_dom = str(os.environ.get('V26_E4_TRAIN_DOMAIN', '') or '').strip()
    _e4_cap = int(os.environ.get('V26_E4_TRAIN_CAP', '0') or 0)
    _e4_clip_cap = int(os.environ.get('V26_E4_TRAIN_CLIP_CAP', '0') or 0)
    _e4_salt = int(os.environ.get('V26_E4_TRAIN_SALT', '0') or 0)
    _e4_vids = set(_vids) - _vval
    if _e4_dom:
        from paper_domain_report import frame_domain as _e4_fd
        if _e4_dom not in ('real', 'syn'):
            _fatal(f"[G-323 E4] V26_E4_TRAIN_DOMAIN={_e4_dom!r} 只接受 real|syn")
        _e4_n_all = len(_e4_vids)
        _e4_sel = {v for v in _e4_vids if _e4_fd(str(v)) == _e4_dom}
        _e4_n_bucket = len(_e4_sel)
        if _e4_cap and _e4_n_bucket > _e4_cap:
            import random as _e4_r
            _e4_sel = set(_e4_r.Random(_e4_salt).sample(sorted(_e4_sel), _e4_cap))
        if not _e4_sel:
            _fatal(f"[G-323 E4] 桶 {_e4_dom} 选出 0 个训练视频 ⇒ 无数据可训, 中止")
        _e4_vids = _e4_sel
    train_clip = _VideoSubset(clip_dataset, _e4_vids)
    # G-323(#9·E4 补口): V26_E4_TRAIN_CAP 对齐的是**视频数**, 而两个桶的 clip/视频 差 ~10×
    #   (real 桶 1,985 视频→2,062 clip; syn 桶 1,624 视频→20,576 clip) ⇒ 只按视频减量,
    #   "域"与"量"会被同时改掉, 单因子对照不成立。这条按 **clip 数**减量, 让两臂数据量真对齐。
    if _e4_clip_cap and not _e4_dom:
        _fatal("[G-323 E4] V26_E4_TRAIN_CLIP_CAP 只在 V26_E4_TRAIN_DOMAIN 已设时才允许生效 "
               "⇒ 否则等于悄悄把全量训练集减量, 基线路径不许被这条改动碰到")
    if _e4_clip_cap and len(train_clip) > _e4_clip_cap:
        import random as _e4_r2
        _e4_keep = sorted(_e4_r2.Random(_e4_salt).sample(range(len(train_clip.idxs)), _e4_clip_cap))
        train_clip.idxs = [train_clip.idxs[_i] for _i in _e4_keep]
        if len(train_clip) != _e4_clip_cap:
            _fatal(f"[G-323 E4] clip 减量没落地: 声明 cap={_e4_clip_cap} 实得 {len(train_clip)}")
    if _e4_dom:
        # 后置条件闸: 不断言"我过滤了", 只断言过滤的**结果** —— 选中的 clip 视频必须全在声明桶内,
        # 且留出集 _vval 一条都不许卷进训练 (否则分域读数与基线不同宇宙)。
        _e4_off = sum(1 for _i in train_clip.idxs
                      if _e4_fd(str(train_clip.ds.clips[_i][1])) != _e4_dom)
        _e4_leak = len(_vval & _e4_vids)
        if _e4_off or _e4_leak:
            _fatal(f"[G-323 E4] 桶外 clip={_e4_off} / 留出集泄漏={_e4_leak} "
                   f"⇒ 过滤器没有牙齿, 中止")
        # CDCR 生死自曝: 用 CDCR 自己的判据 (train_data.py:60-64) 数桶内还剩多少 domain=1
        _e4_cdcr1 = sum(1 for v in _e4_vids if str(v).startswith(('synthetic', 'synth')))
        if rank == 0:
            print(f"  [G-323 E4] 训练域裁剪 声明判据=real_前缀 桶={_e4_dom} "
                  f"视频={len(_e4_vids)}/{_e4_n_all} (桶内共 {_e4_n_bucket}) "
                  f"cap={_e4_cap or '无'} clip_cap={_e4_clip_cap or '无'} salt={_e4_salt} "
                  f"clip={len(train_clip)} 覆盖视频={len({train_clip.ds.clips[_i][1] for _i in train_clip.idxs})} "
                  f"后置闸=桶外 {_e4_off}/泄漏 {_e4_leak} (在 clip 减量之后复核)", flush=True)
            print(f"  [G-323 E4] CDCR 后果自曝: 本臂按 CDCR 判据 domain=1(synthetic 前缀) 视频={_e4_cdcr1} "
                  f"/ domain=0={len(_e4_vids) - _e4_cdcr1} ⇒ domain=1 为 0 时跨域配对退化, "
                  f"本臂不是\"只改数据构成\"的单因子 (读数须带这条)", flush=True)
    # G-303(A2): val 取自去增强通道 (_vval 一字未动 ⇒ 留出集与历史同源);
    # 旧选择通道 clip_dataset 仍留着做逐 clip 对账, 数量不等即中止。
    val_clip = _VideoSubset(val_clip_dataset, _vval)
    _val_aug_cnt = sum(1 for c in clip_dataset.clips if c[1] in _vval)
    if len(val_clip) != _val_aug_cnt:
        _fatal(f"[G-303 val-deaugment] 去增强 val 选出 {len(val_clip)} clip, "
               f"与增强通道同口径应有 {_val_aug_cnt} ⇒ 留出集不一致")
    if rank == 0:
        print(f"  V22 Clip: train={len(train_clip)} val={len(val_clip)}")
        # W10修复(Warning) + V23-fix(cdcr-误报): 域分布验证。
        # 原实现加载前 200 个 clip 采样 — clip 按文件名字母序排列 (real_ < synthetic_),
        # s6 数据下恒得 real=200/synth=0, 误报 "CDCR 无法生效" (3b/3c 实测 cdcr≈0.03-1.1
        # 已正常生效)。改为对全量 clip 索引的 video_id 前缀精确计数 (零 I/O)。
        try:
            _all_vids = [train_clip.ds.clips[i][1] for i in train_clip.idxs]
            _n_synth = sum(1 for v in _all_vids
                           if str(v).startswith(('synthetic', 'synth')))
            _n_real = len(_all_vids) - _n_synth
            print(f"  V22 域分布 (全量 {len(_all_vids)} clips): "
                  f"real={_n_real}, synth={_n_synth}")
            if _n_synth == 0:
                LOGGER.warning(
                    "CDCR 警告: 未检测到 synthetic 域样本! "
                    "CDCR 将无法生效。请检查 video_id 格式是否包含 'synthetic' 前缀。"
                )
        except Exception:
            pass

    train_sampler = (DistributedSampler(train_clip, num_replicas=world_size,
                     rank=rank, shuffle=True) if is_ddp else None)
    # G-303(A5 诚实面): 相位3 的流是**均匀采样**, 与 sampler_ratios / SceneClassBalancedSampler
    # 无关。那个采样器只挂在 core/detection.py:535 的 get_dataloader 里 (P1/P2 通道), 相位3
    # 自建管道不走它; 历史上 A0 让主路径恒崩⇒回退路径 get_dataloader 建了采样器**又只取
    # .dataset 把采样器丢掉**, 于是每个 run 都会打印那条 "(batch 配额 10%)" 横幅却零消费
    # ⇒ 谎值面。A0 修好后主路径不再崩, 那条横幅自然消失, 这里补一句真值, 免得下轮巡检
    # 又把它当成"配平已生效"。挂账见 #271 的 A5 (接采样器=换训练流分布, 属配方级, 不在
    # 本次开机必修内 —— 本轮只带冻结一个变量, 否则 e2/e3 读数无法归因)。
    if rank == 0:
        print(f"  [G-303 A5] 相位3 训练流=均匀 {'DistributedSampler' if is_ddp else 'shuffle'}"
              f" (clip 级), sampler_ratios={getattr(trainer, '_sampler_ratios', None)} "
              f"在相位3 **不消费**; SceneClassBalancedSampler 只作用于 P1/P2 的 "
              f"get_dataloader 通道 ⇒ 类权重 [1,1.5,5] 是单腿补偿")
    clip_loader = torch.utils.data.DataLoader(
        train_clip, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_sampler = (DistributedSampler(val_clip, num_replicas=world_size,
                   rank=rank, shuffle=False) if is_ddp else None)
    val_loader = torch.utils.data.DataLoader(
        val_clip, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=num_workers, pin_memory=True, drop_last=False)

    # ── Scheduler ──
    epochs = args.epochs or 15
    warmup_epochs = 5
    accum_steps = 4
    total_steps = max(epochs, 1) * max((len(clip_loader) + accum_steps - 1) // accum_steps, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[pg['lr'] for pg in param_groups],
        total_steps=total_steps,
        pct_start=min(0.8, warmup_epochs / max(epochs, 1)),
        anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0)

    # ── AMP ──
    use_amp = getattr(args, 'amp', True) and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 训练循环 ──
    best_val_metric = 0.0
    save_dir = Path(args.project) / f'{args.name}_p{args.phase}'
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── V22 训练监控器 ──
    monitor = None
    if rank == 0:
        monitor = TrainingMonitor(
            phase=args.phase, save_dir=save_dir,
            config=V22_MONITOR_CONFIG)
        LOGGER.info(f"  V22 训练监控已启动 (日志: {monitor.log_path})")
        LOGGER.info(f"  检查点评估周期: 每 {V22_MONITOR_CONFIG['checkpoint_every']} epoch")
        LOGGER.info(f"  停滞耐心值: {V22_MONITOR_CONFIG['patience']} 检查点")

    # 断点续传
    start_epoch = 0
    if getattr(args, 'resume', None):
        _resume_path = (Path(args.resume) if isinstance(args.resume, str) and
                        Path(args.resume).is_file() else save_dir / 'last_temporal_v22.pt')
        if _resume_path.is_file():
            try:
                _rest = load_training_state(
                    str(_resume_path), temporal_model, optimizer,
                    scheduler, scaler, device=str(device))
                start_epoch = int(_rest.get('epoch', 0))
                best_val_metric = float(_rest.get('best_metric', 0.0) or 0.0)
                if rank == 0:
                    print(f"  V22 断点续传: {_resume_path} "
                          f"(epoch={start_epoch}, best={best_val_metric:.2f})")
            except Exception as e:
                if rank == 0:
                    print(f"  V22 续传失败: {e}")

    all_trainable = [p for p in temporal_model.parameters() if p.requires_grad]
    # G-278(R2): 检测分支的失败必须留话 —— 原来 except Exception: pass 让它死了也只
    # 表现为 det_loss≈0 (六行 monitor_log 全 0, 无一行说明)。
    _det_fail_seen = set()

    if rank == 0:
        eff_batch = batch_size * accum_steps * world_size
        print(f"\n{'='*60}")
        print(f"V22 Phase {args.phase} 时序训练: {epochs} epochs, "
              f"{len(clip_loader)} batches, accum={accum_steps} "
              f"(effective_batch={eff_batch}, GPUs={world_size})")
        print(f"{'='*60}")

    # W5修复(Warning): CDCR warmup 控制
    cdcr_warmup = min(CDCR_CONFIG.get('warmup_epochs', 10), max(epochs // 3, 1))

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        temporal_model.train()
        # G-303(#271) 三件套之二: BN 运行统计只认 module.training, **不看 requires_grad**
        # (负对照 /tmp/g303/freeze_preflight.py: 仅 requires_grad_(False) 时 318/318
        # BN buffer 照旧被改写), 而 temporal_model.train() 每轮把所有子模块设回 train。
        # 故每轮 train() 之后立刻把检测支路单独钉回 eval。
        if _freeze_det:
            de_parallel(temporal_model).detection_model.eval()
        epoch_losses = {'total': 0, 'scene': 0, 'cause': 0, 'type': 0,
                        # G-278(F2): 原字典没有 'det' 键 ⇒ :748 的累加循环永远收不到
                        # loss_dict['det']，monitor.log_epoch(det_loss=avg_loss.get('det',0))
                        # 恒落 0 —— "没记账"和"没监督"在盘上长得一样。补键后 det 才具备可读性。
                        'det': 0,
                        '3d_geo': 0, 'ttc_3d': 0}
        # V22 创新损失统计
        epoch_v22 = {'cdcr': 0, 'scene_uasc': 0, 'type_ptcl': 0}
        # G-302(#258): det-only 诊断累加器 —— 独立于 epoch_losses, 不参与 total/backward,
        # 只喂 monitor 的 det_loss 键 (该键历史上恒 0 是仪器死路的读数)。
        _det_diag_sum = 0.0
        _det_diag_n = 0
        _det_diag_parts = {'box_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0}
        # G-304(F4): 非有限损失的 batch 数。历史上 nan-guard 把坏 batch 从 epoch 均值里
        #   **滤掉** (:1110-1117), 而 monitor 的 NaN 红线只读均值 ⇒ "NaN=0" 永远是假绿
        #   (读不到"发生过")。这里显式计数并落面, 均值口径一字不动。
        _nonfinite_n = 0
        n_batches = 0
        # V23-fix(grad-monitor): 全局梯度范数统计 (clip_grad_norm_ 返回值, 裁剪前)
        grad_norm_sum = 0.0
        grad_norm_n = 0
        # G-280(clip-audit): clip_grad_norm_ 在 total_norm>max_norm 时是**均匀缩放**
        # (scale=max_norm/total_norm, 方向不变), 而 Adam 对梯度整体缩放近似不变 ⇒
        # 只要"每步都被裁", max_norm 的取值 (0.5 vs 1.0 vs 5) 对训练几乎无影响,
        # 真正有牙的问题是"裁剪到底有没有咬住"。旧日志只报均值, 回答不了这件事。
        # 补 min 与命中数 ⇒ 判"这个旋钮值不值得动"有实测出处, 不再靠拍。
        grad_norm_min = float('inf')
        grad_clip_hit = 0
        _clip_cap = 0.5   # V25-fix: 1.0→0.5 更强梯度裁剪 (G-280: 裁剪调用与本行打印同源取此值)

        scene_correct = scene_total = 0
        frame_correct = frame_total = 0
        lat_sum = lat_count = lat_early = 0.0

        for batch_idx, batch in enumerate(clip_loader):
            imgs = batch['img'].to(device)
            imgs = imgs.float() / 255.0 if imgs.dtype == torch.uint8 else imgs.float()
            gt_boxes = batch['boxes_seq'].to(device)
            gt_classes = batch['classes_seq'].to(device)
            obj_mask = batch['obj_mask'].to(device)
            scene_seq = batch['scene_cls_seq'].to(device)

            # V22: 从 batch 获取碰撞类型 (CTTA 需要)
            type_ids = batch.get('accident_type', None)
            if type_ids is not None:
                type_ids = type_ids.to(device, dtype=torch.long)
                # 仅 crash 帧有有效 type, 其余设为 padding_idx (7=num_types)
                scene_t = batch.get('scene_cls', batch.get('scene_cls_seq'))
                if scene_t is not None:
                    scene_t = scene_t.to(device)
                    # W1修复(Critical): 使用任意帧 crash 判定 crash clip (而非首帧)
                    # 对渐进式事故发展 clip (首帧 normal, 后续 crash) 尤为重要
                    if scene_t.dim() > 1:
                        clip_scene = (scene_t == 3).any(dim=-1).long() * 3
                    else:
                        clip_scene = scene_t
                    non_crash = (clip_scene != 3)
                    type_ids[non_crash] = 7  # padding_idx (7-class)
                    # V26-fix(G-266)(type-OOB): accident_type=-1 的 crash clip (train 侧
                    # 5,776 帧) 会直喂 CTTA 的 type_embed (表宽 NUM_TYPES+1)，CUDA
                    # embedding 对 -1 触发 device-side assert，首个 crash batch 即崩盘。
                    # 非体系内 id ⇒ 与 dropout 同义，落 padding 槽（不改变任何监督标签）。
                    type_ids[(type_ids < 0) | (type_ids >= NUM_TYPES)] = NUM_TYPES
                    # V25-fix(type-leak): crash clip 的 GT type 输入随机 dropout
                    if V25_TYPE_INPUT_DROPOUT > 0:
                        _keep = (torch.rand(type_ids.shape,
                                            device=type_ids.device)
                                 >= V25_TYPE_INPUT_DROPOUT)
                        type_ids[(~_keep) & (~non_crash)] = 7

            # C1修复(Critical): 从 batch 获取 agreement (UASC 需要)
            # ACCIDENT 数据暂无逐 clip 投票分布, 使用 None 走 UASC 回退分支
            agreement = batch.get('agreement', None)
            if agreement is not None:
                agreement = agreement.to(device, dtype=torch.float32)

            with torch.amp.autocast('cuda', enabled=use_amp):
                # V22: 前向传播 (传递 type_ids + agreement 给 CTTA/UASC)
                temporal_out = temporal_model(
                    imgs, temporal_mode=True,
                    type_ids=type_ids,
                    agreement=agreement,  # C1修复: 传入 UASC agreement
                    gt_boxes_seq=gt_boxes,
                    gt_classes_seq=gt_classes,
                    obj_mask_seq=obj_mask)

                det_out, scene_logits, tta_pred, cause_logits, type_logits, \
                    frame_scene_logits = temporal_out[:6]
                geo_3d_dict = temporal_out[6] if len(temporal_out) > 6 else None

                batch_labels = {
                    'scene_cls': batch['scene_cls'].to(device),
                    'scene_cls_seq': scene_seq,
                    'cause': batch['cause'].to(device),
                    # V22-fix(cause-标注缺失): 无标注成因的 clip 不参与成因损失
                    'cause_valid': batch['cause_valid'].to(device)
                    if 'cause_valid' in batch else
                    torch.ones_like(batch['scene_cls']).to(device),
                    'accident_type': batch['accident_type'].to(device),
                }
                # #94: scene 证据档降权 [B]（缺键 ⇒ 不传，损失走全 1.0 等价接入前）
                if 'scene_sup_weight' in batch:
                    batch_labels['scene_sup_weight'] = batch['scene_sup_weight'].to(device)

                # G-302(#258 根治): det 分支从"假监督"改成"零副作用诊断"。三条实测证据来自
                # 同一批真实 crash 帧 (8 帧 / 59 个 GT 框) 的离线探针:
                #  ①原样必抛 AttributeError: 'DetectClassifyModelV18' object has no attribute
                #    'args' —— 时序链自建 DetectionModel + load_state_dict, 不走 ultralytics
                #    的 model.args 挂载点 ⇒ monitor 里 det_loss 恒 0 是仪器死路, 不是"这支没信号"。
                #  ②补上 args 后 model(batch) 返回的是**整个 P2 多任务 total**, 且 8.4 的
                #    v8DetectionLoss.loss() 内部又乘一次 batch_size: 实测 total=103.47, 而同批
                #    scene+type+cause 经 λ 加权后合计 ≈2.7 ⇒ 原样接进 total_loss 就是 ~40x 夺权,
                #    三个核心目标会被定位损失淹没。故显式拒绝入梯度: det_loss_val 恒 0.0
                #    (criterion 调用面一字未动), 读数只取 items 的 box/cls/dfl (增益后均值,
                #    与 P2 val 同尺, 不含 ×B)。
                #  ③train 模式跑这次前向会改动全部 318 个 BN 缓冲 (crash 帧-only 统计偏移),
                #    eval 模式 0 改动且快 3 倍 ⇒ 诊断固定 eval + no_grad, 读完立即恢复
                #    .training 标志 (下一批时序前向必须仍在 train 模式)。
                det_loss_val = 0.0
                _raw_det_model = de_parallel(temporal_model).detection_model if is_ddp \
                    else temporal_model.detection_model
                # 每 16 步一次 (原 4 步): 诊断不需密采样; 实测 0.039s/次@B=8, 按
                # 5659 batches/epoch 计 stride=16 约 +2% 时长
                if batch_idx % 16 == 0 and hasattr(_raw_det_model, 'model'):
                    crash_frames = imgs[:, -1, :, :, :]  # [B, 3, H, W]
                    crash_boxes = gt_boxes[:, -1]       # [B, K, 4]
                    crash_classes = gt_classes[:, -1]    # [B, K]
                    crash_obj_mask = obj_mask[:, -1]    # [B, K]
                    try:
                        # 构建 ultralytics 检测 batch 格式
                        _all_cls = []
                        _all_bbox = []
                        _all_bidx = []
                        for _i in range(crash_frames.size(0)):
                            _valid = crash_obj_mask[_i]
                            if _valid.any():
                                _all_cls.append(crash_classes[_i][_valid].float())
                                _all_bbox.append(crash_boxes[_i][_valid])
                                _all_bidx.append(torch.full(
                                    (_valid.sum(), 1), _i,
                                    dtype=torch.float32, device=device))
                        if _all_cls:
                            _det_batch = {
                                'img': crash_frames,
                                'cls': torch.cat(_all_cls, 0).view(-1, 1),
                                'bboxes': torch.cat(_all_bbox, 0),
                                'batch_idx': torch.cat(_all_bidx, 0),
                            }
                            _prev_training = _raw_det_model.training
                            if not hasattr(_raw_det_model, 'args'):
                                _raw_det_model.args = trainer.args
                            _raw_det_model.eval()
                            try:
                                with torch.no_grad():
                                    _det_out = _raw_det_model(_det_batch)
                            finally:
                                _raw_det_model.train(_prev_training)
                            _items = (_det_out[1] if isinstance(_det_out, tuple)
                                      and len(_det_out) > 1 else None)
                            if isinstance(_items, dict):
                                _dv = []
                                for _k in ('box_loss', 'cls_loss', 'dfl_loss'):
                                    try:
                                        _dv.append(float(_items.get(_k)))
                                    except (TypeError, ValueError):
                                        _dv.append(float('nan'))
                                if all(math.isfinite(_d) for _d in _dv):
                                    _det_diag_n += 1
                                    _det_diag_sum += sum(_dv)
                                    for _j, _k in enumerate(
                                            ('box_loss', 'cls_loss', 'dfl_loss')):
                                        _det_diag_parts[_k] += _dv[_j]
                                    if _det_diag_n == 1 and rank == 0:
                                        print(f"  [Det-Diag] det-only 诊断通道已打通 "
                                              f"(eval+no_grad, 不入 total_loss): "
                                              f"box={_dv[0]:.4f} cls={_dv[1]:.4f} "
                                              f"dfl={_dv[2]:.4f} sum={sum(_dv):.4f} "
                                              f"@batch {batch_idx}")
                    except Exception as _de:
                        # G-278(R2): 不阻断时序训练，但首次见到某类失败就打一行 (去重)
                        _sig = f"{type(_de).__name__}: {_de}"
                        if _sig not in _det_fail_seen:
                            _det_fail_seen.add(_sig)
                            if rank == 0:
                                print(f"  [Det-Branch] 检测损失前向失败 ⇒ det_loss 恒 0 是假读数: "
                                      f"{_sig[:220]}")

                # C2修复(Critical): CDCR 跨域特征拆分
                feat_real_cls = None
                feat_synth_cls = None
                is_cdcr_active = (args.phase in ('3b', '3c')
                                  and epoch >= cdcr_warmup  # W5修复: warmup
                                  and cdcr_lambda > 0)
                if is_cdcr_active and geo_3d_dict is not None:
                    # 从 v24_extras 获取 CLS token (用于域对齐)
                    _extras = geo_3d_dict.get('v24_extras', {}) if isinstance(geo_3d_dict, dict) else {}
                    _cls_feat = _extras.get('cls_token')
                    if _cls_feat is not None:
                        # 尝试从 batch 获取域标签 (若数据集提供)
                        _domain = batch.get('source_domain', None)
                        if _domain is not None:
                            _domain = _domain.to(device)
                            real_mask = (_domain == 0)
                            synth_mask = (_domain == 1)
                            if real_mask.any() and synth_mask.any():
                                # C1修复(Critical): 移除双重 detach — CDCR 需要梯度回传
                                # CDCRLoss 内部已对 target 侧 detach (知识蒸馏标准做法)
                                feat_real_cls = _cls_feat[real_mask]
                                feat_synth_cls = _cls_feat[synth_mask]

                # V22: 使用 V22TemporalMultiTaskLoss (含 UASC + PTCL + CDCR)
                total_loss, loss_dict = criterion(
                    (det_out, scene_logits, tta_pred, cause_logits, type_logits,
                     frame_scene_logits, geo_3d_dict),
                    batch_labels,
                    det_loss=det_loss_val,  # C3修复: 非零检测损失
                    phase=args.phase,
                    feat_real=feat_real_cls,  # C2修复: 跨域特征
                    feat_synth=feat_synth_cls,
                )
                # V25-fix(loss-spike-final): 移除 loss 裁剪 — 裁剪阻断困难样本梯度,
                #   正确方案: 焦点调制 + 梯度裁剪 (max_norm=0.5)
                #   loss 值本身不需要裁剪, 梯度裁剪已足够防爆炸

            # 反向传播
            # 注: 不做 nan 图替换 — GradScaler.step 会检测 nan/inf 梯度并自动
            # 跳过该步更新; 用 sum(p.sum()) 造替代图会破坏 DDP reducer 的图
            # 一致性 (曾导致 "marked ready twice" 崩溃)
            need_sync = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(clip_loader)
            ctx = temporal_model.no_sync() if (is_ddp and not need_sync) else nullcontext()
            with ctx:
                scaler.scale(total_loss / accum_steps).backward()
            if need_sync:
                scaler.unscale_(optimizer)
                # G-280(clip-audit): 上限取自 epoch 级 _clip_cap, 打印与裁剪同源。
                _gn = torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=_clip_cap)
                # V23-fix(grad-monitor): 记录裁剪前全局梯度范数 (异常飙升是崩溃前兆)
                _gn = float(_gn)
                if math.isfinite(_gn):
                    grad_norm_sum += _gn
                    grad_norm_n += 1
                    if _gn > _clip_cap:
                        grad_clip_hit += 1
                    if _gn < grad_norm_min:
                        grad_norm_min = _gn
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                # G-303(#271) 三件套之三: 在体机制闸。首个 optimizer step 后检测支路 BN
                # 计数器必须一字未动; 之后每步一查 (单次设备同步), 一旦漂移说明冻结被
                # 绕过 ⇒ 立刻中止, 不烧完 38h 才发现。
                if _freeze_det and _bn_base_n:
                    _nb_now = _det_bn_now()
                    if _nb_now != _bn_base_sum:
                        _fatal(f"[G-303 det-freeze] epoch{epoch} batch{batch_idx}: "
                               f"检测支路 BN 计数器 sum "
                               f"{_bn_base_sum} → {_nb_now} (Δ={_nb_now-_bn_base_sum}) "
                               f"⇒ 冻结失效, 中止本 run")

            # 统计
            # V25-fix(loss-spike): 检测 loss 突刺 (>5x 中位数) 并记录
            if not hasattr(run_phase3_v24_temporal, '_loss_history'):
                run_phase3_v24_temporal._loss_history = []
            _lt = float(loss_dict.get('total', 0))
            run_phase3_v24_temporal._loss_history.append(_lt)
            if len(run_phase3_v24_temporal._loss_history) > 500:
                run_phase3_v24_temporal._loss_history = run_phase3_v24_temporal._loss_history[-500:]
            _median = sorted(run_phase3_v24_temporal._loss_history)[len(run_phase3_v24_temporal._loss_history)//2]
            if _median > 0 and _lt > 5.0 * _median:
                # G-315(#280) 取证通道: 上面 cause= 印的是**未加权**原值, 而 loss= 是 λ 加权和
                #   (3c: λ_scene=5.243 / λ_cause=1.0 / λ_type=1.0 / λ_cdcr=0.1 / λ_tta=0 ⇒ tta 恒不进
                #    total; 相位3 det_loss_val 按 :921-927 显式恒 0.0) ⇒ 拿 cause/loss 判"谁在尖峰"
                #    必然失真 (3a λ_cause=2.0、3b=1.5 会把 cause 份额低估 2×/1.5×)。就地补加权分解。
                #    纯读数: 只读 loss_dict 与 criterion 的 λ 属性, 不写回任何张量、不碰梯度路径。
                #    resid = total − Σ(加权项) − det ⇒ 明显非 0 说明 total 里还有未上报分量。
                _lam = {'scene': getattr(criterion, 'lambda_scene', None),
                        'cause': getattr(criterion, 'lambda_cause', None),
                        'type': getattr(criterion, 'lambda_type', None),
                        'cdcr': getattr(criterion, 'lambda_cdcr', None)}
                _raw = {k: loss_dict.get(k, 0.0) for k in _lam}
                _w = {k: (_lam[k] * _raw[k]) if (isinstance(_lam[k], (int, float))
                         and isinstance(_raw[k], (int, float))
                         and math.isfinite(_raw[k])) else None for k in _lam}
                _det = loss_dict.get('det', 0.0)
                _det = _det if isinstance(_det, (int, float)) and math.isfinite(_det) else None
                _known = [v for v in list(_w.values()) + [_det] if v is not None]
                _resid = (_lt - sum(_known)) if len(_known) == 5 else None
                _dc = ' '.join((f"{k}_w={_w[k]:.4f}({100.0 * _w[k] / _lt:.1f}%)"
                                if _w[k] is not None else f"{k}_w=na")
                               for k in ('scene', 'cause', 'type', 'cdcr'))
                LOGGER.warning(f"  [V25-loss-spike] batch {batch_idx}: loss={_lt:.4f} "
                             f"(median={_median:.4f}, ratio={_lt/_median:.1f}x) "
                             f"cause={loss_dict.get('cause', 0):.4f}"
                             f" decomp {_dc}"
                             f" det={('%.4f' % _det) if _det is not None else 'na'}"
                             f" resid={('%+.4f' % _resid) if _resid is not None else 'na'}"
                             f" lam=s{_lam['scene']}/c{_lam['cause']}"
                             f"/t{_lam['type']}/d{_lam['cdcr']}")
            # V22-fix(nan-guard): 单个 nan batch 不得污染 epoch 均值;
            # 首次出现时打印全分量诊断 (定位 nan 来源)
            _bad = {k: float(v) for k, v in loss_dict.items()
                    if isinstance(v, (int, float)) and not math.isfinite(v)}
            if _bad:
                # G-304(F4b): 原实现只在本 run **第一次** NaN 时打印 (_nan_reported 是
                #   函数属性, 永不复位) ⇒ 之后每次都悄悄发生、盘上无痕。告警改为逐次打,
                #   重型取证 dump 仍只首一次 (免得刷盘)。
                LOGGER.warning(f"V22-fix(nan-guard): epoch{epoch} batch {batch_idx} "
                               f"非有限损失分量: {_bad} "
                               f"(本轮累计 {_nonfinite_n + 1})")
            if _bad and not getattr(run_phase3_v24_temporal, '_nan_reported', False):
                run_phase3_v24_temporal._nan_reported = True
                # V23-fix(nan-取证): 现场取证 — 模型输出有限性 + 完整张量 dump,
                # 供离线重放 criterion 定位 NaN 真正来源
                try:
                    _fin = {n: (bool(torch.isfinite(t).all().item())
                                if torch.is_tensor(t) else type(t).__name__)
                            for n, t in [('scene_logits', scene_logits),
                                         ('frame_scene_logits', frame_scene_logits),
                                         ('type_logits', type_logits),
                                         ('cause_logits', cause_logits),
                                         ('tta_pred', tta_pred)]}
                    LOGGER.warning(f"V23 取证: 模型输出有限性 {_fin}")
                    if rank == 0:
                        torch.save({
                            'batch_idx': batch_idx, 'epoch': epoch,
                            'imgs': batch['img'].detach().cpu(),
                            'scene_cls': batch['scene_cls'].cpu(),
                            'scene_cls_seq': batch['scene_cls_seq'].cpu(),
                            'cause': batch['cause'].cpu(),
                            'cause_valid': batch.get('cause_valid',
                                                      torch.zeros(0)).cpu(),
                            'accident_type': batch['accident_type'].cpu(),
                            'agreement': batch.get('agreement',
                                                   torch.zeros(0)).cpu(),
                            'video_id': batch.get('video_id', []),
                            'scene_logits': scene_logits.detach().cpu(),
                            'frame_scene_logits':
                                frame_scene_logits.detach().cpu(),
                            'type_logits': type_logits.detach().cpu()
                            if torch.is_tensor(type_logits) else None,
                            'cause_logits': cause_logits.detach().cpu()
                            if torch.is_tensor(cause_logits) else None,
                            'tta_pred': tta_pred.detach().cpu(),
                            'loss_dict': {k: float(v) for k, v in
                                          loss_dict.items()
                                          if isinstance(v, (int, float))},
                        }, save_dir / f'nan_forensics_ep{epoch}_b{batch_idx}.pt')
                        LOGGER.warning(
                            f"V23 取证: 已保存 {save_dir}/"
                            f"nan_forensics_ep{epoch}_b{batch_idx}.pt")
                except Exception as _e:
                    LOGGER.warning(f"V23 取证失败: {_e}")
            # G-304(F4): 记账 —— 有任意分量非有限即计数 (均值仍按 isfinite 过滤, 口径不变)
            if _bad:
                _nonfinite_n += 1
            for k in epoch_losses:
                _v = loss_dict.get(k, 0)
                if isinstance(_v, (int, float)) and math.isfinite(_v):
                    epoch_losses[k] += _v
            for k in epoch_v22:
                _v = loss_dict.get(k, 0)
                if isinstance(_v, (int, float)) and math.isfinite(_v):
                    epoch_v22[k] += _v
            n_batches += 1

            with torch.no_grad():
                pred_cls = scene_logits.argmax(dim=1)
                gt_cls = (batch['scene_cls'].to(device) - 1).clamp(0, 2)
                valid = batch['scene_cls'].to(device) != 0
                scene_correct += ((pred_cls == gt_cls) & valid).sum().item()
                scene_total += valid.sum().item()

                frame_pred = frame_scene_logits.argmax(dim=-1)
                frame_gt = (scene_seq - 1).clamp(-1, 2)
                frame_valid = scene_seq != 0
                frame_correct += ((frame_pred == frame_gt) & frame_valid).sum().item()
                frame_total += frame_valid.sum().item()

                lat, n_lat, early = compute_detection_latency(
                    frame_scene_logits, scene_seq, fps=ds_fps)
                lat_sum += lat * n_lat
                lat_count += n_lat
                lat_early += early * n_lat

            if batch_idx % 50 == 0 and rank == 0:
                print(f"  [{epoch+1}/{epochs}][{batch_idx}/{len(clip_loader)}] "
                      f"loss={loss_dict['total']:.4f} scene={loss_dict['scene']:.4f} "
                      f"type={loss_dict['type']:.4f} cdcr={loss_dict.get('cdcr', 0):.4f} "
                      f"cause={loss_dict.get('cause', 0):.4f}")

        # Epoch 统计
        avg_loss = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        avg_v22 = {k: v / max(n_batches, 1) for k, v in epoch_v22.items()}
        # G-302(#258): 诊断读数按"实际算过的次数"求均值 (不是 n_batches), 否则被 stride 稀释
        _det_diag_avg = _det_diag_sum / max(_det_diag_n, 1)
        acc = 100 * scene_correct / max(scene_total, 1)
        frame_acc = 100 * frame_correct / max(frame_total, 1)
        mean_lat = lat_sum / max(lat_count, 1)
        early_rate = 100 * lat_early / max(lat_count, 1)

        if rank == 0:
            print(f"\n  Epoch {epoch+1}/{epochs}: "
                  f"loss={avg_loss['total']:.4f} scene={avg_loss['scene']:.4f} "
                  f"type={avg_loss['type']:.4f} cause={avg_loss.get('cause', 0):.4f}")
            print(f"  V22 创新: cdcr={avg_v22['cdcr']:.4f} "
                  f"scene_uasc={avg_v22['scene_uasc']:.4f} "
                  f"type_ptcl={avg_v22['type_ptcl']:.4f}")
            # G-304(F4c): 轮末显式报非有限 batch (rank 本地计数, 与 jsonl 的
            #   nonfinite_batches 同源)。历史上这里什么都不打 ⇒ "NaN=0" 看着像没坏,
            #   其实是读数没接。use_amp 一并打出来: AMP 开着时 GradScaler 会自动跳过
            #   inf/nan 梯度的 step, 这类 batch 是"没更新", 不是"更新坏了"。
            if _nonfinite_n > 0:
                print(f"  [G-304 nan] 本轮非有限损失 batch={_nonfinite_n}/{n_batches} "
                      f"(rank{rank}, AMP={use_amp}) ⇒ 已从 epoch 均值滤除, "
                      f"该轮实际更新少这么多个 batch")
            # G-302(#258): det-only 诊断读数 (eval+no_grad, 不进 total_loss)
            _dd_n = max(_det_diag_n, 1)
            print(f"  [Det-Diag] box={_det_diag_parts['box_loss'] / _dd_n:.4f} "
                  f"cls={_det_diag_parts['cls_loss'] / _dd_n:.4f} "
                  f"dfl={_det_diag_parts['dfl_loss'] / _dd_n:.4f} "
                  f"sum={_det_diag_avg:.4f} (n={_det_diag_n}, stride=16, "
                  f"eval+no_grad, 未入 total_loss)")
            print(f"  Scene Acc={acc:.1f}% | Frame Acc={frame_acc:.1f}% | "
                  f"Latency={mean_lat:+.2f}s | Early Rate={early_rate:.1f}%")
            # V23-fix(grad-monitor): 梯度范数/显存监控 (仅日志, 不影响训练行为)
            _avg_gn = grad_norm_sum / max(grad_norm_n, 1)
            _peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
            # G-280(clip-audit): hit% = 裁剪前范数超过上限的步占比。判读规矩:
            # hit%≈100% ⇒ 每步都被均匀缩放 ⇒ max_norm 取值 (0.5/1.0/5) 在 Adam 下
            # 近似无差别, 别再拿它解释训练退步; 只有 hit% 明显 <100% 时这个旋钮才有牙。
            _hit_pct = 100.0 * grad_clip_hit / max(grad_norm_n, 1)
            _min_gn = grad_norm_min if math.isfinite(grad_norm_min) else float('nan')
            print(f"  GradNorm(avg, pre-clip)={_avg_gn:.3f} "
                  f"(min={_min_gn:.3f}, cap={_clip_cap:g}, clip_hit={_hit_pct:.1f}%, "
                  f"{grad_norm_n} steps) | PeakMem={_peak_mem:.1f}GB")
            # G-282(instrument): 步长此前只写在 ckpt 的 optimizer_state_dict 里, 日志无读数
            # ⇒ 复盘只能 torch.load (§16-9-19-5)。这里显式打出每组 lr 与其 weight_decay。
            print("  LR/组: " + " | ".join(
                f"g{i} n={len(pg['params'])} lr={pg['lr']:.3e} wd={pg['weight_decay']:g}"
                for i, pg in enumerate(optimizer.param_groups)))

        # 验证 (每 2 epoch)
        val_frame_acc = 0.0
        val_type_acc = 0.0
        val_cause_recall = 0.0
        val_cause_f1 = 0.0
        # G-312(#273): 只加读数。val_cause_f1 按 9 位计 ⇒ bit4/7 (已摘除, GT 正例实测 0)
        # 上的命中全算假阳性进分母 (3b 实测 899 次), 而论文成因口径只列 7 监督位。
        # 两把尺并列落面; val_metric 一律不动 —— 换它等于换选型公式, 本相即不可比。
        val_cause_f1_7bit = 0.0
        # G-280(type-instrument): 选型公式每 epoch 都会跑, cv 口径同样要兜底, 否则
        # 非验证 epoch 直接 UnboundLocalError (与上面 V23-fix(val-unbound) 同一类坑)。
        # -1 = "本轮不可算" ⇒ 选型自动退回裸口径, 不会把该轮打成 -1 而永久落选。
        val_type_acc_cv = -1.0
        v_type_cv_c = v_type_cv_t = 0
        v_type_unslotable = v_type_labeled_out = 0
        _type_acct_ok = True
        # V23-fix(val-unbound): 验证块每 2 epoch 才执行, 而 monitor.log_epoch
        # 每 epoch 调用并读取 v_loss_n/val_loss_mean — 非验证 epoch 必须兜底,
        # 否则 UnboundLocalError 崩溃
        val_loss_mean = {'total': 0.0, 'scene': 0.0, 'cause': 0.0, 'type': 0.0}
        v_loss_n = 0
        # G-273(adopt-guard): 非验证 epoch 没有帧级计数, 守卫默认不触发
        _pre_collapse = False
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            temporal_model.eval()
            v_correct = v_total = 0
            # G-273(adopt-guard): pre_crash 帧级预测/GT 计数 —— 中间类塌陷守卫的仪器
            v_pred_pre = v_gt_pre = 0
            v_lat_sum = v_lat_count = v_lat_early = 0.0
            # W4修复(Warning): 添加 type 分类精度统计
            v_type_correct = v_type_total = 0
            # G-280(type-instrument): 增记课程内(cv)口径。旧分母把两类**不可评分**样本
            # 算成错: 无输出槽 (GT=-1 或 >=NUM_TYPES, argmax ∈[0,6] 永不等于它) 与
            # 有槽但课程外 (apply_mask 从未给它梯度)。尺② 实测分母 2,323
            # = cv 1,420 + 课程外 560 + 无槽 343 ⇒ 裸 type_acc 被 1,420/2,323=61.1%
            # 的经验天花板压住, 3a 真跌幅被稀释。**只加读数**: v_type_total /
            # val_type_acc 的旧语义与 monitor 的 type_acc_min 闸一律不动。
            v_type_cv_c = v_type_cv_t = 0
            v_type_unslotable = v_type_labeled_out = 0
            _active_types = PTCL_CONFIG.get(f"phase{args.phase}_active_types")
            # V22-fix(eval-cause): 成因 Recall/F1 统计 (sigmoid>0.5, scene>=2 门控)
            v_cause_tp = v_cause_fn = v_cause_fp = 0.0
            v_cause_tp7 = v_cause_fn7 = v_cause_fp7 = 0.0
            _cause_col7 = None
            v_cause_n = 0  # V22-fix(cause-标注缺失): 参与成因评估的已标注 clip 数
            # V23-fix(val-loss): 独立验证损失累计 (此前 monitor 记录的是训练损失)
            v_loss_sum = {'total': 0.0, 'scene': 0.0, 'cause': 0.0, 'type': 0.0}
            v_loss_n = 0
            # V23-fix(cause-诊断): GT 正例位上的最大 sigmoid (Recall=0 时定位差距)
            v_cause_pos_sigmoid_max = 0.0
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for vbatch in val_loader:
                    v_imgs = vbatch['img'].to(device)
                    v_imgs = v_imgs.float() / 255.0 if v_imgs.dtype == torch.uint8 else v_imgs.float()
                    v_boxes = vbatch['boxes_seq'].to(device)
                    v_classes = vbatch['classes_seq'].to(device)
                    v_mask = vbatch['obj_mask'].to(device)
                    v_scene_seq = vbatch['scene_cls_seq'].to(device)

                    # V22: 验证时也需要 type_ids
                    # V22-fix(eval-gate): 与训练侧 W1 修复对齐 — 任意帧 crash 即 crash clip
                    # (旧代码用首帧判定, 27% 渐进式事故 clip 被错判为非 crash)
                    v_type_ids = vbatch.get('accident_type', None)
                    v_scene_t = vbatch.get('scene_cls', vbatch.get('scene_cls_seq'))
                    if v_scene_t is not None:
                        v_scene_t = v_scene_t.to(device)
                        if v_scene_t.dim() > 1:
                            v_clip_scene = (v_scene_t == 3).any(dim=-1).long() * 3
                        else:
                            v_clip_scene = v_scene_t
                    else:
                        v_clip_scene = None
                    # V25-fix(type-leak): 验证一律置 padding 输入 (无泄漏 pass),
                    # 与推理 Pass 1 (TYPE_ID_UNKNOWN=7) 协议对齐
                    if v_type_ids is not None:
                        v_type_ids = torch.full_like(
                            v_type_ids.to(device, dtype=torch.long), 7)

                    v_out = temporal_model(
                        v_imgs, temporal_mode=True,
                        type_ids=v_type_ids,
                        gt_boxes_seq=v_boxes, gt_classes_seq=v_classes,
                        obj_mask_seq=v_mask)
                    v_frame_logits = v_out[5]
                    v_type_logits = v_out[4]  # W4修复: 提取 type_logits
                    v_pred = v_frame_logits.argmax(dim=-1)
                    v_gt = (v_scene_seq - 1).clamp(-1, 2)
                    v_valid = v_scene_seq != 0
                    v_correct += ((v_pred == v_gt) & v_valid).sum().item()
                    v_total += v_valid.sum().item()
                    v_pred_pre += ((v_pred == 1) & v_valid).sum().item()
                    v_gt_pre += ((v_gt == 1) & v_valid).sum().item()

                    # W4修复: 统计 crash 帧 type 分类精度
                    # V22-fix(eval-gate): 使用对齐后的 v_clip_scene (任意帧 crash)
                    v_type_logits = v_out[4]
                    if v_type_logits is not None and 'accident_type' in vbatch:
                        v_type_gt = vbatch['accident_type'].to(device, dtype=torch.long)
                        if v_clip_scene is not None:
                            v_crash = (v_clip_scene == 3)
                            if v_crash.any():
                                v_tp = v_type_logits[v_crash].argmax(dim=-1)
                                v_tg = v_type_gt[v_crash]
                                v_type_correct += (v_tp == v_tg).sum().item()
                                v_type_total += v_crash.sum().item()
                                # G-280(type-instrument): 逐样本三档分桶 (纯函数, 与尺②同源)
                                _tb = [type_score_bucket(int(_g), _active_types, NUM_TYPES)
                                       for _g in v_tg.tolist()]
                                _cv = torch.tensor([_x == 'cv' for _x in _tb],
                                                   dtype=torch.bool, device=v_tg.device)
                                v_type_cv_t += int(_cv.sum().item())
                                if _cv.any():
                                    v_type_cv_c += int((v_tp[_cv] == v_tg[_cv]).sum().item())
                                v_type_unslotable += sum(1 for _x in _tb if _x == 'unslotable')
                                v_type_labeled_out += sum(1 for _x in _tb if _x == 'labeled_out')

                    # V22-fix(eval-cause): 成因 Recall/F1 (scene>=2 门控, sigmoid>0.5)
                    v_cause_logits = v_out[3]
                    if v_cause_logits is not None and 'cause' in vbatch \
                            and v_clip_scene is not None:
                        v_cause_gt = vbatch['cause'].to(device, dtype=torch.float32)
                        if v_cause_gt.dim() == 1:
                            v_cause_gt = v_cause_gt.view(-1, v_cause_logits.shape[-1])
                        v_cgate = (v_clip_scene >= 2)
                        # V22-fix(cause-标注缺失): 仅评估有成因标注的 clip
                        if 'cause_valid' in vbatch:
                            v_cv = vbatch['cause_valid'].to(device, dtype=torch.long)
                            v_cgate = v_cgate & (v_cv == 1)
                        if v_cgate.any():
                            v_cause_n += v_cgate.sum().item()
                            v_csig = torch.sigmoid(v_cause_logits[v_cgate])
                            v_cp = (v_csig > 0.5).float()
                            v_cg = v_cause_gt[v_cgate].clamp(0, 1)
                            v_cause_tp += ((v_cp == 1) & (v_cg == 1)).sum().item()
                            v_cause_fn += ((v_cp == 0) & (v_cg == 1)).sum().item()
                            v_cause_fp += ((v_cp == 1) & (v_cg == 0)).sum().item()
                            if _cause_col7 is None or \
                                    _cause_col7.shape[0] != v_csig.shape[-1]:
                                _cause_col7 = torch.ones(v_csig.shape[-1],
                                                         dtype=torch.bool,
                                                         device=v_csig.device)
                                for _b in sorted(CAUSE_DISABLED_BITS):
                                    if _b < _cause_col7.shape[0]:
                                        _cause_col7[_b] = False
                            _c7 = _cause_col7.unsqueeze(0)
                            v_cause_tp7 += ((v_cp == 1) & (v_cg == 1) & _c7).sum().item()
                            v_cause_fn7 += ((v_cp == 0) & (v_cg == 1) & _c7).sum().item()
                            v_cause_fp7 += ((v_cp == 1) & (v_cg == 0) & _c7).sum().item()
                            # V23-fix(cause-诊断): GT 正例位最大 sigmoid
                            _vpos = v_csig[v_cg == 1]
                            if _vpos.numel() > 0:
                                v_cause_pos_sigmoid_max = max(
                                    v_cause_pos_sigmoid_max,
                                    float(_vpos.max().item()))

                    v_lat, v_n, v_early = compute_detection_latency(
                        v_frame_logits, v_scene_seq, fps=ds_fps)
                    v_lat_sum += v_lat * v_n
                    v_lat_count += v_n
                    v_lat_early += v_early * v_n

                    # V23-fix(val-loss): 用同一 criterion 计算该 batch 验证损失
                    _vlabels = {
                        'scene_cls': vbatch['scene_cls'].to(device),
                        'scene_cls_seq': v_scene_seq,
                        'cause': vbatch['cause'].to(device),
                        'cause_valid': vbatch['cause_valid'].to(device)
                        if 'cause_valid' in vbatch else
                        torch.ones_like(vbatch['scene_cls']).to(device),
                        'accident_type': vbatch['accident_type'].to(device),
                    }
                    _, _vloss_dict = criterion(
                        v_out, _vlabels, det_loss=0.0, phase=args.phase)
                    for _vk in v_loss_sum:
                        _vv = _vloss_dict.get(_vk, 0.0)
                        if isinstance(_vv, (int, float)) and math.isfinite(_vv):
                            v_loss_sum[_vk] += _vv
                    v_loss_n += 1

            if is_ddp:
                _vstats = torch.tensor([v_correct, v_total, v_lat_sum,
                                        v_lat_count, v_lat_early,
                                        v_type_correct, v_type_total,
                                        v_cause_tp, v_cause_fn, v_cause_fp,
                                        v_cause_n, v_pred_pre, v_gt_pre],
                                       device=device, dtype=torch.float64)
                dist.all_reduce(_vstats, op=dist.ReduceOp.SUM)
                v_correct, v_total, v_lat_sum, v_lat_count, v_lat_early, \
                    v_type_correct, v_type_total, \
                    v_cause_tp, v_cause_fn, v_cause_fp, v_cause_n, \
                    v_pred_pre, v_gt_pre = _vstats.tolist()
                # G-280(type-instrument): 新计数走**独立**的第二根 tensor, 不插进上面那 13 项。
                # 理由: 那份表是位置耦合的 (打包/解包靠顺序对齐), 往里插一项就可能把
                # 别的指标静默搬到 type 上; 另开一根则旧 13 项逐字未动。
                _vtcv = torch.tensor([v_type_cv_c, v_type_cv_t,
                                      v_type_unslotable, v_type_labeled_out],
                                     device=device, dtype=torch.float64)
                dist.all_reduce(_vtcv, op=dist.ReduceOp.SUM)
                (v_type_cv_c, v_type_cv_t,
                 v_type_unslotable, v_type_labeled_out) = _vtcv.tolist()
                # G-304(F2b): val_loss 也 all_reduce。原实现 v_loss_sum/v_loss_n 只在
                #   本 rank 累计 (:1228-1229, 注释自陈"损失量级诊断足够"), 但 monitor 的
                #   过拟合早停闸吃的正是这个数 ⇒ 拿 rank0 那半份 clip 宇宙判全局红线,
                #   口径与上面两根 accuracy/type tensor 不一致 (G-277 "两把尺宇宙不同")。
                #   合并后 rank0/rank1 读到同一个全局均值, 早停判定不再依赖哪张卡先看。
                _vkinds = list(v_loss_sum.keys())
                _vloss_t = torch.tensor([v_loss_sum[_k] for _k in _vkinds] + [v_loss_n],
                                        device=device, dtype=torch.float64)
                dist.all_reduce(_vloss_t, op=dist.ReduceOp.SUM)
                _vlv = _vloss_t.tolist()
                for _i, _k in enumerate(_vkinds):
                    v_loss_sum[_k] = _vlv[_i]
                v_loss_n = int(_vlv[len(_vkinds)])

            val_frame_acc = 100 * v_correct / max(v_total, 1)
            val_lat = v_lat_sum / max(v_lat_count, 1)
            val_early = 100 * v_lat_early / max(v_lat_count, 1)
            val_type_acc = 100 * v_type_correct / max(v_type_total, 1)
            # G-280(type-instrument): cv 口径 = 分母只数"可评分"样本 (有输出槽且在课程内)。
            # 分母 0 时置 -1 (显式"不可算"), 不能置 0 —— 0 会被选型公式当成"很差"参与加权。
            val_type_acc_cv = (100.0 * v_type_cv_c / v_type_cv_t
                               if v_type_cv_t > 0 else -1.0)
            # 自我对账: 三档桶必须恰好铺满旧分母, 否则仪器本身错了 (当场喊, 不静默)
            _type_acct_ok = (int(v_type_cv_t + v_type_unslotable
                                 + v_type_labeled_out) == int(v_type_total))
            # V22-fix(eval-cause): 成因 Recall / F1
            val_cause_recall = 100 * v_cause_tp / max(v_cause_tp + v_cause_fn, 1)
            _prec = v_cause_tp / max(v_cause_tp + v_cause_fp, 1)
            _rec = v_cause_tp / max(v_cause_tp + v_cause_fn, 1)
            val_cause_f1 = 200 * _prec * _rec / max(_prec + _rec, 1e-8)
            # G-312(#273): 7 监督位口径 (与论文 per-bit 面同源)。分母 0 ⇒ 0.0,
            # 与 val_cause_f1 的既有语义一致 (不进 val_metric, 无选型副作用)。
            _prec7 = v_cause_tp7 / max(v_cause_tp7 + v_cause_fp7, 1)
            _rec7 = v_cause_tp7 / max(v_cause_tp7 + v_cause_fn7, 1)
            val_cause_f1_7bit = 200 * _prec7 * _rec7 / max(_prec7 + _rec7, 1e-8)
            # V23-fix(val-loss): 验证损失均值 (本 rank 累计; 损失量级诊断足够)
            val_loss_mean = {k: v / max(v_loss_n, 1)
                             for k, v in v_loss_sum.items()}
            # G-273(adopt-guard): 中间类塌陷判据 —— GT 有 pre_crash 正例而帧级预测为 0
            _pre_collapse = (v_gt_pre > 0 and v_pred_pre == 0)
            if rank == 0:
                print(f"  [Val] Frame Acc={val_frame_acc:.1f}% | "
                      f"Latency={val_lat:+.2f}s | Early Rate={val_early:.1f}% | "
                      f"Type Acc={val_type_acc:.1f}%")
                # G-280(type-instrument): 裸 Type Acc 保留 (历史面/monitor 闸同尺度),
                # 另打 cv 口径与其分母构成 —— 以后看 type 一律看这行, 裸数只作连续性对照。
                print(f"  [Val] Type(cv 课程内)={val_type_acc_cv:.2f}% "
                      f"[{int(v_type_cv_c)}/{int(v_type_cv_t)}] | "
                      f"无槽(构造性必错)={int(v_type_unslotable)} "
                      f"课程外(无梯度)={int(v_type_labeled_out)} | "
                      f"分母对账={'OK' if _type_acct_ok else 'MISMATCH!'}")
                print(f"  [Val] pre_crash 帧级 预测={int(v_pred_pre)} "
                      f"GT正例={int(v_gt_pre)}")
                print(f"  [Val] Loss total={val_loss_mean['total']:.4f} "
                      f"scene={val_loss_mean['scene']:.4f} "
                      f"cause={val_loss_mean['cause']:.4f} "
                      f"type={val_loss_mean['type']:.4f}")
                print(f"  [Val] Cause Recall="
                      f"{(f'{val_cause_recall:.1f}%' if v_cause_n > 0 else 'N/A')} | "
                      f"Cause F1="
                      f"{(f'{val_cause_f1:.1f}%' if v_cause_n > 0 else 'N/A')} | "
                      f"成因标注clip={int(v_cause_n)}")
                # G-312(#273): 论文成因口径 = 7 监督位; 上一行的 Cause F1 = 9 位。
                # 两把尺都署名, 严禁互当复验 (位集不同, 差值全部来自摘除位的假阳性)。
                print(f"  [Val] Cause F1(7监督位)="
                      f"{(f'{val_cause_f1_7bit:.1f}%' if v_cause_n > 0 else 'N/A')} "
                      f"[TP/FP/FN={int(v_cause_tp7)}/{int(v_cause_fp7)}/"
                      f"{int(v_cause_fn7)}] vs 9位 TP/FP/FN="
                      f"{int(v_cause_tp)}/{int(v_cause_fp)}/{int(v_cause_fn)}")
                # V23-fix(cause-诊断): Recall=0 时输出正例位 sigmoid 上界,
                # 区分"头未训练"(≈0) 与"阈值不可达"(0.1~0.49)
                if v_cause_n > 0 and val_cause_recall == 0.0:
                    print(f"  [Val] Cause 诊断: GT正例位最大sigmoid="
                          f"{v_cause_pos_sigmoid_max:.3f} (阈值0.5)")

        # 保存最佳模型
        # C4修复(Critical): val_metric 综合帧准确率和类型准确率
        # V22-fix(metric-3a): 3a 的 unified_v21 仅含 type 0 (恒 100%) 且 val 只有
        # 1 个视频 → 改用 frame_acc + cause_recall 选型, 让成因学习真正驱动;
        # 3b/3c 的 s6 有 5 类真实类型, 保留原公式。
        if args.phase == '3a':
            # G-278(R6): 原式 0.6*frame + 0.4*cause_recall 对三目标里的两个失明
            # (类型判定完全没有，成因只算召回不算精确率) ⇒ e4→e8 成因隐含精确率
            # 49.4→35.7 塌方、Type Acc 60.4→58.9 下滑，而闸值反而在 e6 最高 (75.98)。
            # 改为存在/类型/成因三目标齐算，成因用 F1 (含精确率)。
            # 注意: 换了公式 ⇒ 与旧 best_val_metric 不可比，修后必须从 P2 种子重起 3a
            # (不能 --resume 到旧 e8 态，那个主干是随机冻结的，本就是要修的东西)。
            # G-280(type-instrument): type 项取 cv 口径 —— 裸口径被 61.1% 天花板压住,
            # 它那 0.2 权重实际在测"GT 有没有输出槽"。cv 不可算 (分母 0) 时退回裸口径,
            # 免得空分母把该轮 val_metric 打成 -1 而永久排除在采纳之外。
            _type_for_metric = (val_type_acc_cv if val_type_acc_cv >= 0
                                else val_type_acc)
            val_metric = (val_frame_acc * 0.5 + _type_for_metric * 0.2
                          + val_cause_f1 * 0.3) if val_frame_acc > 0 else -1.0
        else:
            # G-301(交付件根治): 兑现 G-280 在此处预登记的决策面 —— "到 P3b 边界时要么同样
            # 切 cv, 要么写下不切的理由"。不切的理由已不成立, 实测反证:
            #   旧惩罚式 20*max(0, val_loss-1) 写在 λ_scene=0.8 时代 (val_loss≈1.5-2.0);
            #   在体 λ_scene=5.243 下 val_loss≈6.5-7.7 ⇒ 惩罚≈110-134 ⇒ val_metric 恒为负
            #   (3b 八轮实测 -37.78/-52.92/-45.25/-46.23), 而选型基线 best_val_metric 初值
            #   0.0 ⇒ 采纳闸 val_metric > best 永不通过 ⇒ 3b 零交付件 (无 best_temporal.pt,
            #   weights/ 空) ⇒ 3c 链式入口 ({name}_p3b/best_temporal.pt 及回退件) 全断。
            # 现与 3a 同尺: 存在 0.5 / 类型 cv 0.2 / 成因 F1 0.3, 跨相位可比。
            # 过拟合观测不归选型尺管 —— monitor 的 val_loss 早停护栏独立负责 (本轮已实测触发)。
            _type_for_metric = (val_type_acc_cv if val_type_acc_cv >= 0
                                else val_type_acc)
            val_metric = (val_frame_acc * 0.5 + _type_for_metric * 0.2
                          + val_cause_f1 * 0.3) if val_frame_acc > 0 else -1.0
        # G-304(F9): 显式声明选型尺的影响面 —— val_metric 只有 存在/类型/成因 三轴,
        #   没有检测轴 (mAP)。这不是漏项: 本 run 检测支路全程冻结 (§16-38⑩ _freeze_det),
        #   检测能力不可能被训练改动, 因此它的守护在训练之外 —— 门一用 CPU 逐张量 diff
        #   (762/762 位等种子) 证明"没被动过", 量级由离线 mAP50 尺 (map_gate1.py) 复核。
        #   把 mAP 塞进 val_metric 反而会让冻结前提失效后自动失真, 故保持"尺外守护"。
        if _pre_collapse and rank == 0:
            print(f"  [Adopt-Guard] pre_crash 帧级预测=0 (GT 正例帧 "
                  f"{int(v_gt_pre)}) ⇒ 本轮不参与 best_temporal 选型 "
                  f"(val_metric={val_metric:.2f}, best={best_val_metric:.2f})")
        if val_metric > best_val_metric and not _pre_collapse:
            best_val_metric = val_metric
            if rank == 0:
                raw_model = temporal_model.module if is_ddp else temporal_model
                # G-304(F8): 件内自带"采纳基线"。此前三只 best 件都没有 best_val_metric/
                #   val_metric/phase 顶层键 ⇒ 拿 best 件当 --resume 起点时,
                #   :754 `_rest.get('best_metric', 0.0)` 静默取 0.0 ⇒ 采纳基线清零
                #   (下一轮只要 val_metric>0 就覆盖 best)。补键不改任何既有键语义。
                _ckpt_tags = {
                    'val_metric': val_metric,
                    'best_val_metric': best_val_metric,
                    'best_metric': best_val_metric,   # load_training_state(:754) 读的键名
                    'phase': args.phase,
                }
                # best_temporal.pt (V22 专用, 含 CTTA 权重)
                save_path = save_dir / 'best_temporal.pt'
                # G-312(#273): 覆盖前抢快照。3b 轮6 交付件 (sha16 be6ab634381ea15c) 被轮8
                #   覆盖时没有快照 ⇒ 那一份字节永久丢失, 论文引用只能指向路径字符串。
                #   把"换代前必抢快照"从人工纪律下沉成代码; 快照失败只出声, 绝不中止训练。
                if save_path.exists():
                    try:
                        import hashlib as _hashlib
                        import shutil as _shutil
                        from datetime import datetime as _dt
                        _snap_dir = save_dir / 'snapshots'
                        _snap_dir.mkdir(parents=True, exist_ok=True)
                        _h = _hashlib.sha256()
                        with open(save_path, 'rb') as _sf:
                            for _blk in iter(lambda: _sf.read(1 << 22), b''):
                                _h.update(_blk)
                        _old16 = _h.hexdigest()[:16]
                        _old_mt = _dt.fromtimestamp(save_path.stat().st_mtime)
                        _dst = _snap_dir / (
                            f'best_temporal_before_r{epoch + 1}_{_old16}.pt')
                        if not _dst.exists():
                            _shutil.copyfile(save_path, _dst)
                        print(f"  [G-312] 换代前快照: {_dst.name} "
                              f"(被覆盖件 sha256_16={_old16}, "
                              f"原保存时间={_old_mt.isoformat(timespec='seconds')}, "
                              f"bytes={_dst.stat().st_size})")
                    except Exception as _snap_err:
                        print(f"  [G-312][WARN] 换代前快照失败 (不影响训练与落件): "
                              f"{_snap_err!r}")
                # G-307(#278): 选型用的是 frame + type_cv + cause_f1 三目标复合, 而件内历史上
                #   只自带 frame 与裸 type ⇒ 拿交付件单独报数会"三缺一"。补两只轴键 (只增键,
                #   既有键语义与字节一字不动; 初值见 :1228/:1232 ⇒ 任何路径都在作用域内)。
                # G-312(#273): 再补 val_cause_f1_7bit ⇒ 顶层键数 13→14 (代际标记:
                #   14 键 = 3c 起的新码; 13 键 = 3b 补键码; 11 键 = 3b 首轮装载的旧副本)。
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_frame_acc': val_frame_acc,
                    'val_type_acc': val_type_acc,
                    'val_type_acc_cv': val_type_acc_cv,
                    'val_cause_f1': val_cause_f1,
                    'val_cause_f1_7bit': val_cause_f1_7bit,
                    'train_loss': avg_loss['total'],
                    'v24_phase': args.phase,
                    **_ckpt_tags,
                }, save_path)
                # weights/best.pt (链式检查点兼容)
                weights_dir = save_dir / 'weights'
                weights_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'val_frame_acc': val_frame_acc,
                    'val_type_acc': val_type_acc,
                    **_ckpt_tags,
                }, weights_dir / 'best.pt')
                # V22-fix(chain-3b): ultralytics 格式链式检查点。
                # _resolve_run_weight 优先命中 weights/best_temporal.pt;
                # 仅存 raw state_dict 会被 ultralytics load_checkpoint 类型检查拒绝
                # (3b 启动即 TypeError)。存检测子模型 Module 供下阶段 setup_model。
                from copy import deepcopy as _deepcopy
                _det = _deepcopy(raw_model.detection_model)
                _ta = getattr(det_model, 'args', None)
                if hasattr(_ta, '__dict__'):
                    _ta = dict(vars(_ta))
                elif not isinstance(_ta, dict):
                    _ta = {}
                _chain = {
                    'epoch': epoch + 1,
                    'best_fitness': val_metric,
                    'model': _det,
                    'ema': _deepcopy(_det),
                    'updates': getattr(det_model, 'updates', 0),
                    'train_args': _ta,
                    'val_frame_acc': val_frame_acc,
                    'val_type_acc': val_type_acc,
                    **_ckpt_tags,
                }
                torch.save(_chain, weights_dir / 'best_temporal.pt')
                print(f"  V22 保存最佳: {save_path} "
                      f"(frame_acc={val_frame_acc:.1f}%, type_acc={val_type_acc:.1f}%)")

        # ── V22 训练监控: 记录 epoch 指标 ──
        _stop, _reason, _diag = False, '', []
        if monitor is not None:
            monitor.log_epoch(
                epoch=epoch,
                train_loss=avg_loss['total'],
                # V23-fix(val-loss): 记录真实验证损失 (此前误记为训练损失,
                # 导致 Train-Val Gap 全程不可观测)
                val_loss=(val_loss_mean['total'] if v_loss_n > 0 else None),
                val_frame_acc=val_frame_acc if val_frame_acc > 0 else None,
                val_type_acc=val_type_acc if val_type_acc > 0 else None,
                val_latency=val_lat if val_frame_acc > 0 else None,
                val_early_rate=val_early if val_frame_acc > 0 else None,
                cdcr_loss=avg_v22.get('cdcr', 0),
                # G-302(#258): 此前取 avg_loss['det'] —— 那条通道恒等于 criterion 收到的
                # det_loss 实参 (0.0), 是假读数; 现取真诊断均值。历史行仍为 0 (口径断裂已登记)。
                det_loss=_det_diag_avg,
                # V23-fix(metric-统一): 停滞判定与检查点选型用同一指标
                # (此前 monitor 用 0.7*frame+0.3*type, 选型用 0.6*frame+0.4*cause,
                # 3a 的 type 恒 100% 使停滞判定对成因退化无感)
                val_metric=val_metric if val_frame_acc > 0 else None,
                extra=({'val_cause_recall': val_cause_recall,
                        # G-304(F4): 本轮被 nan-guard 滤掉的非有限 batch 数 (落面, 巡检现读)
                        'nonfinite_batches': _nonfinite_n,
                        # G-302(#258): det-only 三分量落面 (box/cls/dfl, 增益后均值,
                        # eval+no_grad 诊断口径; det_loss 键 = 三者之和)
                        'det_box': _det_diag_parts['box_loss'] / max(_det_diag_n, 1),
                        'det_cls': _det_diag_parts['cls_loss'] / max(_det_diag_n, 1),
                        'det_dfl': _det_diag_parts['dfl_loss'] / max(_det_diag_n, 1),
                        'det_diag_n': _det_diag_n,
                        # G-303(#271): 冻结在体闸的轮末读数 (只增键)。0 = 检测支路 BN
                        # 计数器一字未动; 非 0 即冻结被绕过 (步内闸会先中止本 run,
                        # 因此能落到面上的非 0 只可能是"该轮没有 step"这一种)
                        'det_bn_nbt_delta': ((_det_bn_now() - _bn_base_sum)
                                             if (_freeze_det and _bn_base_n) else None),
                        'val_cause_f1': val_cause_f1,
                        # G-312(#273): 7 监督位口径成因 F1 落面 (只增键; 上面
                        # val_cause_f1 = 9 位口径, 语义与历史序列一字不动)
                        'val_cause_f1_7bit': val_cause_f1_7bit,
                        # G-280(type-instrument): cv 口径与其分母构成落面 (只增键,
                        # 既有 val_type_acc 保持裸口径 ⇒ 历史序列不断档)
                        'val_type_acc_cv': val_type_acc_cv,
                        'type_cv_denom': int(v_type_cv_t),
                        'type_n_unslotable': int(v_type_unslotable),
                        'type_n_labeled_out': int(v_type_labeled_out),
                        'type_denom_acct_ok': bool(_type_acct_ok),
                        # G-282(instrument): 真实步长落面 (此前 lr 只存在于 ckpt 的
                        # optimizer_state_dict, 日志无读数 ⇒ 复盘必须 torch.load)
                        'lr_g0': optimizer.param_groups[0]['lr'],
                        'lr_g1': optimizer.param_groups[1]['lr'],
                        'lr_g2': optimizer.param_groups[2]['lr'],
                        # G-286(instrument): 只增键。n_nodecay_params 改按"全部 wd=0 的张量数"
                        # 计 (拆组后仍＝77 ⇒ 与 G-282 起的读数连续), 旧三组 lr 键原样不动。
                        'lr_g3': optimizer.param_groups[3]['lr'],
                        'lr_g4': optimizer.param_groups[4]['lr'],
                        'n_nodecay_params': sum(len(pg['params']) for pg in optimizer.param_groups
                                                if pg.get('weight_decay', 0.0) == 0.0),
                        'n_exposed_params': len(optimizer.param_groups[3]['params'])
                                            + len(optimizer.param_groups[4]['params'])}
                       if v_loss_n > 0 else None),
            )

            # 每 N epoch 执行检查点评估与异常诊断
            if (epoch + 1) % V22_MONITOR_CONFIG['checkpoint_every'] == 0 or epoch == epochs - 1:
                monitor.print_checkpoint_summary(epoch)
                _stop, _reason, _diag = monitor.check_anomalies()
                if _diag:
                    for d in _diag:
                        LOGGER.warning(f"  [监控] {d}")

        # V23-fix(nccl-级联): 中止决定广播到所有 rank — 此前仅 rank0 中止,
        # 其余 rank 卡在后续集合通信直到 600s NCCL 超时才崩溃退出
        if is_ddp:
            _stop_flag = torch.tensor([1.0 if _stop else 0.0], device=device)
            dist.broadcast(_stop_flag, src=0)
            _stop = bool(_stop_flag.item() > 0.5)

        if _stop:
            if rank == 0:
                LOGGER.error(f"\n{'='*60}")
                LOGGER.error(f"  ⚠️ V22 Phase {args.phase} 训练中止 @ Epoch {epoch+1}")
                LOGGER.error(f"  原因: {_reason}")
                LOGGER.error(f"  诊断建议:")
                for d in _diag:
                    LOGGER.error(f"    - {d}")
                LOGGER.error(f"  请修复后使用 --resume 恢复训练")
                LOGGER.error(f"{'='*60}")
                # 保存中断状态
                save_training_state(
                    save_dir / f'phase{args.phase}_interrupted.pt',
                    epoch, args.phase, temporal_model,
                    optimizer, scheduler, scaler,
                    best_metric=best_val_metric)
            if is_ddp:
                dist.barrier()
                if _own_pg:
                    dist.destroy_process_group()
            return best_val_metric

        if is_ddp:
            dist.barrier()

        # 每 epoch 保存 last checkpoint (含完整训练状态)
        if rank == 0:
            save_training_state(
                save_dir / 'last_temporal_v22.pt',
                epoch + 1, args.phase, temporal_model,
                optimizer, scheduler, scaler,
                best_metric=best_val_metric)
            # V25-fix: 自动 checkpoint 备份 — 每 3 epoch 备份到独立目录, 防止覆盖丢失
            if (epoch + 1) % 3 == 0:
                import shutil
                backup_dir = save_dir / 'backup'
                backup_dir.mkdir(exist_ok=True)
                _bk_path = backup_dir / f'epoch{epoch+1:03d}_temporal.pt'
                shutil.copy2(save_dir / 'last_temporal_v22.pt', _bk_path)
                LOGGER.info(f"  [备份] checkpoint → {_bk_path}")

    if rank == 0:
        print(f"\nV22 Phase {args.phase} 完成! best_val_metric={best_val_metric:.2f}")
        # ── 阶段验收检查 ──
        if monitor is not None:
            # G-304(F6): 送进验收的读数必须与键名同源。原实现把**训练**均值挂在
            #   'val_loss' 上 (monitor 那侧按 val_loss_max/loss_max 判), 且 val 面的
            #   type cv / early rate / cdcr 一个都没传 ⇒ 三条闸两条形同虚设。
            _has_val_face = val_frame_acc > 0
            final_metrics = {
                'val_frame_acc': val_frame_acc,
                'val_type_acc': val_type_acc,
                'val_type_acc_cv': val_type_acc_cv if _has_val_face else None,
                'val_loss': (val_loss_mean['total'] if v_loss_n > 0 else None),
                'val_early_rate': val_early if _has_val_face else None,
                'cdcr_loss': avg_v22.get('cdcr', 0),
            }
            passed, failures = monitor.evaluate_phase_acceptance(final_metrics)
            if passed:
                print(f"  ✅ Phase {args.phase} 验收通过")
            else:
                print(f"  ⚠️ Phase {args.phase} 验收未达标:")
                for f in failures:
                    print(f"    - {f}")
                print(f"  建议继续训练或调整超参数后进入下一阶段")
            # 打印资源摘要
            res = get_system_resources()
            print(f"  资源摘要: {format_resources(res)}")

    if is_ddp and _own_pg:
        dist.destroy_process_group()

    return best_val_metric
