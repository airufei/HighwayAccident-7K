# -*- coding: utf-8 -*-
"""
phase3_legacy — Phase 3 时序因果推理训练循环 (从 train.py 拆分)

包含:
  - temporal_context 模块导入 (V18TemporalModel / TemporalMultiTaskLoss / _has_temporal)
  - run_phase3_temporal()  Phase 3a/3b/3c DDP 时序训练主循环

V21 拆分时修复:
  - [C-fix] clip_loader NameError: OneCycleLR 创建移至 clip_loader 定义之后
    (原 L1117 在 clip_loader 定义前引用 len(clip_loader), 启动即崩溃)
  - [C-fix] lambda_scene=2.0 硬编码 → DEFAULT_SCENE_CLS_WEIGHT (V21 建议 0.5)
  - [C-fix] warmup_epochs 3 → 5 (与 V21 方案/accident_detect.yaml 对齐)
"""

import os
import sys
from contextlib import nullcontext  # R2-11: DDP no_sync 回退上下文
from pathlib import Path

import torch

from ultralytics.utils import LOGGER
try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    from ultralytics.utils.torch_utils import unwrap_model as de_parallel

from base_config import (
    dist, DDP, DistributedSampler,
    DEFAULT_SCENE_CLS_WEIGHT, get_phase3_loss_weights,
)
from dataset import ClipDataset

# V20: 导入时序推理模块 (Phase 3 核心)
try:
    from temporal_context import V18TemporalModel, TemporalMultiTaskLoss, TemporalContextModule, \
        compute_detection_latency
    _has_temporal = True
except ImportError:
    print("WARNING: 未找到 temporal_context.py, Phase 3 时序模式不可用")
    V18TemporalModel = TemporalMultiTaskLoss = TemporalContextModule = None
    compute_detection_latency = None
    _has_temporal = False


# ============================================================
# 6.6 V20: Phase 3 时序训练循环
# ============================================================

def run_phase3_temporal(trainer, args, clip_len=16, batch_size=4, num_workers=4,
                       class_weights=None):
    """V20: Phase 3 时序因果推理训练循环 (DDP 双卡版本)
    
    使用 V18TemporalModel (YOLO backbone + TemporalContextModule + AccidentDecisionHead)
    对 Clip 序列进行时序推理训练。
    
    Args:
        trainer: V18Trainer 实例 (用于获取数据集和预训练模型)
        args: 命令行参数
        clip_len: Clip 帧数
        batch_size: Clip batch size per GPU
        num_workers: DataLoader workers
    """
    if not _has_temporal:
        print("ERROR: temporal_context.py 未加载, 无法执行 Phase 3 时序训练")
        return

    # ── DDP 初始化 ──
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_ddp = world_size > 1

    # ### V21 MODIFIED R6-01(严重, D1): 全局确定性种子 —
    # 原 Phase 3 循环无任何种子设置 (dropout/DataLoader shuffle/时序模块随机初始化全不可复现)。
    # 各 rank 用同一基础种子初始化模型 (DDP wrap 时会广播 rank0 权重, 双重保险),
    # DistributedSampler 自带 seed+epoch 控制 shuffle。
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
        print(f"  ✓ 全局种子已设置: seed={_seed} (R6-01)")

    if is_ddp:
        _own_pg = False  # R2-07: 跟踪是否由本函数创建进程组
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
            _own_pg = True
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        if rank == 0:
            print(f"  DDP 已启用: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    else:
        device = torch.device(f'cuda:{args.device.split(",")[0]}' if torch.cuda.is_available() else 'cpu')
    
    # ── 构建时序模型 ──
    # ### V21 MODIFIED R7-01(致命, A): BaseTrainer.__init__ 只把 model 存为字符串路径,
    # 真正的 nn.Module 要到 _setup_train→setup_model() 才构建; Phase 3 从不调
    # trainer.train(), 故此处 trainer.model 仍是 str → de_parallel 原样返回 →
    # V18TemporalModel 内 detection_model.model[-1] 抛 AttributeError 启动即崩溃。
    # 显式构建模型 (V20 有此修复, V21 拆分时丢失)。
    if not isinstance(trainer.model, torch.nn.Module):
        if rank == 0:
            print(f"  构建检测模型 (trainer.model 当前为 {type(trainer.model).__name__})...")
        trainer.setup_model()
        trainer.model = trainer.model.to(device)
    det_model = de_parallel(trainer.model)
    # V20: 检查是否启用 3D
    use_3d = getattr(args, 'use_3d', False) or getattr(det_model, 'has_3d', False)
    temporal_model = V18TemporalModel(
        detection_model=det_model,
        feat_dim=256,
        num_heads=8,
        depth=4,
        seq_len=clip_len,
        dropout=0.5,  # V21: 0.3→0.5 增强正则化, 减少过拟合
        use_tracklet=False,  # V25-fix(type-leak): GT tracklet 特征直入决策头, 禁用
        max_objs=8,          # V20: 每帧 Top-8 对象 (按面积降序)
        use_3d=use_3d,       # V20: 3D 几何融合
    ).to(device)

    # V20-fix(C3): 加载前一阶段的完整时序权重 (而非仅 detection_model)
    # 避免 Phase 3b/3c 的 temporal_context + decision_head 从随机权重开始
    # ### D14修复(致命): 加载完整性门禁 —
    # 原逻辑 checkpoint 缺失/加载失败/匹配率过低均仅打印警告继续训练,
    # 时序模块静默从随机权重开始。现改为: 默认硬终止,
    # 匹配率 ≥80% 才放行; 显式 --allow-fallback 保留调试通道。
    prev_phase_map = {'3b': '3a', '3c': '3b'}
    prev_phase = prev_phase_map.get(args.phase)
    if prev_phase:
        allow_fallback = getattr(args, 'allow_fallback', False)
        prev_ckpt = Path(args.project) / f'{args.name}_p{prev_phase}' / 'best_temporal.pt'
        if not prev_ckpt.exists():
            if allow_fallback:
                if rank == 0:
                    print(f"  ⚠ 前阶段检查点不存在: {prev_ckpt}, --allow-fallback 已启用, 时序模块将从随机权重开始")
            else:
                if rank == 0:
                    print(f"ERROR: 前阶段时序检查点不存在: {prev_ckpt}")
                    print(f"  Phase {args.phase} 依赖 Phase {prev_phase} 的 best_temporal.pt, 请先完成前置训练。")
                if is_ddp:
                    dist.destroy_process_group()
                sys.exit(1)
        else:
            try:
                state = torch.load(prev_ckpt, map_location=device, weights_only=False)
                model_state = state.get('model_state_dict', state.get('model', state))
                own_state = temporal_model.state_dict()
                loaded = {k: v for k, v in model_state.items()
                          if k in own_state and own_state[k].shape == v.shape}
                match_rate = len(loaded) / max(len(own_state), 1)
                # D14门禁: 匹配率 < 80% 视为检查点与当前架构不兼容
                if match_rate < 0.8:
                    if allow_fallback:
                        if rank == 0:
                            print(f"  ⚠ 匹配率过低: {len(loaded)}/{len(own_state)} ({match_rate:.1%}), "
                                  f"--allow-fallback 已启用, 未匹配参数保持随机初始化")
                    else:
                        if rank == 0:
                            print(f"ERROR: 检查点匹配率过低: {len(loaded)}/{len(own_state)} ({match_rate:.1%} < 80%)")
                            print(f"  {prev_ckpt} 与当前模型架构不兼容, 拒绝加载。")
                        if is_ddp:
                            dist.destroy_process_group()
                        sys.exit(1)
                else:
                    own_state.update(loaded)
                    temporal_model.load_state_dict(own_state, strict=False)
                    if rank == 0:
                        print(f"  ✓ 加载前阶段 ({prev_phase}) 时序权重: {len(loaded)}/{len(own_state)} keys matched ({match_rate:.1%})")
            except Exception as e:
                if allow_fallback:
                    if rank == 0:
                        print(f"  ⚠ 加载前阶段时序权重失败: {e}, --allow-fallback 已启用, 从随机权重开始")
                else:
                    if rank == 0:
                        print(f"ERROR: 加载前阶段时序权重失败: {e}")
                    if is_ddp:
                        dist.destroy_process_group()
                    sys.exit(1)

    # ── 按子阶段冻结参数 ──
    if args.phase == '3a':
        temporal_model.freeze_for_phase3a()
        if rank == 0:
            print("  冻结: Backbone + Transformer L1-L2")
    elif args.phase == '3b':
        temporal_model.unfreeze_for_phase3b()
        if rank == 0:
            print("  解冻: Transformer 全层 + Backbone 最后 2 stage")
    else:  # 3c
        temporal_model.unfreeze_all()
        if rank == 0:
            print("  解冻: 所有参数")

    # ── DDP 包装 (必须在冻结/解冻之后) ──
    if is_ddp:
        temporal_model = DDP(temporal_model, device_ids=[local_rank],
                             output_device=local_rank,
                             find_unused_parameters=True)  # Phase 3a 冻结部分参数
        if rank == 0:
            print(f"  ✓ DDP 包装完成 (find_unused_parameters=True)")
    
    # ── 损失函数 ──
    # V20-fix(F4): 使用外部传入的 class_weights (来自 train.py DEFAULT_CLASS_WEIGHTS),
    # 回退值对齐单一真相源 [1.0, 1.2, 1.1]
    eff_cls_weight = class_weights if class_weights is not None else [1.0, 1.2, 1.1]
    # V20-fix(F5): 按子阶段 (3a/3b/3c) 获取损失权重
    phase_lw = get_phase3_loss_weights(args.phase)
    criterion = TemporalMultiTaskLoss(
        nc_cls=3, num_causes=9,
        cls_weight=eff_cls_weight,
        # V21-fix: lambda_scene 2.0→DEFAULT_SCENE_CLS_WEIGHT(0.5),
        # 与 Phase 1/2 及 accident_detect.yaml scene_cls_weight 对齐, 降低场景损失主导性
        lambda_scene=DEFAULT_SCENE_CLS_WEIGHT, lambda_tta=0.0,
        lambda_cause=phase_lw['cause'], lambda_type=phase_lw['type'],
        lambda_3d=0.3 if use_3d else 0.0,      # V20: 3D 几何损失权重
        lambda_ttc_3d=0.5 if use_3d else 0.0,  # V20: 3D TTC 损失权重
    ).to(device)
    if rank == 0:
        print(f"  Phase {args.phase} 损失权重: scene={DEFAULT_SCENE_CLS_WEIGHT}, "
              f"cause={phase_lw['cause']}, type={phase_lw['type']}")
        print(f"  cls_weight={eff_cls_weight}")
    
    # ── 优化器 (V20-fix O3: 分层学习率 — backbone 使用 0.1x LR, 避免微调过激破坏检测特征) ──
    base_lr = args.lr0 or 5e-4
    backbone_params = []
    temporal_params = []
    geometry_params = []  # V20: 3D 几何头参数
    for name, p in temporal_model.named_parameters():
        if not p.requires_grad:
            continue
        if 'detection_model' in name:
            backbone_params.append(p)
        elif 'geometry_head' in name or 'geom_fusion' in name or 'ttc_3d' in name:
            geometry_params.append(p)  # V20: 3D 相关参数
        else:
            temporal_params.append(p)
    n_trainable = sum(p.numel() for p in backbone_params) + sum(p.numel() for p in temporal_params) + sum(p.numel() for p in geometry_params)
    if rank == 0:
        print(f"  可训练参数: {n_trainable/1e6:.1f}M (backbone={sum(p.numel() for p in backbone_params)/1e6:.1f}M, "
              f"temporal={sum(p.numel() for p in temporal_params)/1e6:.1f}M, "
              f"geometry={sum(p.numel() for p in geometry_params)/1e6:.1f}M)")
    
    param_groups = [
        {'params': temporal_params, 'lr': base_lr},
        {'params': backbone_params, 'lr': base_lr * 0.05},  # V21: 0.1x→0.05x 更强保护预训练特征
    ]
    if geometry_params:
        param_groups.append({'params': geometry_params, 'lr': base_lr * 0.5})  # V20: 3D 头 0.5x LR
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=0.05,  # V21: 0.01→0.05 增强正则化
        betas=(0.9, 0.999),
    )
    # V21: warmup 5 epochs (3→5 更稳定收敛) + OneCycleLR (替代 CosineAnnealingLR, 文档5.3.1)
    warmup_epochs = 5
    # V20-fix(C11): epochs 必须在 scheduler 创建前定义, 否则 NameError 崩溃
    epochs = args.epochs or 15
    # 注: OneCycleLR 依赖 len(clip_loader), 在下方数据集构建完成后创建 (V21-fix: 修复 clip_loader NameError)
    
    # ── 数据集 (V20-fix: 直接从 data config 构建, 不依赖未初始化的 train_loader) ──
    # V20-fix(R2-1): 使用无增强的数据集构建 Clip —
    # train_loader.dataset 含 fliplr=0.5 等增强, 每帧独立应用会破坏时序一致性
    # V20-fix: trainer.train_loader 在 trainer.train() 前不存在,
    #          改用 trainer.data 配置直接构建数据集路径
    try:
        data_root = trainer._data_root or trainer.data.get('path', '')
        train_subdir = trainer.data.get('train', 'images/train')
        # 兼容绝对路径和相对路径
        if os.path.isabs(train_subdir):
            train_img_path = train_subdir
        else:
            train_img_path = os.path.join(data_root, train_subdir)
        # V21-fix(P3): 使用 mode='train' + augment=False 构建数据集
        # mode='train' 确保 scene_labels_dir 指向 train/ (与图片路径一致)
        # augment=False 确保不构建增强管线 (Clip 帧间时序一致性)
        base_dataset = trainer.build_dataset(train_img_path, mode="train", augment=False, fraction=1.0)
        if rank == 0:
            print(f"  数据集: 使用训练集 (build_dataset mode='train', augment=False) 构建 Clip")
            print(f"  图片路径: {train_img_path}")
            print(f"  scene_labels_dir: {getattr(base_dataset, 'scene_labels_dir', 'N/A')}")
    except Exception as e:
        # 回退: 尝试初始化 dataloader
        LOGGER.warning(f"数据集构建失败: {e}, 尝试通过 get_dataloader 初始化...")
        try:
            # ### V21 MODIFIED R7-02(致命, A): trainer.train_loader 在 trainer.train()
            # 前不存在 (仅 _setup_train 赋值), 原 `trainer.train_loader.dataset` 必抛
            # AttributeError, 使整个 except 回退形同虚设。get_dataloader 返回 loader,
            # 直接取其 .dataset; data_root/train_subdir 用 getattr 兜底防 try 早失败。
            _dr = trainer._data_root or trainer.data.get('path', '')
            fallback_path = os.path.join(_dr, trainer.data.get('train', 'images/train'))
            _loader = trainer.get_dataloader(fallback_path, mode='train')
            base_dataset = _loader.dataset
            data_root, train_subdir = _dr, trainer.data.get('train', 'images/train')
            print(f"  数据集: 回退使用 get_dataloader().dataset (含增强)")
        except Exception as e2:
            raise RuntimeError(
                f"Phase 3 数据集构建失败: {e}\n"
                f"回退也失败: {e2}\n"
                f"请检查 data_root/train 路径是否存在 (trainer._data_root={getattr(trainer, '_data_root', None)})"
            ) from e2
    ds_clip_stride = trainer._ds_config.get('clip_stride', 8) if trainer._ds_config else 8
    ds_fps = trainer._ds_config.get('fps', 10.0) if trainer._ds_config else 10.0  # R3-04
    clip_dataset = ClipDataset(base_dataset, clip_len=clip_len, stride=ds_clip_stride, max_objs=8)
    # 拆分 10% 作为验证集 (V20-fix O1)
    # R1-16: video-level split (whole video -> one subset).
    # clip-level random_split leaks overlapping frames across train/val.
    import random as _random
    _vids = sorted({c[1] for c in clip_dataset.clips})
    _rng = _random.Random(42)
    _rng.shuffle(_vids)
    _n_vval = max(1, int(len(_vids) * 0.1))
    _vval = set(_vids[:_n_vval])
    class _VideoSubset(torch.utils.data.Dataset):
        def __init__(self, ds, vset):
            self.ds = ds
            self.idxs = [i for i, c in enumerate(ds.clips) if c[1] in vset]
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, i):
            return self.ds[self.idxs[i]]
    train_clip = _VideoSubset(clip_dataset, set(_vids) - _vval)
    val_clip = _VideoSubset(clip_dataset, _vval)
    n_train, n_val = len(train_clip), len(val_clip)
    if rank == 0:
        print(f"  Clip 数据集: train={n_train} val={n_val} (clip_len={clip_len}, max_objs=8)")
    
    # DDP: 使用 DistributedSampler 替代 shuffle
    train_sampler = DistributedSampler(train_clip, num_replicas=world_size, rank=rank,
                                       shuffle=True) if is_ddp else None
    clip_loader = torch.utils.data.DataLoader(
        train_clip,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # DDP 时由 sampler 控制 shuffle
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(val_clip, num_replicas=world_size, rank=rank,
                                     shuffle=False) if is_ddp else None
    val_loader = torch.utils.data.DataLoader(
        val_clip,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # ── V21: OneCycleLR — 内含 warmup + cosine 衰减, 无需手动 warmup 逻辑 ──
    # V21-fix: 原代码在 clip_loader 定义前引用 len(clip_loader) → NameError 启动即崩溃,
    # 现移至 clip_loader 定义之后创建
    accum_steps = 4  # R1-05: define before scheduler to size total_steps by optimizer updates
    total_steps = max(epochs, 1) * max((len(clip_loader) + accum_steps - 1) // accum_steps, 1)
    # R2-08: pct_start 必须 <1.0, epochs<5 时 warmup_epochs/epochs 会超过 1 导致 ValueError
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[pg['lr'] for pg in param_groups],
        total_steps=total_steps, pct_start=min(0.8, warmup_epochs / max(epochs, 1)),
        anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0,
    )
    
    # ── V20-fix: AMP 混合精度 (Phase 3 时序训练也必须启用, 节省 ~40% 显存) ──
    use_amp = getattr(args, 'amp', True) and torch.cuda.is_available()
    # ### V21 MODIFIED R7-08(一般, D2): torch.amp.GradScaler('cuda', ...) 需 torch>=2.3,
    # 服务器若为旧版 torch 会抛 TypeError; 回退到 torch.cuda.amp.GradScaler。
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp and rank == 0:
        print("  ✓ AMP 混合精度已启用")
    
    # ── 训练循环 ──
    best_val_metric = 0.0  # R2-06: 原 -inf 导致未验证 epoch (val_metric=-1) 也触发保存
    save_dir = Path(args.project) / f'{args.name}_p{args.phase}'
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ### V21 MODIFIED R6-02(严重, D8): Phase 3 断点续传 —
    # 原代码完全忽略 args.resume (仅 Phase 1/2 生效), checkpoint 存了 optimizer 却从不恢复,
    # scheduler/scaler/best_val_metric 不保存 → 中断后只能从头重训。
    # 现支持: --resume <ckpt路径> 或 --resume 任意真值 (自动找 save_dir/last_temporal.pt)
    start_epoch = 0
    if getattr(args, 'resume', None):
        _resume_path = Path(args.resume) if isinstance(args.resume, str) and \
            Path(args.resume).is_file() else save_dir / 'last_temporal.pt'
        if _resume_path.is_file():
            try:
                _ck = torch.load(_resume_path, map_location=device, weights_only=False)
                _raw = temporal_model.module if is_ddp else temporal_model
                _raw.load_state_dict(_ck['model_state_dict'], strict=False)
                if 'optimizer_state_dict' in _ck:
                    optimizer.load_state_dict(_ck['optimizer_state_dict'])
                if 'scheduler_state_dict' in _ck:
                    scheduler.load_state_dict(_ck['scheduler_state_dict'])
                if 'scaler_state_dict' in _ck:
                    scaler.load_state_dict(_ck['scaler_state_dict'])
                start_epoch = int(_ck.get('epoch', 0))
                best_val_metric = float(_ck.get('best_val_metric', _ck.get('val_frame_acc', 0.0)) or 0.0)
                if rank == 0:
                    print(f"  ✓ 断点续传: {_resume_path} (epoch={start_epoch}, best={best_val_metric:.2f})")
            except Exception as e:
                if rank == 0:
                    print(f"  ⚠ 续传 checkpoint 加载失败, 从头开始: {e}")
        else:
            if rank == 0:
                print(f"  ⚠ 续传 checkpoint 不存在: {_resume_path}, 从头开始")

    # R1-05: accum_steps moved above scheduler creation
    all_trainable = [p for p in temporal_model.parameters() if p.requires_grad]
    
    if rank == 0:
        eff_batch = batch_size * accum_steps * world_size
        print(f"\n{'='*60}")
        print(f"Phase {args.phase} 时序训练: {epochs} epochs, {len(clip_loader)} batches/epoch, "
              f"accum_steps={accum_steps} (effective_batch={eff_batch}, GPUs={world_size})")
        print(f"{'='*60}")
    
    for epoch in range(start_epoch, epochs):  # R6-02: 续传时从 start_epoch 开始
        # DDP: 每个 epoch 设置 sampler 以确保 shuffle 随机性
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        temporal_model.train()
        epoch_losses = {'total': 0, 'scene': 0, 'cause': 0, 'type': 0,
                        '3d_geo': 0, 'ttc_3d': 0}  # V20-fix(F6): 添加 3D 损失跟踪
        n_batches = 0
        
        # 场景分类统计
        scene_correct = 0
        scene_total = 0
        scene_pred_counts = {1: 0, 2: 0, 3: 0}
        scene_gt_counts = {1: 0, 2: 0, 3: 0}
        # V20: 帧级统计 + 检测延迟
        frame_correct = 0
        frame_total = 0
        lat_sum = 0.0
        lat_count = 0
        lat_early = 0.0
        
        for batch_idx, batch in enumerate(clip_loader):
            imgs = batch['img'].to(device)  # [B, T, C, H, W]
            # R1-04: dataset yields uint8; normalize like ultralytics preprocess_batch
            imgs = imgs.float() / 255.0 if imgs.dtype == torch.uint8 else imgs.float()
            # V20: 对象级 GT 框序列 (tracklet 建模)
            gt_boxes = batch['boxes_seq'].to(device)        # [B,T,K,4]
            gt_classes = batch['classes_seq'].to(device)    # [B,T,K]
            obj_mask = batch['obj_mask'].to(device)         # [B,T,K]
            scene_seq = batch['scene_cls_seq'].to(device)   # [B,T] 逐帧标签
            
            # 前向传播 (V20-fix: AMP autocast + 时序模式 + 对象级 tracklet)
            with torch.amp.autocast('cuda', enabled=use_amp):
                # V20-fix: 解包 7 元组 (而非 6), 第 7 个是 geo_3d dict
                temporal_out = temporal_model(imgs, temporal_mode=True,
                                   gt_boxes_seq=gt_boxes,
                                   gt_classes_seq=gt_classes,
                                   obj_mask_seq=obj_mask)
                det_out, scene_logits, tta_pred, cause_logits, type_logits, \
                    frame_scene_logits = temporal_out[:6]
                geo_3d_dict = temporal_out[6] if len(temporal_out) > 6 else None
                
                # 计算损失 (V20: 帧级场景损失为主)
                batch_labels = {
                    'scene_cls': batch['scene_cls'].to(device),
                    'scene_cls_seq': scene_seq,
                    'cause': batch['cause'].to(device),
                    'accident_type': batch['accident_type'].to(device),
                }
                # V20-fix: 传入 7 元组 (含 geo_3d_dict), 使 TemporalMultiTaskLoss 能计算 3D 损失
                total_loss, loss_dict = criterion(
                    (det_out, scene_logits, tta_pred, cause_logits, type_logits,
                     frame_scene_logits, geo_3d_dict),
                    batch_labels,
                    det_loss=0.0,  # Phase 3 不计算检测损失
                )
            
            # 反向传播 (V20-fix O2: 梯度累积 + AMP scaler)
            # R2-11: DDP 梯度累积时前 N-1 步跳过 allreduce (no_sync), 仅最后一步同步
            need_sync = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(clip_loader)
            ctx = temporal_model.no_sync() if (is_ddp and not need_sync) else nullcontext()
            with ctx:
                scaled_loss = total_loss / accum_steps
                scaler.scale(scaled_loss).backward()
            if need_sync:
                scaler.unscale_(optimizer)
                # D3增强(建议, 审查提示词): 梯度范数监控 — 检测梯度爆炸/消失
                # 在 clip 前计算原始范数, clip 后截断 (clip_grad_norm_ 返回截断前的范数)
                grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                # 每 100 步记录一次梯度范数, 异常时告警
                if batch_idx % 100 == 0 and rank == 0:
                    gn_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                    if gn_val > 1000:
                        LOGGER.warning(f'[D3] 梯度范数异常大: {gn_val:.2f} (step {batch_idx}), 可能梯度爆炸')
                    elif gn_val < 1e-6:
                        LOGGER.warning(f'[D3] 梯度范数异常小: {gn_val:.2e} (step {batch_idx}), 可能梯度消失')
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # R1-05: OneCycleLR steps per optimizer update, not per epoch
            
            # 统计
            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, 0)
            n_batches += 1
            
            # 场景分类统计 (clip 级)
            with torch.no_grad():
                pred_cls = scene_logits.argmax(dim=1)  # [B] in {0,1,2}
                gt_cls = (batch['scene_cls'].to(device) - 1).clamp(0, 2)  # [B] in {0,1,2}
                valid = batch['scene_cls'].to(device) != 0
                scene_correct += ((pred_cls == gt_cls) & valid).sum().item()
                scene_total += valid.sum().item()
                for c in range(3):
                    scene_pred_counts[c+1] += ((pred_cls == c) & valid).sum().item()
                    scene_gt_counts[c+1] += ((gt_cls == c) & valid).sum().item()
                
                # V20: 帧级统计
                frame_pred = frame_scene_logits.argmax(dim=-1)      # [B,T] {0,1,2}
                frame_gt = (scene_seq - 1).clamp(-1, 2)             # [B,T]
                frame_valid = scene_seq != 0                        # ignore=0
                frame_correct += ((frame_pred == frame_gt) & frame_valid).sum().item()
                frame_total += frame_valid.sum().item()
                
                # R3-04: 检测延迟使用数据集配置 fps (原硬编码 10.0)
                lat, n_lat, early = compute_detection_latency(
                    frame_scene_logits, scene_seq, fps=ds_fps)
                lat_sum += lat * n_lat
                lat_count += n_lat
                lat_early += early * n_lat
            
            if batch_idx % 50 == 0 and rank == 0:
                print(f"  [{epoch+1}/{epochs}][{batch_idx}/{len(clip_loader)}] "
                      f"loss={loss_dict['total']:.4f} scene={loss_dict['scene']:.4f} "
                      f"cause={loss_dict['cause']:.4f} type={loss_dict['type']:.4f}"
                      f" 3d={loss_dict.get('3d_geo', 0):.4f}")  # V20-fix(F6)
        
        # V21: OneCycleLR 每 epoch 调用 step() (warmup 已内置, 无需手动处理)
        # R1-05: per-epoch scheduler.step() removed (stepped per optimizer update above)
        
        # Epoch 统计
        avg_loss = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        acc = 100 * scene_correct / max(scene_total, 1)
        frame_acc = 100 * frame_correct / max(frame_total, 1)
        mean_lat = lat_sum / max(lat_count, 1)
        early_rate = 100 * lat_early / max(lat_count, 1)
        
        if rank == 0:
            print(f"\n  Epoch {epoch+1}/{epochs}: "
                  f"loss={avg_loss['total']:.4f} scene={avg_loss['scene']:.4f} "
                  f"cause={avg_loss['cause']:.4f} type={avg_loss['type']:.4f}"
                  f" 3d_geo={avg_loss.get('3d_geo', 0):.4f} ttc={avg_loss.get('ttc_3d', 0):.4f}")  # V20-fix(F6)
            print(f"  Scene Acc={acc:.1f}% | Frame Acc={frame_acc:.1f}% | "
                  f"Latency={mean_lat:+.2f}s | Early Rate={early_rate:.1f}% (n={lat_count})")
            print(f"  pred: n={scene_pred_counts[1]} p={scene_pred_counts[2]} c={scene_pred_counts[3]} | "
                  f"gt: n={scene_gt_counts[1]} p={scene_gt_counts[2]} c={scene_gt_counts[3]}")
        
        # 保存最佳模型 (V20-fix O1: 改用验证指标)
        # 验证循环 (V20-fix O1: 每 2 个 epoch 验证一次)
        val_frame_acc = 0.0
        val_lat = 0.0
        val_early = 0.0
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            temporal_model.eval()
            v_correct = v_total = 0
            v_lat_sum = v_lat_count = 0
            v_lat_early = 0.0
            # R2-10: 验证也启用 AMP, 避免 FP32 显存翻倍 + 速度慢 40%
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for vbatch in val_loader:
                    v_imgs = vbatch['img'].to(device)
                    v_imgs = v_imgs.float() / 255.0 if v_imgs.dtype == torch.uint8 else v_imgs.float()  # R1-04
                    v_boxes = vbatch['boxes_seq'].to(device)
                    v_classes = vbatch['classes_seq'].to(device)
                    v_mask = vbatch['obj_mask'].to(device)
                    v_scene_seq = vbatch['scene_cls_seq'].to(device)
                    v_out = temporal_model(
                        v_imgs, temporal_mode=True,
                        gt_boxes_seq=v_boxes, gt_classes_seq=v_classes,
                        obj_mask_seq=v_mask)
                    v_frame_logits = v_out[5]  # V20-fix: 索引取第 6 个元素 (frame_scene_logits), 兼容 7 元组
                    v_pred = v_frame_logits.argmax(dim=-1)
                    v_gt = (v_scene_seq - 1).clamp(-1, 2)
                    v_valid = v_scene_seq != 0
                    v_correct += ((v_pred == v_gt) & v_valid).sum().item()
                    v_total += v_valid.sum().item()
                    v_lat, v_n, v_early = compute_detection_latency(
                        v_frame_logits, v_scene_seq, fps=ds_fps)  # R3-04: 使用数据集配置 fps
                    v_lat_sum += v_lat * v_n
                    v_lat_count += v_n
                    v_lat_early += v_early * v_n
            # R3-02: DDP 聚合各 rank 验证统计量, 避免模型选择基于部分数据
            if is_ddp:
                _vstats = torch.tensor([v_correct, v_total, v_lat_sum, v_lat_count, v_lat_early],
                                       device=device, dtype=torch.float64)
                dist.all_reduce(_vstats, op=dist.ReduceOp.SUM)
                v_correct, v_total, v_lat_sum, v_lat_count, v_lat_early = _vstats.tolist()
            val_frame_acc = 100 * v_correct / max(v_total, 1)
            val_lat = v_lat_sum / max(v_lat_count, 1)
            val_early = 100 * v_lat_early / max(v_lat_count, 1)
            if rank == 0:
                print(f"  [Val] Frame Acc={val_frame_acc:.1f}% | Latency={val_lat:+.2f}s | "
                      f"Early Rate={val_early:.1f}% (n={v_lat_count})")
        
        # 保存最佳模型 (V21-fix: 仅在真正执行了验证时更新, 避免混合量纲)
        if val_frame_acc > 0:
            val_metric = val_frame_acc
        else:
            val_metric = -1.0  # 未验证时不更新最佳模型
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            # DDP: 只在 rank 0 保存模型
            if rank == 0:
                # DDP 包装时用 module.state_dict() 避免 key 带 'module.' 前缀
                raw_model = temporal_model.module if is_ddp else temporal_model
                save_path = save_dir / 'best_temporal.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_frame_acc': val_frame_acc,
                    'val_latency': val_lat,
                    'train_loss': avg_loss['total'],
                    'scene_acc': acc,
                }, save_path)
                # V20-fix: 同时写入 weights/best.pt 供 3b/3c 链式加载
                weights_dir = save_dir / 'weights'
                weights_dir.mkdir(parents=True, exist_ok=True)
                # R1-17: save FULL temporal state_dict (with detection_model. prefix)
                # so the 3b/3c chain (and fallback) restores temporal modules too.
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_frame_acc': val_frame_acc,
                }, weights_dir / 'best.pt')
                print(f"  ✓ 保存最佳模型: {save_path} (val_frame_acc={val_frame_acc:.1f}%)")
        
        # DDP: 等待所有进程完成当前 epoch
        if is_ddp:
            dist.barrier()

        # ### V21 MODIFIED R6-02(严重, D8): 每个 epoch 保存 last_temporal.pt (含完整训练状态)
        if rank == 0:
            _raw_model = temporal_model.module if is_ddp else temporal_model
            # ### V21 MODIFIED R7-09(一般, D8): 原子写 — 直接 torch.save 到目标文件
            # 若在序列化中途被 kill (OOM/抢占), last_temporal.pt 会残缺损坏,
            # 下次 --resume 加载即崩溃。先写 .tmp 再 os.replace 原子替换。
            _last_path = save_dir / 'last_temporal.pt'
            _last_tmp = save_dir / 'last_temporal.pt.tmp'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': _raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_metric': best_val_metric,
                'train_loss': avg_loss['total'],
            }, _last_tmp)
            os.replace(_last_tmp, _last_path)
    
    if rank == 0:
        print(f"\nPhase {args.phase} 时序训练完成! 最佳 val_metric={best_val_metric:.2f}")
    
    # DDP 清理
    if is_ddp and _own_pg:  # R2-07: 仅销毁本函数创建的进程组
        dist.destroy_process_group()
    
    return best_val_metric
