#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — V22 模型评估脚本 (mAP50/Precision/Recall + Type Acc + Scene Acc)

C4修复(Critical): 此脚本提供 V22 缺失的 mAP/Precision/Recall 评估能力,
用于验证训练目标: mAP50 > 50%, Precision > 60%, Recall > 60%

用法:
    python scripts/evaluate.py \
        --checkpoint runs/accident_detect_v24_p3c/best_temporal.pt \
        --data-root datasets/accident_v22 \
        --model-yaml config/yolo26m_detect_cls.yaml

    # 仅评估 Phase 2 检测模型
    python scripts/evaluate.py \
        --checkpoint runs/accident_detect_v24_p2/weights/best.pt \
        --data-root datasets/accident_v22 \
        --mode detection
"""

import os
import sys
import json
import argparse
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# V22 配置 (自动注入 V21 sys.path)
import train_config
from train_config import PROJECT_ROOT_V22, DEFAULT_YAML


def eval_detection(checkpoint, data_root, model_yaml, device='0', imgsz=640,
                   conf=0.25, iou=0.45):
    """评估检测性能 (mAP50/mAP95/Precision/Recall)

    使用 ultralytics 内置验证器计算标准 COCO 指标。
    """
    from detection import V18Trainer

    overrides = dict(
        model=model_yaml,
        data=DEFAULT_YAML,
        imgsz=imgsz,
        batch=16,
        device=device,
        conf=conf,
        iou=iou,
        mode='val',
        project='runs/evaluate',
        name='det_eval',
    )

    trainer = V18Trainer(
        overrides=overrides,
        sampler_ratios=None,
        data_root=data_root,
        dataset_name='accident-both',
    )

    # 加载检查点
    if os.path.exists(checkpoint):
        import torch
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model_state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        # C8修复(Critical): 剥离 detection_model. 前缀 — V24TemporalModel 到 V18Trainer 的键名转换
        normalized_state = {}
        for k, v in model_state.items():
            if k.startswith('detection_model.'):
                normalized_state[k[len('detection_model.'):]] = v
            else:
                normalized_state[k] = v
        if hasattr(trainer.model, 'load_state_dict'):
            own = trainer.model.state_dict()
            loaded = {k: v for k, v in normalized_state.items()
                      if k in own and own[k].shape == v.shape}
            own.update(loaded)
            trainer.model.load_state_dict(own, strict=False)
            print(f"  加载权重: {len(loaded)}/{len(own)} keys ({len(loaded)/max(len(own),1):.1%})")
    else:
        print(f"WARNING: 检查点不存在: {checkpoint}")

    # 运行验证
    metrics = trainer.validate()
    return metrics


def eval_temporal(checkpoint, data_root, model_yaml, device='0',
                  clip_len=8, batch_size=4, num_workers=4):
    """评估时序模型 (Scene Acc + Type Acc + Detection Latency)

    加载 V24TemporalModel 并在 val 集上运行时序推理。
    """
    import torch
    from base_config import dist
    from dataset import ClipDataset
    from detection import V18Trainer
    from phase3_legacy import compute_detection_latency
    from temporal_model import build_temporal_model
    from train_data import wrap_clip_dataset

    dev = torch.device(f'cuda:{device.split(",")[0]}' if torch.cuda.is_available() else 'cpu')

    # 构建检测模型
    overrides = dict(
        model=model_yaml, data=DEFAULT_YAML, imgsz=640, batch=batch_size,
        device=device, mode='val',
    )
    trainer = V18Trainer(
        overrides=overrides, sampler_ratios=None,
        data_root=data_root, dataset_name='accident-both',
    )
    trainer.setup_model()
    det_model = trainer.model.to(dev)

    # 构建 V22 时序模型
    temporal_model = build_temporal_model(
        detection_model=det_model, clip_len=clip_len,
        ctta_config=train_config.CTTA_CONFIG,
        uasc_config=train_config.UASC_CONFIG,
        cdcr_config=train_config.CDCR_CONFIG,
        ptcl_config=train_config.PTCL_CONFIG,
    ).to(dev)

    # 加载检查点
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
        model_state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        own = temporal_model.state_dict()
        loaded = {k: v for k, v in model_state.items()
                  if k in own and own[k].shape == v.shape}
        own.update(loaded)
        temporal_model.load_state_dict(own, strict=False)
        print(f"  加载 V22 权重: {len(loaded)}/{len(own)} keys ({len(loaded)/max(len(own),1):.1%})")
    else:
        print(f"ERROR: 检查点不存在: {checkpoint}")
        sys.exit(1)

    temporal_model.eval()

    # 构建验证数据集
    data_root_abs = os.path.join(PROJECT_ROOT_V22, data_root) \
        if not os.path.isabs(data_root) else data_root
    train_img_path = os.path.join(data_root_abs, 'images', 'val')
    base_dataset = trainer.build_dataset(train_img_path, mode='val', augment=False, fraction=1.0)
    clip_dataset = ClipDataset(base_dataset, clip_len=clip_len, stride=8, max_objs=8)
    clip_dataset = wrap_clip_dataset(clip_dataset, default_agreement=1.0)

    val_loader = torch.utils.data.DataLoader(
        clip_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    ds_fps = 10.0
    scene_correct = scene_total = 0
    frame_correct = frame_total = 0
    type_correct = type_total = 0
    lat_sum = lat_count = lat_early = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            imgs = batch['img'].to(dev)
            imgs = imgs.float() / 255.0 if imgs.dtype == torch.uint8 else imgs.float()
            scene_seq = batch['scene_cls_seq'].to(dev)

            # V25-fix(type-leak): 评估一律用 padding 类型输入 (与推理 Pass 1 对齐),
            # 禁止喂 GT accident_type 再对同一标签算 Type Acc
            type_ids = batch.get('accident_type', None)
            if type_ids is not None:
                type_ids = torch.full_like(type_ids.to(dev, dtype=torch.long), 7)

            out = temporal_model(imgs, temporal_mode=True, type_ids=type_ids)
            frame_logits = out[5]
            type_logits = out[4]
            scene_logits = out[1]

            # Scene accuracy
            pred_cls = scene_logits.argmax(dim=1)
            gt_cls = (batch['scene_cls'].to(dev) - 1).clamp(0, 2)
            valid = batch['scene_cls'].to(dev) != 0
            scene_correct += ((pred_cls == gt_cls) & valid).sum().item()
            scene_total += valid.sum().item()

            # Frame accuracy
            frame_pred = frame_logits.argmax(dim=-1)
            frame_gt = (scene_seq - 1).clamp(-1, 2)
            frame_valid = scene_seq != 0
            frame_correct += ((frame_pred == frame_gt) & frame_valid).sum().item()
            frame_total += frame_valid.sum().item()

            # Type accuracy
            if type_logits is not None and 'accident_type' in batch:
                type_gt = batch['accident_type'].to(dev, dtype=torch.long)
                scene_t2 = batch.get('scene_cls', batch.get('scene_cls_seq')).to(dev)
                cs = scene_t2[:, 0] if scene_t2.dim() > 1 else scene_t2
                crash = (cs == 3)
                if crash.any():
                    tp = type_logits[crash].argmax(dim=-1)
                    tg = type_gt[crash]
                    type_correct += (tp == tg).sum().item()
                    type_total += crash.sum().item()

            # Latency
            lat, n_lat, early = compute_detection_latency(frame_logits, scene_seq, fps=ds_fps)
            lat_sum += lat * n_lat
            lat_count += n_lat
            lat_early += early * n_lat

            if batch_idx % 20 == 0:
                print(f"  [{batch_idx}/{len(val_loader)}] processing...")

    results = {
        'scene_clip_acc': 100 * scene_correct / max(scene_total, 1),
        'scene_frame_acc': 100 * frame_correct / max(frame_total, 1),
        'type_acc': 100 * type_correct / max(type_total, 1),
        'detection_latency_s': lat_sum / max(lat_count, 1),
        'early_detection_rate': 100 * lat_early / max(lat_count, 1),
    }
    # G-306(口径声明): 上面 ds_fps 是硬编码名义 10, 而 s6 子集按 stride-4 抽帧 ⇒ 秒口径被低估 4 倍。
    #   相位3 的延迟正本口径是**有效 fps** (phase3_train.py:573-591), 离线验收面在
    #   eval_acceptance.py (同时落 _nominal_fps 与 _eff 两把键)。本脚本非验收正本,
    #   只加声明键、不改旧键数值 (历史面要能逐位复现)。
    results['detection_latency_s_note'] = ('nominal_fps=10 ⇒ 有效 fps 口径需乘 eff_stride(≈4); '
                                           '验收正本=eval_acceptance.py')
    print('  [口径警告] detection_latency_s 用名义 fps=10, 比有效 fps 口径小约 4 倍; '
          '验收正本=eval_acceptance.py (本脚本 docstring 里的 accident_v22 已不存在)')
    return results


def main():
    parser = argparse.ArgumentParser(
        description='V22 模型评估 — mAP50/Precision/Recall + Type/Scene Acc')
    parser.add_argument('--checkpoint', required=True,
                        help='检查点路径 (.pt)')
    parser.add_argument('--data-root', default='datasets/accident_v22',
                        help='数据集根目录')
    parser.add_argument('--model-yaml', default='config/yolo26m_detect_cls.yaml',
                        help='模型配置 YAML')
    parser.add_argument('--mode', choices=['detection', 'temporal', 'both'],
                        default='both', help='评估模式')
    parser.add_argument('--device', default='0', help='GPU 设备')
    parser.add_argument('--clip-len', type=int, default=8, help='Clip 帧数')
    parser.add_argument('--output', default=None, help='输出 JSON 路径')
    args = parser.parse_args()

    print("=" * 70)
    print("V22 模型评估")
    print("=" * 70)
    print(f"  检查点: {args.checkpoint}")
    print(f"  数据: {args.data_root}")
    print(f"  模式: {args.mode}")

    all_results = {}

    if args.mode in ('detection', 'both'):
        print("\n── 检测指标 (mAP50/mAP95/Precision/Recall) ──")
        try:
            det_metrics = eval_detection(
                args.checkpoint, args.data_root, args.model_yaml, args.device)
            if det_metrics:
                if hasattr(det_metrics, 'box'):
                    box = det_metrics.box
                    all_results['detection'] = {
                        'mAP50': float(box.map50),
                        'mAP95': float(box.map),
                        'precision': float(box.mp),
                        'recall': float(box.mr),
                    }
                    print(f"  mAP50={box.map50:.4f} mAP95={box.map:.4f} "
                          f"Precision={box.mp:.4f} Recall={box.mr:.4f}")
                else:
                    all_results['detection'] = str(det_metrics)
                    print(f"  {det_metrics}")
        except Exception as e:
            print(f"  检测评估失败: {e}")
            import traceback
            traceback.print_exc()

    if args.mode in ('temporal', 'both'):
        print("\n── 时序指标 (Scene Acc + Type Acc + Latency) ──")
        try:
            temporal_results = eval_temporal(
                args.checkpoint, args.data_root, args.model_yaml,
                args.device, args.clip_len)
            all_results['temporal'] = temporal_results
            for k, v in temporal_results.items():
                print(f"  {k}: {v:.2f}")
        except Exception as e:
            print(f"  时序评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 验收标准检查
    print("\n" + "=" * 70)
    print("验收标准检查")
    print("=" * 70)
    det = all_results.get('detection', {})
    if isinstance(det, dict):
        map50 = det.get('mAP50', 0)
        prec = det.get('precision', 0)
        rec = det.get('recall', 0)
        print(f"  mAP50 > 50%:   {'PASS' if map50 > 0.50 else 'FAIL'} ({map50:.2%})")
        print(f"  Precision > 60%: {'PASS' if prec > 0.60 else 'FAIL'} ({prec:.2%})")
        print(f"  Recall > 60%:    {'PASS' if rec > 0.60 else 'FAIL'} ({rec:.2%})")

    temp = all_results.get('temporal', {})
    if temp:
        print(f"  Frame Acc:       {temp.get('scene_frame_acc', 0):.1f}%")
        print(f"  Type Acc:        {temp.get('type_acc', 0):.1f}%")

    # 保存结果
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
