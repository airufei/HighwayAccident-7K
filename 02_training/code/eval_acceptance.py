#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_acceptance.py — V23 验收评估脚本

从 best_temporal.pt 加载 V24TemporalModel, 在 s6 val 集上运行时序推理,
输出:
  - scene_confusion.csv: 3×3 场景混淆矩阵 + PDR/FAR/recall
  - cause_type_metrics.csv: per-cause Recall/Precision + per-type Accuracy
  - eval_acceptance.json: 全部指标汇总

用法:
    python scripts/eval_acceptance.py \
        --checkpoint runs/v26_dw_lane_g304_p3a/best_temporal.pt \
        --data-root datasets/accident_v26 \
        --output-dir /tmp/eval_face
# 注: --data-root 必须是**绝对路径或 PROJECT_ROOT_V22 下真实存在**的目录;
#     旧示例写的 datasets/accident_v24 已不存在 (实测 FileNotFoundError, rc=1)。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import train_config
from train_config import PROJECT_ROOT_V22, DEFAULT_YAML, CAUSE_NAMES, TYPE_NAMES
try:
    import base_config
except ModuleNotFoundError:
    from core import base_config

# 验收必须与训练同口径: run_train.py 的 --clip-len 默认是 16, ClipDataset 默认也是 16。
# 此前本脚本默认 8, 而 shape 不匹配的权重是被静默跳过的 (见 build_model_and_data 里的
# shape_skipped 硬失败), 等于用随机 pos_embed 评估 → 报告 §8.24。
DEFAULT_CLIP_LEN = 16


def build_models(args):
    """构建 det+时序模型并加载检查点；验收与路线 B 打分流共用（clip_len 硬失败闸只此一份）。"""
    import torch
    from detection import DetectClassifyModelV18
    from temporal_model import build_temporal_model

    dev = torch.device(f'cuda:{args.device.split(",")[0]}'
                       if torch.cuda.is_available() else 'cpu')

    # ── 加载 YAML 配置 ──
    yaml_path = args.model_yaml
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(PROJECT_ROOT_V22, yaml_path)

    # ── 构建检测模型 ──
    print("  构建检测模型...")
    det_model = DetectClassifyModelV18(cfg=yaml_path, verbose=False)
    det_model = det_model.to(dev)

    # ── 构建 V22 时序模型 ──
    print("  构建时序模型...")
    temporal_model = build_temporal_model(
        detection_model=det_model, clip_len=args.clip_len,
        ctta_config=train_config.CTTA_CONFIG,
        uasc_config=train_config.UASC_CONFIG,
        cdcr_config=train_config.CDCR_CONFIG,
        ptcl_config=train_config.PTCL_CONFIG,
    ).to(dev)

    # ── 加载检查点 ──
    ckpt_path = args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model_state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    own = temporal_model.state_dict()
    loaded = {k: v for k, v in model_state.items()
              if k in own and own[k].shape == v.shape}
    # V22-fix(clip-len): 权重里 pos_embed / bias_net.2.* 的 shape 直接由 clip_len 决定，
    # 而 shape 不匹配的键会被上面那行**静默跳过** → 拿随机位置编码去评估，且只打印一行百分比。
    # 所以这里必须硬失败，而不是继续跑出一个看着正常的数（历史上默认 --clip-len 8
    # 而训练用 16，就是这么悄悄塌掉的，见报告 §8.24）。
    shape_skipped = [k for k, v in model_state.items()
                     if k in own and own[k].shape != v.shape]
    if shape_skipped:
        raise SystemExit(
            f"检查点有 {len(shape_skipped)} 个键与本模型的 shape 不匹配并被跳过，"
            f"例如 {shape_skipped[:5]}\n"
            f"  → 多半是 --clip-len 与训练时不一致 (本脚本默认 {DEFAULT_CLIP_LEN}，"
            f"run_train.py 亦是 16)。用 `--clip-len <训练值>` 重跑。")
    own.update(loaded)
    temporal_model.load_state_dict(own, strict=False)
    pct = len(loaded) / max(len(own), 1)
    print(f"  加载权重: {len(loaded)}/{len(own)} keys ({pct:.1%})")

    meta = {k: v for k, v in ckpt.items()
            if k not in ('model_state_dict', 'optimizer_state_dict',
                         'model', 'optimizer')}
    print(f"  元数据: epoch={meta.get('epoch','?')}, "
          f"phase={meta.get('v24_phase','?')}, "
          f"val_frame_acc={meta.get('val_frame_acc','?')}")

    temporal_model.eval()
    return det_model, temporal_model, dev, meta


def build_base_dataset(args, det_model, split='val'):
    """按训练侧同一套预处理构建逐帧数据集（验收与路线 B 打分流共用）。"""
    try:
        from ultralytics.utils.torch_utils import de_parallel
    except ImportError:
        from ultralytics.utils.torch_utils import unwrap_model as de_parallel
    from ultralytics.utils import DEFAULT_CFG
    from detection import MultiDatasetAccident

    # ── 加载数据配置 YAML ──
    data_yaml_path = args.data_yaml
    if not os.path.isabs(data_yaml_path):
        data_yaml_path = os.path.join(PROJECT_ROOT_V22, data_yaml_path)
    import yaml
    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_ROOT_V22, data_root)

    # ── 构建数据集 ──
    print(f"  构建 {split} 数据集...")
    img_path = os.path.join(data_root, 'images', split)
    scene_dir = os.path.join(data_root, data_cfg.get('scene_labels_dir', 'scene_labels'), split)
    cause_dir = os.path.join(data_root, data_cfg.get('cause_labels_dir', 'cause_labels'), split)
    type_dir = os.path.join(data_root, data_cfg.get('type_labels_dir', 'type_labels'), split)

    gs = max(int(de_parallel(det_model).stride.max()), 32)

    # 构造 DEFAULT_CFG 的替代（仅需 imgsz, cache, single_cls, fraction 等）
    hyp = deepcopy(DEFAULT_CFG)
    hyp.imgsz = 640
    hyp.cache = False
    hyp.single_cls = False
    hyp.fraction = 1.0
    hyp.classes = None

    base_dataset = MultiDatasetAccident(
        img_path=img_path, imgsz=640, batch_size=args.batch_size,
        augment=False, hyp=hyp, rect=False, cache=False, single_cls=False,
        stride=gs, pad=0.5, prefix=f'{split}: ', task='detect',
        data=data_cfg, classes=None, fraction=1.0,
        scene_labels_dir=scene_dir, cause_labels_dir=cause_dir,
        type_labels_dir=type_dir, dataset_id='cctv_mixed',
    )
    return base_dataset


def build_model_and_data(args):
    """直接构建检测模型和数据集，绕过 ultralytics Trainer（避免全量标签扫描）"""
    import torch
    from dataset import ClipDataset
    from train_data import wrap_clip_dataset

    det_model, temporal_model, dev, meta = build_models(args)
    base_dataset = build_base_dataset(args, det_model, 'val')

    clip_dataset = ClipDataset(base_dataset, clip_len=args.clip_len,
                               stride=8, max_objs=8)
    # 与 phase3_train.py 的 V22-fix(tile-clip) 同口径: 含 tile 帧的 clip 在训练侧被整条排除,
    # 验收若不排就会拿训练从未见过的退化 clip 计分 (且 tile 批的 type 标签有错位, 见报告 §8.22)。
    import re as _re
    _labels = clip_dataset.base.labels
    _tile_idx = {i for i in range(len(_labels))
                 if _re.search(r'_t\d{2}\.[A-Za-z]+$',
                               os.path.basename(str(_labels[i]['im_file'])))}
    if _tile_idx:
        _n_before = len(clip_dataset.clips)
        clip_dataset.clips = [c for c in clip_dataset.clips
                              if not any(fi in _tile_idx for fi in c[0])]
        print(f"  tile-clip 排除 (与 Phase 3 一致): {_n_before} → {len(clip_dataset.clips)} "
              f"clip, 命中 tile 帧 {len(_tile_idx)} 个")
    # G-314(#273 E7): 共同考卷过滤。默认关 (None) ⇒ 与历史全量 val 读数逐位一致。
    # 过滤律与出题尺同源 (paper_e7_universe.filter_clips_by_frame_list, 自测锚 23–25)：
    # 跨界 clip 整条剔, 半条混会让两臂看到不同窗口, "同考卷"就名存实亡。
    if getattr(args, 'frame_list', None):
        from paper_e7_universe import filter_clips_by_frame_list
        _fl = args.frame_list if os.path.isabs(args.frame_list) else \
            os.path.join(PROJECT_ROOT_V22, args.frame_list)
        if not os.path.isfile(_fl):
            raise SystemExit(f'[红] --frame-list 文件不存在: {_fl}')
        _want = {ln.strip() for ln in open(_fl, encoding='utf-8') if ln.strip()}
        if not _want:
            raise SystemExit(f'[红] --frame-list 名单为空 ⇒ fail-closed 会一卷不考, 先查文件: {_fl}')
        _n0 = len(clip_dataset.clips)
        _kept, _partial, _missing = filter_clips_by_frame_list(clip_dataset.clips, _labels, _want)
        if _missing:
            raise SystemExit(f'[红] 名单里有 {_missing} 帧不在本臂 val 树 ⇒ 两臂考卷不是同一张, '
                             f'E7 不得结线 (先重出宇宙: paper_e7_universe.py --emit-face)')
        clip_dataset.clips = _kept
        print(f"  [E7 共同考卷] 名单 {len(_want)} 帧 ← {_fl}")
        print(f"    clips {_n0} → {len(_kept)} (跨界整条剔 {_partial} 条)")
        meta['e7_frame_list'] = {'path_resolved': _fl, 'listed_frames': len(_want),
                                 'clips_before': _n0, 'clips_after': len(_kept),
                                 'clips_partial_dropped': _partial, 'frames_absent_in_tree': 0}
    # G-306(F12-fix): 包装对象没有 .base ⇒ 有效 fps 估计必须用**包装前**的 base labels 与
    #   clip 列表 (与在体 phase3_train.py:573-591 同一实现, 那里就是先抓 _base_labels 再包)。
    _pre_wrap_labels = _labels
    _pre_wrap_clips = list(clip_dataset.clips)
    clip_dataset = wrap_clip_dataset(clip_dataset, default_agreement=1.0)

    val_loader = torch.utils.data.DataLoader(
        clip_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print(f"  验证 clips: {len(clip_dataset)}")
    # G-304(F12): 有效 fps 现读 (与 phase3_train.py:570-590 同一算法)。
    #   配置里的 fps=10 是**源帧率**, 而 s6 子集实际按 stride 抽帧 ⇒ 帧间隔不止 1。
    #   延迟 = (帧号差)/fps, 用源帧率会把秒数低估 stride 倍。旧 ds_fps=10.0 一字不动
    #   (历史 latency_s 面要能逐位复现), 修正值另立新键 + 比值换算。
    meta['fps_nominal'] = 10.0
    meta['eff_stride'] = 1
    try:
        _diffs = []
        _bl = _pre_wrap_labels
        for _fi_list, _vid, _ in _pre_wrap_clips[:200]:
            _fnos = []
            for _fi in _fi_list:
                _m = _re.search(r'_(\d+)\.[A-Za-z]+$', str(_bl[_fi]['im_file']))
                if _m:
                    _fnos.append(int(_m.group(1)))
            _d = [b - a for a, b in zip(_fnos, _fnos[1:]) if b > a]
            if _d:
                _diffs.append(_d[len(_d) // 2])
        if _diffs:
            _eff_stride = sorted(_diffs)[len(_diffs) // 2]
            if _eff_stride > 1:
                meta['eff_stride'] = int(_eff_stride)
    except Exception as _e:
        meta['fps_eff_fail'] = repr(_e)
        print(f"  [fps-eff] 有效 fps 估计失败, 沿用名义 10.0: {_e}")
    meta['fps_eff'] = 10.0 / meta['eff_stride']
    print(f"  [fps] 名义=10.0 有效={meta['fps_eff']:.2f} (帧间隔={meta['eff_stride']}) "
          f"⇒ 旧键 latency_s 用名义(低估 {meta['eff_stride']}x), "
          f"新键 latency_s_eff 才是秒口径")
    return temporal_model, val_loader, meta, dev


def main():
    parser = argparse.ArgumentParser(description='V23 验收评估')
    parser.add_argument('--checkpoint', required=True, help='best_temporal.pt 路径')
    parser.add_argument('--data-root', required=True,
                        help='数据集路径 (e.g. datasets/accident_v26)')
    parser.add_argument('--data-yaml', default='config/accident_detect.yaml',
                        help='数据配置 YAML 路径')
    parser.add_argument('--frame-list', default=None,
                        help='G-314(E7): 只考名单内的帧 (逐行 basename, 出自 paper_e7_universe.py '
                             '--emit-face 的 common_frames.txt)。**默认关＝全量 val, 历史读数不变**。'
                             '跨界 clip 整条剔; 名单帧在本树缺席则当场停 (两臂考卷不是同一张)')
    parser.add_argument('--model-yaml', default='config/yolo26m_detect_cls_fixed.yaml',
                        help='模型配置 YAML 路径')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录 (默认: checkpoint 同目录)')
    parser.add_argument('--device', default='0')
    parser.add_argument('--clip-len', type=int, default=DEFAULT_CLIP_LEN,
                        help='必须与训练该检查点时一致 (run_train.py 默认 16); '
                             '不一致会让 pos_embed / bias_net.2 因 shape 不符被跳过, '
                             '本脚本对此直接硬失败')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    # G-307(#273): 逐 clip 分数流落盘开关。**默认关 ⇒ 本脚本所有既有指标逐位不变**。
    #   根因不在"没有尺"而在"概率当场被扔掉": 实测 `csig` 拿到后立刻 `> 0.5` 阈值化成
    #   TP/FP/FN 累加器 (下方 cause 块), 分数本身从不落盘 ⇒ per-bit AP/AUC (E5)、
    #   lead-time (E1)、bootstrap CI/配对检验 (E6) 三条全都没有输入面。本开关只补那一段。
    parser.add_argument('--dump-scores', default=None,
                        help='逐 clip jsonl 落盘路径 (含 cause 9 位概率/scene 逐帧概率/真值位)。')
    parser.add_argument('--dump-limit', type=int, default=0,
                        help='只落前 N 条 (0=不限)。冒烟验收用, 不影响任何指标计算。')
    args = parser.parse_args()

    import torch

    train_config.ensure_cwd_v22()

    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("V23 验收评估")
    print("=" * 70)
    print(f"  检查点: {args.checkpoint}")
    print(f"  数据:   {args.data_root}")

    temporal_model, val_loader, meta, dev = build_model_and_data(args)

    # G-312(#273): 面必须自证"吃了哪一份字节"。此前 summary 只记 `checkpoint` 路径字符串
    #   ⇒ 今天在册面的 checkpoint=/tmp/g307/best_snapshots/g304_p3b_best_at_0946.pt,
    #   那份原件随 /tmp 一删, 论文引用就指向一个不存在的路径且无从证明它当时是什么。
    #   路径一律记解析后的绝对值 (本函数在 ensure_cwd_v22() 之后跑, 相对参数按包根解析,
    #   与 --data-root/--data-yaml 的既有口径一致)。只增读数, 不动任何指标。
    import hashlib as _hashlib
    from datetime import datetime as _dtm
    _ck_p = Path(args.checkpoint)
    _abs_out_dir = str(Path(args.output_dir).resolve())
    _abs_dump = str(Path(args.dump_scores).resolve()) if args.dump_scores else None
    _ckh = _hashlib.sha256()
    with open(_ck_p, 'rb') as _ckh_fh:
        for _ckh_blk in iter(lambda: _ckh_fh.read(1 << 22), b''):
            _ckh.update(_ckh_blk)
    _instr_meta_keys = {'fps_nominal', 'fps_eff', 'eff_stride', 'fps_eff_fail'}
    _ck_meta = {k: v for k, v in meta.items() if k not in _instr_meta_keys}
    checkpoint_identity = {
        'path_as_given': args.checkpoint,
        'path_resolved': str(_ck_p.resolve()),
        'bytes': int(_ck_p.stat().st_size),
        'sha256_16': _ckh.hexdigest()[:16],
        'mtime': _dtm.fromtimestamp(
            _ck_p.stat().st_mtime).isoformat(timespec='seconds'),
        # 件内顶层键 (已剔本仪器自己塞进去的 fps_* 四只 ⇒ 数得到的就是交付件键数,
        # 与 G-312 代际标记对齐: 11=3b 旧副本 / 13=3b 补键码 / 14=3c 起 7 位尺码)
        'top_level_keys': sorted(str(k) for k in _ck_meta.keys()),
        'scalars': {k: v for k, v in sorted(_ck_meta.items())
                    if isinstance(v, (int, float, str, bool, type(None)))},
    }
    print(f"  [G-312] 权重身份: sha256_16={checkpoint_identity['sha256_16']} "
          f"bytes={checkpoint_identity['bytes']} "
          f"件内键数={len(checkpoint_identity['top_level_keys'])}")

    ds_fps = 10.0
    from phase3_legacy import compute_detection_latency

    # ── 统计容器 ──
    scene_cm = [[0] * 3 for _ in range(3)]  # [gt][pred]: normal=0, pre_crash=1, crash=2
    frame_correct = frame_total = 0

    type_correct = type_total = 0
    type_detail = {}  # {type_idx: [correct, total]}
    # G-280(type-instrument): 三档分桶计数 (尺②, 与尺①共用 train_config.type_score_bucket)。
    # 旧 type_total 把 无输出槽(GT=-1/>=7) 与 有槽但课程外 两类不可评分样本算进分母 ⇒
    # 本尺实测分母 2,323 = cv 1,420 + 课程外 560 + 无槽 343, 裸 type_acc 被 61.1% 天花板压住。
    # 口径从 checkpoint 自带的相位取 (相位未知→只剔无槽)。旧键语义一律不动。
    # G-280b: P3 的件把相位写在**顶层 key 'phase'**（实测 ckpt_e8_G276.pt: phase='3a'、
    # meta['v24_phase'] 不存在 ⇒ 只读 v24_phase 会静默得 None, cv 分母从 1,420 涨到 1,980,
    # 把 87.39% 读成 ~62.7%）。两个键名都取, 取到哪个算哪个。
    type_cv_correct = type_cv_total = 0
    type_n_unslotable = type_n_labeled_out = 0
    _phase_key = meta.get('v24_phase') or meta.get('phase') or ''
    _active_types = train_config.PTCL_CONFIG.get(
        f"phase{_phase_key}_active_types")
    print(f"  [type-instrument] phase={_phase_key or '?'} "
          f"active_types={_active_types} "
          f"(None=不做课程过滤, 只剔无输出槽的 GT)")

    cause_tp = cause_fp = cause_fn = 0.0
    cause_n = 0
    cause_detail = {}  # {idx: [tp, fp, fn, gt_pos]}

    lat_sum = lat_count = lat_early = 0.0

    n_batches = len(val_loader)
    # G-307(#273): 只增读数面 ⇒ 打开与否都不参与任何下面的计数分支。
    _dump_fh = None
    _dump_n = 0
    if args.dump_scores:
        _dp = Path(args.dump_scores)
        _dp.parent.mkdir(parents=True, exist_ok=True)
        _dump_fh = open(_dp, 'w', encoding='utf-8')
        print(f"  [dump-scores] 打开 {_dp} (limit={args.dump_limit}, 0=不限)")
    print(f"\n── 开始评估 ({n_batches} batches) ──")

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        for b_idx, batch in enumerate(val_loader):
            imgs = batch['img'].to(dev)
            imgs = imgs.float() / 255.0 if imgs.dtype == torch.uint8 \
                else imgs.float()
            boxes = batch['boxes_seq'].to(dev)
            classes = batch['classes_seq'].to(dev)
            mask = batch['obj_mask'].to(dev)
            scene_seq = batch['scene_cls_seq'].to(dev)

            # clip-level scene GT (any-frame crash → crash clip)
            scene_t = batch.get('scene_cls', batch.get('scene_cls_seq'))
            if scene_t is not None:
                scene_t = scene_t.to(dev)
                if scene_t.dim() > 1:
                    clip_scene = (scene_t == 3).any(dim=-1).long() * 3
                else:
                    clip_scene = scene_t
            else:
                clip_scene = None

            # V25-fix(type-leak): 评估一律用 padding 类型输入 (与推理 Pass 1 对齐),
            # 禁止喂 GT accident_type 再对同一标签算 Type Acc
            type_ids = batch.get('accident_type', None)
            if type_ids is not None:
                type_ids = torch.full_like(type_ids.to(dev, dtype=torch.long), 7)

            # ── Forward ──
            # V25-fix(type-leak): 不再向模型传 GT boxes/classes/mask
            # (V24 主模型经 **kwargs 忽略, 但防止未来接线后激活泄漏)
            out = temporal_model(
                imgs, temporal_mode=True, type_ids=type_ids)

            frame_logits = out[5]
            type_logits = out[4]
            cause_logits = out[3]

            # ── Frame accuracy + confusion matrix ──
            frame_pred = frame_logits.argmax(dim=-1)
            frame_gt = (scene_seq - 1).clamp(-1, 2)
            frame_valid = scene_seq != 0

            frame_correct += ((frame_pred == frame_gt) & frame_valid).sum().item()
            frame_total += frame_valid.sum().item()

            B = frame_logits.shape[0]
            for i in range(B):
                valid_frames = frame_valid[i]
                if not valid_frames.any():
                    continue
                preds_i = frame_pred[i][valid_frames]
                gt_i = frame_gt[i][valid_frames]
                clip_gt = int(gt_i[-1].item())
                clip_pred = int(preds_i[-1].item())
                if 0 <= clip_gt < 3 and 0 <= clip_pred < 3:
                    scene_cm[clip_gt][clip_pred] += 1

            # ── Type accuracy (crash clips) ──
            if type_logits is not None and 'accident_type' in batch:
                type_gt = batch['accident_type'].to(dev, dtype=torch.long)
                if clip_scene is not None:
                    crash = (clip_scene == 3)
                    if crash.any():
                        tp = type_logits[crash].argmax(dim=-1)
                        tg = type_gt[crash]
                        type_correct += (tp == tg).sum().item()
                        type_total += crash.sum().item()
                        for j in range(tp.shape[0]):
                            ti = int(tg[j].item())
                            if ti not in type_detail:
                                type_detail[ti] = [0, 0]
                            type_detail[ti][1] += 1
                            if int(tp[j].item()) == ti:
                                type_detail[ti][0] += 1
                            # G-280(type-instrument): 在**同一个样本循环**里分桶 ⇒ 新口径
                            # 与旧口径取的是同一批 crash 样本, 不存在"两套宇宙"。
                            _bk = train_config.type_score_bucket(
                                ti, _active_types, len(TYPE_NAMES))
                            _ok = (int(tp[j].item()) == ti)
                            if _bk == 'cv':
                                type_cv_total += 1
                                if _ok:
                                    type_cv_correct += 1
                            elif _bk == 'unslotable':
                                type_n_unslotable += 1
                            else:
                                type_n_labeled_out += 1

            # ── Cause recall/F1 ──
            if cause_logits is not None and 'cause' in batch \
                    and clip_scene is not None:
                cause_gt = batch['cause'].to(dev, dtype=torch.float32)
                if cause_gt.dim() == 1:
                    cause_gt = cause_gt.view(-1, cause_logits.shape[-1])
                cgate = (clip_scene >= 2)
                if 'cause_valid' in batch:
                    cv = batch['cause_valid'].to(dev, dtype=torch.long)
                    cgate = cgate & (cv == 1)
                if cgate.any():
                    cause_n += cgate.sum().item()
                    csig = torch.sigmoid(cause_logits[cgate])
                    cp = (csig > 0.5).float()
                    cg = cause_gt[cgate].clamp(0, 1)
                    # V26: 已摘除因子同时从预测与真值里清零, 不计入 TP/FP/FN
                    #      (位下标保留, 只是不再作为评测项 —— 见 base_config)
                    dis = [i for i in sorted(base_config.CAUSE_DISABLED_BITS)
                           if i < cp.shape[1]]
                    if dis:
                        keep = torch.ones_like(cp)
                        keep[:, dis] = 0.0
                        cp = cp * keep
                        cg = cg * keep
                    cause_tp += ((cp == 1) & (cg == 1)).sum().item()
                    cause_fp += ((cp == 1) & (cg == 0)).sum().item()
                    cause_fn += ((cp == 0) & (cg == 1)).sum().item()
                    for ci in range(csig.shape[1]):
                        c_tp = int(((cp[:, ci] == 1) & (cg[:, ci] == 1)).sum().item())
                        c_fp = int(((cp[:, ci] == 1) & (cg[:, ci] == 0)).sum().item())
                        c_fn = int(((cp[:, ci] == 0) & (cg[:, ci] == 1)).sum().item())
                        c_gt_pos = int((cg[:, ci] == 1).sum().item())
                        if ci not in cause_detail:
                            cause_detail[ci] = [0, 0, 0, 0]
                        cause_detail[ci][0] += c_tp
                        cause_detail[ci][1] += c_fp
                        cause_detail[ci][2] += c_fn
                        cause_detail[ci][3] += c_gt_pos

            # ── G-307(#273) 逐 clip 分数流 (默认关; 开它只读不算, 不动上面任何累加器) ──
            #   落**原始概率 + 原始真值位**, 不落判据: 阈值/门/gate 口径由读数尺自己声明,
            #   这样同一份面能同时喂 E5(0.5 之外的全阈值扫描)与 E1(逐帧)而不互相污染。
            if _dump_fh is not None and (args.dump_limit <= 0
                                         or _dump_n < args.dump_limit):
                _p_fr = torch.softmax(frame_logits, dim=-1).float().cpu()
                # G-319(#273 E7): clip 级场景头 (out[1]) 此前从没进过任何读数面 ⇒ V25 臂
                # 帧头 crash 列全 0 时无法回答"信号在不在 clip 头"。只增读数, 不碰累加器。
                _p_sc = (torch.softmax(out[1], dim=-1).float().cpu()
                         if isinstance(out[1], torch.Tensor) and out[1].dim() == 2 else None)
                _p_ty = (torch.softmax(type_logits, dim=-1).float().cpu()
                         if type_logits is not None else None)
                _p_ca = (torch.sigmoid(cause_logits).float().cpu()
                         if cause_logits is not None else None)
                _g_ca = batch.get('cause', None)
                if _g_ca is not None:
                    _g_ca = _g_ca.cpu()
                    if _g_ca.dim() == 1 and _p_ca is not None:
                        _g_ca = _g_ca.view(-1, _p_ca.shape[1])
                _g_cv = batch.get('cause_valid', None)
                _g_cv = _g_cv.cpu() if _g_cv is not None else None
                _g_ty = batch.get('accident_type', None)
                _g_ty = _g_ty.cpu() if _g_ty is not None else None
                _v_ids = batch.get('video_id', None) or [''] * B
                _n_rf = batch.get('n_real_frames', None)
                _n_rf = _n_rf.cpu() if _n_rf is not None else None
                for _i in range(B):
                    _row = {
                        'row_id': f'{b_idx}:{_i}',
                        'video_id': str(_v_ids[_i]),
                        'clip_scene_gt': (int(clip_scene[_i])
                                          if clip_scene is not None else None),
                        'scene_gt_seq': scene_seq[_i].tolist(),
                        'n_real_frames': (int(_n_rf[_i]) if _n_rf is not None else None),
                        'frame_scene_prob': [[round(float(x), 6) for x in _t]
                                             for _t in _p_fr[_i]],
                        'clip_scene_prob': ([round(float(x), 6) for x in _p_sc[_i]]
                                            if _p_sc is not None else None),
                        'type_gt': (int(_g_ty[_i]) if _g_ty is not None else None),
                        'type_prob': ([round(float(x), 6) for x in _p_ty[_i]]
                                      if _p_ty is not None else None),
                        'cause_prob': ([round(float(x), 6) for x in _p_ca[_i]]
                                       if _p_ca is not None else None),
                        'cause_gt': ([float(x) for x in _g_ca[_i]]
                                     if _g_ca is not None and _g_ca.dim() > 1 else None),
                        'cause_valid': (int(_g_cv[_i]) if _g_cv is not None else None),
                    }
                    _dump_fh.write(json.dumps(_row, ensure_ascii=False) + '\n')
                    _dump_n += 1

            # ── Latency ──
            lat, n_lat, early = compute_detection_latency(
                frame_logits, scene_seq, fps=ds_fps)
            lat_sum += lat * n_lat
            lat_count += n_lat
            lat_early += early * n_lat

            if (b_idx + 1) % 20 == 0 or b_idx == n_batches - 1:
                print(f"  [{b_idx + 1}/{n_batches}] ...")

    # ── G-307(#273): 收尾只关句柄 + 报数, 不影响任何指标 ──
    if _dump_fh is not None:
        _dump_fh.close()
        print(f"  [dump-scores] 已落 {_dump_n} 行 → {args.dump_scores}")

    # ── Compute final metrics ──
    frame_acc = 100 * frame_correct / max(frame_total, 1)
    type_acc = 100 * type_correct / max(type_total, 1)
    # G-280(type-instrument): cv 口径 (分母只数可评分样本; 分母 0 记 -1 = 不可算, 不是 0)
    # 以及三档自对账 —— 三档必须恰好铺满旧分母, 不铺满就是仪器自己错了。
    type_acc_cv = (100 * type_cv_correct / type_cv_total
                   if type_cv_total > 0 else -1.0)
    type_acct_ok = (type_cv_total + type_n_unslotable + type_n_labeled_out
                    == type_total)
    latency = lat_sum / max(lat_count, 1)
    early_rate = 100 * lat_early / max(lat_count, 1)
    # G-304(F12): latency(fps) 与 early_rate 对 fps 是线性/符号不变的 (t_pred、t_gt 不变,
    #   只换除数; lat<=0 的判定不受正数缩放影响) ⇒ 修正口径直接换算, 不重跑前向。
    _fps_ratio = ds_fps / max(meta.get('fps_eff', ds_fps), 1e-6)
    latency_eff = latency * _fps_ratio

    total_clips = sum(scene_cm[gi][pi] for gi in range(3) for pi in range(3))
    scene_acc = 100 * sum(scene_cm[i][i] for i in range(3)) / max(total_clips, 1)

    crash_gt = sum(scene_cm[2])
    pre_crash_gt = sum(scene_cm[1])
    normal_gt = sum(scene_cm[0])
    pre_crash_recall = 100 * scene_cm[1][1] / max(pre_crash_gt, 1)
    crash_recall = 100 * scene_cm[2][2] / max(crash_gt, 1)
    normal_recall = 100 * scene_cm[0][0] / max(normal_gt, 1)
    pdr = 100 * (scene_cm[1][1] + scene_cm[2][2]) / max(pre_crash_gt + crash_gt, 1)
    far = 100 * (scene_cm[0][1] + scene_cm[0][2]) / max(normal_gt, 1)
    pre_crash_pred = sum(scene_cm[gi][1] for gi in range(3))
    pre_crash_precision = 100 * scene_cm[1][1] / max(pre_crash_pred, 1)
    # G-304(F13): scene 轴的多数类安全读数 (G-278 定的口径: raw acc 会被多数类主导,
    #   验收/报数只看 balAcc 与 macro-F1)。只加键, 旧 scene_acc/PDR/FAR 一字未动。
    #   宇宙 = scene_cm 的 clip 级末有效帧口径 (见 :312 clip_gt/clip_pred), 与 raw acc 同分母。
    _s3 = ['normal', 'pre_crash', 'crash']
    _scene_pers = []
    for _ci in range(3):
        _tp = scene_cm[_ci][_ci]
        _fp = sum(scene_cm[g][_ci] for g in range(3)) - _tp
        _fn = sum(scene_cm[_ci]) - _tp
        _rec = _tp / max(_tp + _fn, 1)
        _pre = _tp / max(_tp + _fp, 1)
        _f1 = 2 * _pre * _rec / max(_pre + _rec, 1e-12)
        _scene_pers.append({'label': _s3[_ci], 'recall': 100 * _rec,
                            'precision': 100 * _pre, 'f1': 100 * _f1,
                            'gt_n': _tp + _fn})
    scene_bal_acc = sum(p['recall'] for p in _scene_pers) / 3.0
    scene_macro_f1 = sum(p['f1'] for p in _scene_pers) / 3.0

    cause_precision = cause_tp / max(cause_tp + cause_fp, 1)
    cause_recall = cause_tp / max(cause_tp + cause_fn, 1)
    cause_f1 = 2 * cause_precision * cause_recall / max(cause_precision + cause_recall, 1e-8)

    # ── Output CSVs ──
    labels = ['normal', 'pre_crash', 'crash']

    sc_path = os.path.join(args.output_dir, 'scene_confusion.csv')
    with open(sc_path, 'w') as f:
        f.write('gt\\pred,' + ','.join(labels) + '\n')
        for gi in range(3):
            f.write(f'{labels[gi]},' + ','.join(str(scene_cm[gi][pi]) for pi in range(3)) + '\n')
        f.write('\n# Metrics\n')
        f.write(f'Accuracy,{scene_acc:.2f}%\n')
        f.write(f'PDR,{pdr:.2f}%\n')
        f.write(f'FAR,{far:.2f}%\n')
        f.write(f'pre_crash_Recall,{pre_crash_recall:.2f}%\n')
        f.write(f'pre_crash_Precision,{pre_crash_precision:.2f}%\n')
        f.write(f'crash_Recall,{crash_recall:.2f}%\n')
        f.write(f'normal_Recall,{normal_recall:.2f}%\n')
        # G-304(F13): 新增多数类安全读数 + 逐类并列 (G-278: 三类别均值会抹掉单类退步)
        f.write(f'BalancedAccuracy,{scene_bal_acc:.2f}%\n')
        f.write(f'MacroF1,{scene_macro_f1:.2f}%\n')
        for _p in _scene_pers:
            f.write(f'per_class[{_p["label"]}],'
                    f'gt_n={_p["gt_n"]},R={_p["recall"]:.2f}%,P={_p["precision"]:.2f}%,'
                    f'F1={_p["f1"]:.2f}%\n')
    print(f"\n  场景混淆矩阵 → {sc_path}")

    ct_path = os.path.join(args.output_dir, 'cause_type_metrics.csv')
    with open(ct_path, 'w') as f:
        f.write('task,name,gt_positive,predicted_positive,tp,fp,fn,precision,recall\n')
        for ci in range(len(CAUSE_NAMES)):
            name = CAUSE_NAMES[ci] if ci < len(CAUSE_NAMES) else f'cause_{ci}'
            tp_c, fp_c, fn_c, gt_pos = cause_detail.get(ci, [0, 0, 0, 0])
            pred_pos = tp_c + fp_c
            prec_c = 100 * tp_c / max(pred_pos, 1)
            rec_c = 100 * tp_c / max(gt_pos, 1) if gt_pos > 0 else 0.0
            f.write(f'cause,{name},{gt_pos},{pred_pos},{tp_c},{fp_c},{fn_c},'
                    f'{prec_c:.2f},{rec_c:.2f}\n')
        f.write(f'cause,MICRO_ALL,{int(cause_tp + cause_fn)},'
                f'{int(cause_tp + cause_fp)},{int(cause_tp)},{int(cause_fp)},'
                f'{int(cause_fn)},{cause_precision * 100:.2f},'
                f'{cause_recall * 100:.2f}\n')
        f.write(f'# cause_active_samples,{int(cause_n)}\n')
        _pred_rate = (cause_tp + cause_fp) / max(cause_n * len(CAUSE_NAMES), 1) * 100
        f.write(f'# positive_prediction_rate,{_pred_rate:.2f}%\n')

        _cv_set = set(_active_types) if _active_types else set(range(len(TYPE_NAMES)))
        for ti in range(len(TYPE_NAMES)):
            name = TYPE_NAMES[ti] if ti < len(TYPE_NAMES) else f'type_{ti}'
            correct_t, total_t = type_detail.get(ti, [0, 0])
            rec_t = 100 * correct_t / max(total_t, 1)
            f.write(f'type,{name},{total_t},,,,{correct_t},,\n')
        # G-280(type-instrument): 表外 GT (无输出槽, 如 -1) 也在 type_detail 里, 但旧写法
        # 只循环 range(len(TYPE_NAMES)) ⇒ 它们被静默丢弃, 面上 Σ类GT(1,980) ≠ 分母(2,323)。
        # 补一行让它们可见, 并写下三档对账与口径, 让"分母里有构造性必错"这件事在面上可读。
        _oob_t = sum(t for k, (c, t) in type_detail.items()
                     if not (0 <= k < len(TYPE_NAMES)))
        _oob_c = sum(c for k, (c, t) in type_detail.items()
                     if not (0 <= k < len(TYPE_NAMES)))
        if _oob_t:
            f.write(f'type,no_output_slot,{_oob_t},,,,{_oob_c},,\n')
        f.write(f'# type_total(全部crash clip,含不可评分样本),{type_total}\n')
        f.write(f'# type_cv(有槽且在课程内)=可评分分母,{type_cv_total},'
                f'正确{type_cv_correct},'
                f'{"%.2f" % type_acc_cv if type_acc_cv >= 0 else "N/A"}%\n')
        f.write(f'# type_labeled_out(有槽但课程外,从未给梯度),{type_n_labeled_out}\n')
        f.write(f'# type_unslotable(无槽,构造性必错),{type_n_unslotable}\n')
        f.write(f'# type_acct(三档之和==type_total),{type_acct_ok}\n')
        f.write(f'# type_curriculum(phase={_phase_key or "?"},'
                f'cv类={sorted(_cv_set)})\n')
    print(f"  成因/类型指标 → {ct_path}")

    # ── JSON summary ──
    summary = {
        'checkpoint': args.checkpoint,
        'scene_acc': round(scene_acc, 2),
        'frame_acc': round(frame_acc, 2),
        'type_acc': round(type_acc, 2),
        'PDR': round(pdr, 2),
        'FAR': round(far, 2),
        'pre_crash_recall': round(pre_crash_recall, 2),
        'pre_crash_precision': round(pre_crash_precision, 2),
        'crash_recall': round(crash_recall, 2),
        'normal_recall': round(normal_recall, 2),
        'detection_latency_s': round(latency, 3),
        'early_detection_rate': round(early_rate, 2),
        'cause_recall': round(cause_recall * 100, 2),
        'cause_precision': round(cause_precision * 100, 2),
        'cause_f1': round(cause_f1 * 100, 2),
        'cause_active_clips': int(cause_n),
        'type_total_crash': type_total,
        # G-280(type-instrument): 旧 type_acc/type_total_crash 逐字保留 (历史面不断档),
        # 新口径另落键。以后读 type 一律读 type_acc_cv, 裸 type_acc 只作连续性对照。
        'type_acc_cv': (round(type_acc_cv, 2) if type_cv_total > 0 else None),
        'type_cv_total': int(type_cv_total),
        'type_cv_correct': int(type_cv_correct),
        'type_n_labeled_out': int(type_n_labeled_out),
        'type_n_unslotable': int(type_n_unslotable),
        'type_acct_ok': bool(type_acct_ok),
        'type_phase': _phase_key or '?',
        'type_curriculum_active': (sorted(_active_types)
                                   if _active_types else None),
        'scene_confusion_matrix': scene_cm,
        'total_clips': total_clips,
        # G-304(F12/F13): 只加键，上面每个旧键的值与算法一字未动 ⇒ 历史面不断档。
        #   scene 轴补多数类安全读数 (G-278: raw acc 被多数类主导, 三类别均值会抹掉单类退步);
        #   latency 补有效 fps 口径 (G-302: stride-1 滑窗 ÷clip_len ⇒ 名义 fps 虚报)。
        #   旧 detection_latency_s 保留 = 名义 fps 口径, 引用时必须带口径名。
        'scene_bal_acc': round(scene_bal_acc, 2),
        'scene_macro_f1': round(scene_macro_f1, 2),
        'scene_per_class': [{'label': _p['label'], 'gt_n': int(_p['gt_n']),
                             'recall': round(_p['recall'], 2),
                             'precision': round(_p['precision'], 2),
                             'f1': round(_p['f1'], 2)} for _p in _scene_pers],
        'detection_latency_s_nominal_fps': round(latency, 3),
        'detection_latency_s_eff': round(latency_eff, 3),
        'fps_nominal': meta.get('fps_nominal'),
        'fps_eff': meta.get('fps_eff'),
        'eff_stride': meta.get('eff_stride'),
        # G-306: eff_stride=1 有两种成因 (数据真是 stride-1 / 估计崩了回退名义),
        #   本次 bug 正是被这个歧义掩盖的 ⇒ 把回退原因显式落面。
        'fps_eff_fail': meta.get('fps_eff_fail'),
        # ── G-312(#273) 只增读数 (上面每个旧键的值与算法一字未动) ──
        # 1) 权重身份: 路径字符串不能当出处, 字节 sha 才能。
        'checkpoint_identity': checkpoint_identity,
        'output_dir_resolved': _abs_out_dir,
        # 2) 分母落面: latency/early 的除数是 lat_count, 而旧键在 lat_count=0 时
        #    由 max(.,1) 兜成 0.0 ⇒ "零样本"会伪装成"零延迟"。n=0 即该数无判据力。
        'detection_latency_n': int(lat_count),
        'early_detection_rate_n': int(lat_count),
        # 3) 分数流声明: E1/E5 都吃这盘面, 行数/上限/宇宙必须自带, 否则读数尺分母
        #    只能靠手写 manifest (今天那份就是这么来的)。
        'dump_scores': {
            'path_resolved': _abs_dump,
            'rows_written': int(_dump_n),
            'limit': int(args.dump_limit),
            'truncated': bool(args.dump_limit > 0 and _dump_n >= args.dump_limit),
            'universe_total_clips': int(total_clips),
        },
        # 4) 考卷范围 (G-314/E7): 全量 val 与"两臂共同考卷"是**两张卷子**, 分数不可互比
        #    (V26 全量 val 里有 4,659 帧落在 V25 的训练集) ⇒ 口径必须写在面里, 不能靠记忆。
        'eval_scope': ({'kind': 'e7_common_exam', **meta['e7_frame_list']}
                       if meta.get('e7_frame_list') else {'kind': 'full_val'}),
    }
    json_path = os.path.join(args.output_dir, 'eval_acceptance.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("验收评估结果")
    print("=" * 70)
    print(f"  Scene Acc:       {scene_acc:.2f}%")
    # G-304(F13): raw acc 只作连续性对照, 验收判据用下面两条 (G-278 口径)
    print(f"  Scene balAcc:    {scene_bal_acc:.2f}%   (多数类安全, 与 raw acc 同宇宙)")
    print(f"  Scene macroF1:   {scene_macro_f1:.2f}%   逐类: " +
          ' | '.join(f"{_p['label']} R={_p['recall']:.1f} P={_p['precision']:.1f} "
                     f"F1={_p['f1']:.1f} n={_p['gt_n']}" for _p in _scene_pers))
    print(f"  Frame Acc:       {frame_acc:.2f}%")
    print(f"  Type Acc:        {type_acc:.2f}%")
    # G-280(type-instrument): 裸口径含构造性必错样本 ⇒ 并列打印可评分口径与三档构成
    print(f"  Type Acc(cv):    "
          f"{('%.2f%%' % type_acc_cv) if type_cv_total > 0 else 'N/A(无可评分样本)'}"
          f"  [{type_cv_correct}/{type_cv_total}] | 课程外{type_n_labeled_out} "
          f"无槽{type_n_unslotable} | 三档对账{'OK' if type_acct_ok else 'MISMATCH!'}")
    print(f"  PDR:             {pdr:.2f}%")
    print(f"  FAR:             {far:.2f}%")
    print(f"  pre_crash R/P:   {pre_crash_recall:.2f}% / {pre_crash_precision:.2f}%")
    print(f"  crash Recall:    {crash_recall:.2f}%")
    print(f"  normal Recall:   {normal_recall:.2f}%")
    print(f"  Latency:         {latency:+.3f}s")
    # G-304(F12): 上一条 = 名义 fps 口径 (会低估 stride 倍); 秒口径看这条
    print(f"  Latency(eff fps): {latency_eff:+.3f}s   "
          f"(名义 {meta.get('fps_nominal')} → 有效 {meta.get('fps_eff')}, "
          f"stride={meta.get('eff_stride')}, 比值={_fps_ratio:.2f}x)")
    print(f"  Early Rate:      {early_rate:.2f}%")
    print(f"  Cause R/P/F1:    {cause_recall*100:.2f}% / {cause_precision*100:.2f}% / {cause_f1*100:.2f}%")
    print(f"  Cause clips:     {int(cause_n)}")
    print(f"  Type crash clips:{type_total}")
    print(f"\n  汇总 JSON → {json_path}")
    print("=" * 70)

    return summary


if __name__ == '__main__':
    main()
