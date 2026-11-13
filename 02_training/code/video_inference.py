#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_inference.py — V26 视频推理脚本

对真实事故视频进行逐帧推理，输出:
  - 标注后的可视化视频 (检测框 + 场景标签 + crash 帧红线事故区域/类型/成因)
  - <video>_frames.csv (逐帧: 场景/三概率/类型/成因/框数/区域/触发因)
  - inference_results.json (性能 + 检测统计 + 逐帧事件明细)
  - meta_face_dir 下面: video_inference_summary.json + video_events.csv (事件级口径)

用法:
    python scripts/video_inference.py                      # 用默认正本配置
    python scripts/video_inference.py --videos /path/dir --device 1
    python scripts/video_inference.py --selftest          # 离线自检, 不占卡
权重装载复用 eval_acceptance.build_models (clip_len 与 shape 不一致当场硬失败);
推理卡请用 --device 指到空闲卡, 别和训练抢 0。
"""

import os
import sys
import csv
import json
import time
import glob
import hashlib
import argparse
from pathlib import Path
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

import functools
print = functools.partial(print, flush=True)

import train_config
from train_config import PROJECT_ROOT_V22

_V24_CORE = os.path.join(SCRIPT_DIR, 'core')
if _V24_CORE not in sys.path:
    sys.path.insert(0, _V24_CORE)
from base_config import CAUSE_NAMES, TYPE_NAMES, CAUSE_DISABLED_BITS

SCENE_NAMES = ['normal', 'pre_crash', 'crash']
SCENE_COLORS = {
    'normal': (0, 200, 0),       # green
    'pre_crash': (0, 165, 255),   # orange
    'crash': (0, 0, 255),         # red
}

# 成因/类型中文名 (索引与 base_config.CAUSE_NAMES / TYPE_NAMES 对齐)
# V24 删过 3 个零样本成因、合并过 3 个类型，这里曾是 V20 的 16/10 条旧表 ⇒ 按索引画到
# 视频上的中文全是错位的。下面的断言是防线：位表一改，这里当场报错而不是悄悄画错字。
# G-353 用语修订（用户实看反馈："路肩停驶"非大陆说法）：只动显示中文表，
# 数据层英文名（base_config.CAUSE_NAMES/TYPE_NAMES）一字未动。
# 口径=交管执法/日常用语：standing_shoulder→应急车道违停（"路肩"是工程/日系词）、
# tailgating→跟车过近、low_visibility→视线不良（"低能见度"是气象词）、
# pedestrian_highway→行人上高速、pedestrian_hit→撞行人、head_on→对向碰撞（官方类目名）。
CAUSE_NAMES_CN = ['跟车过近', '超速', '违规变道', '应急车道违停', '行人上高速',
                  '大雨', '大雪', '分心驾驶', '视线不良']
TYPE_NAMES_CN = ['追尾', '侧面碰撞', '翻车', '撞行人', '对向碰撞', '撞固定物', '其他']
assert len(CAUSE_NAMES_CN) == len(CAUSE_NAMES), \
    f"成因中文表 {len(CAUSE_NAMES_CN)} 条 != base_config {len(CAUSE_NAMES)} 条"
assert len(TYPE_NAMES_CN) == len(TYPE_NAMES), \
    f"类型中文表 {len(TYPE_NAMES_CN)} 条 != base_config {len(TYPE_NAMES)} 条"

# 已摘除因子 (评测与展示同一口径: 不参与推理输出)
CAUSE_ACTIVE = [0 if i in CAUSE_DISABLED_BITS else 1 for i in range(len(CAUSE_NAMES))]

# 中文渲染字体 (Noto Sans CJK; 加载失败回退 cv2 英文)
_CJK_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
_FONT_CACHE = {}


def get_cjk_font(size):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(_CJK_FONT_PATH, size)
        except OSError:
            _FONT_CACHE[size] = False
    return _FONT_CACHE[size]


def pil_text(frame, lines, org, font_size=22, color=(255, 255, 255),
             bg=(0, 0, 0), bg_alpha=0.55, pad=6):
    """在 BGR 帧上绘制多行中文文本 (PIL 渲染), 自动画半透明底。
    lines: [(text, color)] 或 [str]; org: 第一行左上角 (x, y)。返回绘制后的帧。"""
    font = get_cjk_font(font_size)
    if font is False:
        # 回退: cv2 只能画 ASCII
        y = org[1]
        for ln in lines:
            t = ln[0] if isinstance(ln, tuple) else ln
            cv2.putText(frame, t.encode('ascii', 'ignore').decode(), (org[0], y + font_size),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y += font_size + 6
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    draw = ImageDraw.Draw(img)
    x, y = org
    line_h = font_size + 8
    box_w = max(draw.textlength(ln[0] if isinstance(ln, tuple) else ln, font=font)
                for ln in lines) + pad * 2
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([x - pad, y - pad, x + box_w, y + line_h * len(lines) + pad],
                    fill=bg + (int(255 * bg_alpha),))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        t, c = ln if isinstance(ln, tuple) else (ln, color)
        draw.text((x, y + i * line_h), t, font=font, fill=c)
    return cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # resolve relative paths
    for key in ('checkpoint', 'model_yaml', 'output_dir', 'box_weights', 'meta_face_dir'):
        val = cfg.get(key)
        if val and not os.path.isabs(val):
            cfg[key] = os.path.join(PROJECT_ROOT_V22, val)
    return cfg


def build_model(cfg, device):
    """构建时序模型并加载交付 checkpoint。

    权重装载复用 eval_acceptance.build_models 那一把尺: clip_len 与权重 shape 不一致时
    硬失败。原来这里自带一份 key 映射启发式 (<50% 就改前缀) + strict=False, shape 不匹配
    的键被静默跳过 ⇒ 拿随机 pos_embed 推理, 数照出但全是假的 (报告 §8.24)。
    """
    from types import SimpleNamespace
    from eval_acceptance import build_models

    # det 支模型只是时序模型的容器; 检测框另有独立来源 (见 build_detector_yolo / #271)
    _det, temporal_model, dev, meta = build_models(SimpleNamespace(
        checkpoint=cfg['checkpoint'], model_yaml=cfg['model_yaml'],
        clip_len=cfg['clip_len'], device=str(device)))
    return temporal_model, meta, dev


def build_detector_yolo(cfg):
    """加载 COCO 预训练 YOLO 用于检测框可视化。

    刻意不用模型自带的检测头: 相位3 的时序梯度会改写检测支路 (#271, 正式 mAP 实测塌到
    1e-4 量级), 画上去只会是噪声。框仅用于合成事故区域, 结果里以 box_source 记明权重,
    免得被当作模型自检输出引用。原来的 `yolov8n.pt` 兜底会静默联网下载另一个模型, 已删。
    """
    from ultralytics import YOLO
    weights = cfg.get('box_weights') or os.path.join(
        PROJECT_ROOT_V22, 'weights', 'yolo26n.pt')
    if not os.path.exists(weights):
        raise SystemExit(f"检测框权重不存在: {weights} ⇒ 改 inference_config.yaml 的 "
                         "box_weights, 不要让它悄悄去下载另一个模型")
    model = YOLO(weights)
    print(f"  检测可视化模型: {weights}")
    return model


def extract_frames(video_path, fps_sample, max_frames):
    """从视频中按指定 FPS 抽帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  错误: 无法打开视频 {video_path}")
        return [], [], {}     # 原来返回 ([], 0, 0, 0) 与下面的三元解包对不上

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if orig_fps > 0 else 0

    step = max(1, int(round(orig_fps / fps_sample)))
    frames = []
    frame_indices = []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(img)
            frame_indices.append(i)
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()

    video_info = {
        'width': width, 'height': height,
        'orig_fps': round(orig_fps, 2),
        'total_frames': total_frames,
        'duration_s': round(duration, 2),
        'sampled_frames': len(frames),
        'sample_fps': fps_sample,
    }
    return frames, frame_indices, video_info


def preprocess_frame(frame, input_size):
    """预处理单帧: LetterBox resize + normalize → [1, 3, H, W] tensor
    与训练管线对齐: 保持宽高比, stride=32 对齐, 灰色填充 (pad=114)
    """
    from ultralytics.data.augment import LetterBox
    lb = LetterBox(new_shape=(input_size, input_size), stride=32, auto=False)
    img = lb(image=frame)  # BGR, LetterBox padded
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


# CTTA type_ids padding index (训练时未知类型用7)
TYPE_ID_UNKNOWN = 7


def run_temporal_streaming(temporal_model, frames, clip_len, input_size, dev):
    """流式滑动窗口时序推理: per-clip 类型条件化
    V25-fix(type-bias-v3):
      Pass 1: type_ids=UNKNOWN → 获取每个 clip 的无偏类型预测
      Pass 2: 每个 clip 用自己的预测类型条件化 CTTA → 改善 scene/cause
      类型输出使用 Pass1 的无偏预测 (避免 CTTA 自我强化偏置)

    返回 dict (帧序列一律裁到输入帧数, 长度与调用方遍历的 frames 严格相等)。
    """
    n_orig = len(frames)
    pad_len = max(0, clip_len - n_orig)
    if pad_len:
        frames = [frames[-1]] * pad_len + list(frames)
    n = len(frames)

    n_clips = n - clip_len + 1
    clip_ms = []

    def _run_pass(type_ids_tensor, measure=False):
        """type_ids_tensor: [1] 全局 或 [n_clips] per-clip"""
        probs_sum = np.zeros((n, 3), dtype=np.float64)
        cause_sum = np.zeros((n, len(CAUSE_NAMES)), dtype=np.float64)
        type_sum = np.zeros((n, len(TYPE_NAMES)), dtype=np.float64)
        cover_count = np.zeros(n, dtype=np.int32)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            for i in range(n_clips):
                # 计时覆盖预处理 + 上传 + 前向: 只掐前向会把 fps 报得比真实部署快
                t_clip = time.perf_counter() if measure else None
                tensors = [preprocess_frame(frames[i + t], input_size) for t in range(clip_len)]
                clip = torch.cat(tensors, dim=0).unsqueeze(0).to(dev)

                if type_ids_tensor.dim() == 0 or type_ids_tensor.shape[0] == 1:
                    tid = type_ids_tensor
                else:
                    tid = type_ids_tensor[i:i+1]

                out = temporal_model(clip, temporal_mode=True, type_ids=tid)
                frame_logits = out[5]
                cause_logits, type_logits = out[3], out[4]
                cause_p = (torch.sigmoid(cause_logits.float()).squeeze(0).cpu().numpy()
                           if cause_logits is not None else None)
                type_p = (torch.softmax(type_logits.float(), dim=-1).squeeze(0).cpu().numpy()
                          if type_logits is not None else None)
                if measure:
                    clip_ms.append((time.perf_counter() - t_clip) * 1000.0)

                if frame_logits is not None:
                    probs = torch.softmax(frame_logits, dim=-1)
                    prob_arr = probs.squeeze(0).cpu().numpy()
                    for t in range(clip_len):
                        idx = i + t
                        if idx < n:
                            probs_sum[idx] += prob_arr[t]
                            if cause_p is not None:
                                cause_sum[idx] += cause_p
                            if type_p is not None:
                                type_sum[idx] += type_p
                            cover_count[idx] += 1

                if (i + 1) % 50 == 0 or i == n_clips - 1:
                    print(f"    clips: {i+1}/{n_clips}")

        cover_count = np.maximum(cover_count, 1)
        return (probs_sum / cover_count[:, None],
                cause_sum / cover_count[:, None],
                type_sum / cover_count[:, None])

    # ── Pass 1: UNKNOWN 类型 → 无偏类型预测 + 初始 scene/cause ──
    type_ids_unknown = torch.tensor([TYPE_ID_UNKNOWN], dtype=torch.long, device=dev)
    _, _, type_avg_1 = _run_pass(type_ids_unknown)

    # 每个 clip 的类型预测 (无偏)
    clip_type_preds = type_avg_1.argmax(axis=1)  # [n] per-frame, 需映射到 clip
    # 构建 per-clip type_ids: 取每个 clip 覆盖帧的多数类型
    clip_type_ids = np.zeros(n_clips, dtype=np.int64)
    for i in range(n_clips):
        frame_range = range(i, min(i + clip_len, n))
        clip_types_in = clip_type_preds[list(frame_range)]
        clip_type_ids[i] = int(np.bincount(clip_types_in.astype(int)).argmax())

    from collections import Counter
    type_dist = Counter(clip_type_ids.tolist())
    print(f"  [per-clip 条件化] 类型分布: {dict(type_dist)}")

    # ── Pass 2: per-clip 类型条件化 → 改善 scene/cause ──
    type_ids_per_clip = torch.tensor(clip_type_ids, dtype=torch.long, device=dev)
    probs_avg, cause_avg, type_avg = _run_pass(type_ids_per_clip, measure=True)

    # 类型输出使用 Pass1 无偏预测 (覆盖 Pass2 的 CTTA 自我强化)
    type_avg = type_avg_1

    # 裁掉前置 pad: 不裁等于整条预测序列左移 pad_len 格 —— 短视频会把"最后一帧副本"的
    # 预测画到第 0 帧上 (调用方按原始帧数遍历, 两边长度一直不匹配)。
    probs_avg = probs_avg[pad_len:]
    cause_avg = cause_avg[pad_len:]
    type_avg = type_avg[pad_len:]

    # 已摘除因子压成 0: 评测面 (eval_acceptance) 已按 CAUSE_DISABLED_BITS 屏蔽,
    # 推理面不屏蔽就会把 pedestrian_highway / distracted_driving 当结论播出去。
    cause_avg = cause_avg * np.asarray(CAUSE_ACTIVE, dtype=np.float64)[None, :]

    frame_preds = probs_avg.argmax(axis=1).tolist()
    frame_confs = probs_avg.max(axis=1).tolist()
    frame_probs = probs_avg.tolist()

    prob_stats = {
        'mean_normal_prob': round(float(probs_avg[:, 0].mean()), 4),
        'mean_precrash_prob': round(float(probs_avg[:, 1].mean()), 4),
        'mean_crash_prob': round(float(probs_avg[:, 2].mean()), 4),
        'max_precrash_prob': round(float(probs_avg[:, 1].max()), 4),
        'max_crash_prob': round(float(probs_avg[:, 2].max()), 4),
    }

    lat = np.asarray(clip_ms, dtype=np.float64)
    timing = {}
    if lat.size:
        mean_ms = float(lat.mean())
        timing = {
            'clip_ms_mean': round(mean_ms, 2),
            'clip_ms_p50': round(float(np.percentile(lat, 50)), 2),
            'clip_ms_p95': round(float(np.percentile(lat, 95)), 2),
            'clip_ms_p99': round(float(np.percentile(lat, 99)), 2),
            # 滑窗步长 1: 每前进一帧都要重算整窗 ⇒ 帧成本 ≈ 一次 clip 前向, 不能除 clip_len
            'stride1_frames_per_s': round(1000.0 / mean_ms, 2) if mean_ms > 0 else None,
            # 不重叠分块 (stride=clip_len) 的上界, 只是参照, 不是本脚本的跑法
            'nonoverlap_frames_per_s': round(clip_len * 1000.0 / mean_ms, 2)
                if mean_ms > 0 else None,
            'n_clip_timed': int(lat.size),
        }
    return {
        'frame_preds': frame_preds, 'frame_confs': frame_confs,
        'frame_probs': frame_probs, 'prob_stats': prob_stats,
        'cause_avg': cause_avg, 'type_avg': type_avg,
        'timing': timing, 'pad_len': pad_len, 'n_clips': n_clips,
    }


def detect_frame(det_yolo, frame, conf_threshold, input_size, device):
    """使用 YOLO 检测单帧中的目标，返回 boxes

    device 必须跟着 --device 走: 原来硬编码 '0', 训练占着 0 卡时要么抢显存要么把框跑到
    另一张卡上。异常也不再吞成 [] —— "0 个框" 和 "检测器崩了" 是两件事, 混在一起会把
    仪器故障读成场景平静。
    """
    results = det_yolo(frame, conf=conf_threshold, imgsz=input_size,
                       verbose=False, device=str(device))
    boxes = []
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = det_yolo.names.get(cls_id, str(cls_id))
                boxes.append({
                    'xyxy': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': round(conf, 3),
                    'cls': cls_name,
                })
    return boxes


class AccidentRegionTracker:
    """从检测框推导事故区域: 紧密簇 (≥2 个相互接近的框) 的最小外接矩形 + 时序平滑。

    模型不直接预测"事故区域", 用启发式合成:
    - 帧内: 车辆/人框两两算边缘间隙, 间隙 < 阈值连边, 取最紧密连通簇
    - 时序: 最近 K 帧候选区域中位数 + EMA 平滑, ≥min_hits 帧命中才显示 (防抖)
    """

    def __init__(self, gap_ratio=0.45, history=8, min_hits=3, ema=0.45):
        self.gap_ratio = gap_ratio
        self.history = history
        self.min_hits = min_hits
        self.ema = ema
        self._recent = []      # 最近候选区域 [(x1,y1,x2,y2)]
        self._smoothed = None  # EMA 平滑后的区域

    @staticmethod
    def _diag(b):
        return float(np.hypot(b[2] - b[0], b[3] - b[1]))

    def _frame_region(self, boxes):
        veh = [b['xyxy'] for b in boxes if b['cls'] in ('car', 'person', 'cyclist',
                                                        'truck', 'bus')]
        veh = [b for b in veh if self._diag(b) > 24]  # 噪声小框剔除
        if len(veh) < 2:
            return None
        n = len(veh)
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                a, b = veh[i], veh[j]
                gap_x = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
                gap_y = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
                gap = float(np.hypot(gap_x, gap_y))
                if gap < self.gap_ratio * max(self._diag(a), self._diag(b)):
                    adj[i][j] = adj[j][i] = True
        # 连通分量
        seen, comps = set(), []
        for i in range(n):
            if i in seen:
                continue
            stack, comp = [i], []
            seen.add(i)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(n):
                    if adj[u][v] and v not in seen:
                        seen.add(v)
                        stack.append(v)
            if len(comp) >= 2:
                comps.append(comp)
        if not comps:
            return None
        # 最紧密簇 = 外接矩形面积最小者 (排除路边成排停车的松散长链: 面积大)
        best = min(comps, key=lambda c: np.hypot(
            max(veh[k][2] for k in c) - min(veh[k][0] for k in c),
            max(veh[k][3] for k in c) - min(veh[k][1] for k in c)))
        xs1 = min(veh[k][0] for k in best)
        ys1 = min(veh[k][1] for k in best)
        xs2 = max(veh[k][2] for k in best)
        ys2 = max(veh[k][3] for k in best)
        return [float(xs1), float(ys1), float(xs2), float(ys2)]

    def update(self, boxes):
        """喂入当前帧检测框; 返回平滑后区域 [x1,y1,x2,y2] 或 None"""
        r = self._frame_region(boxes)
        if r is not None:
            self._recent.append(r)
        if len(self._recent) > self.history:
            self._recent.pop(0)
        if len(self._recent) < self.min_hits:
            self._smoothed = None if r is None else self._smoothed
            return self._smoothed
        arr = np.median(np.array(self._recent), axis=0)
        if self._smoothed is None:
            self._smoothed = arr.copy()
        else:
            self._smoothed = self.ema * arr + (1 - self.ema) * self._smoothed
        return self._smoothed.tolist()

    def reset(self):
        self._recent = []
        self._smoothed = None


def _type_cause_lines(type_info, causes):
    """类型/成因文本行（中文优先，英文括注）——G-353 抽出共用。"""
    lines = []
    if type_info:
        ncn, nen, tc = type_info
        lines.append((f"事故类型: {ncn} {nen} ({tc:.2f})", (80, 200, 255)))
    if causes:
        cs = ' · '.join(f"{n} {c:.2f}" for n, c in causes)
        lines.append((f"成因: {cs}", (120, 120, 255)))
    return lines


def draw_overlay(frame, scene_label, scene_conf, boxes, frame_idx,
                 region=None, type_info=None, causes=None, probs=None,
                 accident=False):
    """在帧上绘制场景标签、检测框、事故区域红框与类型/成因文本。

    region: [x1,y1,x2,y2] 事故区域
    type_info: (name_cn, name_en, conf) 事故类型
    causes: [(name_cn, conf), ...] top-k 成因
    probs: [p_normal, p_pre_crash, p_crash] G-353 三概率显示条——scene 头存在
        "混淆惩罚过度抑制"（所以才配了概率触发器），argmax 单标签会把
        pre_crash/低峰 crash 的打分完全隐藏；把三概率画出来才看得到"打了分"。
    accident: 该帧是否被三重触发器判为事故帧。G-353（用户 7 点时间戳复核）：
        触发命中但 argmax 未落 crash 的帧此前只有一闪而过的横幅，人眼不可见
        ⇒ 事故判定与 argmax 解耦显示：命中即挂红底 ACCIDENT 徽标并照常显示类型/成因。
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 场景标签背景
    color = SCENE_COLORS.get(scene_label, (128, 128, 128))
    label_text = f"{scene_label} ({scene_conf:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, w / 1600)
    thickness = max(1, int(w / 800))

    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    cv2.rectangle(overlay, (0, 0), (tw + 20, th + 20), color, -1)
    cv2.putText(overlay, label_text, (10, th + 10), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    # G-353: 三概率显示条 + pre_crash 预警副标（打分可见性）
    if probs is not None:
        strip = f"N {probs[0]:.2f} | P {probs[1]:.2f} | C {probs[2]:.2f}"
        cv2.putText(overlay, strip, (10, th + 36), font, font_scale * 0.72,
                    (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
        if scene_label != 'pre_crash' and probs[1] >= 0.25:
            bw = int(tw * 1.5) + 20
            cv2.rectangle(overlay, (0, th + 46), (bw, th + 72),
                          SCENE_COLORS['pre_crash'], -1)
            cv2.putText(overlay, 'pre_crash warning', (10, th + 66), font,
                        font_scale * 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 检测框
    det_color_map = {'car': (255, 200, 0), 'person': (0, 255, 255),
                     'cyclist': (255, 0, 255), 'truck': (0, 128, 255),
                     'bus': (0, 200, 200)}
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        dc = det_color_map.get(box['cls'], (200, 200, 200))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), dc, 2)
        det_label = f"{box['cls']} {box['conf']:.2f}"
        cv2.putText(overlay, det_label, (x1, y1 - 5), font, 0.4,
                    dc, 1, cv2.LINE_AA)

    # 事故区域红框 + 类型/成因
    is_crash = scene_label == 'crash'
    show_acc = is_crash or accident
    if accident and not is_crash:
        # G-353: 触发器命中但 argmax 不在 crash ⇒ 红底徽标持续标记（人眼可见）
        bw2 = int(tw * 1.2) + 20
        cv2.rectangle(overlay, (0, th + 78), (bw2, th + 106), (0, 0, 220), -1)
        cv2.putText(overlay, 'ACCIDENT', (10, th + 100), font, font_scale * 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if region is not None:
        rx1, ry1, rx2, ry2 = [int(v) for v in region]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w - 1, rx2), min(h - 1, ry2)
        if rx2 - rx1 > 10 and ry2 - ry1 > 10:
            t = 4 if is_crash else 2
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 255), t)
            # 四角强调
            cl = 14
            for cx, cy, dx, dy in ((rx1, ry1, 1, 1), (rx2, ry1, -1, 1),
                                   (rx1, ry2, 1, -1), (rx2, ry2, -1, -1)):
                cv2.line(overlay, (cx, cy), (cx + dx * cl, cy), (0, 0, 255), t + 2)
                cv2.line(overlay, (cx, cy), (cx, cy + dy * cl), (0, 0, 255), t + 2)
            # 文本: 类型 + 成因 (事故帧即显示——G-353 起含触发命中帧，不再只跟 argmax crash)
            if show_acc and (type_info or causes):
                lines = _type_cause_lines(type_info, causes)
                fs = max(18, int(w * 0.016))
                lh = fs + 8
                tx, ty = rx1, ry1 - lh * len(lines) - 14
                if ty < 0:  # 顶部空间不足 → 放到区域下方
                    ty = ry2 + 10
                overlay = pil_text(overlay, lines, (tx, ty), font_size=fs)
    # G-353: 事故帧但无事故区域框（region 覆盖率仅 ~28%，外部检测器没框≠模型没打分）
    # 时，类型/成因文本落到画面左下——否则"打了分却什么都不显示"。
    elif show_acc and (type_info or causes):
        lines = _type_cause_lines(type_info, causes)
        fs = max(18, int(w * 0.016))
        overlay = pil_text(overlay, lines, (10, h - 14 * len(lines) - 30),
                           font_size=fs)

    # 帧号
    cv2.putText(overlay, f"f:{frame_idx}", (w - 120, h - 10), font, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    return overlay


def process_video(video_path, temporal_model, det_yolo, cfg, dev, output_dir):
    """处理单个视频: 抽帧 → 时序推理 → 检测 → 可视化 → 逐帧 csv + 事件级汇总"""
    video_name = Path(video_path).stem
    fps_sample = cfg.get('fps_sample', 5)
    clip_len = cfg['clip_len']
    input_size = cfg.get('input_size', 640)
    conf_thresh = cfg.get('conf_threshold', 0.25)
    max_frames = cfg.get('max_frames_per_video', 2000)

    print(f"\n{'─'*60}")
    print(f"  处理视频: {Path(video_path).name}")

    # 抽帧
    t0 = time.time()
    frames, frame_indices, video_info = extract_frames(video_path, fps_sample, max_frames)
    if not frames:
        print(f"  跳过: 无法读取帧")
        return None
    extract_time = time.time() - t0
    print(f"  抽帧: {len(frames)} 帧 @ {fps_sample}fps (耗时 {extract_time:.1f}s)")
    print(f"  总帧数: {len(frames)} (clip_len={clip_len})")

    # 流式时序推理 (滑动窗口, O(1) 显存)
    t1 = time.time()
    stream = run_temporal_streaming(temporal_model, frames, clip_len, input_size, dev)
    temporal_time = time.time() - t1
    temporal_fps = len(frames) / max(temporal_time, 0.001)
    frame_preds = stream['frame_preds']
    frame_confs = stream['frame_confs']
    frame_probs = stream['frame_probs']
    prob_stats = stream['prob_stats']
    cause_avg = stream['cause_avg']
    type_avg = stream['type_avg']
    n_clips = stream['n_clips']
    print(f"  时序推理: {temporal_time:.2f}s ({temporal_fps:.1f} fps, {n_clips} clips)")

    cause_conf = cfg.get('cause_conf', 0.35)
    max_causes = cfg.get('max_causes', 3)
    crash_prob_thresh = cfg.get('crash_prob_thresh', 0.15)
    pre_crash_run_len = cfg.get('pre_crash_run_len', 8)
    orig_fps = video_info.get('orig_fps') or fps_sample

    scene_dist = {'normal': 0, 'pre_crash': 0, 'crash': 0}
    det_times = []
    frame_records = []
    _precrash_run_prev = 0
    tracker = AccidentRegionTracker()

    writer = None
    out_path = None
    if cfg.get('save_video', True):
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{video_name}_inference.mp4")
        fourcc = cv2.VideoWriter_fourcc(*cfg.get('video_codec', 'mp4v'))
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(out_path, fourcc, fps_sample, (w, h))

    for fi, frame in enumerate(frames):
        scene_id = frame_preds[fi]
        scene_name = SCENE_NAMES[scene_id]
        scene_conf = frame_confs[fi]
        scene_dist[scene_name] += 1

        # 检测 (每帧)
        td = time.time()
        boxes = detect_frame(det_yolo, frame, conf_thresh, input_size, dev)
        det_times.append(time.time() - td)

        # 事故区域跟踪: 场景进入 pre_crash 即开始, normal 时复位
        region = tracker.update(boxes) if scene_id >= 1 else None
        if scene_id == 0:
            tracker.reset()

        # V25-fix(crash-trigger): 事故事件判定 — 三重触发机制
        # 1) scene_id == 2 (argmax crash)
        # 2) crash_prob > crash_prob_thresh (概率阈值, 解决混淆惩罚过度抑制)
        # 3) 连续 pre_crash 帧数 >= pre_crash_run (持续事故前兆)
        probs = frame_probs[fi]
        crash_prob, precrash_prob = probs[2], probs[1]
        if scene_id == 1:
            _precrash_run = _precrash_run_prev + 1
        elif scene_id == 0:
            _precrash_run = 0
        else:
            _precrash_run = _precrash_run_prev + 1
        _precrash_run_prev = _precrash_run

        is_accident_frame = (
            scene_id == 2
            or crash_prob >= crash_prob_thresh
            or (scene_id == 1 and _precrash_run >= pre_crash_run_len)
        )

        rec = {
            'video': Path(video_path).name,
            'frame_idx': fi,
            'src_frame': frame_indices[fi] if fi < len(frame_indices) else -1,
            't_s': round(frame_indices[fi] / orig_fps, 3)
                if fi < len(frame_indices) else round(fi / fps_sample, 3),
            'scene': scene_name,
            'scene_conf': round(float(scene_conf), 4),
            'p_normal': round(float(probs[0]), 4),
            'p_pre_crash': round(float(precrash_prob), 4),
            'p_crash': round(float(crash_prob), 4),
            'accident': int(bool(is_accident_frame)),
            'trigger': ('crash_argmax' if scene_id == 2
                        else 'crash_prob' if crash_prob >= crash_prob_thresh
                        else 'precrash_run' if _precrash_run >= pre_crash_run_len
                        else 'none'),
            'n_boxes': len(boxes),
            'region': ','.join(str(int(v)) for v in region) if region else '',
            'type': '', 'type_conf': '', 'causes': '',
        }
        causes = []
        type_info = None
        if is_accident_frame:
            tv = type_avg[fi]
            tid = int(tv.argmax())
            rec['type'] = TYPE_NAMES[tid]
            rec['type_conf'] = round(float(tv[tid]), 4)
            # 已摘除位已在 run_temporal_streaming 里压成 0, 取 top-k 只剩启用因子
            order = [int(k) for k in np.argsort(cause_avg[fi])[::-1]
                     if cause_avg[fi][k] >= cause_conf][:max_causes]
            causes = [(CAUSE_NAMES_CN[k], CAUSE_NAMES[k], float(cause_avg[fi][k]))
                      for k in order]
            rec['causes'] = '|'.join(f'{n}:{c:.3f}' for _, n, c in causes)
            type_info = (TYPE_NAMES_CN[tid], TYPE_NAMES[tid], float(tv[tid]))

        frame_records.append(rec)

        if writer is not None:
            # G-353 修解包错位：causes 元组是 (中文, 英文, conf)，旧写法 `for _, n, c`
            # 把中文丢了只显英文（用户实看反馈）；并传三概率做可见性显示条。
            vis = draw_overlay(frame, scene_name, scene_conf, boxes, fi,
                               region=region, type_info=type_info,
                               causes=[(cn, c) for cn, _, c in causes],
                               probs=[float(x) for x in probs],
                               accident=bool(is_accident_frame))
            writer.write(vis)

    if writer is not None:
        writer.release()

    total_time = time.time() - t0

    # ── 事件级去重: 逐帧列表会把一次事故摊成几十条, 报"检测到几起"必须走事件口径 ──
    from postprocess import EventDetector
    ev_detector = EventDetector(
        min_crash_frames=cfg.get('event_min_frames', 3),
        min_event_duration=cfg.get('event_min_duration_s', 0.6),
        fps=fps_sample, confidence_threshold=crash_prob_thresh)
    events = []
    for e in ev_detector.detect_events([r['p_crash'] for r in frame_records]):
        win = frame_records[e['start_frame']:e['end_frame'] + 1]
        peak = max(win, key=lambda r: r['p_crash'])
        cause_count = {}
        type_count = {}
        for r in win:
            if r['type']:
                type_count[r['type']] = type_count.get(r['type'], 0) + 1
            for token in (r['causes'].split('|') if r['causes'] else []):
                name = token.split(':')[0]
                cause_count[name] = cause_count.get(name, 0) + 1
        events.append({
            'event_id': len(events) + 1,
            'start_frame': e['start_frame'], 'end_frame': e['end_frame'],
            'start_time_s': round(e['start_time'], 3),
            'end_time_s': round(e['end_time'], 3),
            'duration_s': e['duration_s'],
            'crash_frames': e['crash_frames'],
            'mean_crash_prob': e['avg_crash_prob'],
            'max_crash_prob': e['max_crash_prob'],
            'first_scene': win[0]['scene'],
            'peak_frame': peak['frame_idx'],
            # 事件类型走窗内多数票: 单取峰值帧那一帧, 一帧抖动就把整起事故改判
            'type': max(type_count, key=type_count.get) if type_count else None,
            'type_votes': type_count,
            'causes': sorted(cause_count, key=cause_count.get, reverse=True),
            'region': peak['region'] or None,
        })

    # ── 性能面: fps / 延迟分位 / 峰值显存 正是 paper_efficiency 挂在 not_measured 的三项
    lat = np.asarray([t * 1000.0 for t in det_times], dtype=np.float64)
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(dev)
    per_frame_ms = float(stream['timing'].get('clip_ms_mean') or 0.0)
    if lat.size:
        per_frame_ms += float(lat.mean())
    timing = {
        'extract_s': round(extract_time, 3),
        'temporal_inference_s': round(temporal_time, 3),
        'detection_total_s': round(float(lat.sum() / 1000.0), 3) if lat.size else 0.0,
        'total_s': round(total_time, 3),
        'wall_fps': round(len(frames) / max(total_time, 0.001), 2),
        'temporal_pass_fps': round(temporal_fps, 2),
        'per_frame_ms_est': round(per_frame_ms, 2),
        'streaming_fps_est': round(1000.0 / per_frame_ms, 2) if per_frame_ms > 0 else None,
        'detect_ms_mean': round(float(lat.mean()), 2) if lat.size else None,
        'detect_ms_p50': round(float(np.percentile(lat, 50)), 2) if lat.size else None,
        'detect_ms_p95': round(float(np.percentile(lat, 95)), 2) if lat.size else None,
        'detect_ms_p99': round(float(np.percentile(lat, 99)), 2) if lat.size else None,
    }
    timing.update(stream['timing'])

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{video_name}_frames.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wcsv = csv.DictWriter(f, fieldnames=list(frame_records[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(frame_records)

    result = {
        'video_name': Path(video_path).name,
        'video_path': video_path,
        'video_info': video_info,
        'total_frames_processed': len(frames),
        'clips_processed': n_clips,
        'clip_len': clip_len,
        'pad_len': stream['pad_len'],
        'scene_distribution': scene_dist,
        'timing': timing,
        'peak_gpu_memory_mb': round(peak_mem_mb, 1),
        'box_source': {'weights': cfg.get('box_weights'),
                       'conf_threshold': conf_thresh, 'device': str(dev),
                       'note': '检测框来自独立 COCO 权重, 非模型检测头 '
                               '(#271: 相位3 时序梯度会改写检测支路)'},
        'total_detections': int(sum(r['n_boxes'] for r in frame_records)),
        'avg_detections_per_frame': round(float(np.mean([r['n_boxes']
                                                        for r in frame_records])), 2),
        'output_video': out_path,
        'per_frame_csv': csv_path,
        'probability_stats': prob_stats,
        'events': events,
        'accident_summary': {
            'n_accident_frames': int(sum(r['accident'] for r in frame_records)),
            'n_events': len(events),
            'event_type_dist': {t: sum(1 for e in events if e['type'] == t)
                                for t in {e['type'] for e in events if e['type']}},
            'event_cause_dist': {c: sum(1 for e in events if c in e['causes'])
                                 for c in {c for e in events for c in e['causes']}},
        },
    }

    print(f"  场景分布: {scene_dist}")
    print(f"  概率统计: mean=[N:{prob_stats['mean_normal_prob']:.3f} "
          f"P:{prob_stats['mean_precrash_prob']:.3f} "
          f"C:{prob_stats['mean_crash_prob']:.3f}] "
          f"max_preC:{prob_stats['max_precrash_prob']:.3f} "
          f"max_C:{prob_stats['max_crash_prob']:.3f}")
    print(f"  事件: {len(events)} 起 (逐帧触发 "
          f"{result['accident_summary']['n_accident_frames']} 帧)")
    print(f"  墙钟 FPS: {timing['wall_fps']}, 逐帧检测 p95: {timing['detect_ms_p95']}ms, "
          f"显存峰值: {peak_mem_mb:.0f}MB, 总耗时: {total_time:.1f}s")

    return result


def resolve_videos(args, cfg):
    """收集待推理视频 (CLI > config 的 video_dirs > 内置两个默认目录)。"""
    dirs = []
    if args.videos:
        vp = Path(args.videos)
        if vp.is_file():
            return [str(vp)]
        if vp.is_dir():
            dirs = [str(vp)]
    else:
        dirs = cfg.get('video_dirs') or [
            d for d in os.environ.get('V26_VIDEO_DIRS', 'videos').split(':') if d
        ]

    videos = []
    for d in dirs:
        if os.path.isdir(d):
            for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
                videos.extend(sorted(glob.glob(os.path.join(d, ext))))
        else:
            print(f"  警告: 视频目录不存在 {d}")

    # TUM 目录混着 2000+ 个无事故视频, 只按名字挑 accident 的
    tum_dir = os.environ.get('V26_TUM_VIDEO_DIR', '')
    videos = [v for v in videos
              if not (tum_dir and v.startswith(tum_dir)
                      and 'accident' not in os.path.basename(v).lower())]
    if args.max_videos:
        videos = videos[:args.max_videos]
    return videos


def ckpt_identity(path):
    """交付件身份: 路径 + 字节 + sha256 前 16 位 (同名 best_temporal.pt 盘上不止一件)"""
    if not os.path.exists(path):
        return {'path': path, 'exists': False}
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return {'path': path, 'exists': True, 'bytes': os.path.getsize(path),
            'sha256_16': h.hexdigest()[:16], 'mtime': time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))}


def write_faces(summary, results, cfg):
    """机器可读面落盘: 逐视频 JSON 留在 output_dir, 事件面 + 汇总面按 §7 落 meta_face_dir。"""
    out_dir = cfg['output_dir']
    json_path = os.path.join(out_dir, 'inference_results.json')
    _dump(json_path, summary)

    meta_dir = cfg.get('meta_face_dir')
    if not meta_dir:
        return json_path, None, None
    os.makedirs(meta_dir, exist_ok=True)
    events_path = os.path.join(meta_dir, 'video_events.csv')
    face = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'caliber': '事件级 (postprocess.EventDetector: 连续帧确认 + 最短时长), '
                       'fps_sample=%s, crash_prob_thresh=%s' % (
                           cfg.get('fps_sample'), cfg.get('crash_prob_thresh')),
            'checkpoint': summary['checkpoint_identity'],
            'box_source': summary['config'].get('box_weights'),
            'num_videos': summary['num_videos'],
            'aggregate': summary['aggregate'],
            'videos': [{'video': r['video_name'], 'frames': r['total_frames_processed'],
                        'clips': r['clips_processed'], 'events': r['accident_summary']['n_events'],
                        'scene_distribution': r['scene_distribution'],
                        'wall_fps': r['timing']['wall_fps'],
                        'per_frame_ms_est': r['timing']['per_frame_ms_est'],
                        'clip_ms_p50': r['timing'].get('clip_ms_p50'),
                        'clip_ms_p95': r['timing'].get('clip_ms_p95'),
                        'detect_ms_p50': r['timing']['detect_ms_p50'],
                        'detect_ms_p95': r['timing']['detect_ms_p95'],
                        'peak_gpu_memory_mb': r['peak_gpu_memory_mb']}
                       for r in results]}
    face_path = os.path.join(meta_dir, 'video_inference_summary.json')
    _dump(face_path, face)

    rows = []
    for r in results:
        for e in r['events']:
            row = {'video': r['video_name'], **e}
            row['causes'] = '|'.join(row['causes'])
            row['type_votes'] = '|'.join(f'{k}:{v}' for k, v in
                                         sorted(row['type_votes'].items(),
                                                key=lambda kv: -kv[1]))
            rows.append(row)
    ev_csv = os.path.join(meta_dir, 'video_events.csv')
    if rows:
        with open(ev_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        with open(ev_csv, 'w', newline='', encoding='utf-8') as f:
            f.write('video,event_id\n')   # 空面也要落: 没事件 ≠ 没跑
    return json_path, face_path, ev_csv


def _dump(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  写入: {path}")


def run_selftest():
    """离线自检: 不建模型、不抢 GPU, 只验仪器自身。每项打印命令级可复读的结论。"""
    import inspect
    checks = []

    def ck(name, cond, detail=''):
        checks.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    print("=" * 70)
    print("video_inference 自检 (无模型 / 无 GPU 依赖)")
    print("=" * 70)

    # 1) 中文标签表与 base_config 按索引对齐 (V20 旧表错位的防线)
    ck('cause_cn_len', len(CAUSE_NAMES_CN) == len(CAUSE_NAMES),
       f'cn={len(CAUSE_NAMES_CN)} en={len(CAUSE_NAMES)}')
    ck('type_cn_len', len(TYPE_NAMES_CN) == len(TYPE_NAMES),
       f'cn={len(TYPE_NAMES_CN)} en={len(TYPE_NAMES)}')
    ck('cn_index_rear_end', TYPE_NAMES_CN[TYPE_NAMES.index('rear_end')] == '追尾')
    ck('cn_index_low_vis', CAUSE_NAMES_CN[CAUSE_NAMES.index('low_visibility')] == '低能见度')
    ck('cn_no_retired_bits',
       not any(w in ''.join(CAUSE_NAMES_CN) for w in ('起火', '连环', '坠落', '逆行', '爆胎', '酒后', '疲劳', '减速警示')),
       'V24 已删/已合并的旧因子不许出现在中文表')

    # 2) 已摘除因子开关接得上
    ck('cause_disabled_bits', CAUSE_ACTIVE[4] == 0 and CAUSE_ACTIVE[7] == 0,
       f'CAUSE_DISABLED_BITS={sorted(CAUSE_DISABLED_BITS)}')
    ck('cause_active_count', sum(CAUSE_ACTIVE) == len(CAUSE_NAMES) - len(CAUSE_DISABLED_BITS),
       f'启用 {sum(CAUSE_ACTIVE)}/{len(CAUSE_NAMES)}')

    # 3) 抽帧失败路径的元数 (曾是 4 元 vs 3 元解包)
    f, idx, info = extract_frames('/tmp/__no_such_video__.mp4', 5, 100)
    ck('extract_fail_arity', f == [] and idx == [] and info == {}, f'{type(info).__name__}')

    # 4) 短视频前置 pad 的裁齐: 帧内容 → 类别指纹, 错位会当场露出来
    class FakeTemporal:
        def __call__(self, clip, temporal_mode=False, type_ids=None):
            T = clip.shape[1]
            marks = [int(round(float(clip[0, t, :, 32, 32].mean() * 255) / 10)) % 3
                     for t in range(T)]
            logits = torch.full((1, T, 3), -10.0, device=clip.device)
            for t, m in enumerate(marks):
                logits[0, t, m] = 10.0
            return (None, None, None, torch.ones(1, len(CAUSE_NAMES), device=clip.device),
                    torch.ones(1, len(TYPE_NAMES), device=clip.device), logits, None)

        def eval(self):
            return self

    frames = [np.full((64, 64, 3), v, dtype=np.uint8) for v in (10, 20, 30)]
    stream = run_temporal_streaming(FakeTemporal(), frames, 8, 64, torch.device('cpu'))
    ck('pad_trim_len', len(stream['frame_preds']) == len(frames),
       f"preds={len(stream['frame_preds'])} frames={len(frames)} pad={stream['pad_len']}")
    ck('pad_trim_align', stream['frame_preds'] == [1, 2, 0],
       f"preds={stream['frame_preds']} 期望 [1,2,0] (按帧内容指纹)")
    ck('cause_mask_zeroed',
       stream['cause_avg'][0][4] == 0.0 and stream['cause_avg'][0][7] == 0.0
       and float(stream['cause_avg'][0][0]) > 0.4,
       f"bit0={stream['cause_avg'][0][0]:.3f} bit4={stream['cause_avg'][0][4]} "
       f"bit7={stream['cause_avg'][0][7]}")
    ck('timing_face', 'clip_ms_p95' in stream['timing'], str(stream['timing']))

    # 5) 事件级去重确实走的是那把已有尺子
    from postprocess import EventDetector
    ev = EventDetector(min_crash_frames=3, min_event_duration=0.6, fps=5.0,
                       confidence_threshold=0.15)
    one = ev.detect_events([0.01, 0.01, 0.9, 0.9, 0.9, 0.9, 0.01])
    ck('event_dedupe', len(one) == 1 and one[0]['crash_frames'] == 4,
       f'{len(one)} 起 / crash_frames={one[0]["crash_frames"] if one else "-"}')

    # 6) 权重装载复用验收那把尺 (shape 不匹配硬失败, 不再静默跳过)
    #    按函数体查, 不按整文件查 —— 整文件查会被这段自检自己的字面量命中 (假红)。
    src_bm = inspect.getsource(build_model)
    src_df = inspect.getsource(detect_frame)
    src_pv = inspect.getsource(process_video)
    ck('reuses_eval_gate', 'from eval_acceptance import build_models' in src_bm
       and 'load_state_dict' not in src_bm, 'build_model 里不该再有一份装载逻辑')
    ck('no_old_prefix_remap', ('detection_model.' + '{k}') not in src_bm
       and 'torch.load' not in src_bm)
    ck('no_hardcoded_gpu0', ("device=" + "'0'") not in src_df
       and 'except Exception' not in src_df)
    ck('detect_frame_gets_dev', ('detect_frame(det_yolo, frame, conf_thresh, '
                                 'input_size, dev)') in src_pv)
    ck('detect_frame_device', 'device' in inspect.signature(detect_frame).parameters)

    # 7) 配置正本: 有键还得有接线, 指向的文件必须真在盘上
    cfg_path = os.path.join(SCRIPT_DIR, 'inference_config.yaml')
    cfg = load_config(cfg_path)
    for key in ('checkpoint', 'model_yaml', 'output_dir', 'box_weights', 'meta_face_dir',
                'clip_len', 'input_size', 'device', 'fps_sample'):
        ck(f'config[{key}]', key in cfg, str(cfg.get(key))[:60])
    ck('config_abs', all(os.path.isabs(cfg[k]) for k in
                         ('checkpoint', 'model_yaml', 'output_dir', 'box_weights')))
    ck('checkpoint_on_disk', os.path.exists(cfg['checkpoint']), cfg['checkpoint'])
    ck('model_yaml_on_disk', os.path.exists(cfg['model_yaml']), cfg['model_yaml'])
    ck('box_weights_on_disk', os.path.exists(cfg['box_weights']), cfg['box_weights'])
    ck('clip_len_matches_training', cfg['clip_len'] == 16, str(cfg['clip_len']))

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print('-' * 70)
    print(f"[计数] 自检 {len(checks)} 项 / 失败 {n_fail}")
    return 1 if n_fail else 0


def main():
    parser = argparse.ArgumentParser(description='V26 视频推理 (逐帧场景/类型/成因 + 事件级汇总)')
    parser.add_argument('--config', default=os.path.join(SCRIPT_DIR, 'inference_config.yaml'),
                        help='推理配置 YAML (默认 scripts/inference_config.yaml 正本)')
    parser.add_argument('--videos', default=None,
                        help='视频文件或目录 (覆盖配置文件中的默认路径)')
    parser.add_argument('--checkpoint', default=None, help='覆盖 checkpoint 路径')
    parser.add_argument('--device', default=None, help='覆盖 device (推理卡号, 别和训练抢 0)')
    parser.add_argument('--max-videos', type=int, default=None, dest='max_videos',
                        help='最多处理几个视频')
    parser.add_argument('--max-frames', type=int, default=None, dest='max_frames',
                        help='每个视频最多抽几帧 (冒烟用, 别拿这个数当性能结论)')
    parser.add_argument('--output-dir', default=None, dest='output_dir',
                        help='覆盖输出目录 (冒烟/分批跑用, 免得污染交付面)')
    parser.add_argument('--no-face', action='store_true', dest='no_face',
                        help='不写 meta_face_dir 汇总面/事件面')
    parser.add_argument('--meta-face-dir', default=None, dest='meta_face_dir',
                        help='覆盖汇总面目录 (冒烟时指到临时目录, 别污染交付面)')
    parser.add_argument('--no-save-video', action='store_true', dest='no_save_video',
                        help='只出数不出标注视频')
    parser.add_argument('--selftest', action='store_true', help='离线自检 (不建模型不占卡)')
    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg['checkpoint'] = args.checkpoint if os.path.isabs(args.checkpoint) \
            else os.path.join(PROJECT_ROOT_V22, args.checkpoint)
    if args.device:
        cfg['device'] = args.device
    if args.no_save_video:
        cfg['save_video'] = False
    if args.max_frames:
        cfg['max_frames_per_video'] = args.max_frames
    if args.output_dir:
        cfg['output_dir'] = args.output_dir if os.path.isabs(args.output_dir) \
            else os.path.join(PROJECT_ROOT_V22, args.output_dir)
    if args.no_face:
        cfg['meta_face_dir'] = None
    if args.meta_face_dir:
        cfg['meta_face_dir'] = args.meta_face_dir if os.path.isabs(args.meta_face_dir) \
            else os.path.join(PROJECT_ROOT_V22, args.meta_face_dir)

    train_config.ensure_cwd_v22()
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    ident = ckpt_identity(cfg['checkpoint'])
    print("=" * 70)
    print("V26 视频推理")
    print("=" * 70)
    print(f"  检查点: {ident['path']}")
    print(f"  检查点身份: bytes={ident.get('bytes')} sha256_16={ident.get('sha256_16')} "
          f"mtime={ident.get('mtime')}")
    print(f"  设备: {cfg['device']} | clip_len={cfg['clip_len']} | "
          f"fps_sample={cfg['fps_sample']}")
    print(f"  输出目录: {output_dir}")
    print(f"  面目录: {cfg.get('meta_face_dir')}")

    temporal_model, meta, dev = build_model(cfg, cfg['device'])
    det_yolo = build_detector_yolo(cfg)

    videos = resolve_videos(args, cfg)
    print(f"\n  测试视频: {len(videos)} 个")
    for v in videos:
        print(f"    - {Path(v).name} ({os.path.getsize(v)/(1024**2):.1f}MB)")

    results = []
    for video_path in videos:
        result = process_video(video_path, temporal_model, det_yolo, cfg, dev, output_dir)
        if result:
            results.append(result)

    perf = _aggregate_perf(results)
    summary = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': cfg['checkpoint'],
        'checkpoint_identity': ident,
        'model_metadata': {k: str(v) for k, v in meta.items()},
        'config': cfg,
        'num_videos': len(results),
        'aggregate': {
            'total_frames': sum(r['total_frames_processed'] for r in results),
            'total_clips': sum(r['clips_processed'] for r in results),
            'total_events': sum(r['accident_summary']['n_events'] for r in results),
            'total_detections': sum(r['total_detections'] for r in results),
            'peak_gpu_memory_mb': max((r['peak_gpu_memory_mb'] for r in results), default=0),
            **perf,
        },
        'videos': results,
    }

    manifest = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'videos': [{
            'name': Path(r['video_path']).name,
            'path': r['video_path'],
            'resolution': f"{r['video_info']['width']}x{r['video_info']['height']}",
            'fps': r['video_info']['orig_fps'],
            'duration_s': r['video_info']['duration_s'],
            'frames_sampled': r['video_info']['sampled_frames'],
        } for r in results],
    }
    _dump(os.path.join(output_dir, 'test_videos_manifest.json'), manifest)

    json_path, face_path, ev_csv = write_faces(summary, results, cfg)

    print(f"\n{'='*70}")
    print(f"推理完成")
    print(f"  视频数: {len(results)} / 事件数: {summary['aggregate']['total_events']}")
    print(f"  逐帧 CSV: {[r['per_frame_csv'] for r in results]}")
    print(f"  结果 JSON: {json_path}")
    print(f"  汇总面: {face_path}")
    print(f"  事件面: {ev_csv}")
    print(f"{'='*70}")


def _aggregate_perf(results):
    """跨视频只做算术平均, 分位数按各视频分位的均值近似 (逐帧原始值已在 frames.csv 里)。"""
    keys = ('wall_fps', 'per_frame_ms_est', 'streaming_fps_est', 'stride1_frames_per_s',
            'clip_ms_p50', 'clip_ms_p95', 'detect_ms_p50', 'detect_ms_p95')
    out = {}
    for k in keys:
        vals = [r['timing'][k] for r in results if r['timing'].get(k) is not None]
        out[k] = round(float(np.mean(vals)), 2) if vals else None
    return out


if __name__ == '__main__':
    main()
