#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_data.py — V26 数据层优化模块 (独立于 V25)

核心优化:
  [1] PseudoLabelPropagator — 伪标签传播 (补充类型标签)
  [2] OHEMSampler — 在线难样本挖掘
  [3] AdvancedVideoAugment — 视频级增强 (TimeWarp/Cutout/MixUp)
  [4] CrossDomainMixer — 跨域数据混入
  [5] TypeAwareBalance — 类型感知重采样

不修改 V25 任何数据文件。
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1. 伪标签传播 (解决类型标签稀疏问题)
# ============================================================
class PseudoLabelPropagator:
    """半监督伪标签传播 — 用模型预测补充无标签帧的类型标签

    V25 问题: 仅 crash 帧有类型标签 (大量 -1)
    V26 解决: 高置信度预测传播到相邻帧
    """
    def __init__(self, confidence_threshold=0.8, propagation_window=5):
        self.confidence_threshold = confidence_threshold
        self.propagation_window = propagation_window

    def propagate(self, frame_predictions: List[Dict],
                  frame_labels: List[int]) -> List[int]:
        """
        frame_predictions: [{'type_probs': [0.3,...], 'type_pred': 2, 'confidence': 0.9}, ...]
        frame_labels: [2, -1, -1, 2, -1, ...] (-1 = 无标签)
        返回: 补充后的标签列表
        """
        new_labels = list(frame_labels)
        n = len(frame_labels)

        for i, pred in enumerate(frame_predictions):
            if frame_labels[i] != -1:
                continue  # 已有标签跳过

            if pred.get('confidence', 0) >= self.confidence_threshold:
                # 向前向传播
                for j in range(max(0, i - self.propagation_window),
                               min(n, i + self.propagation_window + 1)):
                    if new_labels[j] == -1:
                        new_labels[j] = pred['type_pred']

        return new_labels

    def batch_propagate(self, clips: List[Dict]) -> List[Dict]:
        """批量处理多个 clip"""
        results = []
        for clip in clips:
            preds = clip.get('frame_predictions', [])
            labels = clip.get('type_labels', [-1] * len(preds))
            new_labels = self.propagate(preds, labels)
            clip['type_labels'] = new_labels
            clip['pseudo_label_count'] = sum(
                1 for a, b in zip(labels, new_labels) if a == -1 and b != -1)
            results.append(clip)
        return results


# ============================================================
# 2. OHEM 难样本挖掘
# ============================================================
class OHEMSampler:
    """Online Hard Example Mining — 自动聚焦难样本

    V25 问题: 随机采样, 难样本被淹没
    V26 解决: 按 loss 排序, 过采样高 loss 样本
    """
    def __init__(self, hard_ratio=0.3, min_keep=16):
        self.hard_ratio = hard_ratio
        self.min_keep = min_keep
        self.sample_losses = {}

    def update_loss(self, sample_id: str, loss: float):
        self.sample_losses[sample_id] = loss

    def get_weights(self, sample_ids: List[str]) -> np.ndarray:
        """返回采样权重 (难样本权重更高)"""
        losses = np.array([self.sample_losses.get(sid, 0.0) for sid in sample_ids])
        if losses.max() == 0:
            return np.ones(len(sample_ids)) / len(sample_ids)

        # 归一化 + 软排序
        normalized = losses / (losses.max() + 1e-8)
        weights = normalized ** 1.5  # 幂律加权
        weights = weights / weights.sum()
        return weights

    def sample(self, sample_ids: List[str], n: int) -> List[str]:
        """按难度加权采样"""
        weights = self.get_weights(sample_ids)
        indices = np.random.choice(len(sample_ids), size=min(n, len(sample_ids)),
                                    replace=True, p=weights)
        return [sample_ids[i] for i in indices]


# ============================================================
# 3. 高级视频增强
# ============================================================
class AdvancedVideoAugment:
    """视频级数据增强 — V26 增强版 (V25 Mosaic=0.0 是重大缺失)"""
    def __init__(self, mixup_alpha=0.2, cutout_prob=0.3,
                 timewarp_prob=0.2, mosaic_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutout_prob = cutout_prob
        self.timewarp_prob = timewarp_prob
        self.mosaic_prob = mosaic_prob

    def mixup(self, frames1: np.ndarray, frames2: np.ndarray,
              label1: int, label2: int) -> Tuple[np.ndarray, np.ndarray]:
        """MixUp 样本混合"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_frames = lam * frames1 + (1 - lam) * frames2
        mixed_label = lam * np.eye(self.num_classes)[label1] + \
                      (1 - lam) * np.eye(self.num_classes)[label2]
        return mixed_frames, mixed_label

    def timewarp(self, frames: np.ndarray, warp_factor: float = 0.3) -> np.ndarray:
        """时间扭曲增强 — 模拟变速"""
        n = len(frames)
        if n < 2:
            return frames

        # 生成时间扭曲路径
        orig_t = np.linspace(0, 1, n)
        warp = np.cumsum(np.random.uniform(1 - warp_factor, 1 + warp_factor, n))
        warp = warp / warp[-1]

        # 插值
        warped = np.zeros_like(frames)
        for c in range(frames.shape[1]):
            for h in range(frames.shape[2]):
                for w in range(frames.shape[3]):
                    warped[:, c, h, w] = np.interp(warp, orig_t, frames[:, c, h, w])
        return warped

    def cutout_video(self, frames: np.ndarray) -> np.ndarray:
        """时空 Cutout — 随机遮挡时空块"""
        if random.random() > self.cutout_prob:
            return frames

        n, c, h, w = frames.shape
        # 随机时空块
        t1, t2 = sorted(random.sample(range(n), 2))
        y1, y2 = sorted(random.sample(range(h), 2))
        x1, x2 = sorted(random.sample(range(w), 2))

        frames[t1:t2, :, y1:y2, x1:x2] = 0
        return frames

    def mosaic4(self, clips: List[np.ndarray]) -> np.ndarray:
        """4-clip Mosaic — V26 核心增强"""
        if len(clips) < 4:
            return clips[0]

        n, c, h, w = clips[0].shape
        mosaic = np.zeros_like(clips[0])
        half_h, half_w = h // 2, w // 2

        positions = [(0, 0), (0, half_w), (half_h, 0), (half_h, half_w)]
        for i, clip in enumerate(clips[:4]):
            y_off, x_off = positions[i]
            mosaic[:, :, y_off:y_off+half_h, x_off:x_off+half_w] = \
                clip[:, :, :half_h, :half_w]

        return mosaic


# ============================================================
# 4. 跨域数据混入
# ============================================================
class CrossDomainMixer:
    """跨域数据混入 — 缩小训练-推理域差距

    V25 问题: 仅自建数据, 域差距大
    V26 解决: 混入 CCD/DADA 等外部数据
    """
    def __init__(self, external_data_dir: str = os.environ.get('V26_EXTERNAL_DIR', 'external')):
        self.external_dir = Path(external_dir)
        self.datasets = {}

    def load_ccd(self) -> Optional[Dict]:
        """加载 CCD 数据集"""
        ccd_dir = self.external_dir / 'ccd_data_yolo'
        if not ccd_dir.exists():
            return None

        samples = []
        meta_file = ccd_dir / 'scene_metadata_train.csv'
        if meta_file.exists():
            import csv
            with open(meta_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(row)

        self.datasets['CCD'] = {
            'samples': samples,
            'domain': 'youtube_dashcam',
            'n_samples': len(samples),
        }
        return self.datasets.get('CCD')

    def mix_datasets(self, primary_samples: List[Dict],
                     mix_ratio: float = 0.3) -> List[Dict]:
        """混合主数据 + 外部数据"""
        mixed = list(primary_samples)

        for name, data in self.datasets.items():
            n_external = int(len(primary_samples) * mix_ratio / max(len(self.datasets), 1))
            external_samples = random.sample(data['samples'],
                                             min(n_external, len(data['samples'])))
            for s in external_samples:
                s['source_domain'] = name
            mixed.extend(external_samples)

        random.shuffle(mixed)
        return mixed


# ============================================================
# 5. 类型感知重采样
# ============================================================
class TypeAwareBalance:
    """类型感知重采样 — 平衡类型分布

    V25 问题: head_on 过采样, rear_end/side_impact 欠采样
    V26 解决: 精确控制每类采样率
    """
    def __init__(self, target_distribution: Optional[Dict] = None):
        self.target_dist = target_distribution or {
            'rear_end': 0.25, 'side_impact': 0.25, 'rollover': 0.12,
            'pedestrian_hit': 0.10, 'head_on': 0.13, 'fixed_object': 0.08,
            'other': 0.07,
        }

    def compute_weights(self, type_counts: Dict[str, int]) -> Dict[str, float]:
        """计算每类采样权重"""
        total = sum(type_counts.values())
        weights = {}
        for type_name, target_ratio in self.target_dist.items():
            actual_ratio = type_counts.get(type_name, 0) / max(total, 1)
            if actual_ratio > 0:
                weights[type_name] = target_ratio / actual_ratio
            else:
                weights[type_name] = 5.0  # 缺失类给高权重
        return weights

    def resample(self, samples: List[Dict], type_counts: Dict[str, int]) -> List[Dict]:
        """按目标分布重采样"""
        weights = self.compute_weights(type_counts)
        resampled = []

        for sample in samples:
            type_name = sample.get('accident_type', 'other')
            w = weights.get(type_name, 1.0)
            # 按权重重复采样
            repeat = max(1, int(w * random.uniform(0.8, 1.2)))
            for _ in range(repeat):
                resampled.append(sample)

        random.shuffle(resampled)
        return resampled
