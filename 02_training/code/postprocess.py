#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess.py — V26 事件级后处理 (解决 crash 误检/漏检)

核心机制:
  [1] TemporalSmoothing — 时序平滑 (消除帧级抖动)
  [2] EventDetector — 事件级判定 (连续帧确认, 消除误检)
  [3] AdaptiveThreshold — 自适应阈值 (平衡漏检/误检)
  [4] EventEvaluator — 事件级评估 (替代帧级)

V25 问题: 帧级判定 → 漏检+误检+抖动
V26 解决: 帧级概率 → 时序平滑 → 事件级判定
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


# ============================================================
# 1. 时序平滑 (消除帧级抖动)
# ============================================================
class TemporalSmoothing:
    """滑动窗口概率平滑 — 消除帧级跳变

    V25 问题: 帧级独立判定, normal→crash→normal 跳变
    V26 解决: 滑动窗口平均 + 中值滤波
    """
    def __init__(self, window_size=15, use_median=False):
        self.window_size = window_size
        self.use_median = use_median
        self.prob_buffer = deque(maxlen=window_size)

    def smooth(self, prob: float) -> float:
        """单帧概率平滑"""
        self.prob_buffer.append(prob)
        if self.use_median:
            return float(np.median(list(self.prob_buffer)))
        return float(np.mean(list(self.prob_buffer)))

    def smooth_sequence(self, probs: List[float]) -> List[float]:
        """序列概率平滑"""
        smoothed = []
        for p in probs:
            smoothed.append(self.smooth(p))
        return smoothed

    def reset(self):
        self.prob_buffer.clear()


# ============================================================
# 2. 事件级判定 (消除误检)
# ============================================================
class EventDetector:
    """事件级判定 — 连续帧确认机制

    V25 问题: 单帧 crash 即触发事件 → 误检
    V26 解决: 连续 N 帧确认 + 事件级聚合
    """
    def __init__(self, min_crash_frames=5, min_event_duration=3.0,
                 fps=5.0, confidence_threshold=0.3):
        self.min_crash_frames = min_crash_frames
        self.min_event_duration = min_event_duration
        self.fps = fps
        self.confidence_threshold = confidence_threshold

    def detect_events(self, frame_probs: List[float],
                      frame_times: List[float] = None) -> List[Dict]:
        """从帧级概率检测事件

        frame_probs: 每帧 crash 概率 (0-1)
        frame_times: 每帧时间戳 (秒), 默认按 fps 均匀
        返回: 事件列表
        """
        n = len(frame_probs)
        if n == 0:
            return []

        if frame_times is None:
            frame_times = [i / self.fps for i in range(n)]

        events = []
        in_event = False
        event_start = 0
        crash_frame_count = 0

        for i in range(n):
            is_crash = frame_probs[i] > self.confidence_threshold

            if is_crash:
                crash_frame_count += 1
                if not in_event:
                    # 开始新事件
                    in_event = True
                    event_start = i
            else:
                if in_event:
                    # 结束事件
                    event_end = i - 1
                    self._finalize_event(
                        events, frame_probs, frame_times,
                        event_start, event_end, crash_frame_count)
                    in_event = False
                    crash_frame_count = 0

        # 处理末尾事件
        if in_event:
            self._finalize_event(
                events, frame_probs, frame_times,
                event_start, n - 1, crash_frame_count)

        return events

    def _finalize_event(self, events: List, frame_probs: List,
                        frame_times: List, start: int, end: int,
                        crash_count: int):
        """事件确认与属性计算"""
        duration = frame_times[end] - frame_times[start]
        n_frames = end - start + 1

        # 确认条件: 足够多 crash 帧 + 足够长持续时间
        if crash_count >= self.min_crash_frames and duration >= self.min_event_duration:
            event_probs = frame_probs[start:end + 1]
            events.append({
                'start_frame': start,
                'end_frame': end,
                'start_time': frame_times[start],
                'end_time': frame_times[end],
                'duration_s': round(duration, 2),
                'n_frames': n_frames,
                'crash_frames': crash_count,
                'avg_crash_prob': round(float(np.mean(event_probs)), 4),
                'max_crash_prob': round(float(np.max(event_probs)), 4),
                'confidence': round(float(np.mean(event_probs)), 4),
            })


# ============================================================
# 3. 自适应阈值 (平衡漏检/误检)
# ============================================================
class AdaptiveThreshold:
    """自适应阈值 — 根据上下文动态调整

    V25 问题: 固定阈值 → 漏检/误检失衡
    V26 解决: 基于场景/天气的自适应阈值
    """
    def __init__(self, base_threshold=0.3):
        self.base_threshold = base_threshold
        self.context_adjustments = {
            'rain': -0.05,       # 雨天降低阈值 (增加召回)
            'snow': -0.05,
            'night': -0.03,
            'fog': -0.05,
            'highway': 0.0,      # 高速正常
            'urban': 0.02,       # 城市提高阈值 (降低误报)
        }

    def get_threshold(self, weather: str = 'normal',
                      scene_type: str = 'highway') -> float:
        """获取自适应阈值"""
        threshold = self.base_threshold

        if weather in self.context_adjustments:
            threshold += self.context_adjustments[weather]
        if scene_type in self.context_adjustments:
            threshold += self.context_adjustments[scene_type]

        return max(0.1, min(0.8, threshold))  # 限制范围


# ============================================================
# 4. 事件级评估 (替代帧级)
# ============================================================
class EventEvaluator:
    """事件级评估 — 公平评价 crash 检测

    V25 问题: 帧级评估 → 与实际需求不匹配
    V26 解决: 事件级 IoU 匹配
    """
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold

    def evaluate(self, pred_events: List[Dict],
                 gt_events: List[Dict]) -> Dict:
        """事件级评估

        pred_events: 预测事件 [{start_frame, end_frame, ...}]
        gt_events: 真实事件 [{start_frame, end_frame, ...}]
        """
        if not gt_events:
            return {
                'n_gt': 0, 'n_pred': len(pred_events),
                'precision': 0, 'recall': 0, 'f1': 0,
                'missed': 0, 'false_alarm': 0,
            }

        # 事件匹配 (IoU)
        matched_pred = set()
        matched_gt = set()
        ious = []

        for i, gt in enumerate(gt_events):
            best_iou = 0
            best_j = -1
            for j, pred in enumerate(pred_events):
                if j in matched_pred:
                    continue
                iou = self._temporal_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= self.iou_threshold and best_j >= 0:
                matched_gt.add(i)
                matched_pred.add(best_j)
                ious.append(best_iou)

        # 统计
        tp = len(matched_gt)
        fn = len(gt_events) - tp
        fp = len(pred_events) - len(matched_pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'n_gt': len(gt_events),
            'n_pred': len(pred_events),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'avg_iou': round(float(np.mean(ious)), 4) if ious else 0,
        }

    def _temporal_iou(self, gt: Dict, pred: Dict) -> float:
        """时间 IoU"""
        gt_start, gt_end = gt['start_frame'], gt['end_frame']
        pred_start, pred_end = pred['start_frame'], pred['end_frame']

        inter_start = max(gt_start, pred_start)
        inter_end = min(gt_end, pred_end)
        inter = max(0, inter_end - inter_start)

        union = (gt_end - gt_start) + (pred_end - pred_start) - inter
        return inter / union if union > 0 else 0


# ============================================================
# 5. V26 完整后处理流水线
# ============================================================
class V26PostProcess:
    """V26 事件级后处理流水线 — 帧级概率 → 事件级判定

    流程: 帧概率 → 时序平滑 → 自适应阈值 → 事件检测 → 事件评估
    """
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.smoother = TemporalSmoothing(
            window_size=config.get('smooth_window', 15),
            use_median=config.get('use_median', False))
        self.detector = EventDetector(
            min_crash_frames=config.get('min_crash_frames', 5),
            min_event_duration=config.get('min_duration', 3.0),
            fps=config.get('fps', 5.0),
            confidence_threshold=config.get('conf_threshold', 0.3))
        self.threshold_adapter = AdaptiveThreshold(
            base_threshold=config.get('base_threshold', 0.3))
        self.evaluator = EventEvaluator(
            iou_threshold=config.get('iou_threshold', 0.3))

    def process(self, frame_probs: List[float],
                weather: str = 'normal',
                scene_type: str = 'highway',
                frame_times: List[float] = None) -> Dict:
        """完整后处理: 帧概率 → 事件列表"""
        # 1. 时序平滑
        smoothed_probs = self.smoother.smooth_sequence(frame_probs)

        # 2. 自适应阈值
        threshold = self.threshold_adapter.get_threshold(weather, scene_type)

        # 3. 事件检测
        events = self.detector.detect_events(smoothed_probs, frame_times)

        # 4. 结果汇总
        n_frames = len(frame_probs)
        n_crash_frames = sum(1 for p in smoothed_probs if p > threshold)

        return {
            'events': events,
            'n_events': len(events),
            'n_frames': n_frames,
            'n_crash_frames': n_crash_frames,
            'crash_ratio': round(n_crash_frames / max(n_frames, 1), 4),
            'threshold_used': round(threshold, 4),
            'smoothed_probs': smoothed_probs,
        }

    def evaluate(self, pred_result: Dict, gt_events: List[Dict]) -> Dict:
        """事件级评估"""
        return self.evaluator.evaluate(pred_result['events'], gt_events)
