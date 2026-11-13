# -*- coding: utf-8 -*-
"""
dataset — V21 数据管线 (从 train.py 拆分)

包含:
  - SceneClassBalancedSampler  DDP 感知场景类别均衡采样器
  - MultiDatasetAccident       多数据集时序感知 Dataset (scene/cause/type/3D 标签)
  - ClipDataset                单帧数据集 → Clip 序列包装 (Phase 3 时序训练)
"""

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

from base_config import dist, _has_dist, extract_video_id, lane_class_encode, LANE_IGNORE

# #94: scene 证据档降权（唯一权重实现见 core/evidence_weights.py；文件缺⇒全 1.0 等价接入前）
try:
    from core import evidence_weights as _ew
except Exception:  # 兼容扁平导入路径
    import evidence_weights as _ew


# ============================================================
# 5. SceneClassBalancedSampler (场景类别均衡采样)
# ============================================================

class SceneClassBalancedSampler(torch.utils.data.Sampler):
    """DDP 感知的场景类别均衡采样器 (同 V17)"""

    def __init__(self, dataset, batch_size, target_ratios=None, seed=0,
                 ignore_ratio=0.10, frame_weights=None):
        self.dataset = dataset
        self.batch_size = batch_size
        # R3-05: 默认值对齐 1:1:1 均衡配比 (rebalance_dataset.py RECOMMENDED_SAMPLER_RATIOS)
        self.target_ratios = target_ratios or [0.34, 0.33, 0.33]
        # V22-fix(sampler-ignore): scene=0(ignore) 帧 (realv 真实事故合集, 带 COCO
        # 伪框) 原被完全排除在训练外 → 真实域检测零监督, 但 val 含 3421 帧 realv
        # GT。按 ignore_ratio 留出 batch 配额给 ignore 帧 (仅检测损失生效,
        # 场景损失 ignore_index=0 自动跳过)。
        self.ignore_ratio = ignore_ratio
        # V25: 逐帧采样权重 (accident_type × risk_level 均衡)
        self.frame_weights = frame_weights
        # R2-12: 硬编码假设 3 个元素 (n_per_class[c-1] 中 c∈{1,2,3}), 非 3 元素会 IndexError
        if len(self.target_ratios) != 3:
            raise ValueError(f"target_ratios 必须为 3 个值 (normal/pre_crash/crash), 实际: {self.target_ratios}")
        # ### V21 MODIFIED R7-04(严重, C采样层): 保存不可变基准比例 —
        # update_weights 原将 boost 累乘到 self.target_ratios 并回写, 每 epoch
        # 触发 (见 detection 修复) 会复利漂移形成"采样棘轮" (pre_crash 比例单调
        # 逼近 0.60 上限)。改为始终从 _base_ratios 重算, 使 boost 幂等可回退。
        self._base_ratios = list(self.target_ratios)
        self.seed = seed
        self.epoch = 0
        self._build_class_groups()

    def _build_class_groups(self):
        # V20: 使用 1/2/3 代替 0/1/2 (0=ignore_index)
        self.class_indices = {0: [], 1: [], 2: [], 3: []}
        for i in range(len(self.dataset)):
            try:
                label = self.dataset.labels[i]
                scene_cls = label.get("scene_cls", 0)  # V20-fix(XV): default ignore
                if isinstance(scene_cls, (list, torch.Tensor)):
                    scene_cls = int(scene_cls[0] if hasattr(scene_cls, '__getitem__') else scene_cls)
                scene_cls = int(scene_cls)
                if scene_cls in (0, 1, 2, 3):
                    self.class_indices[scene_cls].append(i)
                # V20-fix(XV): 越界值不归入任何类
            except (IndexError, AttributeError, TypeError):
                pass  # V20-fix(XV): 异常样本不归入任何类

        # V25: 为每个类构建权重向量 (用于加权采样)
        self.class_weights = {}
        if self.frame_weights:
            for c in (0, 1, 2, 3):
                indices = self.class_indices[c]
                w = [self.frame_weights.get(i, 1.0) for i in indices]
                self.class_weights[c] = torch.tensor(w, dtype=torch.float64)

        n1, n2, n3 = [len(self.class_indices[c]) for c in (1, 2, 3)]
        n0 = len(self.class_indices[0])
        total = n1 + n2 + n3
        if total > 0:
            LOGGER.info(f'SceneClassBalancedSampler: normal={n1}({100*n1/total:.1f}%), '
                        f'pre_crash={n2}({100*n2/total:.1f}%), crash={n3}({100*n3/total:.1f}%), '
                        f'ignore={n0} (batch 配额 {100*self.ignore_ratio:.0f}%)')
            if self.frame_weights:
                LOGGER.info(f'SceneClassBalancedSampler: VLM frame weights enabled')
            # R1-10: 数据质量警告 — pre_crash 占比过低说明 expand_pre_crash 未充分执行
            if n2 > 0 and n2 / total < 0.02:
                LOGGER.warning(f'[V21] pre_crash 占比仅 {100*n2/total:.1f}% (<2%), '
                               f'建议先运行 expand_pre_crash.py 扩充窗口再训练')

    def __iter__(self):
        rank = dist.get_rank() if _has_dist and dist.is_initialized() else 0
        world_size = dist.get_world_size() if _has_dist and dist.is_initialized() else 1

        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch * 1000 + rank)

        # V22-fix(sampler-ignore): ignore 桶配额 (realv 帧检测监督)
        n_ignore = max(0, int(self.batch_size * self.ignore_ratio)) \
            if len(self.class_indices.get(0, [])) else 0
        rest = self.batch_size - n_ignore
        n_per_class = [max(1, int(rest * r)) for r in self.target_ratios]
        diff = rest - sum(n_per_class)
        n_per_class[1] += diff  # pre_crash gets remainder

        shuffled = {}
        for c in (0, 1, 2, 3):
            indices = self.class_indices[c][:]
            if len(indices) == 0:
                shuffled[c] = []
                continue
            need = n_ignore if c == 0 else n_per_class[c - 1]

            # V25: 加权采样 (有 frame_weights 时用 multinomial, 否则用均匀 shuffle)
            if self.frame_weights and c in self.class_weights and self.class_weights[c].numel() > 0:
                weights = self.class_weights[c]
                # 需要的总样本数 (考虑 rank 和 world_size)
                total_need = need * len(indices)
                # multinomial 采样: 权重越高被采到概率越大
                sampled = torch.multinomial(
                    weights, total_need, replacement=True, generator=rng
                )
                shuffled[c] = [indices[i] for i in sampled.tolist()]
            else:
                perm = torch.randperm(len(indices), generator=rng).tolist()
                shuffled[c] = [indices[i] for i in perm]
                while len(shuffled[c]) < need * len(indices):
                    shuffled[c].extend(shuffled[c][:len(indices)])

        # 仅对非空类计算 ptrs (避免 ZeroDivisionError)
        ptrs = {}
        n_per_class_effective = {}
        for c in (0, 1, 2, 3):
            if len(shuffled[c]) == 0:
                n_per_class_effective[c] = 0
                ptrs[c] = 0
            else:
                take = n_ignore if c == 0 else n_per_class[c - 1]
                n_per_class_effective[c] = take
                ptrs[c] = (rank * take) % len(shuffled[c])

        indices = []
        total_needed = sum(len(self.class_indices[c]) for c in (0, 1, 2, 3))
        total_needed = (total_needed // world_size + self.batch_size - 1) // self.batch_size * self.batch_size

        while len(indices) < total_needed:
            for c in (0, 1, 2, 3):
                if len(shuffled[c]) == 0:
                    continue
                for _ in range(n_per_class_effective[c]):
                    if ptrs[c] >= len(shuffled[c]):
                        ptrs[c] = 0
                    indices.append(shuffled[c][ptrs[c]])
                    ptrs[c] += 1

        indices = indices[:total_needed]
        return iter(indices)

    def __len__(self):
        total = sum(len(self.class_indices[c]) for c in (1, 2, 3))
        world_size = dist.get_world_size() if _has_dist and dist.is_initialized() else 1
        return total // world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_weights(self, boost_factors):
        """### V21 MODIFIED R5-01(严重, C采样层): 困难类采样提升 —
        _v24_scene_aware_augmentation 回调依赖本方法, 原类未实现导致
        hasattr(sampler, 'update_weights') 永假, 整段逻辑死代码。
        boost_factors: [b_normal, b_pre, b_crash], 乘到 _base_ratios 后
        归一化并 clamp 到 [0.15, 0.60], 避免单类占据 batch。
        ### V21 MODIFIED R7-04(严重, C采样层): 始终基于 _base_ratios 重算 (而非
        累乘 self.target_ratios), 消除复利漂移; 相同 boost 幂等, boost=[1,1,1] 回到基准。
        """
        if not isinstance(boost_factors, (list, tuple)) or len(boost_factors) != 3:
            return
        raw = [r * max(float(b), 0.1) for r, b in zip(self._base_ratios, boost_factors)]
        total = sum(raw)
        if total <= 0:
            return
        norm = [x / total for x in raw]
        clamped = [min(max(x, 0.15), 0.60) for x in norm]
        s = sum(clamped)
        new_ratios = [x / s for x in clamped]
        if new_ratios == self.target_ratios:
            return  # 无变化, 避免刷屏日志
        self.target_ratios = new_ratios
        LOGGER.info(f'SceneClassBalancedSampler: ratios → '
                    f'{[round(x, 3) for x in self.target_ratios]} (boost={boost_factors})')


# ============================================================
# 6. 多数据集时序感知 Dataset
# ============================================================

class MultiDatasetAccident(YOLODataset):
    """多数据集时序感知 Dataset: scene_cls + cause + type + dataset_id"""

    def __init__(self, *args, scene_labels_dir=None, cause_labels_dir=None,
                 type_labels_dir=None, dataset_id='unknown',
                 semantic_labels_file=None, tokenizer=None,
                 lane_labels_dir=None, **kwargs):
        self.scene_labels_dir = scene_labels_dir
        # 注意: cause 标签当前内嵌于 scene_labels 文件第2行 (16-bit), cause_labels_dir 保留供未来独立目录扩展
        self.cause_labels_dir = cause_labels_dir
        self.type_labels_dir = type_labels_dir
        self.dataset_id = dataset_id
        # V26-G247: 车道线折线标注 (lane_labels/{stem}.json), 折线几何在
        # get_image_and_label 按需栅格化成 semantic_mask
        self.lane_labels_dir = lane_labels_dir
        self.lane_map = {}  # stem -> [(role, ltype, color, points[N,2]float32), ...]

        # V25: 语义对比学习 — 加载语义标注和分词器
        self.tokenizer = tokenizer
        self.semantic_map = {}  # clip_id -> text description
        if semantic_labels_file and os.path.exists(semantic_labels_file):
            import json as _json
            with open(semantic_labels_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = _json.loads(line)
                    if r.get('status') == 'success' and r.get('vlm', {}).get('description'):
                        clip_id = r.get('clip_id', '')
                        self.semantic_map[clip_id] = r['vlm']['description']

        super().__init__(*args, **kwargs)

    def get_labels(self):
        labels = super().get_labels()

        # ── 场景分类标签 ──
        if self.scene_labels_dir and os.path.isdir(self.scene_labels_dir):
            n_scene_loaded = 0
            n_scene_missing = 0
            n_orphan_tile_cause = 0  # #48/#61: 孤儿 tile 成因屏蔽计数
            _debug_missing_samples = []  # V21-diag: 记录前5个缺失样本
            for label in labels:
                img_stem = Path(label["im_file"]).stem
                scene_file = os.path.join(self.scene_labels_dir, f"{img_stem}.txt")
                if os.path.exists(scene_file):
                    with open(scene_file) as f:
                        line = f.read().strip()
                        # 兼容双行格式: 第一行 scene_cls, 第二行 V4 bit编码
                        lines = line.split('\n')
                        try:
                            scene_cls = int(lines[0])
                            # V20-fix: 1=normal, 2=pre_crash, 3=crash (0=ignore)
                            if scene_cls not in (0, 1, 2, 3):
                                scene_cls = 0  # V20-fix(XV): 越界值→ignore (不污染 normal)
                        except ValueError:
                            scene_cls = 0  # V20-fix(XV): 解析失败→ignore (不污染 normal)
                        label["scene_cls"] = scene_cls
                        n_scene_loaded += 1
                        # V4 第二行: 成因编码 (修复: 支持位掩码整数 或 逗号分隔 multi-hot)
                        # 转换脚本写入的是位掩码整数 (如 3 = bit0|bit1 = tailgating+speeding),
                        # 必须展开成 16-bit multi-hot, 否则 target=3 喂给 BCE 会产生负 loss
                        if len(lines) >= 2 and scene_cls >= 2:  # V20: pre_crash=2, crash=3
                            raw = lines[1].strip()
                            try:
                                if ',' in raw:
                                    # 逗号分隔 multi-hot 格式
                                    cause_bits = [int(b) for b in raw.split(',')]
                                    cause_bits = cause_bits[:9]
                                    while len(cause_bits) < 9:
                                        cause_bits.append(0)
                                else:
                                    # 位掩码整数 (V4 格式): bit i → cause[i]
                                    factor_bits = int(raw)
                                    cause_bits = [(factor_bits >> i) & 1 for i in range(9)]
                                # 强制 0/1 (防止脏数据导致 BCE 异常)
                                cause_bits = [1 if b else 0 for b in cause_bits]
                                label["cause"] = cause_bits
                            except (ValueError, IndexError):
                                label["cause"] = None  # V22-fix: 损坏的成例行 → 视为无标注
                        else:
                            # V22-fix(cause-标注缺失): scene>=2 但无第二行 → None 哨兵,
                            # 下游以 cause_valid=0 排除, 避免全零目标毒化成因头
                            label["cause"] = None if scene_cls >= 2 else [0] * 9
                        # #48/#61 (2026-10-04 用户裁决: 改损失通道、文件保留): 孤儿 tile
                        # (父帧 scene_labels 文件不存在) 的成因行无从对账 (标准 §9 #61、
                        # 报告 §8.34⑪: standing_shoulder 召回 1.5% 且方向相反、bit6 假正例),
                        # 而成因损失通道 (losses.py 按 cause_valid 过滤、不看 scene_valid)
                        # 此前会把它们打进梯度 → 置 None 走 cause_valid=0。
                        # 有父帧的 tile 不受影响 (#60 已保证 tile 第2行 = 父帧第2行)。
                        if label.get("cause") is not None:
                            _mp = re.match(r'^(.*)_t\d{2}$', img_stem)
                            if _mp and not os.path.exists(os.path.join(
                                    self.scene_labels_dir, _mp.group(1) + '.txt')):
                                label["cause"] = None
                                n_orphan_tile_cause += 1
                else:
                    # V20-fix(W8): 缺失标签文件 → scene_cls=0(ignore), 不污染 normal 类
                    label["scene_cls"] = 0
                    label["cause"] = [0] * 9  # V21-fix: 9→16 对齐 num_causes
                    n_scene_missing += 1
                    if len(_debug_missing_samples) < 5:
                        _debug_missing_samples.append(img_stem)
            # V21-diag: 场景标签加载诊断报告 (仅 RANK 0 输出, 避免 DDP 多卡重复)
            from ultralytics.utils import RANK as _RANK
            if n_scene_missing > 0 and _RANK in (-1, 0):
                try:
                    _dir_count = sum(1 for _ in os.scandir(self.scene_labels_dir))
                except OSError:
                    _dir_count = -1
                LOGGER.warning(
                    f'\u26a0\ufe0f 场景标签匹配诊断 [{self.dataset_id}]:\n'
                    f'   scene_labels_dir: {self.scene_labels_dir}\n'
                    f'   图像总数: {len(labels)}, 匹配成功: {n_scene_loaded}, '
                    f'匹配失败: {n_scene_missing} ({100*n_scene_missing/max(len(labels),1):.1f}%)\n'
                    f'   缺失样本示例(前5): {_debug_missing_samples}\n'
                    f'   目录下实际文件数: {_dir_count}\n'
                    f'   → 匹配失败原因: 图像stem与scene_labels目录下.txt文件名不一致!'
                )
            elif n_scene_loaded > 0 and _RANK in (-1, 0):
                # 统计分布
                _sc_dist = {0: 0, 1: 0, 2: 0, 3: 0}
                for lb in labels:
                    _sc = lb.get("scene_cls", 0)
                    _sc_dist[_sc] = _sc_dist.get(_sc, 0) + 1
                LOGGER.info(
                    f'\u2705 场景标签加载成功 [{self.dataset_id}]: '
                    f'{n_scene_loaded}/{len(labels)} | '
                    f'ignore={_sc_dist[0]} normal={_sc_dist[1]} '
                    f'pre_crash={_sc_dist[2]} crash={_sc_dist[3]}'
                )
            if n_orphan_tile_cause > 0 and _RANK in (-1, 0):
                LOGGER.info(
                    f'#48/#61: 孤儿 tile 成因屏蔽 {n_orphan_tile_cause} 帧 '
                    f'(cause_valid=0, 文件保留) [{self.dataset_id}]')
        else:
            # V20-fix: 场景标签目录缺失时发出显著警告 (而非静默默认 normal)
            LOGGER.warning(
                f'\u26a0\ufe0f 场景标签目录不存在: {self.scene_labels_dir}\n'
                f'   所有 {len(labels)} 个样本的 scene_cls 将默认为 1(normal)!\n'
                f'   场景分类头将不会学到任何有效信息! '
                f'请检查 data_root 路径和 scene_labels 目录结构.')
            for label in labels:
                label["scene_cls"] = 1  # V20-fix: 默认 normal
                label["cause"] = [0] * 9  # V21-fix: 9→16 对齐 num_causes

        # ── 事故类型标签 ──
        if self.type_labels_dir and os.path.isdir(self.type_labels_dir):
            for label in labels:
                img_stem = Path(label["im_file"]).stem
                type_file = os.path.join(self.type_labels_dir, f"{img_stem}.txt")
                if os.path.exists(type_file):
                    with open(type_file) as f:
                        try:
                            val = int(f.read().strip())
                            # V21-fix: TYPE_NAMES 有 10 类 (0-9), 原范围 (-1..5) 会把 6-9 误置 -1
                            label["accident_type"] = val if -1 <= val <= 9 else -1
                        except ValueError:
                            label["accident_type"] = -1
                else:
                    label["accident_type"] = -1
        else:
            for label in labels:
                label["accident_type"] = -1

        # ── V26-G247: 车道线折线标注 (lane_labels/{stem}.json) ──
        # 折线只进 self.lane_map (不进 label dict), 避免 deepcopy/队列污染;
        # 栅格化推迟到 get_image_and_label (按当次加载尺寸), 借 semantic_mask
        # 通道走全部几何增广 (RandomPerspective/MixUp/LetterBox/Flip 同步变换)。
        self.lane_map = {}
        if self.lane_labels_dir and os.path.isdir(self.lane_labels_dir):
            n_lane_lines = 0
            n_lane_dropped = 0
            for label in labels:
                img_stem = Path(label["im_file"]).stem
                lane_file = os.path.join(self.lane_labels_dir, f"{img_stem}.json")
                if not os.path.exists(lane_file):
                    continue
                try:
                    with open(lane_file) as f:
                        lane_json = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                parsed = []
                n_lane_bad = 0
                for ln in lane_json.get("lines", []):
                    pts = ln.get("points") or []
                    if len(pts) < 2:
                        continue
                    if ln.get("role") not in ('divider', 'edge') or \
                            ln.get("color") not in ('white', 'yellow'):
                        n_lane_bad += 1  # 宁缺勿滥: 键值非法的线丢弃, 不让运行期崩
                        continue
                    parsed.append((ln.get("role"), ln.get("lane_type"),
                                   ln.get("color"),
                                   np.asarray(pts, dtype=np.float32).reshape(-1, 2)))
                if parsed:  # 空折线帧不进 lane_map → 全 255 掩码 (不拿"空"当"无车道线")
                    self.lane_map[img_stem] = parsed
                    n_lane_lines += len(parsed)
                n_lane_dropped += n_lane_bad
            from ultralytics.utils import RANK as _RANK
            if _RANK in (-1, 0):
                LOGGER.info(
                    f'✅ 车道线标注加载 [{self.dataset_id}]: '
                    f'{len(self.lane_map)}/{len(labels)} 帧, {n_lane_lines} 条折线'
                    f'{f", 丢弃非法 {n_lane_dropped} 条" if n_lane_dropped else ""} '
                    f'(lane_labels_dir={self.lane_labels_dir})')

        # ── dataset_id + frame_idx ──
        for label in labels:
            label["dataset_id"] = self.dataset_id

        video_frame_counter = {}
        for label in labels:
            stem = Path(label["im_file"]).stem
            # V20-fix: video_id 必须包含摄像头标识, 否则多摄像头帧会交错混合,
            # 破坏 ClipDataset 的时序连续性 (不同视角帧交替进入同一 clip)
            # 文件名格式: r09_s03_s040_camera_basler_north_16mm_1648480750_204000000_...
            # ### B7修复(严重, 复审N2): 行内正则 → base_config.extract_video_id 真相源
            # (正则 ^(.*?)_(\d{10})_ 提取时间戳前的摄像头级前缀, 回退 rsplit)
            if '_' in stem:
                vid_key = extract_video_id(stem)
            else:
                # R1-06: 无下划线 stem 归入 dataset_id 组 (保持原行为)
                vid_key = self.dataset_id
            label["video_id"] = vid_key
            if vid_key not in video_frame_counter:
                video_frame_counter[vid_key] = 0
            label["frame_idx"] = video_frame_counter[vid_key]
            video_frame_counter[vid_key] += 1

        # ── V20: 解析扩展标签字段 (14/18 字段格式) ──
        # 格式: cls x y w h depth l w h sin cos [vx vy vz speed] in_acc ttc scene_cls
        for label in labels:
            label_file = label.get("txt_file", None)
            if label_file is None:
                # 尝试从 im_file 推断标签文件路径
                im_file = label["im_file"]
                label_file = im_file.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
                label_file = Path(label_file).with_suffix('.txt')
            else:
                label_file = Path(label_file)

            has_3d = False
            objs_3d = []
            if label_file.exists():
                try:
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 14:
                                # 14+ 字段: 含 3D 信息
                                vals = [float(x) for x in parts]
                                obj_3d = {
                                    'depth': vals[5],
                                    'dims': vals[6:9],      # l, w, h
                                    'orient': vals[9:11],   # sin, cos
                                }
                                if len(vals) >= 18:
                                    # 18 字段: 含速度
                                    obj_3d['vel'] = vals[11:14]   # vx, vy, vz
                                    obj_3d['speed'] = vals[14]
                                    obj_3d['in_acc'] = int(vals[15])
                                    obj_3d['ttc'] = vals[16]
                                    obj_3d['scene_cls'] = int(vals[17])
                                else:
                                    # 14 字段: cls x y w h depth l w h sin cos in_acc ttc scene_cls
                                    # 无速度数据, 对齐 geometry_3d.py parse_3d_labels 的 14 字段解析
                                    obj_3d['vel'] = [0.0, 0.0, 0.0]
                                    obj_3d['speed'] = 0.0
                                    obj_3d['in_acc'] = int(vals[11])
                                    obj_3d['ttc'] = vals[12]
                                    obj_3d['scene_cls'] = int(vals[13])
                                objs_3d.append(obj_3d)
                                has_3d = True
                except Exception:
                    pass

            label["has_3d"] = has_3d
            label["objs_3d"] = objs_3d

        return labels

    def get_image_and_label(self, index):
        """V26-G247: 超类取图后把车道线折线栅格化成 semantic_mask (类值 1..16)

        - 有折线: 掩码背景=0 (监督负类), 折线 stroke=2px 写入编码类值
        - 无折线: 全 255 (ignore), 不把"未标注"当"无车道线"
        掩码随后随 img/instances 走 RandomPerspective/MixUp/LetterBox/Flip 的
        apply_semantic 同步几何变换 (borderValue=255=ignore), 最终由 collate_fn
        按 semantic_mask 键 torch.stack。
        """
        label = super().get_image_and_label(index)
        h, w = label["img"].shape[:2]
        lines = self.lane_map.get(Path(self.im_files[index]).stem)
        if lines:
            mask = np.zeros((h, w), dtype=np.uint8)
            scale = np.array([w, h], dtype=np.float32)
            for role, ltype, color, pts in lines:
                poly = np.clip(np.round(pts * scale), [0, 0], [w - 1, h - 1])
                cv2.polylines(mask, [poly.astype(np.int32).reshape(-1, 1, 2)],
                              False, int(lane_class_encode(role, ltype, color)),
                              thickness=2, lineType=cv2.LINE_8)
        else:
            mask = np.full((h, w), LANE_IGNORE, dtype=np.uint8)
        label["semantic_mask"] = mask
        return label

    def __getitem__(self, index):
        label = self.labels[index]
        scene_cls = label.get("scene_cls", 0)  # V20-fix(XV): default ignore (不污染 normal)
        cause_bits = label.get("cause", [0] * 9)  # V21-fix: 9→16 对齐 num_causes
        # V22-fix(cause-标注缺失): None 哨兵 → 零张量 + cause_valid=0 (损失中屏蔽)
        cause_valid = 0 if cause_bits is None else 1
        if cause_bits is None:
            cause_bits = [0] * 9
        acc_type = label.get("accident_type", -1)

        result = super().__getitem__(index)

        # scene_cls
        if "scene_cls" not in result:
            result["scene_cls"] = torch.tensor(scene_cls, dtype=torch.long)  # V20-fix(W9): 标量 shape []
        elif not isinstance(result["scene_cls"], torch.Tensor):
            result["scene_cls"] = torch.tensor(result["scene_cls"], dtype=torch.long)

        # cause multi-label
        ct = torch.tensor(cause_bits, dtype=torch.float32)
        # V20-fix: 保持 [16] 形状, 不做 view(1,-1) — 避免 BCE 广播为 [B,B,16]
        result["cause"] = ct
        result["cause_valid"] = torch.tensor(cause_valid, dtype=torch.long)  # V22-fix(cause-标注缺失)

        # accident type
        result["accident_type"] = torch.tensor(acc_type, dtype=torch.long)  # V20-fix(W9): 标量

        # V22-fix(tile-mask): 2x2 瓦片 (_tNN) 继承父块 scene 标签, 但裁切区常无
        # 事故证据 (标签-内容解耦) → scene_valid=0 屏蔽场景损失/指标, 保留检测监督
        _im = str(result.get("im_file", ""))
        _is_tile = re.search(r'_t\d{2}\.[A-Za-z]+$', _im) is not None
        result["scene_valid"] = torch.tensor(0 if _is_tile else 1, dtype=torch.long)

        # #94: 逐帧 scene 证据档权重（有支撑=1.0 / 质量档=WEAK / 非 crash 帧或表缺=1.0）。
        # 与质检面同源：core/evidence_weights 读 meta/qc_scene_frame_tier.jsonl（由
        # qc_scene_frame_weight.py 用 census 同一份 attr_of 生成）。split 从 im_file 路径取。
        _ms = re.search(r'/images/([^/]+)/', _im) or re.search(r'/(train|val)/', _im)
        _sp = _ms.group(1) if _ms else ''
        _stem = os.path.basename(_im).rsplit('.', 1)[0]
        result["scene_sup_weight"] = torch.tensor(
            _ew.frame_weight(_sp, _stem), dtype=torch.float32)

        result["video_id"] = label.get("video_id", '')
        result["frame_idx"] = label.get("frame_idx", 0)
        result["dataset_id"] = label.get("dataset_id", 'unknown')

        # V20: 3D 标签 (可选)
        # V20-fix(C9): 固定形状 [MAX_OBJ_3D, 10] + 有效掩码, 避免变长张量无法 collate
        # 原始问题: gt_3d 形状 [N_obj, 10] 每样本不同 → collate 变为 list → 损失代码跳过 → 3D 头永远不训练
        MAX_OBJ_3D = 8
        gt_3d_padded = torch.zeros(MAX_OBJ_3D, 10, dtype=torch.float32)
        gt_3d_mask = torch.zeros(MAX_OBJ_3D, dtype=torch.bool)
        if label.get("has_3d", False) and label.get("objs_3d"):
            objs_3d = label["objs_3d"]
            n_obj = min(len(objs_3d), MAX_OBJ_3D)
            for idx, obj in enumerate(objs_3d[:n_obj]):
                gt_3d_padded[idx] = torch.tensor([
                    obj['depth'],
                    *obj['dims'],       # l, w, h
                    *obj['orient'],     # sin, cos
                    *obj['vel'],        # vx, vy, vz
                    obj['speed'],
                ], dtype=torch.float32)
                gt_3d_mask[idx] = True
            result["has_3d"] = True
        else:
            result["has_3d"] = False
        result["gt_3d"] = gt_3d_padded        # [MAX_OBJ_3D, 10] 固定形状, collate 安全
        result["gt_3d_mask"] = gt_3d_mask     # [MAX_OBJ_3D] 有效对象掩码

        # V25: 语义对比学习 — 加载文本 embedding
        if self.tokenizer and self.semantic_map:
            # 从文件名解析 clip_id
            stem = Path(label.get("im_file", "")).stem
            import re as _re
            m = _re.search(r'_(\d{6})(?:_\w+)?$', stem)
            clip_id = stem[:m.start()] if m else stem

            description = self.semantic_map.get(clip_id, '')
            if description:
                token_ids, attn_mask = self.tokenizer.encode(description)
                result["text_token_ids"] = torch.tensor(token_ids, dtype=torch.long)
                result["text_attn_mask"] = torch.tensor(attn_mask, dtype=torch.long)
                result["has_text"] = torch.tensor(1, dtype=torch.long)
            else:
                max_len = self.tokenizer.max_len
                result["text_token_ids"] = torch.zeros(max_len, dtype=torch.long)
                result["text_attn_mask"] = torch.zeros(max_len, dtype=torch.long)
                result["has_text"] = torch.tensor(0, dtype=torch.long)
        else:
            # 无语义标注时, 使用零向量作为 text embedding
            result["text_emb"] = torch.zeros(256, dtype=torch.float32)
            result["has_text"] = torch.tensor(0, dtype=torch.long)

        # V26-G247: semantic_mask 掩码 → 张量 (collate_fn 按键 torch.stack)
        # uint8 保体积; 损失侧查表解码。缺失时补全 255 掩码保证 batch 键集合一致。
        if "semantic_mask" in result:
            if not isinstance(result["semantic_mask"], torch.Tensor):
                result["semantic_mask"] = torch.from_numpy(
                    np.ascontiguousarray(result["semantic_mask"]))
        else:
            result["semantic_mask"] = torch.full(
                result["img"].shape[-2:], LANE_IGNORE, dtype=torch.uint8)

        return result


# ============================================================
# 6.5 V20: ClipDataset — 将单帧数据集包装为 Clip 序列数据集 (Phase 3)
# ============================================================

# video_sampling 补帧 (integrate_rare_frames.py): video_id 形如 tv{md5前8位}
_TV_STEM_RE = re.compile(r'^tv[0-9a-f]{8}$')
# 抽帧间隔 4s; 命中帧之间通常相隔数十~数百秒, 超过该间隔即视为不同事件瞬间
_TV_MAX_GAP_S = 5.0

# TUMTraf: stem 尾段是相机名而非时间戳, 时刻要从中间的 {10位秒}_{9位纳秒} 取。
# 25fps → 组内正常步长 4cs; 缺口 >1s 表示源视频在这里丢了 25+ 帧, 属不同瞬间。
# 阈值只能这么定: 实测 tumtraf 组内相邻帧 gap>10s 的对数为 0 (train/val 皆无),
# 所以它不存在 tv 族那种"跨事件拼接", 需要拦的是 >1s 的丢帧缺口
# (train 51 个滑窗 / val 15 个, 其中 >5s 的 8 / 4)。
_TUM_STEM_RE = re.compile(r'^tum_r\d+_s\d+_')
_TUM_MAX_GAP_CS = 100

_TUM_TS_RE = re.compile(r'_(\d{10})_(\d{9})')


def _frame_ts_cs(stem):
    """帧时刻 (百分之一秒); 取不到返回 None。"""
    m = _TUM_TS_RE.search(stem)
    if m:
        return int(m.group(1)) * 100 + int(m.group(2)[:2])
    tail = stem.rsplit('_', 1)[1] if '_' in stem else ''
    return int(tail) if tail.isdigit() else None


def _split_by_ts_gap(base, frame_indices, max_gap_cs=_TV_MAX_GAP_S * 100):
    """按帧时刻 (cs) 的断点把一组帧切成连续 run.

    同一 video_id 的帧若时刻相隔很远, 直接滑窗会造出时间上不连续的假 clip,
    而 clip 场景聚合取最高风险等级, crash 标签会外溢到 clip 内的无关帧。
    时刻解析不出来的帧一律不切 (退回整组, 与未启用该保护等价)。
    """
    stamped = []
    unstamped = []
    for i in frame_indices:
        stem = Path(base.labels[i].get('im_file', '')).stem
        ts = _frame_ts_cs(stem)
        (stamped.append((ts, i)) if ts is not None else unstamped.append(i))
    if not stamped:
        return [list(frame_indices)]
    stamped.sort()
    runs, cur, prev = [], [], None
    for ts, i in stamped:
        if prev is not None and (ts - prev) > max_gap_cs:
            runs.append(cur)
            cur = []
        cur.append(i)
        prev = ts
    if cur:
        runs.append(cur)
    if unstamped:
        runs.append(unstamped)
    return runs


class ClipDataset(torch.utils.data.Dataset):
    """V20: 将 MultiDatasetAccident 的单帧数据包装为 Clip 序列    
    按 video_id 分组连续帧, 构建 [T, C, H, W] 的 Clip 输入,
    供 V18TemporalModel 进行时序因果推理。
    
    场景标签聚合策略: 取 clip 中最高风险等级 (crash > pre_crash > normal)
    
    V20 新增:
      - 逐帧场景标签 scene_cls_seq [T] — 帧级监督 + 检测延迟评估
      - GT 框序列 boxes_seq/classes_seq/obj_mask — 对象级 tracklet 建模
    """
    
    def __init__(self, base_dataset, clip_len=16, stride=8, max_objs=8):
        """
        Args:
            base_dataset: MultiDatasetAccident 实例
            clip_len: 每个 clip 的帧数
            stride: clip 滑动步长
            max_objs: V20 每帧最大对象数 (按面积降序取 Top-K)
        """
        self.base = base_dataset
        self.clip_len = clip_len
        self.stride = stride
        self.max_objs = max_objs
        self.clips = []  # [(start_idx, end_idx, video_id)]
        self._build_clips()
    
    def _build_clips(self):
        """按 video_id 分组, 构建 clip 索引
        
        V20-fix(C2): 直接存储帧索引列表, 而非 start/end 端点,
        避免 range() 在非连续数据集索引上跨视频混帧。
        """
        # 按 video_id 分组帧索引
        video_frames = {}
        for i in range(len(self.base)):
            label = self.base.labels[i]
            vid = label.get('video_id', 'unknown')
            if vid not in video_frames:
                video_frames[vid] = []
            video_frames[vid].append(i)
        
        # 每个视频生成 clip
        for vid, frames in video_frames.items():
            # video_sampling 补的稀有类型帧: 同一源视频但命中时刻相隔数十~数百秒,
            # 若照旧滑窗会拼出时间上不连续的假 clip, 且 clip 场景聚合取最高风险
            # (crash) 会把标签外溢到无关帧 → 按时间戳断点切成孤立 run。
            # TUMTraf 同理但要按它自己的步长定阈 (>1s 丢帧缺口, 而非 >5s)。
            runs = [frames]
            if _TV_STEM_RE.match(vid):
                runs = _split_by_ts_gap(self.base, frames)
            elif _TUM_STEM_RE.match(vid):
                runs = _split_by_ts_gap(self.base, frames, _TUM_MAX_GAP_CS)
            for frames in runs:
                n = len(frames)
                if n < self.clip_len:
                    # 短视频: 单 clip, 存储完整帧索引列表
                    if n > 0:
                        self.clips.append((list(frames), vid, True))
                    continue
                # 滑动窗口: 直接存储帧索引切片
                last_start = 0
                for start in range(0, n - self.clip_len + 1, self.stride):
                    self.clips.append((frames[start:start + self.clip_len], vid, False))
                    last_start = start
                # #86: 补末窗。原 `range(0, n-clip_len+1, stride)` 无末窗 ⇒ run 尾部不足
                # 一步长的帧从不进任何窗口（实测 369 个 crash 帧掉出供给）。补一条与最后一条
                # 重叠的末窗：不新增假时序、不碰标签、不碰 split（判据同 #83 windows_of）。
                if last_start + self.clip_len < n:
                    self.clips.append((frames[n - self.clip_len:], vid, False))
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        # V20-fix(C2): 直接使用存储的帧索引列表, 不再 range() 重建
        frame_indices, vid, is_padded = self.clips[idx]
        frame_indices = list(frame_indices)  # 复制, 避免修改原始列表
        K = self.max_objs
        
        # 收集 clip 帧
        frames = []
        scene_labels = []
        cause_labels = []
        cause_valid_flags = []  # V22-fix(cause-标注缺失)
        type_labels = []
        scene_sup_weights = []  # #94: 逐帧 scene 证据档权重（clip 降权取 crash 帧的 max）
        
        # V20: 逐帧 GT 框收集 (对象级 tracklet 建模)
        boxes_seq = torch.zeros(self.clip_len, K, 4, dtype=torch.float32)   # 归一化 xywh
        classes_seq = torch.full((self.clip_len, K), -1, dtype=torch.long)
        obj_mask = torch.zeros(self.clip_len, K, dtype=torch.bool)
        
        # G-304(F10): 先记下**真实帧数**, 再补帧 —— 短视频的复制帧不是观测,
        # 必须在逐帧标签与对象槽里打成 ignore/padding, 否则:
        #   · 帧级 focal 监督 (temporal_context.py:1109-1118) 只跳 scene=0,
        #     末帧若是 crash, 最多 15 份 crash 标签被复制进损失 (教出"画面不动=出事");
        #   · 帧级 acc 分母 (phase3_train.py frame_valid=scene_seq!=0) 同样被灌copies。
        # 实测在体宇宙 (tile 排除后 23,614 clips, 与 phase3_train.py 在体读数逐字相等):
        #   padded clip 2,375 条 = 10.06%; 复制帧槽 19,435 / 377,824 = 5.14%
        #   (量具 /tmp/g304/pad_measure.py, 只读元数据)。
        # 注: 下面 :869 的 V20-fix(W6) 早就声明"padding 用 ignore 而非末帧复制",
        #   但它在本循环之后才计算 pad_len —— 那时 len 已恒等于 clip_len ⇒
        #   **该保护从来没生效过 (死代码)**, 本次是把它真正接上, 不是新配方。
        _n_real_frames = min(len(frame_indices), self.clip_len)
        # 短视频填充: 重复最后一帧
        if is_padded:
            while len(frame_indices) < self.clip_len:
                frame_indices.append(frame_indices[-1])
        # V20-fix(C2): 防御性截断, 确保不超过 clip_len
        frame_indices = frame_indices[:self.clip_len]
        
        for slot, fi in enumerate(frame_indices):
            item = self.base[fi]
            frames.append(item['img'])
            
            sc = item.get('scene_cls', torch.tensor([1], dtype=torch.long))
            if isinstance(sc, torch.Tensor):
                sc = sc.item()
            scene_labels.append(int(sc))
            
            cause = item.get('cause', torch.zeros(9, dtype=torch.float32))
            if not isinstance(cause, torch.Tensor):
                cause = torch.zeros(9, dtype=torch.float32)
            cause_labels.append(cause)

            # V22-fix(cause-标注缺失): 逐帧成因有效性
            cv = item.get('cause_valid', torch.tensor(1, dtype=torch.long))
            if isinstance(cv, torch.Tensor):
                cv = cv.item()
            cause_valid_flags.append(int(cv))
            
            at = item.get('accident_type', torch.tensor([-1], dtype=torch.long))
            if isinstance(at, torch.Tensor):
                at = at.item()
            type_labels.append(int(at))

            # #94: 该帧 scene 证据档权重（MultiDatasetAccident.__getitem__ 已挂；缺⇒1.0）
            sw = item.get('scene_sup_weight', None)
            if isinstance(sw, torch.Tensor):
                sw = sw.item()
            scene_sup_weights.append(float(sw) if sw is not None else 1.0)
            
            # V20: GT 框 (归一化 xywh), 按面积降序取 Top-K
            bboxes = item.get('bboxes', None)
            classes = item.get('cls', None)
            if bboxes is not None and isinstance(bboxes, torch.Tensor) \
                    and bboxes.numel() > 0:
                n = bboxes.shape[0]
                areas = (bboxes[:, 2] * bboxes[:, 3])
                order = torch.argsort(areas, descending=True)[:K]
                nk = len(order)
                boxes_seq[slot, :nk] = bboxes[order].float()
                if classes is not None and isinstance(classes, torch.Tensor) \
                        and classes.numel() > 0:
                    cls_flat = classes.view(-1).long()
                    classes_seq[slot, :nk] = cls_flat[order].clamp(0, 2)
                obj_mask[slot, :nk] = True
        
        # 堆叠为 [T, C, H, W]
        clip_tensor = torch.stack(frames, dim=0)
        
        # V20-fix(W6): 场景标签聚合 — 剔除 ignore(0) 后取最高风险等级
        # T110(2026-10-08): 整条 clip 零标注 ⇒ ignore(0)，不再回退 normal(1)。
        # 依据 meta/clipnorm/ 实测：train 29,597 / val 11,100 条零标注 clip 中
        # 93.9% / 96.0% 来自"该视频一帧都没标"的面，其中 synthetic 侧 27,679 条
        # 全是 1-4 帧碎片且 video_id 就是 rear-end/head-on/t-bone 事故剧情
        # ⇒ 断言 normal 会把事故画面教成正常。三路损失 (focal ignore_index=0 /
        # cause scene>=2 / type scene==3) 本就跳过 0，无需改配方。
        valid_scene = [s for s in scene_labels if s != 0]
        clip_scene = max(valid_scene) if valid_scene else 0
        
        # 成因标签: 取最高风险帧的成因 (与场景标签一致)
        # V20-fix(C10): 安全查找 — 全 ignore 时 clip_scene=0，index(0) 命中首帧
        try:
            max_risk_idx = scene_labels.index(clip_scene)
        except ValueError:
            max_risk_idx = 0  # 全 ignore 时回退第一帧
        # V22-fix(cause-标注缺失): 优先取同风险等级且有成因标注的帧
        clip_cause_valid = 1
        if clip_scene == 0:
            clip_cause_valid = 0  # 整条无标注: 不声称任何帧的成因有效
        elif cause_valid_flags and not cause_valid_flags[max_risk_idx]:
            _alt = [i for i, s in enumerate(scene_labels)
                    if s == clip_scene and i < len(cause_valid_flags)
                    and cause_valid_flags[i]]
            if _alt:
                max_risk_idx = _alt[0]
            else:
                clip_cause_valid = 0
        clip_cause = cause_labels[max_risk_idx] if cause_labels else torch.zeros(9)
        clip_type = type_labels[max_risk_idx] if type_labels else -1

        # #94: clip scene 降权 = max(该 clip 内 scene==3 帧的帧权重)。
        # 有支撑帧(1.0)会抵消 ⇒ 仅"整条 clip 的 crash 帧全是质量档"才 <1，逐 clip 复现 #89 的
        # scene_best 分层（1,624 有支撑 / 1,060 质量档）。无 crash 帧 ⇒ DEFAULT_W。
        _cw = [w for s, w in zip(scene_labels, scene_sup_weights) if s == 3]
        clip_scene_sup_weight = max(_cw) if _cw else _ew.DEFAULT_W
        
        # V20-fix(W6): 逐帧场景标签序列 — padding 用 ignore(0) 而非末帧复制
        # 避免短视频末帧为 crash 时填充帧膨胀 crash 样本
        # G-304(F10): 原写法 `pad_len = clip_len - len(scene_labels[:clip_len])` 恒为 0
        #   (帧槽在上游已被补齐), 于是 [0]*pad_len 从未出现 ⇒ 注释与盘上不符。
        #   现按真实帧数补齐; clip 级聚合 (:839 取最高风险) 一字未动 —— 复制帧的标签
        #   本就等于某个真实帧, max 不受影响, 变的只有**逐帧**监督与逐帧分母。
        scene_seq_list = list(scene_labels[:_n_real_frames]) + \
            [0] * (self.clip_len - _n_real_frames)
        scene_seq = torch.tensor(scene_seq_list, dtype=torch.long)

        # G-304(F10): 对象槽同样屏蔽复制帧 —— 上面 per-slot 循环把复制帧的 GT 框又写了
        #   一遍并置 obj_mask=True (像素相同 ⇒ 同一目标被计 N 次), 帧级对象监督/统计
        #   会把"静止 copy"当成独立观测。padding 槽清空。
        if _n_real_frames < self.clip_len:
            boxes_seq[_n_real_frames:, :, :] = 0.0
            classes_seq[_n_real_frames:, :] = -1
            obj_mask[_n_real_frames:, :] = False

        # G-303(A2) 在体不变量: 逐帧标签长度必须恰好 clip_len, 且 ignore 槽数
        #   必须等于 padding 槽数 (多了说明真实帧被误屏蔽 —— 那才是真事故)。
        assert scene_seq.numel() == self.clip_len, \
            f"scene_cls_seq 长度 {scene_seq.numel()} != clip_len {self.clip_len}"
        _n_ignore_real = sum(1 for s in scene_labels[:_n_real_frames] if s == 0)
        _n_ignore_total = int((scene_seq == 0).sum())
        assert _n_ignore_total == _n_ignore_real + (self.clip_len - _n_real_frames), (
            f"ignore 槽数不对: 总 {_n_ignore_total} ≠ 真实零标注 {_n_ignore_real} "
            f"+ padding {self.clip_len - _n_real_frames}")
        
        return {
            'img': clip_tensor,           # [T, C, H, W]
            'scene_cls': torch.tensor(clip_scene, dtype=torch.long),
            'scene_cls_seq': scene_seq,   # V20 [T] 逐帧场景标签
            'boxes_seq': boxes_seq,       # V20 [T, K, 4] 归一化 xywh
            'classes_seq': classes_seq,   # V20 [T, K] 对象类别
            'obj_mask': obj_mask,         # V20 [T, K] 有效对象掩码
            'cause': clip_cause,
            'cause_valid': torch.tensor(clip_cause_valid, dtype=torch.long),  # V22-fix(cause-标注缺失)
            'accident_type': torch.tensor(clip_type, dtype=torch.long),
            'scene_sup_weight': torch.tensor(clip_scene_sup_weight, dtype=torch.float32),  # #94
            'video_id': vid,
            'n_frames': len(frame_indices),
            # G-304(F10): 真实帧数 (只增键)。padding 槽已在 scene_cls_seq/obj_mask 里
            #   打成 ignore; 下游若要按"有效帧"口径读数, 用这个键, 别再用 n_frames
            #   (它在补齐后恒等于 clip_len, 保留只为不断历史面)。
            'n_real_frames': torch.tensor(_n_real_frames, dtype=torch.long),
        }
