# -*- coding: utf-8 -*-
"""evidence_weights — #89 scene 证据档 → 训练侧逐样本损失的**唯一**权重实现。

背景（#89 / 标准 127）：全库真 crash clip 里，scene 标签的证据强度分两档：
  · 有支撑   E0_gt / E1_dual_scale / E2_film / E6_source / E6b_gt_unreconciled  → 权重 1.0
  · 质量档   E3_model / E4_derived / E7_pipeline（流水/模型/继承，非 GT）      → 权重 WEAK（弱标签）
质检面 `qc_loss_unit_census.py` 按 **clip** 用 `scene_best = 该 clip 内 crash 帧的最强档` 分层，
得到 1,624 有支撑 / 1,060 质量档。本模块把同一份逐帧证据档落到训练损失上：

  帧权重 = TIER_WEIGHT[该帧 scene 证据档]（不在表里 / 非 crash 帧 ⇒ 1.0）
  clip 权重 = max(该 clip 内所有 scene==3 帧的帧权重)   # 最强支撑抵消 ⇒ 与 #89 的 scene_best 同判据

⇒ "整条 clip 的 crash 帧全是质量档" 才会被降权，逐 clip 复现 #89 的 1,624/1,060 拆分。
默认全 1.0（表缺 / WEAK=1.0）时，所有损失与接入前**数值等价**（回归闸保证）。
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent                     # .../scripts/core
# meta 目录：默认取与本模块同包的 accident_v26，可用环境变量覆盖（多数据集/测试）。
_META = Path(os.environ.get(
    'V26_SCENE_TIER_META',
    str(HERE.parent.parent / 'datasets' / 'accident_v26' / 'meta')))
TIER_FILE = _META / 'qc_scene_frame_tier.jsonl'

# 降权系数：质量档 clip 的 scene 监督进损失的倍率。单一可调面（消融/复审用），默认 0.3。
WEAK = float(os.environ.get('V26_SCENE_WEAK_WEIGHT', '0.3'))
DEFAULT_W = 1.0

# 证据档 → 权重。高证据=1.0；#89 质量档=WEAK。键与 qc_loss_unit_census.CLASS_ORDER 对齐。
TIER_WEIGHT = {
    'E0_gt': 1.0,
    'E1_dual_scale': 1.0,
    'E2_film': 1.0,
    'E6_source': 1.0,
    'E6b_gt_unreconciled': 1.0,
    'E3_model': WEAK,
    'E4_derived': WEAK,
    'E7_pipeline': WEAK,
}

_lock = threading.Lock()
_cache = {'mtime': None, 'size': None, 'by_split': {}}    # {split: {stem: weight}}


def _load() -> dict:
    """按 (mtime,size) 惰性加载逐帧证据档 → {split:{stem:weight}}；文件缺失返回空表（全 1.0）。"""
    if not TIER_FILE.exists():
        return {}
    st = TIER_FILE.stat()
    if _cache['mtime'] == st.st_mtime and _cache['size'] == st.st_size:
        return _cache['by_split']
    with _lock:
        if _cache['mtime'] == st.st_mtime and _cache['size'] == st.st_size:
            return _cache['by_split']
        by_split: dict = {}
        with TIER_FILE.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if '_note' in r:
                    continue
                sp, tier = r.get('split'), r.get('tier')
                stem = r.get('stem')
                if not (sp and stem):
                    continue
                by_split.setdefault(sp, {})[stem] = TIER_WEIGHT.get(tier, DEFAULT_W)
        _cache.update(mtime=st.st_mtime, size=st.st_size, by_split=by_split)
    return by_split


def frame_weight(split: str, stem: str) -> float:
    """单帧 scene 监督权重；无记录（非 crash 帧 / 表缺）⇒ DEFAULT_W。"""
    m = _load().get(split)
    if not m:
        return DEFAULT_W
    return m.get(stem, DEFAULT_W)


def clip_scene_weight(pairs) -> float:
    """clip 权重 = max(scene==3 帧的帧权重)。pairs: [(split, stem, scene_label), ...]。
    无 crash 帧 ⇒ DEFAULT_W。与 #89 `scene_best=最强支撑` 同判据。

    G-323(#273・E2) 修尺：原写法把累加器初始化成 DEFAULT_W(=1.0) 再取 `if fw > w` ⇒
    任何 <1.0 的降权**永远上不了 max** ⇒ 本函数对真实 clip 恒返回 1.0，与它自己第一行的
    "max(该 clip 内 scene==3 帧的帧权重)" 和 dataset.py:878-879 的在用算法矛盾。
    唯一在册调用方 qc_scene_weight_gate.py:52 传的是两只不存在的 stem（帧权重兜底=1.0）
    ⇒ 考卷只覆盖了"漏配=1.0"这一条后置条件，恰好看不见这个洞（E2 三臂若拿它出 clip 级
    读数会把"降权生效"读成"没生效"）。现按 core/dataset.py 同算法重写：只对 crash 帧取 max，
    无 crash 帧才回 DEFAULT_W。既有调用点输出不变（见本轮 EDIT 凭据的验收）。
    """
    ws = [frame_weight(sp, stem) for sp, stem, sc in pairs if sc == 3]
    return max(ws) if ws else DEFAULT_W


def is_enabled() -> bool:
    return TIER_FILE.exists()


def stats() -> dict:
    """加载态读数（回归闸用）：{split: {n_frames, n_weak, n_support}}。"""
    m = _load()
    out = {}
    for sp, d in m.items():
        weak = sum(1 for v in d.values() if v < 1.0)
        out[sp] = {'n_frames': len(d), 'n_weak': weak, 'n_support': len(d) - weak}
    return out
