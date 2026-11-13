#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""lane_finetune.py — V26-G350: 车道线头 only 微调档（V26_LANE_HEAD_ONLY=1）

背景（在册事实，见训练记录 §16-36/G-247 与 run_train.py 接线表）：
  · lane loss 只在检测侧 criterion（core/losses.py），即 Phase 1/2 回路；
  · Phase 3（3a/3b/3c）检测支路整体冻结（G-303 #271 三件套），λ_lane 不入时序梯度；
  · 事故库 lane_labels 种子仅 36 帧/88 线、val 侧 0 帧（债 #247）⇒ lane 头近随机。
⇒ CULane 车道线补训必须走 Phase-2 regime（检测侧），且**只放开 lane_head 可训**：
  其余权重逐字节不动 ⇒ P3c 时序头的特征面不受扰动，事故三目标既有读数/交付件
  不被这次微调污染；车道线能力单独记账。

护栏三条（fail-closed，任一不满足即中止发车）：
  1) 模型里必须找到 model.23.lane_head.* 参数（找不到 ⇒ 种子不对，拒跑）；
  2) requires_grad=False 挡不住 BN running stats（G-303 实测教训：BN 只认 .training）
     ⇒ 包装 model.train()，每次调用后把 lane_head 之外的 BN 全部打回 eval；
     打完当场清点，仍有 training=True 的冻结 BN ⇒ 中止；
  3) 冻结后非 lane 可训练参数必须为 0。

用法：train.py phase-2 分支在 V26_LANE_HEAD_ONLY=1 时调 install_lane_head_only(trainer)；
自检：python3 scripts/lane_finetune.py --exam（纯 CPU，加载 P3c 检测件真跑回调）。
"""
import os
import sys

import torch
import torch.nn as nn

LANE_MARK = 'lane_head'


def install_lane_head_only(trainer):
    """挂 on_train_start 回调：optimizer 建好之后冻结非 lane 参数 + BN 守护。

    时序依据（ultralytics 8.4.112 trainer.py:417-446）：_setup_train（建模型+优化器）
    先于 on_train_start ⇒ 冻结时优化器已含全部参数；torch SGD 对 grad=None 的参数
    直接跳过 ⇒ 冻结参数永不更新（动量缓冲空占内存，可接受，不重建优化器以免
    触发 V26_CHAIN_OPTIM_STATE 二次恢复）。
    """
    def _on_train_start(tr):
        try:
            from ultralytics.utils.torch_utils import de_parallel
        except ImportError:      # 8.4.112: 同 core/detection.py:32-34 的兜底
            from ultralytics.utils.torch_utils import unwrap_model as de_parallel
        m = de_parallel(tr.model)
        lane_np = [(n, p) for n, p in m.named_parameters() if LANE_MARK in n]
        if not lane_np:
            raise RuntimeError('[G-350 lane-only] 模型无 lane_head 参数 ⇒ 链式种子不对，中止')
        n_frozen = 0
        for name, p in m.named_parameters():
            keep = LANE_MARK in name
            p.requires_grad = keep
            n_frozen += int(not keep)
        trainable = sum(p.numel() for _, p in lane_np)
        total = sum(p.numel() for p in m.parameters())
        frozen_bn = [b for n, b in m.named_modules()
                     if isinstance(b, (nn.BatchNorm2d, nn.BatchNorm1d))
                     and LANE_MARK not in n]
        _orig_train = m.train

        def _guarded_train(mode=True):
            _orig_train(mode)
            if mode:
                for b in frozen_bn:
                    b.eval()
            return m

        m.train = _guarded_train
        m.train(True)
        still = [n for n, b in m.named_modules()
                 if isinstance(b, (nn.BatchNorm2d, nn.BatchNorm1d))
                 and LANE_MARK not in n and b.training]
        if still:
            raise RuntimeError(f'[G-350 lane-only] {len(still)} 只冻结 BN 仍处 training 模式'
                               f'（示例 {still[:3]}）⇒ 守护失效，中止')
        n_lane_bn = sum(1 for n, b in m.named_modules()
                        if isinstance(b, (nn.BatchNorm2d, nn.BatchNorm1d)) and LANE_MARK in n)
        print(f'  [G-350 lane-only] 可训练 = lane_head only: {len(lane_np)} 张量 / '
              f'{trainable:,} 参数（冻结 {n_frozen} 张量 / {total - trainable:,} 参数）')
        print(f'  [G-350 lane-only] BN 守护生效: 冻结侧 {len(frozen_bn)} 全 eval'
              f'（lane 侧 BN/GN {n_lane_bn} 不受管）；优化器含冻结参数但 grad=None 被 SGD 跳过')
        return {'trainable': trainable, 'lane_tensors': len(lane_np),
                'frozen': n_frozen, 'frozen_bn': len(frozen_bn)}

    trainer.add_callback('on_train_start', _on_train_start)
    return _on_train_start


class _FakeTrainer:
    """exam 用最小替身：只提供 add_callback/model 两点接口。"""
    def __init__(self, model):
        self.model = model
        self._cb = []
    def add_callback(self, ev, fn):
        self._cb.append((ev, fn))


def _exam():
    """离线证真：真加载 P3c 检测件 → 跑回调 → 逐条断言。CPU 即可。"""
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ck = os.path.join(pkg, 'runs/v26_dw_lane_g304_p3c/weights/best_temporal.pt')
    if not os.path.exists(ck):
        print('[exam] SKIP: P3c 检测件不存在', ck)
        return 1
    # pickled 类名是顶层模块（detection.*）⇒ 按 run_train.py 同法注入 sys.path 才能反序列化
    sys.path.insert(0, os.path.join(pkg, 'scripts'))
    sys.path.insert(0, os.path.join(pkg, 'scripts', 'core'))
    os.environ.setdefault('V26_MODE', '1')
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    m = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
    m = m.eval()
    tr = _FakeTrainer(m)
    cb = install_lane_head_only(tr)
    assert len(tr._cb) == 1 and tr._cb[0][0] == 'on_train_start', '回调未挂'
    stats = cb(tr)
    lane_keys = [n for n, _ in m.named_parameters() if LANE_MARK in n]
    assert len(lane_keys) == 10, f'lane 张量应 10 只, 实得 {len(lane_keys)}: {lane_keys}'
    assert all(p.requires_grad for n, p in m.named_parameters() if LANE_MARK in n)
    assert not any(p.requires_grad for n, p in m.named_parameters() if LANE_MARK not in n)
    m.train(True)
    assert not any(b.training for n, b in m.named_modules()
                   if isinstance(b, (nn.BatchNorm2d,)) and LANE_MARK not in n), 'BN 守护漏'
    m.eval(); m.train(True)
    assert not any(b.training for n, b in m.named_modules()
                   if isinstance(b, (nn.BatchNorm2d,)) and LANE_MARK not in n), 'BN 守护二次调用漏'
    print(f'[exam] PASS: lane 10 张量可训 / {stats["trainable"]:,} 参数；'
          f'冻结 {stats["frozen"]} 张量；冻结 BN {stats["frozen_bn"]} 只全 eval')
    return 0


if __name__ == '__main__':
    sys.exit(_exam() if '--exam' in sys.argv else int(print(
        '用法: 由 train.py phase2 在 V26_LANE_HEAD_ONLY=1 时 import；自检 --exam') or 0))
