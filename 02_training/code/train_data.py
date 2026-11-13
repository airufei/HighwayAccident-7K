# -*- coding: utf-8 -*-
"""
train_data — V22 数据管道扩展

在 V21 ClipDataset 基础上增加:
  - source_domain: 标注 clip 来自 real(0) 还是 synthetic(1)
  - agreement: 标注者投票一致性 (若无, 使用默认值走 UASC 回退)

设计原则:
  - 不修改 V21 任何文件
  - 通过包装(wrap) V21 ClipDataset 扩展功能
  - source_domain 从 clip 文件名前缀推断 (real_ / synthetic_)
"""

import torch
from torch.utils.data import Dataset

# 通过 train_config 的 sys.path 注入引入 V21 模块
import train_config  # noqa: F401
from dataset import ClipDataset


class V24ClipDataset(Dataset):
    """V22 扩展 ClipDataset — 增加 source_domain 和 agreement 字段

    包装 V21 ClipDataset, 在其输出基础上添加:
      - source_domain: [B] tensor (0=real, 1=synthetic) — CDCR 需要
      - agreement: [B] tensor float (标注者一致性, 默认 0.7) — UASC 需要

    推断逻辑:
      source_domain 从 clip 的 video_id 前缀推断:
        "real_*"      → 0 (real CCTV)
        "synthetic_*" → 1 (CARLA synthetic)
        其他           → 0 (默认 real)
    """

    def __init__(self, base_clip_dataset, default_agreement=1.0):
        """Args:
            base_clip_dataset: V21 ClipDataset 实例
            default_agreement: 默认标注者一致性
                W3修复: 默认 1.0 (≥ threshold 0.8) 走高一致性硬标签分支
                避免对全部 clip 施加 0.3 过度平滑 (原 0.7 < 0.8 全走 low 路径)
        """
        self.base = base_clip_dataset
        self.default_agreement = default_agreement

    def __len__(self):
        return len(self.base)

    @property
    def clips(self):
        return self.base.clips

    def _infer_domain(self, video_id):
        """从 video_id 前缀推断域来源

        convert_accident.py 生成的 stem 格式:
          real_{clip_id}_{frame:06d}      → domain=0
          synthetic_{clip_id}_{frame:06d}  → domain=1
        """
        vid_str = str(video_id) if video_id is not None else ''
        if vid_str.startswith('synthetic') or vid_str.startswith('synth'):
            return 1
        return 0  # 默认 real

    def __getitem__(self, idx):
        item = self.base[idx]

        # 推断域来源
        vid = item.get('video_id', '')
        domain = self._infer_domain(vid)
        item['source_domain'] = torch.tensor(domain, dtype=torch.long)

        # 标注者一致性 (ACCIDENT 无逐 clip 数据, 使用默认值)
        item['agreement'] = torch.tensor(self.default_agreement, dtype=torch.float32)

        return item


def wrap_clip_dataset(base_dataset, default_agreement=1.0):
    """便捷函数: 包装 V21 ClipDataset 为 V24ClipDataset

    Args:
        base_dataset: V21 ClipDataset 实例
        default_agreement: 默认标注者一致性

    Returns:
        V24ClipDataset 实例
    """
    return V24ClipDataset(base_dataset, default_agreement=default_agreement)
