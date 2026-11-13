#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_config.py — V26 数据集配置 (独立于 V25)

数据集结构 (V26 自包含, 不依赖 V25 的任何路径):
  ${V26_DATA_ROOT:-datasets/accident_v26}/
  ├── images/       → 真实文件 (硬链接去重, 同 xfs 卷零额外字节)
  ├── labels/       → 真实文件 (独立副本)
  ├── scene_labels/ → 真实文件 (独立副本)
  ├── type_labels/  → 真实文件 (独立副本)
  ├── meta/         → 真实文件
  └── corrections/  → 独立目录 (标注修正覆盖层)

标签改写只发生在 V26 树内; images/ 是硬链接, 改写文本标签不会波及 V25。

标注修正机制:
  corrections/ 目录存放修正后的标注, 加载时覆盖原始标注。
  格式: corrections/{split}/{stem}.txt → 同标签格式

用法:
  from dataset_config import V26DatasetConfig
  config = V26DatasetConfig()
  labels = config.load_labels('train')  # 自动应用修正
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional


class V26DatasetConfig:
    """V26 数据集配置 — 数据全部自包含, 修正层独立"""

    BASE_DIR = Path(os.environ.get('V26_DATA_ROOT', 'datasets/accident_v26'))
    CORRECTIONS_DIR = BASE_DIR / 'corrections'

    # V26 类别定义 (与 V25 对齐)
    TYPE_NAMES = ['rear_end', 'side_impact', 'rollover', 'pedestrian_hit',
                  'head_on', 'fixed_object', 'other']
    CAUSE_NAMES = ['tailgating', 'speeding', 'dangerous_lane_change',
                   'standing_shoulder', 'pedestrian_highway',
                   'heavy_rain', 'heavy_snow', 'distracted_driving',
                   'low_visibility']
    SCENE_NAMES = ['normal', 'pre_crash', 'crash']

    # V26 类别权重 (反 head_on 偏置)
    # 非训练消费方(标注工具用); 训练真相源=core/base_config.TYPE_CLASS_WEIGHT, 两处须同步
    # 2026-10-22 反 side_impact 过预测重平衡: side 2.0→1.2 / head_on 0.8→1.2 / other 1.0→1.2
    TYPE_CLASS_WEIGHT = [1.8, 1.2, 1.8, 2.5, 1.2, 1.8, 1.2]

    def __init__(self):
        self._correction_cache = {}

    def get_split_dir(self, split: str) -> Path:
        """获取数据划分目录"""
        return self.BASE_DIR / 'images' / split

    def get_labels_dir(self, split: str) -> Path:
        """获取标注目录 (含修正)"""
        return self.BASE_DIR / 'labels' / split

    def get_corrections_dir(self, split: str) -> Path:
        """获取修正标注目录"""
        return self.CORRECTIONS_DIR / split

    def load_correction(self, split: str, stem: str) -> Optional[Dict]:
        """加载标注修正 (覆盖原始标注)"""
        cache_key = f'{split}/{stem}'
        if cache_key in self._correction_cache:
            return self._correction_cache[cache_key]

        correction_file = self.get_corrections_dir(split) / f'{stem}.txt'
        if not correction_file.exists():
            return None

        corrections = {}
        with open(correction_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    key = parts[0]
                    values = [int(v) for v in parts[1:]]
                    corrections[key] = values

        self._correction_cache[cache_key] = corrections
        return corrections

    def save_correction(self, split: str, stem: str, corrections: Dict):
        """保存标注修正"""
        correction_dir = self.get_corrections_dir(split)
        correction_dir.mkdir(parents=True, exist_ok=True)

        correction_file = correction_dir / f'{stem}.txt'
        with open(correction_file, 'w') as f:
            for key, values in corrections.items():
                f.write(f'{key} {" ".join(str(v) for v in values)}\n')

    def apply_corrections(self, split: str, original_labels: Dict,
                          stem: str) -> Dict:
        """应用修正到原始标注"""
        corrections = self.load_correction(split, stem)
        if not corrections:
            return original_labels

        corrected = dict(original_labels)
        for key, values in corrections.items():
            if key == 'accident_type':
                corrected['accident_type'] = values[0] if values else -1
            elif key == 'scene_class':
                corrected['scene_class'] = values[0] if values else -1
            elif key == 'causes':
                corrected['causes'] = values

        return corrected

    def get_dataset_stats(self) -> Dict:
        """获取数据集统计"""
        stats = {
            'type_distribution': {},
            'scene_distribution': {},
            'cause_distribution': {},
            'corrections_count': 0,
        }

        for split in ['train', 'val']:
            labels_dir = self.get_labels_dir(split)
            corrections_dir = self.get_corrections_dir(split)

            if labels_dir.exists():
                for f in labels_dir.glob('*.txt'):
                    stats['corrections_count'] += 1

            if corrections_dir.exists():
                for f in corrections_dir.glob('*.txt'):
                    stats['corrections_count'] += 1

        return stats

    def validate_structure(self) -> Dict:
        """验证数据集结构完整性"""
        results = {}

        for subdir in ['images', 'labels', 'scene_labels', 'type_labels', 'meta']:
            path = self.BASE_DIR / subdir
            if path.exists():
                items = list(path.iterdir())
                results[subdir] = {
                    'exists': True,
                    'items': len(items),
                    'is_symlink': any(item.is_symlink() for item in items),
                }
            else:
                results[subdir] = {'exists': False}

        # corrections 独立目录
        corr_path = self.CORRECTIONS_DIR
        results['corrections'] = {
            'exists': corr_path.exists(),
            'items': len(list(corr_path.iterdir())) if corr_path.exists() else 0,
            'is_symlink': False,
        }

        return results


def main():
    config = V26DatasetConfig()
    print("=" * 60)
    print("V26 数据集配置验证")
    print("=" * 60)

    validation = config.validate_structure()
    for subdir, info in validation.items():
        status = "✅" if info.get('exists') else "❌"
        symlink = " (→V25)" if info.get('is_symlink') else " (独立)"
        items = info.get('items', 0)
        print(f"  {status} {subdir}: {items} 项{symlink}")

    stats = config.get_dataset_stats()
    print(f"\n  修正文件数: {stats['corrections_count']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
