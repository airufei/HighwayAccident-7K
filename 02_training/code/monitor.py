# -*- coding: utf-8 -*-
"""
monitor — V22 训练监控与诊断系统

功能:
  1. 定期检查点评估 (每 3 epoch)
  2. 关键指标监控: Scene Acc, Frame Acc, Type Acc, Detection Latency, Loss 趋势
  3. 自动停止: NaN/Inf, 指标偏离, 训练停滞
  4. 详细日志: 训练损失, 验证指标, GPU/内存资源
  5. 问题诊断: 异常原因定位建议
  6. 恢复训练: 检查点保存与加载

设计原则:
  - 不修改 V21 任何文件
  - 作为独立模块被 phase3_train.py 和 train.py 导入
  - 所有阈值可通过 V22_MONITOR_CONFIG 覆盖
"""

import os
import time
import math
import json
import psutil
from pathlib import Path
from datetime import datetime
from collections import deque

import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from ultralytics.utils import LOGGER
except ImportError:
    import logging
    LOGGER = logging.getLogger(__name__)


# ============================================================
# 监控配置: 各 Phase 的阈值与验收标准
# ============================================================

V22_MONITOR_CONFIG = {
    # ── 通用监控参数 ──
    'checkpoint_every': 3,         # 每 N epoch 执行检查点评估
    'val_every': 2,                # 每 N epoch 执行轻量验证
    'patience': 8,                 # 指标停滞耐心值 (检查点数)
    'stagnation_delta': 0.001,     # 停滞判定最小改善量
    # V23-fix(overfit-stop): 验证损失连续上升 (过拟合) 早停参数 —
    # 精度复合指标持续改善会掩盖 val_loss 发散 (3b/3c 实测), 需独立判定
    'overfit_patience': 3,         # 连续 N 次验证损失超 best 即中止 (V25-fix: 2→3 给更多收敛机会)
    'overfit_margin': 0.02,        # 相对 best 的恶化容忍度 (2%)
    'loss_history_len': 10,        # Loss 历史记录长度
    'nan_check': True,             # 是否检查 NaN/Inf
    'resource_check': True,        # 是否检查系统资源
    'gpu_memory_warn_mb': 2048,    # GPU 剩余显存警告阈值
    'ram_usage_warn_pct': 90,      # RAM 使用率警告阈值

    # ── Phase 1 验收标准 (空间 3D 感知预热) ──
    'phase1': {
        'val_loss_max': 3.0,       # 验证损失上限
        'scene_acc_min': 50.0,     # 场景分类准确率下限 (%)
        'det_loss_max': 2.5,       # 检测损失上限
    },

    # ── Phase 2 验收标准 (时序因果推理) ──
    'phase2': {
        'val_loss_max': 2.5,
        'scene_acc_min': 60.0,
        'frame_acc_min': 55.0,     # 帧级场景准确率 (%)
        'det_loss_max': 2.0,
    },

    # ── Phase 3a 验收标准 (CTTA + UASC 启动) ──
    'phase3a': {
        'frame_acc_min': 55.0,
        'type_acc_min': 15.0,      # 类型准确率 (%), 仅部分类型激活
        'loss_max': 2.0,
        'cdcr_active': False,
    },

    # ── Phase 3b 验收标准 (CDCR 启动) ──
    'phase3b': {
        'frame_acc_min': 58.0,
        'type_acc_min': 25.0,
        'loss_max': 1.8,
        'cdcr_active': True,
        'cdcr_loss_max': 2.0,      # CDCR 损失上限
    },

    # ── Phase 3c 验收标准 (全系统优化) ──
    'phase3c': {
        'frame_acc_min': 60.0,     # 最终目标 Recall > 60%
        'type_acc_min': 30.0,
        'loss_max': 1.5,
        'cdcr_active': True,
        'cdcr_loss_max': 1.5,
        'early_detection_min': 50.0,  # 早期检测率 (%)
    },
}


# ============================================================
# 系统资源监控
# ============================================================

def get_system_resources():
    """获取当前系统资源使用情况

    Returns:
        dict: CPU/RAM/GPU 使用率
    """
    if not _HAS_PSUTIL:
        # psutil 不可用时仅返回 GPU 信息
        res = {'cpu_pct': -1, 'ram_pct': -1,
               'ram_used_gb': -1, 'ram_total_gb': -1}
    else:
        res = {
            'cpu_pct': psutil.cpu_percent(interval=None),
            'ram_pct': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / 1024**3,
            'ram_total_gb': psutil.virtual_memory().total / 1024**3,
        }

    # GPU 监控
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                gpu_mem = torch.cuda.memory_allocated(i) / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(i) / 1024**3
                # G-304(F3): 属性名曾是 total_mem (不存在) ⇒ 整块被下面的 except Exception
                #   吞掉 ⇒ 面里从来没有 gpu*_* 键、显存红线一次没判过。正名 total_memory,
                #   并保留 getattr 兜底 (跨版本)。
                _props = torch.cuda.get_device_properties(i)
                gpu_total = (getattr(_props, 'total_memory', None)
                             or getattr(_props, 'total_mem', 0)) / 1024**3
                res[f'gpu{i}_mem_alloc_gb'] = round(gpu_mem, 2)
                res[f'gpu{i}_mem_reserved_gb'] = round(gpu_reserved, 2)
                res[f'gpu{i}_mem_total_gb'] = round(gpu_total, 2)
                res[f'gpu{i}_mem_pct'] = round(100 * gpu_mem / max(gpu_total, 1), 1)
                # G-304(F3b): memory_allocated/reserved 只数**本进程**的池子, DDP 对端
                #   与任何占卡服务一概看不见 ⇒ "可能 OOM"红线当时是假绿。
                #   mem_get_info 给驱动视角的 (free, total) ⇒ 补设备级读数 (含所有进程),
                #   旧键逐字保留 (只做加法, 防历史面断线)。
                try:
                    _free_b, _total_b = torch.cuda.mem_get_info(i)
                    _used_dev = (_total_b - _free_b) / 1024**3
                    res[f'gpu{i}_mem_used_dev_gb'] = round(_used_dev, 2)
                    res[f'gpu{i}_mem_pct_dev'] = round(
                        100 * _used_dev / max(_total_b / 1024**3, 1), 1)
                except Exception:
                    pass
            except Exception:
                pass

    return res


def format_resources(res):
    """格式化资源信息为一行日志"""
    if res.get('cpu_pct', -1) < 0:
        parts = ["psutil N/A"]
    else:
        parts = [f"CPU={res['cpu_pct']:.0f}%", f"RAM={res['ram_pct']:.0f}%"]
    if 'gpu0_mem_pct' in res:
        parts.append(f"GPU0={res['gpu0_mem_pct']:.0f}%({res['gpu0_mem_alloc_gb']:.1f}G)")
    if 'gpu1_mem_pct' in res:
        parts.append(f"GPU1={res['gpu1_mem_pct']:.0f}%({res['gpu1_mem_alloc_gb']:.1f}G)")
    return ' '.join(parts)


# ============================================================
# 训练监控器
# ============================================================

class TrainingMonitor:
    """V22 训练监控器

    功能:
      - 记录训练/验证指标历史
      - 检测异常 (NaN/Inf, 指标偏离, 训练停滞)
      - 生成诊断建议
      - 保存监控日志

    使用方式:
        monitor = TrainingMonitor(phase='3a', save_dir=save_dir)
        monitor.log_epoch(epoch, train_loss=0.5, val_metrics={...})
        monitor.check_and_act(epoch)
        if monitor.should_stop:
            print(monitor.stop_reason)
    """

    def __init__(self, phase, save_dir, config=None):
        self.phase = str(phase)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or V22_MONITOR_CONFIG
        self.phase_config = self.config.get(f'phase{self.phase}', {})

        # 指标历史
        self.train_loss_history = deque(maxlen=self.config['loss_history_len'])
        self.val_loss_history = deque(maxlen=self.config['loss_history_len'])
        self.val_metric_history = deque(maxlen=self.config['loss_history_len'])

        # 停滞检测
        self.best_metric = -float('inf')
        self.stagnation_count = 0
        # V23-fix(overfit-stop): 验证损失 best / 连续上升计数 (独立于精度复合指标)
        self.val_loss_best = float('inf')
        self.val_loss_rise = 0
        # G-304(F2a): 本轮连升起点时的选型指标锚点 (合取项用), None = 当前不在连升中
        self.rise_metric_anchor = None

        # 状态
        self.should_stop = False
        self.stop_reason = ""
        self.epochs_trained = 0

        # 监控日志文件
        self.log_path = self.save_dir / 'monitor_log.jsonl'

    def log_epoch(self, epoch, train_loss=None, val_loss=None,
                  val_frame_acc=None, val_type_acc=None,
                  val_latency=None, val_early_rate=None,
                  cdcr_loss=None, det_loss=None, extra=None,
                  val_metric=None):
        """记录一个 epoch 的指标

        Args:
            epoch: 当前 epoch (0-based)
            train_loss: 训练平均损失
            val_loss: 验证平均损失
            val_frame_acc: 验证帧级准确率 (%)
            val_type_acc: 验证类型准确率 (%)
            val_latency: 验证检测延迟 (秒)
            val_early_rate: 早期检测率 (%)
            cdcr_loss: CDCR 损失
            det_loss: 检测损失
            extra: 额外指标 dict
            val_metric: 训练循环的检查点选型指标 (提供时优先用于
                最佳/停滞跟踪, 保证停滞判定与选型口径一致)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'phase': self.phase,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_frame_acc': val_frame_acc,
            'val_type_acc': val_type_acc,
            'val_latency': val_latency,
            'val_early_rate': val_early_rate,
            'cdcr_loss': cdcr_loss,
            'det_loss': det_loss,
        }
        if val_metric is not None:
            entry['val_metric'] = val_metric
        if extra:
            entry.update(extra)

        # 资源信息
        if self.config.get('resource_check', True):
            entry['resources'] = get_system_resources()

        # 更新历史
        if train_loss is not None:
            self.train_loss_history.append(train_loss)
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
            # V23-fix(overfit-stop) → G-304(F2a) 更正"连续"的语义:
            #   原实现只在"val_loss 创新高"时清零, 落在带内 (best < v ≤ best×1.02) 时
            #   既不累加也不清零 ⇒ rise 是**跨轮累计** (实测: 一轮带内 + 一轮超带也算
            #   "连续 2 次上升"), 与 stop_reason 的措辞不符, 会把良性波动攒成早停。
            #   修法 = 只有真·连续超带才累计; 带内即视为未恶化 ⇒ 清零。
            if math.isfinite(val_loss):
                _band = self.val_loss_best * (
                    1.0 + self.config.get('overfit_margin', 0.02))
                if val_loss < self.val_loss_best:
                    self.val_loss_best = val_loss
                    self.val_loss_rise = 0
                    self.rise_metric_anchor = None
                elif val_loss > _band:
                    self.val_loss_rise += 1
                    if self.val_loss_rise == 1:
                        # 记下这段连升起点时的选型指标 ⇒ 连升期间若仍在创新高就不该杀
                        self.rise_metric_anchor = self.best_metric
                else:
                    self.val_loss_rise = 0
                    self.rise_metric_anchor = None

        # 更新最佳指标
        # V23-fix(metric-统一): 优先使用训练循环传入的选型指标
        if val_metric is not None:
            composite = val_metric
            validated = True
        else:
            composite = 0
            validated = val_frame_acc is not None and val_frame_acc > 0
            if validated:
                composite = val_frame_acc
                if val_type_acc is not None and val_type_acc > 0:
                    composite = val_frame_acc * 0.7 + val_type_acc * 0.3
        self.val_metric_history.append(composite)

        # V23-fix(停滞计数): 仅在真实验证的 epoch 跟踪最佳/停滞 —
        # 此前非验证 epoch 以 composite=0 计入, 停滞计数被约 3 倍放大,
        # patience=8 实际 ~3 个验证周期即触发 (3a/3c 均因此在 ep12 早夭)
        if validated:
            if composite > self.best_metric + self.config['stagnation_delta']:
                self.best_metric = composite
                self.stagnation_count = 0
            elif composite < self.best_metric - self.config['stagnation_delta']:
                self.stagnation_count += 1

        # G-304(F2d): 把早停闸的**内部状态**落到面里 — 此前 rise/best/锚点只存进程内,
        #   巡检只能靠 val_loss 手算, 算出来的和真闸不同源 (G-276 "预钉判据须先注册仪器")。
        entry['overfit_rise'] = self.val_loss_rise
        entry['overfit_patience'] = self.config.get('overfit_patience', 0)
        entry['val_loss_best'] = (None if self.val_loss_best == float('inf')
                                  else round(self.val_loss_best, 4))
        entry['overfit_band'] = None if self.val_loss_best == float('inf') else round(
            self.val_loss_best * (1.0 + self.config.get('overfit_margin', 0.02)), 4)
        entry['overfit_still_improving'] = self._overfit_suppressed_by_metric()

        # 追加到日志文件 (G-304: 写盘挪到历史/闸状态更新之后, 好把闸状态一并落面)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')

        self.epochs_trained = epoch + 1

    def _overfit_suppressed_by_metric(self):
        """G-304(F2a) 合取项: 这段连升期间选型指标是否仍在创新高。

        True = val_loss 虽连升, 但 best.pt 仍在被刷新 ⇒ 不是终局过拟合, 不中止。
        False = 指标自连升起点后没再涨 ⇒ 过拟合早停按预案兑现。
        """
        if self.rise_metric_anchor is None:
            return False
        return self.best_metric > self.rise_metric_anchor

    def check_anomalies(self):
        """检查训练异常，返回 (should_stop, reason, diagnosis)

        Returns:
            tuple: (是否应停止, 停止原因, 诊断建议列表)
        """
        diagnosis = []

        # ── 检查 1: NaN/Inf ──
        if self.config.get('nan_check', True):
            for name, hist in [('train_loss', self.train_loss_history),
                               ('val_loss', self.val_loss_history)]:
                if hist and (math.isnan(hist[-1]) or math.isinf(hist[-1])):
                    self.should_stop = True
                    self.stop_reason = f"{name} 出现 NaN/Inf @ epoch {self.epochs_trained}"
                    diagnosis.extend([
                        "可能原因 1: 学习率过高导致梯度爆炸 → 降低 lr0 到 1/10",
                        "可能原因 2: 数据中存在异常值 → 检查 scene_labels 中的极端值",
                        "可能原因 3: AMP 混合精度溢出 → 检查 scaler.unscale_ 后的 grad_norm",
                        "可能原因 4: PTCL 返回空张量 (类型不在激活集) → 检查 phase_masks 配置",
                    ])
                    return self.should_stop, self.stop_reason, diagnosis

        # ── 检查 1.5: 验证损失连续上升 (过拟合早停) ──
        # V23-fix(overfit-stop): 精度复合指标可能持续改善而 val_loss 发散 (3b/3c 实测),
        #   停滞检测对此盲区 ⇒ 独立判定。
        # G-304(F2a) 更正: "独立"不等于"单条就杀"。交付物是 best.pt (每创新高即落件),
        #   val_loss 连升而选型指标仍在创新高 = 还在学到东西, 截断只会砍掉后半程增益
        #   (本 run e4 实测: rise=1 同时 val_metric 创新高 78.330→81.112)。
        #   ⇒ 改为合取: 连升达上限 **且** 这段连升期间指标没再创新高 才中止。
        _op = self.config.get('overfit_patience', 0)
        if _op > 0 and self.val_loss_rise >= _op:
            if self._overfit_suppressed_by_metric():
                diagnosis.append(
                    f"登记: val_loss 连续 {self.val_loss_rise} 次超带 "
                    f"(band={self.val_loss_best * (1.0 + self.config.get('overfit_margin', 0.02)):.4f}), "
                    f"但选型指标自连升起点仍在创新高 (best_metric={self.best_metric:.3f}) "
                    f"⇒ 按 G-304(F2a) 合取口径**不中止**; 若指标再不涨即兑现早停")
            else:
                self.should_stop = True
                self.stop_reason = (
                    f"过拟合早停: 验证损失连续 {self.val_loss_rise} 次高于最佳值 "
                    f"(best={self.val_loss_best:.4f}, "
                    f"当前={self.val_loss_history[-1]:.4f}) 且选型指标期间未创新高")
                diagnosis.extend([
                    "best_temporal.pt 已锁定在最佳验证点, 产物无需重训",
                    "下一轮可选: 缩短 epochs / 调整正则强度 / 降低学习率",
                ])
                return self.should_stop, self.stop_reason, diagnosis

        # ── 检查 2: 训练损失持续上升 ──
        if len(self.train_loss_history) >= 5:
            recent = list(self.train_loss_history)[-5:]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                diagnosis.append(
                    "警告: 训练损失连续 5 epoch 上升 → 可能过拟合或学习率过高"
                )

        # ── 检查 3: 指标停滞 ──
        if self.stagnation_count >= self.config['patience']:
            self.should_stop = True
            self.stop_reason = (
                f"指标停滞: 连续 {self.stagnation_count} 个检查点无改善 "
                f"(best={self.best_metric:.2f})"
            )
            diagnosis.extend([
                "可能原因 1: 学习率过低 → 尝试增大 lr0 或减小 warmup",
                "可能原因 2: 数据增强不足 → 增加 hsv/flip/erasing 强度",
                "可能原因 3: 模型容量不足 → 尝试 --backbone l (45M) 或 x (68M)",
                f"当前最佳指标: {self.best_metric:.2f}",
            ])
            return self.should_stop, self.stop_reason, diagnosis

        # ── 检查 4: Phase 特定阈值 ──
        pc = self.phase_config
        if pc:
            if self.train_loss_history and 'loss_max' in pc:
                if self.train_loss_history[-1] > pc['loss_max']:
                    diagnosis.append(
                        f"警告: 训练损失 {self.train_loss_history[-1]:.3f} "
                        f"超过 {self.phase} 阈值 {pc['loss_max']}"
                    )

            if self.val_loss_history and 'val_loss_max' in pc:
                if self.val_loss_history[-1] > pc['val_loss_max']:
                    self.should_stop = True
                    self.stop_reason = (
                        f"验证损失 {self.val_loss_history[-1]:.3f} "
                        f"超过 {self.phase} 上限 {pc['val_loss_max']}"
                    )
                    diagnosis.extend([
                        "可能原因: 过拟合 → 减小 batch_size 或增加 weight_decay",
                        "可能原因: 验证集数据分布与训练集差异过大",
                    ])
                    return self.should_stop, self.stop_reason, diagnosis

        # ── 检查 5: 系统资源 ──
        if self.config.get('resource_check', True):
            res = get_system_resources()
            if res.get('ram_pct', -1) > self.config.get('ram_usage_warn_pct', 90):
                diagnosis.append(
                    f"警告: RAM 使用率 {res['ram_pct']:.0f}% "
                    f"({res['ram_used_gb']:.1f}/{res['ram_total_gb']:.1f}GB) "
                    f"→ 减少 --workers 或启用 cache='disk'"
                )
            # G-304(F3c): 旧键 gpu*_mem_pct 是本进程池占比 (看不见 DDP 对端),
            #   设备级 gpu*_mem_pct_dev 才含所有进程 ⇒ 两条都判, 并写明口径。
            for key in list(res.keys()):
                if key.startswith('gpu') and key.endswith('_mem_pct'):
                    if res[key] > 95:
                        diagnosis.append(
                            f"警告: {key} = {res[key]}% (本进程池口径) → 可能 OOM, "
                            f"减小 batch_size 或 clip_len"
                        )
                elif key.startswith('gpu') and key.endswith('_mem_pct_dev'):
                    if res[key] > 95:
                        diagnosis.append(
                            f"警告: {key} = {res[key]}% (设备级, 含对端/占卡服务) "
                            f"→ 真 OOM 前兆, 先查是谁在占卡"
                        )

        return self.should_stop, self.stop_reason, diagnosis

    def evaluate_phase_acceptance(self, final_metrics):
        """评估阶段验收是否通过

        Args:
            final_metrics: dict 包含 val_frame_acc, val_type_acc, val_type_acc_cv,
                           val_loss (真 val 面), val_early_rate, cdcr_loss 等

        Returns:
            tuple: (是否通过, 未达标项列表)

        G-304(F6) 口径修正 (只改读数来源与措辞, 阈值一字未动):
          1. 损失键名两套并存: 相位1/2 用 `val_loss_max`, 相位3a/3b/3c 用 `loss_max`
             —— 原实现只读 `val_loss_max` ⇒ 相位3 的损失线**一次都没判过** (死旋钮)。
          2. 原调用方把**训练**平均损失挂在 `val_loss` 键上传进来 (phase3_train.py
             final_metrics['val_loss']=avg_loss['total']) ⇒ 拿 train 面比 val 红线,
             同名不同源。现在以真 val 面为准, 拿不到就明说"无读数"。
          3. 验收方案 §6 已裁决: λ_scene 换代 (0.8→5.243) 后绝对损失量级线**判停**
             (同尺度不可比) ⇒ 损失项降级为"登记", 不进 failures, 不据此定不通过。
          4. type 轴按 G-280 口径以 cv (课程内) 为主读数, 裸 Type Acc 只作连续性对照;
             两个数并列打印, 带口径名。
          5. `early_detection_min` (3c=50.0) / `cdcr_loss_max` (3b/3c) 原实现从不读取
             ⇒ 现按报告项兑现 (3c 收官时能看见差距, 但不作为中止依据)。
        """
        pc = self.phase_config
        if not pc:
            return True, []

        failures = []
        notes = []
        frame_acc = final_metrics.get('val_frame_acc', None)
        type_acc = final_metrics.get('val_type_acc', None)
        type_acc_cv = final_metrics.get('val_type_acc_cv', None)
        val_loss = final_metrics.get('val_loss', None)
        early_rate = final_metrics.get('val_early_rate', None)
        cdcr_loss = final_metrics.get('cdcr_loss', None)

        if 'frame_acc_min' in pc:
            if frame_acc is None:
                notes.append(f"Frame Acc 无读数 (闸 {pc['frame_acc_min']}% 未判)")
            elif frame_acc < pc['frame_acc_min']:
                failures.append(
                    f"Frame Acc(帧级裸口径) {frame_acc:.1f}% < {pc['frame_acc_min']}%")
            else:
                notes.append(
                    f"Frame Acc(帧级裸口径) {frame_acc:.1f}% ≥ {pc['frame_acc_min']}% ✓")

        if 'type_acc_min' in pc:
            _t = type_acc_cv if (type_acc_cv is not None and type_acc_cv >= 0) else type_acc
            _cal = 'cv 课程内' if (type_acc_cv is not None and type_acc_cv >= 0) else '裸口径(无 cv 读数)'
            if _t is None:
                notes.append(f"Type Acc 无读数 (闸 {pc['type_acc_min']}% 未判)")
            elif _t < pc['type_acc_min']:
                failures.append(
                    f"Type Acc({_cal}) {_t:.1f}% < {pc['type_acc_min']}%"
                    + (f" [裸口径={type_acc:.1f}%]" if type_acc is not None else ""))
            else:
                notes.append(
                    f"Type Acc({_cal}) {_t:.1f}% ≥ {pc['type_acc_min']}% ✓"
                    + (f" [裸口径={type_acc:.1f}%]" if type_acc is not None else ""))

        # 损失线: 键名两套都认 (§6 已判停 ⇒ 只登记)
        _loss_gate = pc.get('val_loss_max', pc.get('loss_max', None))
        if _loss_gate is not None:
            if val_loss is None:
                notes.append(f"损失线 {_loss_gate} 未判 (无 val_loss 读数)")
            else:
                notes.append(
                    f"Val Loss(真 val 面, 全局均值) {val_loss:.3f} vs 线 {_loss_gate} "
                    f"→ §6 已判停 (λ_scene 换代不可比), **仅登记不定不通过**")

        if pc.get('cdcr_active') and 'cdcr_loss_max' in pc:
            notes.append(
                f"CDCR 报告项: cdcr_loss={cdcr_loss} vs 线 {pc['cdcr_loss_max']} "
                f"(#272: CDCR/UASC 支路当前无读者, 只登记)")
        if 'early_detection_min' in pc:
            if early_rate is None:
                notes.append(f"早期检测率线 {pc['early_detection_min']}% 未判 (无读数)")
            else:
                notes.append(
                    f"Early Rate(报告项) {early_rate:.1f}% vs 线 "
                    f"{pc['early_detection_min']}%"
                    + (" ✓" if early_rate >= pc['early_detection_min'] else " ✗(未达)"))

        for _n in notes:
            LOGGER.info(f"  [验收·登记] {_n}")
        return len(failures) == 0, failures

    def print_checkpoint_summary(self, epoch):
        """打印检查点评估摘要"""
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"  检查点评估 @ Epoch {epoch+1} (Phase {self.phase})")
        LOGGER.info(f"  最佳指标: {self.best_metric:.2f}")
        LOGGER.info(f"  停滞计数: {self.stagnation_count}/{self.config['patience']}")
        if self.train_loss_history:
            LOGGER.info(f"  训练损失趋势: "
                        f"{list(self.train_loss_history)[-3:]}")
        LOGGER.info(f"{'='*60}")


# ============================================================
# 恢复训练: 检查点保存与加载
# ============================================================

def save_training_state(save_path, epoch, phase, model, optimizer=None,
                        scheduler=None, scaler=None, best_metric=0.0,
                        monitor_state=None):
    """保存训练状态 (支持恢复训练)

    Args:
        save_path: 保存路径
        epoch: 当前 epoch (0-based)
        phase: 当前 Phase
        model: 模型 (含 DDP unwrap)
        optimizer: 优化器
        scheduler: 学习率调度器
        scaler: AMP scaler
        best_metric: 最佳验证指标
        monitor_state: 监控器状态
    """
    raw_model = model.module if hasattr(model, 'module') else model

    state = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': raw_model.state_dict(),
        'best_val_metric': best_metric,
        'timestamp': datetime.now().isoformat(),
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    if monitor_state is not None:
        state['monitor_state'] = monitor_state

    # 原子写入: 先写 .tmp 再 rename
    tmp_path = str(save_path) + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, save_path)
    LOGGER.info(f"  训练状态已保存: {save_path} (epoch={epoch}, phase={phase})")


def load_training_state(ckpt_path, model, optimizer=None, scheduler=None,
                        scaler=None, device='cpu'):
    """加载训练状态

    Args:
        ckpt_path: 检查点路径
        model: 目标模型
        optimizer: 目标优化器
        scheduler: 目标调度器
        scaler: 目标 scaler
        device: 加载设备

    Returns:
        dict: {'epoch': start_epoch, 'phase': phase, 'best_metric': float,
               'monitor_state': dict or None}
    """
    if not os.path.exists(ckpt_path):
        LOGGER.warning(f"  恢复训练: 检查点不存在 {ckpt_path}")
        return {'epoch': 0, 'phase': None, 'best_metric': 0.0,
                'monitor_state': None}

    LOGGER.info(f"  恢复训练: 从 {ckpt_path} 加载...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    raw_model = model.module if hasattr(model, 'module') else model

    # 加载模型权重
    own = raw_model.state_dict()
    loaded = {k: v for k, v in ckpt.get('model_state_dict', {}).items()
              if k in own and own[k].shape == v.shape}
    own.update(loaded)
    raw_model.load_state_dict(own, strict=False)
    LOGGER.info(f"  模型权重: {len(loaded)}/{len(own)} keys "
                f"({len(loaded)/max(len(own),1):.1%})")

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            LOGGER.info("  优化器状态已恢复")
        except Exception as e:
            LOGGER.warning(f"  优化器恢复失败: {e}")

    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            LOGGER.info("  调度器状态已恢复")
        except Exception as e:
            LOGGER.warning(f"  调度器恢复失败: {e}")

    if scaler is not None and 'scaler_state_dict' in ckpt:
        try:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            LOGGER.info("  AMP scaler 状态已恢复")
        except Exception as e:
            LOGGER.warning(f"  Scaler 恢复失败: {e}")

    return {
        'epoch': ckpt.get('epoch', 0),
        'phase': ckpt.get('phase', ckpt.get('v24_phase', None)),
        'best_metric': ckpt.get('best_val_metric', 0.0),
        'monitor_state': ckpt.get('monitor_state', None),
    }


def find_resume_checkpoint(project_dir, name, phase):
    """自动查找恢复检查点

    查找顺序:
      1. {project}/{name}_p{phase}/last_temporal_v22.pt (Phase 3)
      2. {project}/{name}_p{phase}/weights/last.pt (Phase 1/2)

    Returns:
        str or None: 检查点路径
    """
    candidates = [
        os.path.join(project_dir, f'{name}_p{phase}', 'last_temporal_v22.pt'),
        os.path.join(project_dir, f'{name}_p{phase}', 'weights', 'last.pt'),
        os.path.join(project_dir, f'{name}_p{phase}', 'best_temporal.pt'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None
