# -*- coding: utf-8 -*-
"""
losses — V21 多任务损失函数 (从 train.py 拆分)

包含:
  - AsymmetricFocalLoss   场景三分类 (ignore_index=0, 标签 1/2/3 → 索引 0/1/2)
  - CauseMultiLabelLoss   9-bit 事故成因多标签 (BCEWithLogits + pos_weight + scene_mask)
  - AccidentTypeLoss      10 类事故类型分类 (仅 crash 场景)
  - v18MultiTaskLoss      L_det + λ₂L_scene + λ₄L_cause + λ₅L_type + λ₆L_3d
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss

import base_config
from base_config import DEFAULT_SCENE_CLS_WEIGHT, DEFAULT_CLASS_WEIGHTS, TYPE_CLASS_WEIGHT, dist, _has_dist
from core.pfrpn import PFRPNLoss

# V25: 对比学习损失 (可选)
try:
    from core.contrastive import ContrastiveLoss
    HAS_CONTRASTIVE = True
except ImportError:
    try:
        from contrastive import ContrastiveLoss
        HAS_CONTRASTIVE = True
    except ImportError:
        HAS_CONTRASTIVE = False


# ### D13修复(严重): 零损失可见性 — 原各损失在有效样本为零时静默返回 0,
# 无效损失被总 loss 掩盖, 曲线看似正常下降实则子任务无学习。
# 现按步节流告警 (首次 + 每500次), 既保持可见性又不淹没日志。
_zero_loss_counters = {}

# ### D13增强(严重, 审查提示词): 跨 epoch 零损失持续检测 —
# 单步告警不足以发现“子任务在整个 epoch 内静默零收敛”的隐蔽故障。
# 机制: 每个 epoch 结束时调用 _d13_epoch_check(), 检查各子任务零损失步占比;
# 若某子任务连续 3 个 epoch 零损失步占比 > 90%, 触发显著 WARNING。
_d13_epoch_zero_steps = {}  # {loss_name: [zero_steps_in_epoch_1, zero_steps_in_epoch_2, ...]}
_d13_epoch_total_steps = {}  # {loss_name: [total_steps_in_epoch_1, ...]}
_d13_consecutive_zero_epochs = {}  # {loss_name: consecutive_count}
D13_ZERO_EPOCH_THRESHOLD = 3  # 连续 N 个 epoch 零损失即报警
D13_ZERO_STEP_RATIO = 0.90  # epoch 内零损失步占比超过此值视为“该 epoch 零损失”


def _warn_zero_loss(name):
    c = _zero_loss_counters.get(name, 0) + 1
    _zero_loss_counters[name] = c
    if c == 1 or c % 500 == 0:
        LOGGER.warning(f'[D13] {name} 零返回 (有效样本=0) 累计 {c} 次 — 该损失未生效, '
                       f'请检查标签分布/采样配比; 若持续发生需警惕子任务静默零收敛。')


def _d13_epoch_mark(total_steps=None):
    """每个 epoch 结束时调用, 记录当前 epoch 各子任务的零损失步数 (增量)。

    V22-fix(D13): total_steps 由调用方传入 (len(train_loader)), 供占比判定;
    未传入时退化为下限估计。
    """
    for name, count in _zero_loss_counters.items():
        if name not in _d13_epoch_zero_steps:
            _d13_epoch_zero_steps[name] = []
            _d13_epoch_total_steps[name] = []
            _d13_consecutive_zero_epochs[name] = 0
        # 更精确: 用累计值减去上一次记录值
        if _d13_epoch_zero_steps[name]:
            prev_zero = sum(_d13_epoch_zero_steps[name])
            delta = count - prev_zero
        else:
            delta = count
        delta = max(delta, 0)
        _d13_epoch_zero_steps[name].append(delta)
        _d13_epoch_total_steps[name].append(total_steps if total_steps else max(delta, 1))


def _d13_epoch_check():
    """每个 epoch 结束时调用, 检查连续零损失 epoch 数。"""
    for name in list(_d13_epoch_zero_steps.keys()):
        zero_list = _d13_epoch_zero_steps[name]
        if not zero_list or zero_list[-1] == 0:
            # 本 epoch 无零损失步, 重置连续计数
            _d13_consecutive_zero_epochs[name] = 0
            continue
        # V22-fix(D13): 原实现把"存在任意零损失步"判为零损失 epoch, crash 稀疏时误报。
        # 现按占比判定: 零损失步占比 >= D13_ZERO_STEP_RATIO 才视为零损失 epoch。
        ratio = zero_list[-1] / max(_d13_epoch_total_steps[name][-1], 1)
        if ratio < D13_ZERO_STEP_RATIO:
            _d13_consecutive_zero_epochs[name] = 0
            continue
        _d13_consecutive_zero_epochs[name] = _d13_consecutive_zero_epochs.get(name, 0) + 1
        # 触发告警
        consec = _d13_consecutive_zero_epochs[name]
        if consec >= D13_ZERO_EPOCH_THRESHOLD:
            LOGGER.warning(
                f'[D13-CRITICAL] {name} 连续 {consec} 个 epoch 零损失步占比 '
                f'{ratio:.0%} (>={D13_ZERO_STEP_RATIO:.0%})! '
                f'子任务可能已退化为平凡解 (全零预测)。'
                f'请检查: (1) 标签覆盖率 (2) 采样配比 (3) 损失权重是否为 0。'
                f'建议: 中止训练并运行数据质量门禁。'
            )


# ============================================================
# 2b. V26-G254: scene 表征辅助损失 (纯函数, 可单测)
# ============================================================

def _supcon_loss(feat, labels, temperature=0.07):
    """监督对比 (SupCon, mean-of-logs): 同 scene 类 embedding 拉近、异类推远。
    feat [B,D] 未归一化; labels [B] in {0..3}, 0=ignore 既不作锚也不进分母/分子。
    批内无正例的锚跳过; B<2 返回 0。纯函数, 无内部集合通信 (DDP 安全)。
    """
    if feat is None or feat.dim() != 2 or feat.shape[0] < 2:
        dev = feat.device if feat is not None else 'cpu'
        return torch.zeros(1, device=dev).squeeze()
    z = F.normalize(feat.float(), dim=1)
    B = z.shape[0]
    valid = labels > 0
    pair = valid.view(-1, 1) & valid.view(1, -1)
    pair = pair & ~torch.eye(B, dtype=torch.bool, device=feat.device)
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & pair
    sim = z @ z.t() / temperature
    # 行平移不改 softmax 比值, 只保 exp 不溢出
    s = sim - sim.max(dim=1, keepdim=True).values.detach()
    denom = torch.exp(s).masked_fill(~pair, 0.0).sum(dim=1, keepdim=True).clamp(min=1e-12)
    log_prob = s - torch.log(denom)
    has_pos = pos.any(dim=1)
    pos_cnt = pos.sum(dim=1).clamp(min=1)
    loss_i = -(log_prob * pos.float()).sum(dim=1) / pos_cnt
    return (loss_i * has_pos.float()).sum() / has_pos.sum().clamp(min=1)


def _rdrop_kl(logits1, logits2):
    """R-Drop 对称 KL: 同输入两次 dropout 视图的预测互相对齐 (Liang 2021)。
    手写 KL 保证对两路 logits 都有梯度 (F.kl_div 的 target 梯度口径有版本差异)。
    """
    p1 = F.log_softmax(logits1.float(), dim=1)
    p2 = F.log_softmax(logits2.float(), dim=1)
    kl12 = (p1.exp() * (p1 - p2)).sum(dim=1).mean()
    kl21 = (p2.exp() * (p2 - p1)).sum(dim=1).mean()
    return (kl12 + kl21) * 0.5


def _scene_aux_losses(aux, scene_logits, scene_targets, supcon_w, consist_w,
                      temperature=0.07):
    """L2b 装配 (纯函数): 从 head 尾部 aux dict 取 embedding/第二视图,
    返回 (supcon_loss, consist_loss) 0 维张量。缺件或权重 0 ⇒ 对应项 0。"""
    dev = scene_logits.device if isinstance(scene_logits, torch.Tensor) else 'cpu'
    zero = torch.zeros(1, device=dev).squeeze()
    supcon = consist = zero
    if not isinstance(aux, dict):
        return supcon, consist
    sf = aux.get('supcon_feat')
    if supcon_w > 0 and isinstance(sf, torch.Tensor) and sf.dim() == 2 \
            and isinstance(scene_targets, torch.Tensor) \
            and scene_targets.numel() == sf.shape[0]:
        supcon = _supcon_loss(sf, scene_targets, temperature) * supcon_w
    v2 = aux.get('scene_view2')
    if consist_w > 0 and isinstance(v2, torch.Tensor) and isinstance(scene_logits, torch.Tensor) \
            and v2.shape == scene_logits.shape:
        consist = _rdrop_kl(scene_logits, v2) * consist_w
    return supcon, consist


# ============================================================
# 2. AsymmetricFocalLoss (场景三分类)
# ============================================================

class AsymmetricFocalLoss(nn.Module):
    """非对称 Focal Loss: 不同类别使用不同 gamma
    
    V20: 支持 ignore_index=0, 标签 0 的样本不参与损失计算
    场景标签: 0=ignore, 1=normal, 2=pre_crash, 3=crash
    模型输出 nc_cls=3 个 logits, 对应 normal/pre_crash/crash
    标签映射: target=1->cls_idx=0, target=2->cls_idx=1, target=3->cls_idx=2
    """

    # V21: label smoothing 0.05→0.1 提升泛化 (改进方案§4.3.2)
    # V25-ep4: smoothing 0.1→0.02 — crash 类 smoothing 0.1 将目标从1.0降到0.9,
    # 消散了10%的梯度信号, 对少数类(20k样本)损害远大于多数类。
    def __init__(self, alpha=None, gamma_per_class=None, smoothing=0.02, reduction='mean',
                 ignore_index=0):
        super().__init__()
        # V22-fix(gamma): 原 [2.0, 1.0, 1.5] 在 1:1:1 均衡采样下把 normal 的
        # 易样本梯度几乎清零 → 模型塌缩到预测 pre_crash (FAR 72%)。
        # V25-ep4: crash gamma 2.0→3.0 更强难样本挖掘
        # V25-ep7b: crash gamma 3.0→1.5 — gamma=3.0 过度压制 crash 易样本梯度
        # V25-ep32: crash gamma 1.5→2.0 — 1.5 导致 scene_loss 过拟合 (Gap=4.03),
        #   2.0 是平衡点: 保留难样本挖掘 + 减少过拟合
        # [normal, pre_crash, crash]
        self.gamma_per_class = gamma_per_class or [1.0, 1.0, 2.0]
        # V25-ablation: B3 消融开关 (gamma=0 禁用 focal)
        import os as _os
        if _os.environ.get('V25_ABLATE_FOCAL', '0') == '1':
            self.gamma_per_class = [0.0, 0.0, 0.0]
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index  # V20: 标签为 0 的样本不计算损失
        self.register_buffer('gamma', torch.tensor(self.gamma_per_class, dtype=torch.float32))
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
            elif isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha', alpha.clone().detach())
            else:
                self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets, sample_weight=None):
        device = inputs.device
        n_cls = inputs.size(1)
        
        # V20: 创建 ignore mask — 标签为 ignore_index 的样本不参与损失
        valid_mask = (targets != self.ignore_index)  # [B]
        if valid_mask.sum() == 0:
            # V22-fix(D13): 验证期全 ignore batch (真实帧 scene=0 连续排列) 属预期, 不告警
            if self.training:
                _warn_zero_loss('AsymmetricFocalLoss')
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 将标签映射到分类索引: 1->0, 2->1, 3->2
        cls_targets = (targets - 1).clamp(0, n_cls - 1)  # [B]
        
        if self.smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(inputs, self.smoothing / max(n_cls - 1, 1))
                smooth.scatter_(1, cls_targets.unsqueeze(1), 1.0 - self.smoothing)
            ce_loss = -(smooth * torch.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, cls_targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # 确保 gamma 和 alpha 在正确的设备上
        gamma = self.gamma.to(device)[cls_targets]
        focal_weight = (1 - pt) ** gamma

        if self.alpha is not None:
            at = self.alpha.to(device)[cls_targets]
            focal_loss = at * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # V20: 应用 ignore mask
        focal_loss = focal_loss * valid_mask.float()

        # #94: 逐样本 scene 证据档权重（质量档 clip 少进损失）。weight=1 ⇒ 与接入前数值等价。
        if sample_weight is not None:
            sw = sample_weight.view(-1).to(device).float()
            if sw.numel() == focal_loss.numel():
                focal_loss = focal_loss * sw
                eff = valid_mask.float() * sw
                if self.reduction == 'mean':
                    return focal_loss.sum() / (eff.sum() + 1e-8)
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                return focal_loss

        if self.reduction == 'mean':
            return focal_loss.sum() / (valid_mask.float().sum() + 1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

    def forward_unweighted(self, inputs, targets):
        """V21: 验证用 — 不应用 alpha/gamma 权重, 返回真实 CE loss (用于监控)
        
        解决 V20 问题: val/cls_loss 是训练的 20-400 倍
        原因: 训练用均衡采样 + alpha 权重, 验证用自然分布 + 同样权重 → 分布不匹配
        修复: 验证时记录无权重 CE, 反映模型真实性能
        """
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        cls_targets = (targets - 1).clamp(0, inputs.size(1) - 1)
        ce_loss = F.cross_entropy(inputs, cls_targets, reduction='none')
        ce_loss = ce_loss * valid_mask.float()
        return ce_loss.sum() / (valid_mask.float().sum() + 1e-8)


# ============================================================
# 3. 事故成因多标签损失 (BCEWithLogitsLoss)
# ============================================================

class CauseMultiLabelLoss(nn.Module):
    """9-bit 事故成因多标签分类损失

    - 使用 BCEWithLogitsLoss (内置 Sigmoid + BCE, 数值稳定)
    - 仅对 pre_crash(2) 和 crash(3) 场景计算损失 (normal=1 / ignore=0 时掩码)
    - per-factor pos_weight 处理类别不均衡 (行人权重 > 超速权重)
    """

    def __init__(self, pos_weight=None, num_causes=9):
        super().__init__()
        self.num_causes = num_causes
        if pos_weight is None:
            # V21-fix: 通过模块属性引用, 拿到 main() 动态计算后的最新权重
            pos_weight = base_config.CAUSE_POS_WEIGHT
        self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))
        # V26: 因子开关 —— 位下标不变, 只把 base_config.CAUSE_DISABLED_BITS 里的位
        # 从损失里摘掉 (见 base_config 的摘除依据)。全部启用时等价于原来的 num_causes 归一。
        self.register_buffer('factor_mask',
                             torch.tensor(base_config.cause_active_mask()[:num_causes],
                                          dtype=torch.float32).unsqueeze(0))  # [1, C]

    def forward(self, logits, targets, scene_mask, valid_mask=None):
        """
        Args:
            logits:     [B, 9] 成因 logits (no sigmoid)
            targets:    [B, 9] 成因标签 (0/1 多标签)
            scene_mask: [B]    场景标签, 2=pre_crash, 3=crash 时计算, 0=ignore / 1=normal 时跳过
            valid_mask: [B]    可选, cause_valid=0 的帧 (scene>=2 但无成因标注) 从损失中屏蔽
        Returns:
            scalar loss
        """
        if targets.numel() == 0:
            _warn_zero_loss('CauseMultiLabelLoss(empty_targets)')
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        device = logits.device
        # scene_mask: 仅 pre_crash(2)/crash(3) 的场景参与成因计算
        # V20: normal=1, pre_crash=2, crash=3, ignore=0
        active_mask = scene_mask >= 2
        if valid_mask is not None and valid_mask.numel() == scene_mask.numel():
            active_mask = active_mask & valid_mask.view(-1).bool()
        active_mask = active_mask.float().unsqueeze(1)  # [B, 1]
        if active_mask.sum() == 0:
            _warn_zero_loss('CauseMultiLabelLoss(no_active_scene)')
            return torch.tensor(0.0, device=device, requires_grad=True)
        # V26: 逐因子开关 → [B, C]; 摘除位不再计入分子分母 (全启用时与原式数值等价)
        active_mask = active_mask * self.factor_mask.to(device)

        # 防御: 确保 targets 在 [0,1] (脏数据可能导致 BCE 异常)
        targets = targets.float().clamp(0.0, 1.0)

        # BCEWithLogitsLoss with pos_weight (只作用于正样本, 处理类别不均衡)
        # pos_weight 作为专用参数传入 (而非 weight 逐元素乘)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight.to(device).unsqueeze(0),
            reduction='none'
        )  # [B, 9]

        # active_mask 已是 [B, C] 的逐元素掩码, 其和即参与计算的元素个数
        loss = (bce * active_mask).sum() / (active_mask.sum() + 1e-8)
        return loss


# ============================================================
# 4. 事故类型分类损失 (CrossEntropy)
# ============================================================

class AccidentTypeLoss(nn.Module):
    """7-class 事故类型分类损失

    V25-fix(type-bias): 加入焦点调制 (focal modulation) —
    对容易样本(head_on过度预测)降低梯度贡献, 对困难样本(rear_end/side_impact)提升。
    - CrossEntropyLoss with class weights + focal modulation
    - 仅对 crash(3) 场景计算损失
    """

    def __init__(self, class_weight=None, num_types=7, focal_gamma=1.0):
        super().__init__()
        if class_weight is None:
            class_weight = TYPE_CLASS_WEIGHT
        self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float32))
        self.num_types = num_types
        # V25-ablation: B3 消融开关 (gamma=0 禁用 focal)
        import os as _os
        if _os.environ.get('V25_ABLATE_FOCAL', '0') == '1':
            focal_gamma = 0.0
        self.focal_gamma = focal_gamma

    def forward(self, logits, targets, scene_mask):
        """
        Args:
            logits:     [B, 7] 类型 logits
            targets:    [B]    类型标签 (0-6, -1 表示无标签)
            scene_mask: [B]    场景标签, 仅 crash(3) 时计算
        Returns:
            scalar loss
        """
        # 仅 crash(3) 场景参与计算
        crash_mask = (scene_mask == 3)
        if crash_mask.sum() == 0:
            _warn_zero_loss('AccidentTypeLoss(no_crash)')
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[crash_mask]
        valid_targets = targets[crash_mask].long()

        # 过滤无标签 (-1) 和越界值
        valid_idx = (valid_targets >= 0) & (valid_targets < self.num_types)
        if valid_idx.sum() == 0:
            _warn_zero_loss('AccidentTypeLoss(no_valid_label)')
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = valid_logits[valid_idx]
        valid_targets = valid_targets[valid_idx]

        # V25-fix(type-bias): 焦点调制 — 对易样本(高置信度正确)降低梯度
        ce_per_sample = F.cross_entropy(
            valid_logits, valid_targets,
            weight=self.class_weight.to(valid_logits.device),
            reduction='none')
        with torch.no_grad():
            pt = torch.exp(-ce_per_sample)
            focal_weight = (1 - pt) ** self.focal_gamma
        loss = (ce_per_sample * focal_weight).mean()
        return loss


# ============================================================
# 6.9 V26-G247: 车道线掩码解码 (损失与评测共用, 口径唯一实现)
# ============================================================

_LANE_BIT_CACHE = None


def lane_bit_tables():
    """(tgt_bits[256], val_bits[256]) uint8: 掩码类值 → 7 通道 target/valid 位打包"""
    global _LANE_BIT_CACHE
    if _LANE_BIT_CACHE is None:
        t_tab, v_tab = base_config.lane_class_decode_table()
        tb = np.zeros(256, dtype=np.uint8)
        vb = np.zeros(256, dtype=np.uint8)
        for c in range(base_config.NUM_LANE_CHANNELS):
            tb |= ((t_tab[:, c] > 0.5).astype(np.uint8) << c)
            vb |= ((v_tab[:, c] > 0.5).astype(np.uint8) << c)
        _LANE_BIT_CACHE = (torch.from_numpy(tb), torch.from_numpy(vb))
    return _LANE_BIT_CACHE


LANE_IGNORE_RING = 1  # G-244: 正块 ±1 块邻域忽略带 (仅训练损失用, 评测仍严格口径)


def lane_target_valid(sem_mask, out_hw, ignore_ring=0):
    """掩码类值 [B,H,W] → P3 块级 (target, valid) [B,7,oh,ow] float32

    聚合口径 (与 LANE_POS_WEIGHT 实测一致):
      target_c = 块内任一有效像素通道 c 为正 (max-pool)
      valid_c  = 块内所有像素通道 c 均有效 (min-pool) —— 未知类型线像素使
                 type 通道无效, 混进背景块的未知类型线整块放弃 type 监督
                 (宁缺勿滥); role/color 仅在 255 像素处无效。
    ignore_ring>0 (G-244 监督容差): 正块邻域(环宽 ignore_ring 块)内的负块
      置为无效 —— 标注折线残余几像素抖动时, 命中真线的预测不再被当 FP 罚。
      仅训练损失传 >0; 评测传 0 保持严格口径, 与历史读数可比。
    位打包逐通道展开, 避免 [B,H,W,7] 全展开的 GB 级瞬时内存。
    """
    B, H, W = sem_mask.shape
    oh, ow = out_hw
    tgt_bits, val_bits = lane_bit_tables()
    m = sem_mask.long()
    tgt_bits = tgt_bits.to(m.device)[m]  # [B,H,W] uint8, bit c = 通道 c 为正
    val_bits = val_bits.to(m.device)[m]
    targets, valids = [], []
    for c in range(base_config.NUM_LANE_CHANNELS):
        vp = ((val_bits >> c) & 1).to(torch.float32).view(B, 1, H, W)
        tp = ((tgt_bits >> c) & 1).to(torch.float32).view(B, 1, H, W) * vp
        t = F.adaptive_max_pool2d(tp, (oh, ow))
        v = (F.adaptive_avg_pool2d(vp, (oh, ow)) >= 1.0 - 1e-6).to(torch.float32)
        if ignore_ring > 0:
            k = 2 * ignore_ring + 1
            near_pos = F.max_pool2d(t, k, stride=1, padding=ignore_ring)
            v = v * (1.0 - (near_pos - t).clamp(0.0, 1.0))  # 只豁免负块, 正块照旧监督
        targets.append(t)
        valids.append(v)
    return torch.cat(targets, dim=1), torch.cat(valids, dim=1)


# ============================================================
# 7. V18 多任务损失函数 (5 Losses)
# ============================================================

class v18MultiTaskLoss(v8DetectionLoss):
    """V20 多任务损失: L_det + λ₂L_scene + λ₄L_cause + λ₅L_type + λ₆L_3d
    
    V20 变更:
      - 移除 TTA Loss (无标签数据, loss 始终为 0)
      - CLASS_WEIGHTS 修正: pre_crash 获得最高权重 (核心目标类)
    V20 变更:
      - 新增 3D 几何损失 (depth + dims + orient + vel)
    """

    # V21-fix: 调整类别权重 — crash 获得最高权重, normal 适度降低
    # V20 [0.5, 3.0, 2.0] + Sampler [0.25, 0.45, 0.30] = 54x 过补偿 → 只预测 pre_crash
    # V21 [1.0, 1.2, 1.1] + Sampler [0.34, 0.33, 0.33] = ~1.0x 均衡 (1:1:1 数据均衡 + 温和补偿)
    # V21-refined 曾尝试 [1.5, 2.5, 0.8] 抑制 scene_loss, 但未同步到 train.py →
    # ### D15修复(严重): 运行时 train.py 经 V21_CLASS_WEIGHTS env 覆盖本值,
    # 实际生效的一直是 train.py 侧 [1.0,1.2,1.1]; 现统一到 base_config.DEFAULT_CLASS_WEIGHTS
    # 单一真相源, 消除两套权重条件性生效的混乱。
    CLASS_WEIGHTS = list(DEFAULT_CLASS_WEIGHTS)  # [normal, pre_crash, crash]

    def __init__(self, model, cls_weight=DEFAULT_SCENE_CLS_WEIGHT,
                 tta_weight=0.0, cause_weight=1.0, type_weight=0.1,
                 lambda_3d=1.0, lane_weight=0.3):
        super().__init__(model)
        self.model = model
        self.cls_weight = cls_weight
        self.tta_weight = 0.0  # V20: TTA 永久禁用 (无标签数据)
        self.cause_weight = cause_weight
        self.type_weight = type_weight
        self.lambda_3d = lambda_3d  # V20: 3D 损失权重
        # V26-G247: 车道线多标签分割权重 λ_lane (0 = 关闭损失, 但头始终前向)
        self.lane_weight = lane_weight

        # V22-fix(gamma): 对称 gamma=1.0 — 任何非对称 gamma 在 1:1:1 均衡采样下
        # 都会制造类别梯度偏差。focal 难样本聚焦由对称 gamma 保留。
        # V25-ep4: 恢复非对称 gamma — crash recall=8.4% 需要更强难样本挖掘
        # V25-ep32: crash gamma 1.5→2.0 — 平衡过拟合与难样本挖掘
        self.focal = AsymmetricFocalLoss(
            alpha=self.CLASS_WEIGHTS,
            gamma_per_class=[1.0, 1.0, 2.0],
            ignore_index=0
        )

        # 成因多标签损失 (V21-fix: 模块属性引用, 支持 main() 动态更新)
        self.cause_loss_fn = CauseMultiLabelLoss(pos_weight=base_config.CAUSE_POS_WEIGHT)

        # 事故类型损失
        self.type_loss_fn = AccidentTypeLoss(class_weight=TYPE_CLASS_WEIGHT)

        # V26-G247: 车道线 BCE pos_weight (解码/池化口径见 lane_target_valid)
        lane_bit_tables()  # 预热模块级位表缓存
        self._lane_pos_weight = torch.tensor(
            base_config.LANE_POS_WEIGHT, dtype=torch.float32).view(1, -1, 1, 1)

        # 统计计数器
        self._pred_counts = torch.zeros(3, dtype=torch.long)
        self._gt_counts = torch.zeros(3, dtype=torch.long)
        self._step_count = 0

        # V25: PF-RPN 损失
        self.pfrpn_loss_fn = PFRPNLoss(strides=[8, 16, 32])
        self.pfrpn_weight_obj = 1.0
        self.pfrpn_weight_box = 2.0
        self.pfrpn_weight_cen = 0.5
        self.pfrpn_warmup_epochs = 15
        self.pfrpn_warmup_scale = 0.0  # linear ramp from 0 to1 over warmup_epochs

        # V25: 对比学习损失 (可选)
        # V25-ep4: contrastive weight 0.1→0.3 — 增强特征区分能力
        self.contrastive_weight = getattr(model, 'data', {}).get('contrastive_weight', 0.3) if hasattr(model, 'data') else 0.3
        if HAS_CONTRASTIVE:
            self.contrastive_loss_fn = ContrastiveLoss(temperature=0.07)

        # V26-G254: scene 表征辅助损失 (监督对比 + R-Drop)。真生效权重读 yaml
        # (scene_supcon_weight / scene_consist_weight), base_config 仅兜底对齐 (防双真相源)。
        _md = getattr(model, 'data', None) or {}
        self.supcon_weight = float(_md.get('scene_supcon_weight', base_config.DEFAULT_SCENE_SUPCON_WEIGHT))
        self.consist_weight = float(_md.get('scene_consist_weight', base_config.DEFAULT_SCENE_CONSIST_WEIGHT))
        self.supcon_temperature = 0.07

    def update_class_weights(self, new_weights):
        """动态更新 CLASS_WEIGHTS"""
        # V20-fix(S10): 原地更新 alpha, 避免重复 register_buffer 和 CPU/GPU 设备不匹配
        w = torch.tensor(new_weights, dtype=torch.float32,
                         device=self.focal.alpha.device)
        self.focal.alpha.data.copy_(w)
        self.CLASS_WEIGHTS = list(new_weights)
        LOGGER.info(f'  Class weights updated: {new_weights}')

    def update_loss_weights(self, tta=None, cause=None, type_w=None,
                            lane_w=None, **kwargs):
        """Phase 3a/3b/3c 动态调整损失权重"""
        # 修复: get_phase3_loss_weights 返回的字典 key 是 'type', 这里兼容 'type' 与 'type_w'
        if type_w is None:
            type_w = kwargs.get('type', None)
        if tta is not None:
            self.tta_weight = tta
        if cause is not None:
            self.cause_weight = cause
        if type_w is not None:
            self.type_weight = type_w
        if lane_w is not None:
            self.lane_weight = lane_w
        LOGGER.info(f'  Loss weights: λ_scene={self.cls_weight:.2f}, '
                    f'λ_tta={self.tta_weight:.2f}, λ_cause={self.cause_weight:.2f}, '
                    f'λ_type={self.type_weight:.2f}, λ_lane={self.lane_weight:.2f}, '
                    f'λ_supcon={self.supcon_weight:.2f}, λ_consist={self.consist_weight:.2f}')

    def __call__(self, preds, batch):
        """preds: (det, scene_logits, tta_pred, cause_logits, type_logits, reid_maps)"""
        # ── B9 标签值域断言 (审查提示词强制项, 每 Phase 首次触发, 后续跳过) ──
        # 防止跨阶段标签语义冲突 (如 P1 ignore_index=0 vs P3 ignore_index=-100)
        if not getattr(self, '_b9_asserted', False):
            self._b9_asserted = True
            scene_t = batch.get('scene_cls', None)
            if scene_t is not None and isinstance(scene_t, torch.Tensor):
                s_min, s_max = scene_t.min().item(), scene_t.max().item()
                assert s_min >= 0 and s_max <= 3, \
                    f'[B9] scene_cls 值域异常: [{s_min}, {s_max}], 期望 [0, 3]'
            cause_t = batch.get('cause', None)
            if cause_t is not None and isinstance(cause_t, torch.Tensor):
                c_min, c_max = cause_t.min().item(), cause_t.max().item()
                assert c_min >= 0 and c_max <= 1, \
                    f'[B9] cause 值域异常: [{c_min}, {c_max}], 期望 [0, 1] (multi-hot)'
            type_t = batch.get('accident_type', None)
            if type_t is not None and isinstance(type_t, torch.Tensor):
                t_min, t_max = type_t.min().item(), type_t.max().item()
                assert t_min >= -1 and t_max <= 9, \
                    f'[B9] accident_type 值域异常: [{t_min}, {t_max}], 期望 [-1, 9]'
            sem_t = batch.get('semantic_mask', None)
            if sem_t is not None and isinstance(sem_t, torch.Tensor):
                # V26-G247: 车道线掩码值域 = 类值 0..16 或 255(ignore), 除此以外必是管线坏了
                _sv = sem_t.unique()
                assert bool(((_sv <= 16) | (_sv == 255)).all()), \
                    f'[B9] semantic_mask 值域异常: {[_ for _ in _sv.tolist() if _ > 16 and _ != 255][:8]}, 期望 0..16 或 255'

        det_preds = None
        scene_logits = tta_pred = cause_logits = type_logits = None

        if isinstance(preds, (tuple, list)):
            det_preds = preds[0]
            if len(preds) >= 2:
                scene_logits = preds[1]
            if len(preds) >= 3:
                tta_pred = preds[2]
            if len(preds) >= 4:
                cause_logits = preds[3]
            if len(preds) >= 5:
                type_logits = preds[4]

        # ── L1: 检测 Loss ──
        # DDP修复: 不在 error path 提前 return, 否则 scene/cause/type 头参数
        # 在一个 rank 上有梯度、另一个 rank 上无梯度 → DDP backward 死锁。
        # 改为标记 det_failed, 始终执行完整损失计算。
        det_failed = False
        try:
            det_total, det_items = super().__call__(det_preds, batch)
            # Ultralytics ≥8.4 returns det_items as dict{'box','cls','dfl'};
            # V21 expects a 1-D tensor for torch.cat downstream.
            if isinstance(det_items, dict):
                det_items = torch.stack([v.detach() if isinstance(v, torch.Tensor)
                                         else torch.tensor(v, device=det_total.device)
                                         for v in det_items.values()])
                if det_items.dim() > 1:
                    det_items = det_items.view(-1)
            # V24-fix(ultralytics-8.4): 该版本 det_total 为 [3] 向量 (box/cls/dfl × bs),
            # 非标量 → total.backward() 在首个 step 即崩溃; 归一化为标量
            if isinstance(det_total, torch.Tensor) and det_total.numel() > 1:
                det_total = det_total.sum()
        except RuntimeError as e:
            if "invalid" in str(e):
                # R1-15: det_preds 含 NaN/invalid 时回退为零损失;
                # 必须保持与模型输出的图连接, 否则 backward 崩溃 (无 grad_fn)
                device = batch['img'].device if isinstance(batch.get('img'), torch.Tensor) else torch.device('cuda')
                if isinstance(det_preds, (tuple, list)) and len(det_preds) > 0:
                    # R2-02: NaN×0=NaN (IEEE 754), 必须先用 nan_to_num 清除 NaN;
                    # 乘 0 保持计算图 (梯度为零但路径存在), 避免 DDP allreduce 死锁
                    det_total = sum(
                        torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0).sum()
                        for p in det_preds if isinstance(p, torch.Tensor)
                    ) * 0.0
                else:
                    det_total = torch.zeros(1, device=device, requires_grad=True)
                det_items = torch.zeros(3, device=device)
                det_failed = True
                # 不 return — 继续执行 scene/cause/type 损失, 确保所有头参数
                # 在所有 rank 上都有梯度, 避免 DDP backward 死锁
            else:
                raise

        scene_loss = torch.zeros(1, device=det_total.device).squeeze()
        cause_loss = torch.zeros(1, device=det_total.device).squeeze()
        type_loss = torch.zeros(1, device=det_total.device).squeeze()
        supcon_loss = torch.zeros(1, device=det_total.device).squeeze()
        consist_loss = torch.zeros(1, device=det_total.device).squeeze()

        # ── L2: 场景分类 Loss (V20: ignore_index=0, labels 1/2/3 -> cls 0/1/2) ──
        if scene_logits is not None:
            scene_targets = batch.get("scene_cls", None)
            if scene_targets is not None:
                scene_targets = self._process_targets(scene_targets, scene_logits.device).clamp(0, 3)
                # V22-fix(tile-mask) + V26-fix(tile-mask 统一): 2x2 瓦片 (_tNN) 标签-内容解耦,
                # scene 位置 0 不参与场景损失; 检测监督不受影响。三头共用 _apply_tile_mask。
                scene_targets = self._apply_tile_mask(batch, scene_targets)
                # #94: scene 证据档降权（质量档 clip 少进 scene 损失）；缺表/None ⇒ 全 1.0 等价接入前
                scene_w = self._process_targets(batch.get("scene_sup_weight", None),
                                                scene_targets.device)
                if scene_w is not None and scene_w.numel() != scene_targets.numel():
                    scene_w = None
                # AsymmetricFocalLoss 内部处理 ignore_index=0 和标签映射 1->0, 2->1, 3->2
                scene_loss = self.focal(scene_logits, scene_targets,
                                        sample_weight=scene_w) * self.cls_weight

                # V25-ep4: confusion penalty — 直接惩罚 crash→pre_crash 混淆
                # V25-ep4c: margin 1.5→2.0, weight 0.35→0.5 — 平衡版crash_Recall卡在25%
                # V25-ep5: margin 2.0→3.0, weight 0.5→0.8 — crash_Recall=34%距目标75%太远
                # V25-ep7: 回退 m=3.0→2.0/w=0.8→0.5 是灾难 — crash_Recall 39.9%→17.3%
                # V25-ep7b: 恢复 m=3.0/w=0.8, 同时降 AsymmetricFocal gamma_crash 3.0→1.5
                # V25-ep32: 降低 penalty weight 0.8→0.5 — scene_loss 被 penalty 拉至 4.8-8.8,
                #   Train-Val Gap=4.03 严重过拟合, 降低 penalty 让 scene_loss 更真实反映模型状态
                crash_mask = (scene_targets == 3)  # crash samples
                if crash_mask.any():
                    crash_logits = scene_logits[crash_mask]  # [N_crash, 3]
                    # 目标: crash logit (idx=2) > pre_crash logit (idx=1)
                    # 惩罚: max(0, logit_pre_crash - logit_crash + margin)
                    margin = 3.0
                    _pen = F.relu(crash_logits[:, 1] - crash_logits[:, 2] + margin)
                    if scene_w is not None:  # #94: 质量档 crash clip 的混淆惩罚同样按证据档降权
                        _pw = scene_w.view(-1)[crash_mask]
                        confusion_penalty = (_pen * _pw).sum() / (_pw.sum() + 1e-8)
                    else:
                        confusion_penalty = _pen.mean()
                    scene_loss = scene_loss + confusion_penalty * self.cls_weight * 0.5

                with torch.no_grad():
                    # V20: 预测统计 (模型输出 3 个 logits: normal/pre_crash/crash)
                    # 将预测映射回标签空间: pred_idx 0->1, 1->2, 2->3
                    pred_cls = scene_logits.argmax(dim=1) + 1  # [B] in {1,2,3}
                    # 仅统计有效样本 (排除 ignore_index)
                    valid = (scene_targets != 0)
                    pc = torch.zeros(4, device=pred_cls.device, dtype=torch.long)
                    gc = torch.zeros(4, device=scene_targets.device, dtype=torch.long)
                    for c in (1, 2, 3):
                        pc[c] = ((pred_cls == c) & valid).sum()
                        gc[c] = ((scene_targets == c) & valid).sum()
                    if self._pred_counts.device != pc.device:
                        self._pred_counts = self._pred_counts.to(pc.device)
                        self._gt_counts = self._gt_counts.to(pc.device)
                    # 累加索引 1/2/3 (对应 normal/pre_crash/crash)
                    self._pred_counts += pc[1:4]
                    self._gt_counts += gc[1:4]
                    self._step_count += 1

                    # V25: 自适应权重更新已移至 epoch-end callback (_v24_log_scene_dist)。
                    # 原在此处的 all_reduce + EMA 是4次 NCCL 死锁的根源 —
                    # DDP forward 内的集体通信与 DDP backward 的梯度同步产生竞争。
                    # 损失函数现在零集体操作, 权重更新完全在 callback 中完成。

        # ── L2b: V26-G254 scene 表征辅助 Loss ──
        # ① 监督对比: 同 scene 类 embedding 拉近/异类推远, 直接压实 crash↔normal 边界
        #    (G-249 根因: 边界向 normal 漂移); ② R-Drop: 同前向两次 dropout 视图预测对齐。
        # 无新参数, 无集合通信; aux dict 由 head 训练态尾部提供 (不占固定索引)。
        _aux = next((p for p in preds if isinstance(p, dict) and 'supcon_feat' in p), None)
        if _aux is not None and scene_logits is not None:
            _st_aux = self._process_targets(batch.get('scene_cls', None), scene_logits.device)
            if _st_aux is not None:
                _st_aux = self._apply_tile_mask(batch, _st_aux.clamp(0, 3))
            supcon_loss, consist_loss = _scene_aux_losses(
                _aux, scene_logits, _st_aux,
                self.supcon_weight, self.consist_weight, self.supcon_temperature)

        # ── L3: TTA 回归 Loss — V20: 永久移除 (无标签数据, 已从 items/total 中删除) ──

        # ── L4: 成因多标签 Loss ──
        if cause_logits is not None and self.cause_weight > 0:
            cause_targets = batch.get("cause", None)
            if cause_targets is not None:
                cause_targets = self._process_cause_targets(cause_targets, cause_logits.device)
                scene_t = self._process_targets(batch.get("scene_cls", None), cause_logits.device)
                if scene_t is None:
                    scene_t = torch.full((cause_targets.size(0),), 1, dtype=torch.long, device=cause_logits.device)  # V20-fix(NEW-2): default normal=1 (scene_mask>=2 过滤后 cause_loss=0, 安全)
                scene_t = scene_t.clamp(0, 3)
                # V26-fix(tile-mask-cause): 与 L2 对齐。V22-fix(tile-mask) 的理由是
                # "2x2 瓦片裁切区常无事故证据 (标签-内容解耦)"——该理由对成因头同样成立，
                # 但当初只在 L2 上了 scene_valid，L4 取的是原始 scene_cls ⇒ tile 若 scene>=2
                # 就会进成因损失，教一个常无事故证据的裁切块打成因位。这里补同一道掩码。
                scene_t = self._apply_tile_mask(batch, scene_t)
                # V24: 无成因标注的帧 (cause_valid=0) 屏蔽, 避免全零目标毒化成因头
                cv_t = self._process_targets(batch.get("cause_valid", None), cause_logits.device)
                if cv_t is not None and cv_t.numel() != cause_targets.size(0):
                    cv_t = None
                cause_loss = self.cause_loss_fn(
                    cause_logits, cause_targets, scene_t, valid_mask=cv_t) * self.cause_weight

        # ── L5: 事故类型 Loss ──
        if type_logits is not None and self.type_weight > 0:
            type_targets = batch.get("accident_type", None)
            if type_targets is not None:
                type_targets = self._process_targets(type_targets, type_logits.device).long()
                scene_t = self._process_targets(batch.get("scene_cls", None), type_logits.device)
                if scene_t is None:
                    scene_t = torch.full((type_targets.size(0),), 1, dtype=torch.long, device=type_logits.device)  # V20-fix(NEW-2): default normal=1 (crash_mask==3 过滤后 type_loss=0, 安全)
                scene_t = scene_t.clamp(0, 3)
                # V26-fix(tile-mask-type): 同 L4——瓦片裁切区常无事故证据，不该教类型头
                # "这是 t-bone/head-on"。与 L2 的 scene_valid 掩码同源同口径。
                scene_t = self._apply_tile_mask(batch, scene_t)
                type_loss = self.type_loss_fn(type_logits, type_targets, scene_t) * self.type_weight

        # ── V26-G247: L5b: 车道线多标签分割 Loss (P3 稠密 7 通道 BCE) ──
        # 监督: semantic_mask 类值 → 7 通道 (role2+type3+color2)。无折线帧掩码全 255
        # ⇒ 有效块 0 ⇒ 损失 0 (但保持计算图, DDP 全 rank 参数都有梯度路径)。
        lane_loss = torch.zeros(1, device=det_total.device).squeeze()
        lane_logits = None
        if len(preds) > 7 and isinstance(preds[7], torch.Tensor) and preds[7].dim() == 4:
            lane_logits = preds[7]  # [B, 7, H/8, W/8]
        if lane_logits is not None:
            sem_mask = batch.get("semantic_mask", None)
            if isinstance(sem_mask, torch.Tensor) and sem_mask.dim() == 3:
                lane_loss = self._lane_bce(lane_logits, sem_mask) * self.lane_weight
            else:
                # 掩码缺 (理论不应发生) → 图连接零损失, 不许整头断梯度
                lane_loss = lane_logits.sum() * 0.0

        # ── V20: L6: 3D 几何 Loss ──
        loss_3d = torch.zeros(1, device=det_total.device).squeeze()
        if self.lambda_3d > 0 and len(preds) >= 7 and preds[6] is not None:
            geo_3d_preds = preds[6]  # list of [B, A*10, Hi, Wi] 或 None
            gt_3d = batch.get("gt_3d", None)
            gt_3d_mask = batch.get("gt_3d_mask", None)  # V20-fix(C9): [B, MAX_OBJ] 有效掩码
            if geo_3d_preds is not None and gt_3d is not None:
                # V20-fix(C9): gt_3d 现在是固定形状 [B, MAX_OBJ_3D, 10], 始终为 Tensor
                try:
                    if isinstance(gt_3d, torch.Tensor) and gt_3d.numel() > 0:
                        device_3d = det_total.device
                        gt_3d = gt_3d.to(device_3d)
                        # 取 P3 尺度预测, 全局平均池化
                        p3_pred = geo_3d_preds[0]  # [B, A*10, H, W]
                        B_pred = p3_pred.shape[0]
                        # V20-fix(F3): 原 code 取 p3_pred[:, :10] 是原始未解码值
                        # depth 初始化在 log-space (bias=log(30)), 直接取 log() 会双重-log
                        # 正确做法: 对原始值做 decode (exp/normalize/clamp) 后再与 GT 比较
                        raw_10 = p3_pred[:, :10, :, :].mean(dim=(2, 3))  # [B, 7] raw
                        pred_depth = torch.exp(raw_10[:, 0:1].clamp(-5, 5)).clamp(0.5, 200.0)
                        pred_dims = torch.exp(raw_10[:, 1:4].clamp(-5, 5)).clamp(0.1, 25.0)
                        pred_orient = F.normalize(raw_10[:, 4:6], dim=-1)
                        pred_vel = raw_10[:, 6:9].clamp(-50.0, 50.0)
                        # GT: 使用掩码取有效对象的均值 (V20-fix(C9))
                        if gt_3d.dim() == 3:  # [B, MAX_OBJ, 10]
                            if gt_3d_mask is not None:
                                gt_3d_mask = gt_3d_mask.to(device_3d)
                                # 每个样本取有效对象的均值
                                mask_f = gt_3d_mask.unsqueeze(-1).float()  # [B, MAX_OBJ, 1]
                                n_valid = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
                                gt_mean = (gt_3d * mask_f).sum(dim=1) / n_valid  # [B, 7]
                            else:
                                gt_mean = gt_3d.mean(dim=1)  # [B, 7]
                        elif gt_3d.dim() == 2:  # [N, 10] 兼容旧格式
                            gt_mean = gt_3d.mean(dim=0, keepdim=True).expand(B_pred, -1)
                        else:
                            gt_mean = gt_3d
                        gt_mean = gt_mean.to(device_3d)
                        # 仅对有 3D 标签的样本计算损失 (V20-fix(C9))
                        has_3d_batch = batch.get("has_3d", None)
                        if has_3d_batch is not None:
                            if isinstance(has_3d_batch, torch.Tensor):
                                valid_3d = has_3d_batch.bool().to(device_3d)
                            elif isinstance(has_3d_batch, list):
                                valid_3d = torch.tensor(has_3d_batch, dtype=torch.bool, device=device_3d)
                            else:
                                valid_3d = torch.ones(B_pred, dtype=torch.bool, device=device_3d)
                        else:
                            valid_3d = torch.ones(B_pred, dtype=torch.bool, device=device_3d)
                        if valid_3d.any():
                            # V20-fix(F3): 使用解码后的预测值 (而非原始 log-space 值)
                            p_d = pred_depth[valid_3d]
                            p_dm = pred_dims[valid_3d]
                            p_o = pred_orient[valid_3d]
                            p_v = pred_vel[valid_3d]
                            gt_v = gt_mean[valid_3d]
                            # depth: log-space L1 (pred 已解码为米, 取 log 回到 log-space)
                            loss_depth = F.l1_loss(
                                torch.log(p_d.clamp(min=0.5)),
                                torch.log(gt_v[:, 0:1].clamp(min=0.5)))
                            loss_dims = F.l1_loss(p_dm, gt_v[:, 1:4])
                            loss_orient = 1.0 - F.cosine_similarity(
                                p_o, F.normalize(gt_v[:, 4:6], dim=-1), dim=-1).mean()
                            loss_vel = F.smooth_l1_loss(p_v, gt_v[:, 6:9])
                            loss_3d = (loss_depth + 0.5 * loss_dims + 0.3 * loss_orient + 0.5 * loss_vel) * self.lambda_3d
                except Exception as e:
                    if not hasattr(self, '_3d_warn_count'):
                        self._3d_warn_count = 0
                    self._3d_warn_count += 1
                    if self._3d_warn_count <= 3:
                        LOGGER.warning(f'[3D Loss] 计算异常 (#{self._3d_warn_count}): {type(e).__name__}: {e}')
                    loss_3d = torch.zeros(1, device=det_total.device).squeeze()

        # ── V25: L7: PF-RPN Loss ──
        pfrpn_loss_obj = torch.zeros(1, device=det_total.device).squeeze()
        pfrpn_loss_box = torch.zeros(1, device=det_total.device).squeeze()
        pfrpn_loss_cen = torch.zeros(1, device=det_total.device).squeeze()
        # Search from end: PFRPN dict is appended last after any contrastive outputs
        pfrpn_preds = None
        for p in reversed(preds):
            if isinstance(p, dict) and 'supcon_feat' not in p:
                pfrpn_preds = p
                break
        if pfrpn_preds is not None:
            img_size = (batch['img'].shape[2], batch['img'].shape[3])
            try:
                l_obj, l_box, l_cen = self.pfrpn_loss_fn(pfrpn_preds, batch, img_size)
                # warmup: linear ramp 0→1 over warmup_epochs
                phase_epoch = int(os.environ.get('V25_PFRPN_EPOCH', '0'))
                if phase_epoch < self.pfrpn_warmup_epochs:
                    scale = phase_epoch / max(self.pfrpn_warmup_epochs, 1)
                else:
                    scale = 1.0
                pfrpn_loss_obj = l_obj * self.pfrpn_weight_obj * scale
                pfrpn_loss_box = l_box * self.pfrpn_weight_box * scale
                pfrpn_loss_cen = l_cen * self.pfrpn_weight_cen * scale
            except Exception as e:
                if not hasattr(self, '_pfrpn_warn_count'):
                    self._pfrpn_warn_count = 0
                self._pfrpn_warn_count += 1
                if self._pfrpn_warn_count <= 3:
                    LOGGER.warning(f'[PFRPN Loss] 计算异常 (#{self._pfrpn_warn_count}): {type(e).__name__}: {e}')

        # ── V25: L8: Contrastive Loss (语义对比学习) ──
        contrastive_loss = torch.zeros(1, device=det_total.device).squeeze()
        # V26-G247: lane_logits 占索引 7 后, vis_proj/cls_feat 后移到 8,9
        _vis_proj = preds[8] if len(preds) > 8 else None
        _cls_feat = preds[9] if len(preds) > 9 else None
        # Guard: skip if these are dicts (PFRPN) or None
        if HAS_CONTRASTIVE and _vis_proj is not None and _cls_feat is not None \
                and isinstance(_vis_proj, torch.Tensor) and isinstance(_cls_feat, torch.Tensor) \
                and _vis_proj.dim() == 2 and _cls_feat.dim() == 2:
            vis_proj = _vis_proj   # [B, 256] visual projection
            cls_feat = _cls_feat   # [B, 1024] scene cls_feat

            # 方式1: 使用 batch 中的预计算 text_emb (离线模式)
            text_emb = batch.get('text_emb', None)
            if isinstance(text_emb, (list, tuple)):
                text_emb = torch.stack(text_emb) if text_emb and isinstance(text_emb[0], torch.Tensor) else None
            if text_emb is not None and text_emb.dim() == 2 and text_emb.sum() != 0:
                try:
                    contrastive_loss = self.contrastive_loss_fn(vis_proj, text_emb) * self.contrastive_weight
                except Exception as e:
                    if not hasattr(self, '_contrastive_warn_count'):
                        self._contrastive_warn_count = 0
                    self._contrastive_warn_count += 1
                    if self._contrastive_warn_count <= 3:
                        LOGGER.warning(f'[Contrastive Loss] 计算异常: {e}')
            # 方式2: 使用模型的 text_encoder 端到端训练
            elif hasattr(self.model, 'text_encoder') and self.model.text_encoder is not None:
                text_token_ids = batch.get('text_token_ids', None)
                text_attn_mask = batch.get('text_attn_mask', None)
                has_text = batch.get('has_text', None)
                if text_token_ids is not None and has_text is not None:
                    # collate_fn 不 stack 自定义 key → 可能是 tuple, 需转为 tensor
                    if isinstance(has_text, (list, tuple)):
                        has_text = torch.stack(has_text) if isinstance(has_text[0], torch.Tensor) \
                            else torch.tensor(has_text, device=vis_proj.device)
                    if isinstance(text_token_ids, (list, tuple)):
                        text_token_ids = torch.stack(text_token_ids).to(vis_proj.device)
                    if isinstance(text_attn_mask, (list, tuple)):
                        text_attn_mask = torch.stack(text_attn_mask).to(vis_proj.device)
                    text_mask = has_text.squeeze().bool()
                    if text_mask.any():
                        try:
                            text_feat = self.model.text_encoder(
                                text_token_ids[text_mask],
                                text_attn_mask[text_mask] if text_attn_mask is not None else None
                            )
                            text_proj = self.model.text_proj_head(text_feat)  # [N, 256]
                            vis_matched = vis_proj[text_mask]  # [N, 256]
                            contrastive_loss = self.contrastive_loss_fn(vis_matched, text_proj) * self.contrastive_weight
                        except Exception as e:
                            if not hasattr(self, '_contrastive_warn_count'):
                                self._contrastive_warn_count = 0
                            self._contrastive_warn_count += 1
                            if self._contrastive_warn_count <= 3:
                                LOGGER.warning(f'[Contrastive Loss] 计算异常: {e}')

        # ── 总 Loss ──
        total = det_total + scene_loss + cause_loss + type_loss + lane_loss + loss_3d \
                + pfrpn_loss_obj + pfrpn_loss_box + pfrpn_loss_cen \
                + contrastive_loss + supcon_loss + consist_loss
        # V21-fix: 验证时 items 记录无加权 scene loss, 避免 val/cls_loss 虚高
        # 训练时 items 仍记录加权值 (与 total 一致, 便于监控梯度)
        scene_loss_log = scene_loss.detach().view(1)
        if not self.model.training and scene_logits is not None:
            scene_targets_log = batch.get("scene_cls", None)
            if scene_targets_log is not None:
                scene_targets_log = self._process_targets(scene_targets_log, scene_logits.device).clamp(0, 3)
                scene_loss_log = self.focal.forward_unweighted(scene_logits, scene_targets_log).detach().view(1)
        items = torch.cat([
            det_items,
            scene_loss_log,
            cause_loss.detach().view(1),
            type_loss.detach().view(1),
            lane_loss.detach().view(1),
            loss_3d.detach().view(1),
            pfrpn_loss_obj.detach().view(1),
            pfrpn_loss_box.detach().view(1),
            pfrpn_loss_cen.detach().view(1),
            contrastive_loss.detach().view(1),
            supcon_loss.detach().view(1),
            consist_loss.detach().view(1),
        ])
        return total, items

    def _lane_bce(self, lane_logits, sem_mask):
        """V26-G247: 掩码类值 → P3 块级 7 通道 (target, valid) → 掩码 BCE
        G-244: 训练侧带 ±1 块忽略带 (评测口径不变, 传 0)"""
        target, valid = lane_target_valid(sem_mask, lane_logits.shape[-2:],
                                          ignore_ring=LANE_IGNORE_RING)
        bce = F.binary_cross_entropy_with_logits(
            lane_logits, target, reduction='none',
            pos_weight=self._lane_pos_weight.to(lane_logits.device))
        return (bce * valid).sum() / valid.sum().clamp(min=1.0)

    @staticmethod
    def _process_targets(targets, device):
        """标准化 batch 目标: list/Tensor → Tensor[B]"""
        if targets is None:
            return None
        if isinstance(targets, (list, tuple)):
            tensors = []
            for t in targets:
                if isinstance(t, torch.Tensor):
                    tensors.append(t.view(-1))
                else:
                    tensors.append(torch.tensor([t], dtype=torch.long))
            return torch.cat(tensors).to(device)
        elif isinstance(targets, torch.Tensor):
            return targets.view(-1).to(device)
        return targets

    @staticmethod
    def _apply_tile_mask(batch, scene_t):
        """V26-fix(tile-mask): 2x2 瓦片 (_tNN) 的 scene 位不进**任何语义头**。

        口径唯一实现，三个头共用，不许各写一份。依据是 V22-fix(tile-mask) 的原话
        "瓦片裁切区常无事故证据 (标签-内容解耦) → 屏蔽场景损失/指标, 保留检测监督"——
        该理由对成因头/类型头同样成立，而 L4/L5 此前取的是**原始** scene_cls，
        未过 scene_valid ⇒ tile 一旦被标成 scene>=2 就会进成因损失、标成 3 就进类型损失，
        等于拿一个常无事故证据的裁切块去教语义头。这里补上与 L2 同源的掩码：
        scene_valid==0 的样本 scene 位置 0 ⇒ L4 的 `scene_mask>=2` 与 L5 的
        `scene_mask==3` 都自动不激活，检测监督不受影响（与注释声明一致）。

        掩码缺省/长度不符时**不改**（fail-open 到旧行为），避免因 batch 组装差异
        把整批样本静默丢出损失。
        """
        if scene_t is None or not isinstance(scene_t, torch.Tensor):
            return scene_t
        sv = v18MultiTaskLoss._process_targets(batch.get('scene_valid', None), scene_t.device)
        if sv is None or sv.numel() != scene_t.numel():
            return scene_t
        return torch.where(sv.view(-1) > 0, scene_t.view(-1), torch.zeros_like(scene_t.view(-1)))

    @staticmethod
    def _process_cause_targets(cause_targets, device):
        """标准化 cause targets: → Tensor[B, 9]"""
        if cause_targets is None:
            return None
        if isinstance(cause_targets, (list, tuple)):
            tensors = []
            for ct in cause_targets:
                if isinstance(ct, torch.Tensor):
                    if ct.dim() == 1:
                        tensors.append(ct.view(1, -1))
                    elif ct.dim() == 2:
                        tensors.append(ct)
                elif isinstance(ct, (list, tuple)):
                    tensors.append(torch.tensor([ct], dtype=torch.float32))
                else:
                    tensors.append(torch.zeros(1, 16, dtype=torch.float32))
            return torch.cat(tensors, dim=0).to(device)
        elif isinstance(cause_targets, torch.Tensor):
            if cause_targets.dim() == 1:
                return cause_targets.view(1, -1).to(device)
            return cause_targets.to(device)
        return cause_targets
