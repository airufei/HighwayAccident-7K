"""
PF-RPN: Proposal-Free Region Proposal Network (V25)

无锚框、无类别约束的候选区域生成器。
在 FPN 多尺度特征上预测 objectness + box + centerness,
能自动发现画面中所有潜在物体, 补充 3 类检测头的盲区。

架构: 类似 FCOS, 每个空间位置直接预测物体存在性和边界距离 (l,t,r,b)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['PFRPN', 'PFRPNLoss', 'decode_pfrpn_proposals', 'fuse_proposals']


# ============================================================
# 1. 子模块
# ============================================================

class _FeatureBlock(nn.Module):
    """Conv→GroupNorm→SiLU→Conv→GroupNorm→SiLU"""

    def __init__(self, in_ch, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, hidden), hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, hidden), hidden)

    def forward(self, x):
        x = F.silu(self.gn1(self.conv1(x)))
        x = F.silu(self.gn2(self.conv2(x)))
        return x


class _ObjectnessHead(nn.Module):
    """二值 objectness 预测"""

    def __init__(self, hidden):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden // 2), hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class _BoxHead(nn.Module):
    """box 回归: 预测 (l, t, r, b) 距离"""

    def __init__(self, hidden):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden // 2), hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden // 2, 4, 1),
        )

    def forward(self, x):
        return F.relu(self.conv(x))


class _CenterNessHead(nn.Module):
    """centerness 质量评分"""

    def __init__(self, hidden):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden // 2), hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)  # raw logits; sigmoid in BCEWithLogitsLoss (autocast-safe)


# ============================================================
# 2. PFRPN 主模块
# ============================================================

class PFRPN(nn.Module):
    """Proposal-Free Region Proposal Network.

    Args:
        ch_list: FPN 通道数列表 [256, 512, 1024]
        hidden_dim: 隐藏层维度 (default 256)
        strides: FPN 步长 [8, 16, 32]
    """

    def __init__(self, ch_list=(256, 512, 1024), hidden_dim=256, strides=(8, 16, 32)):
        super().__init__()
        self.strides = list(strides)
        self.n_scales = len(ch_list)

        self.feat_blocks = nn.ModuleList()
        self.obj_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()
        self.cen_heads = nn.ModuleList()

        for c in ch_list:
            self.feat_blocks.append(_FeatureBlock(c, hidden_dim))
            self.obj_heads.append(_ObjectnessHead(hidden_dim))
            self.box_heads.append(_BoxHead(hidden_dim))
            self.cen_heads.append(_CenterNessHead(hidden_dim))

    def forward(self, features):
        """前向传播.

        Args:
            features: list of [B, C_s, H_s, W_s] — P3, P4, P5

        Returns:
            dict:
                'objectness': list of [B, 1, H_s, W_s]
                'boxes':      list of [B, 4, H_s, W_s]  (l,t,r,b in pixels)
                'centerness': list of [B, 1, H_s, W_s]
        """
        obj_list, box_list, cen_list = [], [], []
        for i, feat in enumerate(features):
            f = self.feat_blocks[i](feat)
            obj_list.append(self.obj_heads[i](f))
            box_list.append(self.box_heads[i](f))
            cen_list.append(self.cen_heads[i](f))
        return {
            'objectness': obj_list,
            'boxes': box_list,
            'centerness': cen_list,
        }


# ============================================================
# 3. PFRPN 损失
# ============================================================

class PFRPNLoss(nn.Module):
    """PF-RPN 损失: focal objectness + GIoU box + BCE centerness.

    Args:
        strides: FPN 步长
        focal_alpha: focal loss alpha
        focal_gamma: focal loss gamma
        area_ranges: 每个 FPN 级别的面积范围 (像素面积)
    """

    def __init__(self, strides=(8, 16, 32), focal_alpha=0.25, focal_gamma=2.0,
                 area_ranges=((0, 1024), (1024, 9216), (9216, 1e8))):
        super().__init__()
        self.strides = list(strides)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.area_ranges = area_ranges

    def forward(self, preds, batch, img_size):
        """计算 PFRPN 损失.

        Args:
            preds: PFRPN.forward() 输出 dict
            batch: dict, 需要 'bboxes' [N,4] (normalized xywh), 'batch_idx' [N]
            img_size: (H, W) 图像尺寸

        Returns:
            obj_loss, box_loss, cen_loss (标量)
        """
        device = preds['objectness'][0].device
        bboxes = batch.get('bboxes', torch.zeros(0, 4, device=device))
        batch_idx = batch.get('batch_idx', torch.zeros(0, dtype=torch.long, device=device))

        B = preds['objectness'][0].shape[0]
        H_img, W_img = img_size

        # 归一化 xywh → 像素 xyxy
        if bboxes.numel() > 0:
            boxes_xyxy = self._xywh2xyxy(bboxes, H_img, W_img)
        else:
            boxes_xyxy = torch.zeros(0, 4, device=device)

        # 生成 GT
        obj_gts, box_gts, cen_gts, pos_masks = self._generate_gt(
            boxes_xyxy, batch_idx, B, img_size, device)

        # 计算损失
        obj_loss = torch.zeros(1, device=device).squeeze()
        box_loss = torch.zeros(1, device=device).squeeze()
        cen_loss = torch.zeros(1, device=device).squeeze()
        n_pos_total = 0
        n_scales = len(self.strides)

        for s in range(n_scales):
            obj_pred = preds['objectness'][s]  # [B, 1, H, W]
            box_pred = preds['boxes'][s]       # [B, 4, H, W]
            cen_pred = preds['centerness'][s]  # [B, 1, H, W]

            obj_gt = obj_gts[s]     # [B, 1, H, W]
            box_gt = box_gts[s]     # [B, 4, H, W]
            cen_gt = cen_gts[s]     # [B, 1, H, W]
            pos_mask = pos_masks[s] # [B, 1, H, W]

            n_pos = pos_mask.sum().clamp(min=1)
            n_pos_total += n_pos.item()

            # focal objectness loss (mean-reduced, no n_pos division needed)
            obj_loss = obj_loss + self._focal_loss(obj_pred, obj_gt)

            # GIoU box loss (仅正样本, mean-reduced)
            if pos_mask.any():
                pos_box_pred = box_pred[pos_mask.expand_as(box_pred)].view(-1, 4)
                pos_box_gt = box_gt[pos_mask.expand_as(box_gt)].view(-1, 4)
                box_loss = box_loss + self._giou_loss(pos_box_pred, pos_box_gt)

                # BCE centerness loss (仅正样本, mean-reduced)
                pos_cen_pred = cen_pred[pos_mask]
                pos_cen_gt = cen_gt[pos_mask]
                cen_loss = cen_loss + F.binary_cross_entropy_with_logits(
                    pos_cen_pred, pos_cen_gt, reduction='mean')

        return obj_loss, box_loss, cen_loss

    @staticmethod
    def _xywh2xyxy(bboxes, H, W):
        """归一化 center-xywh → 像素 xyxy"""
        cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _generate_gt(self, boxes_xyxy, batch_idx, B, img_size, device):
        """为每个 FPN 级别生成 GT objectness / box / centerness."""
        H_img, W_img = img_size
        obj_gts, box_gts, cen_gts, pos_masks = [], [], [], []
        n_scales = len(self.strides)

        for s in range(n_scales):
            stride = self.strides[s]
            H = H_img // stride
            W = W_img // stride

            obj_gt = torch.zeros(B, 1, H, W, device=device)
            box_gt = torch.zeros(B, 4, H, W, device=device)
            cen_gt = torch.zeros(B, 1, H, W, device=device)
            pos_mask = torch.zeros(B, 1, H, W, dtype=torch.bool, device=device)

            if boxes_xyxy.numel() == 0:
                obj_gts.append(obj_gt)
                box_gts.append(box_gt)
                cen_gts.append(cen_gt)
                pos_masks.append(pos_mask)
                continue

            # 为每个 GT box 分配 FPN 级别
            areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            lo, hi = self.area_ranges[s]
            scale_mask = (areas >= lo) & (areas < hi)

            for idx in torch.where(scale_mask)[0]:
                b = int(batch_idx[idx].item())
                x1, y1, x2, y2 = boxes_xyxy[idx]

                # 映射到特征图坐标
                cx_feat = ((x1 + x2) / 2 / stride).long().clamp(0, W - 1)
                cy_feat = ((y1 + y2) / 2 / stride).long().clamp(0, H - 1)

                # 正样本范围: 中心在 GT 框内的空间位置
                x1f = int((x1 / stride).floor().long().clamp(0, W - 1).item())
                y1f = int((y1 / stride).floor().long().clamp(0, H - 1).item())
                x2f = int((x2 / stride).ceil().long().clamp(0, W - 1).item())
                y2f = int((y2 / stride).ceil().long().clamp(0, H - 1).item())

                if x2f <= x1f or y2f <= y1f:
                    continue

                obj_gt[b, 0, y1f:y2f, x1f:x2f] = 1.0

                # box targets: (l, t, r, b) 距离
                ys = torch.arange(y1f, y2f, device=device, dtype=torch.float32)
                xs = torch.arange(x1f, x2f, device=device, dtype=torch.float32)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
                centers_x = (grid_x + 0.5) * stride
                centers_y = (grid_y + 0.5) * stride

                l = (centers_x - x1).clamp(min=0)
                t = (centers_y - y1).clamp(min=0)
                r = (x2 - centers_x).clamp(min=0)
                b_dist = (y2 - centers_y).clamp(min=0)

                region = torch.stack([l, t, r, b_dist], dim=0)  # [4, h, w]
                box_gt[b, :, y1f:y2f, x1f:x2f] = region

                # centerness targets
                min_lr = torch.min(l, r)
                max_lr = torch.max(l, r).clamp(min=1e-6)
                min_tb = torch.min(t, b_dist)
                max_tb = torch.max(t, b_dist).clamp(min=1e-6)
                cen = torch.sqrt(min_lr / max_lr * min_tb / max_tb)
                cen_gt[b, 0, y1f:y2f, x1f:x2f] = cen

                pos_mask[b, 0, y1f:y2f, x1f:x2f] = True

            obj_gts.append(obj_gt)
            box_gts.append(box_gt)
            cen_gts.append(cen_gt)
            pos_masks.append(pos_mask)

        return obj_gts, box_gts, cen_gts, pos_masks

    def _focal_loss(self, pred, target):
        """Sigmoid focal loss (所有空间位置)."""
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        loss = -alpha_t * ((1 - pt) ** gamma) * (pt + 1e-8).log()
        return loss.mean()

    @staticmethod
    def _giou_loss(pred, target):
        """GIoU loss (正样本)."""
        x1 = torch.max(pred[:, 0], target[:, 0])
        y1 = torch.max(pred[:, 1], target[:, 1])
        x2 = torch.min(pred[:, 2], target[:, 2])
        y2 = torch.min(pred[:, 3], target[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
        area_gt = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
        union = area_pred + area_gt - inter + 1e-6
        iou = inter / union

        # enclosing box
        cx1 = torch.min(pred[:, 0], target[:, 0])
        cy1 = torch.min(pred[:, 1], target[:, 1])
        cx2 = torch.max(pred[:, 2], target[:, 2])
        cy2 = torch.max(pred[:, 3], target[:, 3])
        c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-6

        giou = iou - (c_area - union) / c_area
        return (1 - giou).mean()


# ============================================================
# 4. 解码与融合
# ============================================================

def decode_pfrpn_proposals(preds, img_size, strides=(8, 16, 32),
                           conf_thresh=0.3, nms_thresh=0.6, top_k=200):
    """解码 PFRPN 预测为提案 (per image).

    Args:
        preds: PFRPN.forward() 输出
        img_size: (H, W)
        strides: FPN 步长
        conf_thresh: 置信度阈值
        nms_thresh: NMS IoU 阈值
        top_k: 每张图最多保留的提案数

    Returns:
        list of [N_i, 5] tensors (x1, y1, x2, y2, confidence)
    """
    H_img, W_img = img_size
    B = preds['objectness'][0].shape[0]
    results = []

    for b in range(B):
        all_boxes = []
        all_scores = []

        for s in range(len(strides)):
            stride = strides[s]
            obj = preds['objectness'][s][b, 0]  # [H, W]
            box = preds['boxes'][s][b]           # [4, H, W]
            cen = torch.sigmoid(preds['centerness'][s][b, 0])  # [H, W]

            H, W = obj.shape
            conf = obj * cen  # [H, W]

            # 阈值过滤
            mask = conf > conf_thresh
            if not mask.any():
                continue

            scores = conf[mask]
            ys, xs = torch.where(mask)

            # 解码 box (l,t,r,b → x1,y1,x2,y2)
            l = box[0, ys, xs]
            t = box[1, ys, xs]
            r = box[2, ys, xs]
            b_dist = box[3, ys, xs]

            cx = (xs + 0.5).float() * stride
            cy = (ys + 0.5).float() * stride

            x1 = (cx - l).clamp(0, W_img)
            y1 = (cy - t).clamp(0, H_img)
            x2 = (cx + r).clamp(0, W_img)
            y2 = (cy + b_dist).clamp(0, H_img)

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            all_boxes.append(boxes)
            all_scores.append(scores)

        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)

            # Top-k
            if all_scores.numel() > top_k:
                topk_idx = all_scores.topk(top_k).indices
                all_boxes = all_boxes[topk_idx]
                all_scores = all_scores[topk_idx]

            # NMS
            keep = _nms(all_boxes, all_scores, nms_thresh)
            proposals = torch.cat([all_boxes[keep], all_scores[keep, None]], dim=1)
        else:
            proposals = torch.zeros(0, 5, device=preds['objectness'][0].device)

        results.append(proposals)

    return results


def _nms(boxes, scores, thresh):
    """Simple NMS (torchvision.ops.nms fallback)."""
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, thresh)
    except ImportError:
        # fallback: 贪心 NMS
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            ious = _box_iou(boxes[i:i+1], boxes[order[1:]])[0]
            mask = ious <= thresh
            order = order[1:][mask]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _box_iou(box1, box2):
    """计算两组 box 的 IoU."""
    x1 = torch.max(box1[:, 0:1], box2[:, 0].T)
    y1 = torch.max(box1[:, 1:2], box2[:, 1].T)
    x2 = torch.min(box1[:, 2:3], box2[:, 2].T)
    y2 = torch.min(box1[:, 3:4], box2[:, 3].T)
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2.T - inter
    return inter / (union + 1e-6)


def fuse_proposals(det_preds, pfrpn_proposals, boost_alpha=0.1,
                   iou_match=0.5, iou_suppress=0.3):
    """融合检测结果与 PFRPN 提案.

    Args:
        det_preds: 检测输出 tensor [B, N_det, 6] (x1,y1,x2,y2,score,cls)
        pfrpn_proposals: decode_pfrpn_proposals() 输出 (list of [N,5])
        boost_alpha: 匹配到的检测置信度提升系数
        iou_match: IoU >= 此值 → 匹配
        iou_suppress: IoU >= 此值 → PFRPN 提案被抑制

    Returns:
        list of [M, 6] tensors (x1, y1, x2, y2, score, class)
        其中 class=-1 表示 PFRPN-only 未知物体
    """
    results = []
    B = len(pfrpn_proposals)

    for b in range(B):
        pfrpn = pfrpn_proposals[b]  # [N, 5]
        if isinstance(det_preds, (list, tuple)):
            det = det_preds[b] if b < len(det_preds) else torch.zeros(0, 6, device=pfrpn.device)
        else:
            det = det_preds[b] if det_preds.dim() == 3 else det_preds

        if det.numel() == 0 and pfrpn.numel() == 0:
            results.append(torch.zeros(0, 6, device=pfrpn.device))
            continue

        merged = []

        if det.numel() > 0 and pfrpn.numel() > 0:
            det_boxes = det[:, :4]
            det_scores = det[:, 4]
            det_cls = det[:, 5] if det.shape[1] > 5 else torch.zeros(len(det), device=det.device)

            pfrpn_boxes = pfrpn[:, :4]
            pfrpn_scores = pfrpn[:, 4]

            ious = _box_iou(det_boxes, pfrpn_boxes)  # [N_det, N_pfrpn]

            # Boost 匹配到的检测
            max_iou_per_det, _ = ious.max(dim=1)
            boosted_scores = det_scores.clone()
            boost_mask = max_iou_per_det >= iou_match
            if boost_mask.any():
                max_obj_per_det = (ious[boost_mask] * pfrpn_scores[None, :]).max(dim=1).values
                boosted_scores[boost_mask] = (det_scores[boost_mask] +
                                              boost_alpha * max_obj_per_det).clamp(max=1.0)

            for i in range(len(det)):
                merged.append(torch.cat([det_boxes[i], boosted_scores[i:i+1], det_cls[i:i+1]]))

            # 添加未匹配的 PFRPN 提案
            for j in range(len(pfrpn)):
                if ious[:, j].max() < iou_suppress:
                    merged.append(torch.cat([pfrpn_boxes[j], pfrpn_scores[j:j+1],
                                             torch.full((1,), -1, device=pfrpn.device)]))

        elif det.numel() > 0:
            det_cls = det[:, 5] if det.shape[1] > 5 else torch.zeros(len(det), device=det.device)
            for i in range(len(det)):
                merged.append(torch.cat([det[i, :4], det[i, 4:5], det_cls[i:i+1]]))

        elif pfrpn.numel() > 0:
            for j in range(len(pfrpn)):
                merged.append(torch.cat([pfrpn[j, :4], pfrpn[j, 4:5],
                                         torch.full((1,), -1, device=pfrpn.device)]))

        if merged:
            results.append(torch.stack(merged))
        else:
            results.append(torch.zeros(0, 6, device=pfrpn.device))

    return results
