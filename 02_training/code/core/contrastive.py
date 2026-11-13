# -*- coding: utf-8 -*-
"""
contrastive — VLM 语义对比学习模块

提供:
  - SimpleTextEncoder: 轻量级文本编码器 (Embedding + GRU)
  - TextProjectionHead: 文本→视觉空间投影
  - ContrastiveLoss: InfoNCE 对比损失

集成到 scene 分支: 视觉特征 (cls_feat 1024-dim) 与文本特征对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleTextEncoder(nn.Module):
    """轻量级文本编码器: Embedding + Bidirectional GRU

    输入: token ids [B, seq_len]
    输出: text embedding [B, embed_dim]

    不依赖外部预训练模型, 从零训练.
    词表基于 VLM 语义标注中的高频词构建.
    """

    def __init__(self, vocab_size=4096, embed_dim=256, hidden_dim=256,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, attention_mask=None):
        """
        Args:
            token_ids: [B, seq_len] token ids
            attention_mask: [B, seq_len] 1=valid, 0=padding
        Returns:
            text_feat: [B, hidden_dim * 2]
        """
        emb = self.dropout(self.embedding(token_ids))  # [B, seq_len, embed_dim]

        if attention_mask is not None:
            # Pack padded sequences for efficiency
            lengths = attention_mask.sum(dim=1).cpu().long()
            # Ensure minimum length of 1
            lengths = lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths, batch_first=True, enforce_sorted=False
            )
            output, hidden = self.gru(packed)
            hidden = self.layer_norm(hidden)  # [num_layers*2, B, hidden_dim]
        else:
            output, hidden = self.gru(emb)
            hidden = self.layer_norm(hidden)

        # 取最后一层的前向+后向隐藏状态
        text_feat = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, hidden_dim*2]
        return text_feat


class TextProjectionHead(nn.Module):
    """文本特征投影到视觉空间

    文本编码器输出 (512-dim) → 投影到 scene head 的 1024-dim 空间
    """

    def __init__(self, text_dim=512, visual_dim=1024, proj_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(visual_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, text_feat):
        """
        Args:
            text_feat: [B, text_dim]
        Returns:
            proj_feat: [B, proj_dim] L2-normalized
        """
        return F.normalize(self.proj(text_feat), dim=-1)


class VisualProjectionHead(nn.Module):
    """视觉特征投影到对比学习空间

    scene head 的 cls_feat (1024-dim) → 投影到同一对比学习空间
    """

    def __init__(self, visual_dim=1024, proj_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(visual_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, cls_feat):
        """
        Args:
            cls_feat: [B, visual_dim]
        Returns:
            proj_feat: [B, proj_dim] L2-normalized
        """
        return F.normalize(self.proj(cls_feat), dim=-1)


class ContrastiveLoss(nn.Module):
    """InfoNCE 对比损失 (Symmetric Cross-Entropy)

    正样本: 配对的 (图像, 文本) 对
    负样本: batch 内其他样本

    L = -log(exp(sim(v_i, t_i)/τ) / Σ_j exp(sim(v_i, t_j)/τ))
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: [B, proj_dim] L2-normalized visual features
            text_feat: [B, proj_dim] L2-normalized text features
        Returns:
            loss: scalar contrastive loss
        """
        # Clamp logit_scale for stability
        logit_scale = self.logit_scale.exp().clamp(max=100)

        # Similarity matrix [B, B]
        sim_v2t = visual_feat @ text_feat.t() * logit_scale  # image→text
        sim_t2v = text_feat @ visual_feat.t() * logit_scale  # text→image

        # Labels: diagonal is positive
        B = visual_feat.shape[0]
        labels = torch.arange(B, device=visual_feat.device)

        # Symmetric loss
        loss_v2t = F.cross_entropy(sim_v2t, labels)
        loss_t2v = F.cross_entropy(sim_t2v, labels)

        return (loss_v2t + loss_t2v) / 2


class SemanticTokenizer:
    """简单的语义文本分词器

    将 VLM 生成的语义描述文本转换为 token ids.
    基于字级别的中文+英文混合词表.
    """

    def __init__(self, vocab_size=4096, max_len=128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2id = {'<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3}
        self.next_id = 4

    def build_vocab(self, texts):
        """从文本列表构建词表"""
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        # 取最高频的词
        for word, _ in word_counts.most_common(self.vocab_size - self.next_id):
            if word not in self.word2id:
                self.word2id[word] = self.next_id
                self.next_id += 1

    def encode(self, text, max_len=None):
        """文本 → token ids"""
        max_len = max_len or self.max_len
        words = self._tokenize(text)
        ids = [self.word2id.get(w, self.word2id['<unk>']) for w in words]
        ids = ids[:max_len - 1]
        ids.append(self.word2id['<sep>'])

        # Padding
        attention_mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(0)
            attention_mask.append(0)

        return ids, attention_mask

    def _tokenize(self, text):
        """中英文混合分词: 按字符级别切分中文, 按空格切分英文"""
        tokens = []
        current_word = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                # 中文字符: 单独成词
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                tokens.append(char)
            elif char.isalnum() or char in '-_':
                current_word.append(char)
            else:
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
        if current_word:
            tokens.append(''.join(current_word))
        return tokens

    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump({'word2id': self.word2id, 'vocab_size': self.vocab_size,
                       'max_len': self.max_len}, f)

    def load(self, path):
        import json
        with open(path) as f:
            data = json.load(f)
        self.word2id = data['word2id']
        self.vocab_size = data['vocab_size']
        self.max_len = data['max_len']
        self.next_id = max(self.word2id.values()) + 1
