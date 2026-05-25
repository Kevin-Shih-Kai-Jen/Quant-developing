"""
models/experts/tabular_tabnet.py — 輕量級 TabNet Expert（純 PyTorch 實作）
=========================================================================

Architecture Report 第二節「結構化基本面」專家。

TabNet 核心機制：
    1. Attentive Transformer — 每步動態選擇最重要的特徵子集
    2. Feature Transformer  — 對選中特徵做非線性映射
    3. Multi-Step 序列注意力 — 逐步精煉預測

優勢：
    - 白盒可解釋性：每步的注意力遮罩可以直接解讀為特徵重要性
    - 稀疏特徵選擇：不是所有特徵都參與每步計算（自然正則化）
    - CPU 友善：純全連接層，無需 GPU

Reference: Arik & Pfister (2021) "TabNet: Attentive Interpretable Tabular Learning"

Author : Nexus Quant OS — Model Engineering
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """GLU 門控線性單元 — TabNet Feature Transformer 的基礎積木。

    將輸入分成兩半：一半做線性變換，另一半做 Sigmoid 門控，
    逐元素相乘後輸出。比 ReLU 更適合表格型數據。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc    = nn.Linear(in_dim, out_dim * 2)
        self.norm  = nn.LayerNorm(out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.fc(x))
        h1, h2 = h.chunk(2, dim=-1)
        return h1 * torch.sigmoid(h2)


class AttentiveTransformer(nn.Module):
    """注意力遮罩生成器 — 動態選擇特徵子集。

    輸入前一步的 processed features（先驗資訊），
    輸出 sparse attention mask (softmax + sparsemax 近似)。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        processed: torch.Tensor,   # [B, in_dim] — 前一步輸出
        prior_scales: torch.Tensor, # [B, out_dim] — 累積注意力（防止重複選）
    ) -> torch.Tensor:
        h    = self.norm(self.fc(processed))   # [B, out_dim]
        # Mathematical scaling in softmax space: add log(prob)
        h    = h + torch.log(prior_scales + 1e-8)
        mask = F.softmax(h, dim=-1)             # [B, out_dim] — 稀疏注意力
        return mask


class TabNetExpert(nn.Module):
    """TabNet Sequential Attention Expert — 純 PyTorch 實作。

    Architecture
    ------------
    N_steps 步序列注意力：
        Step 1: AttentiveTransformer → 選擇特徵子集 → FeatureTransformer → 殘差累加
        Step 2: ...
        Step N: ...
        Output: 累積所有步的殘差 → 線性映射 → 資產權重

    Parameters
    ----------
    input_dim   : int   輸入特徵維度（e.g. 4 for macro-only, 8 for all）
    output_dim  : int   輸出維度（= N_assets = 7）
    n_steps     : int   序列注意力步數（越多 = 越精細，但也越慢）
    hidden_dim  : int   Feature Transformer 隱藏維度
    gamma       : float 注意力稀疏係數（越大 = 越少重複選同特徵）
    """

    def __init__(
        self,
        input_dim:  int   = 8,
        output_dim: int   = 7,
        n_steps:    int   = 3,
        hidden_dim: int   = 64,
        gamma:      float = 1.5,
    ) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.n_steps    = n_steps
        self.gamma      = gamma

        # 初始 BatchNorm — 對輸入特徵做標準化
        self.initial_norm = nn.LayerNorm(input_dim)

        # 每步的 Attentive + Feature Transformer
        self.attentive_layers = nn.ModuleList([
            AttentiveTransformer(hidden_dim, input_dim)
            for _ in range(n_steps)
        ])
        self.feature_layers = nn.ModuleList([
            GatedLinearUnit(input_dim, hidden_dim)
            for _ in range(n_steps)
        ])

        # 第一步需要特殊的初始處理（因為沒有前一步的輸出）
        self.initial_transform = nn.Linear(input_dim, hidden_dim)

        # 最終輸出映射
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向推論。

        Parameters
        ----------
        x : torch.Tensor  [B, input_dim]

        Returns
        -------
        torch.Tensor  [B, output_dim]
        """
        B = x.shape[0]
        x = self.initial_norm(x)

        # 初始先驗尺度（全 1，表示所有特徵都可選）
        prior_scales = torch.ones(B, self.input_dim, device=x.device)

        # 初始處理（用於第一步 AttentiveTransformer 的輸入）
        processed = self.initial_transform(x)  # [B, hidden_dim]

        # 累積殘差輸出
        aggregated = torch.zeros(B, processed.shape[-1], device=x.device)

        for step in range(self.n_steps):
            # 1. Attentive Transformer — 決定看哪些特徵
            mask = self.attentive_layers[step](processed, prior_scales)  # [B, input_dim]

            # 2. 更新先驗尺度（已選過的特徵降低權重）
            prior_scales = prior_scales * (self.gamma - mask)

            # 3. Feature Transformer — 對選中特徵做非線性映射
            masked_x  = mask * x                                        # [B, input_dim]
            processed = self.feature_layers[step](masked_x)             # [B, hidden_dim]

            # 4. 殘差累加
            aggregated = aggregated + processed

        # 最終映射
        out = self.output_layer(F.relu(aggregated))  # [B, output_dim]
        return out
