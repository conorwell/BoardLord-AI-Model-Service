"""
BoardCNN extracted from board_lord_ai_model/model.py.
Contains only the model architecture needed for inference.
"""
import torch
import torch.nn as nn
from app import utils as tb_util


class BoardCNN(nn.Module):
    """
    Small CNN that predicts climb difficulty from a 5-channel board grid + angle.

    Input:
        grid      — (batch, 5, 35, 33) float tensor
        angle_idx — (batch,) long, integer angle index 0–13 (degrees // 5)
        nomatch   — (batch,) float, 1.0 if the problem has a matchable hold, 0.0 if not

    Output:
        (batch, 11) logits — one per V-grade class (≤V2, V3–V10, V11+)
    """

    NUM_ANGLES = 14
    ANGLE_EMB_DIM = 16

    def __init__(self, dropout=0.4):
        super().__init__()
        self.angle_emb = nn.Embedding(self.NUM_ANGLES, self.ANGLE_EMB_DIM)

        self.conv = nn.Sequential(
            nn.Conv2d(tb_util.NUM_CHANNELS, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024 + self.ANGLE_EMB_DIM + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, tb_util.NUM_CLASSES),
        )

    def forward(self, grid, angle_idx, nomatch):
        x = self.conv(grid)
        x = x.flatten(1)
        angle_vec = self.angle_emb(angle_idx)
        x = torch.cat([x, angle_vec, nomatch.unsqueeze(1)], dim=1)
        return self.fc(x)
