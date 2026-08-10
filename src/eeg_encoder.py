# src/eeg_encoder.py 파일 생성
import torch
import torch.nn as nn

class EEGMapper(nn.Module):
    def __init__(self, clip_dim=768):
        super().__init__()
        # 64채널, 600 타임스텝의 EEG 신호를 처리하는 1D-CNN 인코더
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.projection = nn.Linear(256, clip_dim)

    def forward(self, eeg_data):
        x = self.encoder(eeg_data)
        x = x.squeeze(-1)
        clip_embeds = self.projection(x)
        return clip_embeds