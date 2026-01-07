import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RefAwareFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Input channel x2 vì nối 2 frame lại
        self.fusion_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, 1, 3, 1, 1), # Ra 1 channel score
            nn.Sigmoid() # Score từ 0 đến 1
        )

    def forward(self, x, t):
        # x: [Batch*t, C, H, W]
        b_frames, c, h, w = x.size()
        n_frames = t
        b_size = b_frames // n_frames
        
        x_view = x.view(b_size, n_frames, c, h, w)
        
        # Lấy frame giữa làm chuẩn
        ref_idx = n_frames // 2
        ref = x_view[:, ref_idx:ref_idx+1, :, :, :] # Giữ dim [B, 1, C, H, W]
        ref = ref.repeat(1, n_frames, 1, 1, 1) # Lặp lại t lần [B, t, C, H, W]
        
        # Gộp Batch và Frame lại để đưa vào Conv2d
        # Input cho Conv2d sẽ là [Batch*t, 2C, H, W]
        concat_feat = torch.cat([x_view, ref], dim=2).view(b_frames, c*2, h, w)
        
        # Tính Score
        scores = self.fusion_net(concat_feat).view(b_size, n_frames, 1, h, w)
        
        # Cộng gộp có trọng số
        att_map = F.softmax(scores, dim=1) 
        
        return torch.sum(x_view * att_map, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        return x + self.pe[:, :x.size(1)]

class ChannelSpatialFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(channels, channels, 1)

    # SỬA: Thêm tham số t (số lượng frames) vào forward để không bị cứng t=5
    def forward(self, x, t):
        bt, c, h, w = x.size()
        b = bt // t

        # Reshape về dạng [Batch, Time, Channel, H, W] để tính attention giữa các frames
        x_view = x.view(b, t, c, h, w)

        # Spatial Attention
        scores = self.spatial_att(x).view(b, t, 1, h, w)
        att_map = F.softmax(scores, dim=1) # Softmax trên chiều thời gian (frames)

        # Fuse: Cộng gộp các frame lại dựa trên attention map
        fused_spatial = torch.sum(x_view * att_map, dim=1) # Kết quả: [B, C, H, W]

        # Channel Attention
        y = self.avg_pool(fused_spatial).view(b, c)
        y = self.channel_att(y).view(b, c, 1, 1) # [B, C, 1, 1]

        # Kết hợp Spatial và Channel
        return self.fusion_conv(fused_spatial * y)
class AttentionFusion(nn.Module):
    """
    Attention-based temporal fusion module.
    
    Learns attention weights for each frame and produces a weighted combination
    of features from multiple frames.
    """
    
    def __init__(self, channels):
        """
        Args:
            channels: Number of input feature channels
        """
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )
    
    def forward(self, x, t):
        """
        Args:
            x: Input tensor of shape [B*t, C, H, W]
            t: Number of frames per sample
        
        Returns:
            Fused tensor of shape [B, C, H, W]
        """
        bt, c, h, w = x.size()
        b = bt // t
        
        # Reshape for frame-wise processing
        x_view = x.view(b, t, c, h, w)
        
        # Compute attention scores
        scores = self.score_net(x).view(b, t, 1, h, w)
        
        # Apply softmax and weighted sum
        att_map = F.softmax(scores, dim=1)
        return torch.sum(x_view * att_map, dim=1)
