# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CrossModalAlignment(nn.Module):
#     def __init__(self, feature_dim=512, num_heads=8):
#         super(CrossModalAlignment, self).__init__()
        
#         # 1. Cơ chế Chú ý Ngôn ngữ - Video (Học từ tài liệu TC-MGC)
#         # Giúp AI tập trung vào đúng đối tượng "xe máy", "màu áo" được nhắc đến
#         self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        
#         # Lớp chuẩn hóa để ổn định việc huấn luyện
#         self.norm1 = nn.LayerNorm(feature_dim)
#         self.norm2 = nn.LayerNorm(feature_dim)
        
#         # 2. Span-aware Temporal Aggregation (Học từ tài liệu STAN)
#         # Giúp gom nhóm các khung hình lại thành một hành động thống nhất
#         self.temporal_conv = nn.Sequential(
#             nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2) # Giảm chiều thời gian để lấy đặc trưng cô đọng
#         )

#     def forward(self, v_feat, t_feat):
#         """
#         v_feat: Đặc trưng video [Batch, Frames, 512]
#         t_feat: Đặc trưng văn bản [Batch, 512]
#         """
#         # Bước 1: Căn chỉnh văn bản vào từng khung hình (TC-MGC)
#         # Chúng ta coi văn bản như "Query" và video như "Key/Value"
#         # Mở rộng t_feat để khớp với số lượng Frames
#         t_feat_expanded = t_feat.unsqueeze(1) # [Batch, 1, 512]
        
#         # Attention: AI nhìn vào video và lọc ra những gì giống với mô tả trong Text
#         attn_output, _ = self.attention(t_feat_expanded, v_feat, v_feat)
        
#         # Cộng và Chuẩn hóa (Residual Connection)
#         x = self.norm1(attn_output + t_feat_expanded) # [Batch, 1, 512]
        
#         # Bước 2: Gom nhóm thông tin thời gian (STAN)
#         # Chuyển về dạng [Batch, 512, Frames] để dùng Conv1d
#         v_feat_transposed = v_feat.transpose(1, 2)
#         temporal_info = self.temporal_conv(v_feat_transposed)
#         temporal_info = temporal_info.transpose(1, 2) # [Batch, Frames/2, 512]
        
#         # Bước 3: Kết hợp đặc trưng đã lọc và đặc trưng thời gian
#         # Ở đây chúng ta tính độ tương đồng (Similarity)
#         # Đây chính là giá trị để AI quyết định đoạn nào khớp với nhãn nhangoc.jsonl nhất
#         similarity = torch.matmul(x, v_feat.transpose(1, 2)) # [Batch, 1, Frames]
        
#         return similarity.squeeze(1), temporal_info

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlignment(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8):
        super(CrossModalAlignment, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, v_feat, t_feat):
        """
        v_feat: [B, F, 512]
        t_feat: [B, 512]
        """

        # ========================
        # 1. CROSS ATTENTION FIX
        # ========================
        t_feat_expanded = t_feat.unsqueeze(1).expand(-1, v_feat.size(1), -1)

        attn_output, _ = self.attention(
            v_feat,                # Query (frame-level)
            t_feat_expanded,       # Key
            t_feat_expanded        # Value
        )

        x = self.norm1(attn_output + v_feat)  # residual

        # ========================
        # 2. TEMPORAL
        # ========================
        v_feat_t = v_feat.transpose(1, 2)
        temporal_info = self.temporal_conv(v_feat_t)
        temporal_info = temporal_info.transpose(1, 2)

        # pad lại nếu bị giảm frame
        if temporal_info.size(1) < v_feat.size(1):
            temporal_info = F.pad(
                temporal_info,
                (0, 0, 0, v_feat.size(1) - temporal_info.size(1))
            )

        v_feat_enhanced = v_feat + temporal_info

        # ========================
        # 3. SIMILARITY MAP
        # ========================
        similarity = torch.sum(x * v_feat_enhanced, dim=-1)

        return similarity, temporal_info