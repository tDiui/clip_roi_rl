# import torch
# import clip
# from torch import nn

# class CLIPHybridEncoder(nn.Module):
#     def __init__(self, model_name="ViT-L/14"):
#         super().__init__()
#         # Tải mô hình CLIP mạnh nhất (ViT-L/14)
#         self.model, self.preprocess = clip.load(model_name, device="cuda")
        
#         # Đóng băng các lớp dưới để giữ "tri thức" cũ (Học chuyển đổi) [cite: 42, 44]
#         for name, param in self.model.named_parameters():
#             if "visual.transformer.resblocks.23" in name or \
#             "visual.ln_post" in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False

#     def forward(self, videos, text_queries):
#         # videos shape: [Batch, Frames, C, H, W]
#         b, f, c, h, w = videos.shape
        
#         # 1. Xử lý Video (Visual Modality) [cite: 31]
#         # Gom Batch và Frames lại để đưa qua CLIP
#         videos = videos.view(-1, c, h, w) 
#         visual_features = self.model.encode_image(videos) # [B*F, 768]
#         visual_features = visual_features.view(b, f, -1) # [B, F, 768]

#         # 2. Xử lý Văn bản (Textual Modality) [cite: 35]
#         text_tokens = clip.tokenize(text_queries).to(videos.device)
#         text_features = self.model.encode_text(text_tokens) # [B, 768]

#         visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

import torch
import clip
from torch import nn

class CLIPHybridEncoder(nn.Module):
    def __init__(self, model_name="ViT-L/14"):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device="cuda")
        
        # Freeze + fine-tune layer cuối
        for name, param in self.model.named_parameters():
            if "visual.transformer.resblocks.23" in name or \
               "visual.ln_post" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Temporal modeling nhẹ
        self.temporal = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )

    def forward(self, videos, text_queries):
        b, f, c, h, w = videos.shape
        
        videos = videos.view(-1, c, h, w)
        visual_features = self.model.encode_image(videos)
        visual_features = visual_features.view(b, f, -1)

        # Temporal enhancement
        visual_features = self.temporal(visual_features)

        device = visual_features.device
        text_tokens = clip.tokenize(text_queries).to(device)
        text_features = self.model.encode_text(text_tokens)

        # Normalize (QUAN TRỌNG)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return visual_features, text_features