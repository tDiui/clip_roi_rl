# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResidualBlock(nn.Module):
#     """Khối Residual giúp tín hiệu truyền sâu hơn mà không bị mất mát."""
#     def __init__(self, dim):
#         super().__init__()
#         self.ln = nn.LayerNorm(dim)
#         self.net = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.GELU(), # GELU cho hiệu năng tốt hơn ReLU trong các tác vụ CLIP
#             nn.Linear(dim * 2, dim),
#             nn.Dropout(0.1)
#         )

#     def forward(self, x):
#         return x + self.net(self.ln(x))

# class ActorCriticAgent(nn.Module):
#     def __init__(self, input_dim=2236, hidden_dim=768):
#         super(ActorCriticAgent, self).__init__()
        
#         # 1. Feature Fusion Layer (Gated Linear Unit)
#         # Giúp model tự học cách "nhấn mạnh" phần đặc trưng quan trọng
#         self.fusion = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim)
#         )
        
#         # 2. Backbone mạnh mẽ hơn với Residual Blocks
#         # Tăng khả năng tư duy logic cho Agent
#         self.backbone = nn.Sequential(
#             ResidualBlock(hidden_dim),
#             ResidualBlock(hidden_dim)
#         )
        
#         # 3. Actor Head (Chính sách)
#         self.actor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 2, 5)
#         )
        
#         # 4. Critic Head (Đánh giá)
#         self.critic = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 2, 1)
#         )

#     def forward(self, state):
#         # Bước 1: Hòa trộn đặc trưng Ảnh + Text
#         x = self.fusion(state)
        
#         # Bước 2: Đưa qua các lớp tư duy sâu
#         x = self.backbone(x)
        
#         # Bước 3: Đưa ra quyết định và đánh giá
#         action_probs = F.softmax(self.actor(x), dim=-1)
#         state_value = self.critic(x)
        
#         return action_probs, state_value

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return x + self.net(self.ln(x))


class ActorCriticAgent(nn.Module):
    def __init__(self, input_dim=2306, hidden_dim=768):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.backbone = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 5)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    # ✅ PHẢI nằm trong class
    def forward(self, state):
        x = self.fusion(state)
        x = self.backbone(x)

        logits = self.actor(x)                  # dùng logits cho Categorical
        state_value = self.critic(x).squeeze(-1)

        return logits, state_value