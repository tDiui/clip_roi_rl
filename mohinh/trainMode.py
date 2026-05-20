import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import NhangocDataset
from models.agent import ActorCriticAgent
import torch.nn.functional as F
from config import settings
import clip
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt # --- [MỚI] ---

# ========================================================
# 🚩 CẤU HÌNH CHẾ ĐỘ CHẠY (Thái đổi ở đây)
# TRAIN_MODE = "B1" # CLIP Only (Dùng Full Frame)
# TRAIN_MODE = "B2" # CLIP + ROI (Dùng ROI Frame)
TRAIN_MODE = "B1"   # Proposed: CLIP + ROI + RL (Agent)
# ========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = settings.BATCH_SIZE
EPOCHS = settings.EPOCHS
LEARNING_RATE = settings.LEARNING_RATE 
GAMMA = settings.GAMMA

# --- Mạng Regressor cho Baseline 1 & 2 (Đoán tọa độ tĩnh) ---
class SupervisedBaseline(nn.Module):
    def __init__(self, input_dim=2306):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2), 
            nn.Sigmoid() # Đưa Start/End về [0, 1]
        )
    def forward(self, x):
        return self.network(x)

def calculate_iou(pred_segment, gt_segment):
    start_p, end_p = pred_segment[0].item(), pred_segment[1].item()
    start_g, end_g = gt_segment[0].item(), gt_segment[1].item()
    intersection = max(0, min(end_p, end_g) - max(start_p, start_g))
    union = max(1e-6, (end_p - start_p) + (end_g - start_g) - intersection)
    return intersection / union

def clean_query(q):
    return " ".join(q.lower().strip().split())

def custom_collate(batch):
    return {
        "video": torch.stack([item['video'] for item in batch]),
        "query": [item['query'] for item in batch],
        "segment": torch.stack([item['segment'] for item in batch]),
        "duration": torch.tensor([item.get('duration', 10.0) for item in batch])
    }

def train():
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(f'logs/VMR_{TRAIN_MODE}_Final')

    print(f"--- 🚀 KHỞI TẠO HỆ THỐNG: {TRAIN_MODE} ---")
    print(f"--- 💻 GPU: {torch.cuda.get_device_name(0)} | VRAM: 12GB ---")
    
    clip_model, _ = clip.load("ViT-L/14", device=DEVICE)
    clip_model.eval()
    scaler = torch.amp.GradScaler('cuda')

    # Khởi tạo mô hình
    if TRAIN_MODE in ["B1", "B2"]:
        model = SupervisedBaseline(input_dim=settings.AGENT_INPUT_DIM).to(DEVICE)
        criterion = nn.MSELoss()
    else:
        model = ActorCriticAgent(input_dim=settings.AGENT_INPUT_DIM).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Dataset & Dataloader
    train_dataset = NhangocDataset(label_file="data/train_annotations.jsonl", use_cache=True)
    val_dataset = NhangocDataset(label_file="data/val_annotations.jsonl", use_cache=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=custom_collate, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=custom_collate)

    best_iou = 0.0
    
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache() # Dọn rác VRAM mỗi Epoch
        model.train()
        total_train_loss, total_train_iou = 0, 0
        
        for batch_idx, batch in enumerate(train_loader):
            videos = batch['video'].to(DEVICE)
            raw_queries = batch['query']
            gt_segments = batch['segment'].to(DEVICE)
            durations = batch['duration'].to(DEVICE)
            cleaned_queries = [clean_query(q) for q in raw_queries]

            with torch.amp.autocast('cuda'):
                b = videos.shape[0]
                with torch.no_grad():
                    video_input = videos.view(-1, 3, 224, 224)
                    # Trích xuất đặc trưng CLIP (Chiếm nhiều VRAM nhất)
                    v_feat = clip_model.encode_image(video_input).view(b, settings.NUM_FRAMES, -1).float()
                    text_tokens = clip.tokenize(cleaned_queries, truncate=True).to(DEVICE)
                    t_feat = clip_model.encode_text(text_tokens).float()
                    v_feat = F.normalize(v_feat, p=2, dim=-1)
                    t_feat = F.normalize(t_feat, p=2, dim=-1)

                v_feat_global = v_feat.mean(dim=1)
                gt_norm = gt_segments / durations.unsqueeze(1)

                # --- MODE B1 & B2: SUPERVISED REGRESSION ---
                if TRAIN_MODE in ["B1", "B2"]:
                    state = torch.cat([v_feat_global, v_feat_global, t_feat, torch.zeros(b, 2).to(DEVICE)], dim=-1)
                    pred_boundaries = model(state)
                    loss = criterion(pred_boundaries, gt_norm)
                    
                    batch_iou_sum = 0
                    for i in range(b):
                        batch_iou_sum += calculate_iou(pred_boundaries[i], gt_norm[i])
                    avg_batch_iou = batch_iou_sum / b

                # --- MODE B3: RL AGENT ---
                else:
                    batch_rl_loss = 0
                    batch_iou_sum = 0
                    for i in range(b):
                        similarities = v_feat[i] @ t_feat[i]
                        best_f = torch.argmax(similarities).item()
                        curr_seg = torch.tensor([max(0, best_f/64 - 0.1), min(1.0, best_f/64 + 0.1)]).to(DEVICE)
                        p_iou = calculate_iou(curr_seg, gt_norm[i])

                        for step in range(settings.MAX_STEPS):
                            s_idx = max(0, min(63, int(curr_seg[0]*64)))
                            e_idx = max(s_idx+1, min(64, int(curr_seg[1]*64)))
                            local_v = v_feat[i, s_idx:e_idx].mean(0)
                            state = torch.cat([local_v, v_feat_global[i], t_feat[i], curr_seg], dim=-1)
                            
                            logits, val = model(state)
                            m = torch.distributions.Categorical(logits=F.softmax(logits, dim=-1))
                            act = m.sample()

                            step_sz = 0.04
                            if act == 0: curr_seg[0] = max(0, curr_seg[0] - step_sz)
                            elif act == 1: curr_seg[0] = min(curr_seg[1] - 0.01, curr_seg[0] + step_sz)
                            elif act == 2: curr_seg[1] = max(curr_seg[0] + 0.01, curr_seg[1] - step_sz)
                            elif act == 3: curr_seg[1] = min(1.0, curr_seg[1] + step_sz)

                            c_iou = calculate_iou(curr_seg, gt_norm[i])
                            reward = (c_iou - p_iou) * settings.REWARD_SCALE
                            
                            advantage = (reward + GAMMA * val.detach()) - val
                            batch_rl_loss += -m.log_prob(act) * advantage.detach() + F.mse_loss(val, torch.tensor([reward + GAMMA * val.item()]).to(DEVICE))
                            
                            p_iou = c_iou
                            if act == 4: break
                        batch_iou_sum += p_iou
                    loss = batch_rl_loss / b
                    avg_batch_iou = batch_iou_sum / b

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            total_train_iou += avg_batch_iou
            
            if batch_idx % 20 == 0:
                print(f"[{TRAIN_MODE}] E{epoch+1} B{batch_idx} | Loss: {loss.item():.2f} | IoU: {avg_batch_iou:.4f}")

        # === VALIDATION EPOCH (SỬA LỖI TRẮNG IOU) ===
        model.eval()
        val_iou_sum = 0
        with torch.no_grad():
            for val_batch in val_loader:
                v_v = val_batch['video'].to(DEVICE)
                q_v = [clean_query(q) for q in val_batch['query']]
                gt_v = val_batch['segment'].to(DEVICE)
                d_v = val_batch['duration'].to(DEVICE)
                b_v = v_v.shape[0]

                v_f = clip_model.encode_image(v_v.view(-1, 3, 224, 224)).view(b_v, settings.NUM_FRAMES, -1).float()
                t_f = clip_model.encode_text(clip.tokenize(q_v, truncate=True).to(DEVICE)).float()
                v_f_global = v_f.mean(dim=1)
                gt_n = gt_v / d_v.unsqueeze(1)

                if TRAIN_MODE in ["B1", "B2"]:
                    state = torch.cat([v_f_global, v_f_global, t_f, torch.zeros(b_v, 2).to(DEVICE)], dim=-1)
                    pred = model(state)
                    for i in range(b_v): val_iou_sum += calculate_iou(pred[i], gt_n[i])
                else:
                    for i in range(b_v):
                        similarities = v_f[i] @ t_f[i]
                        best_f = torch.argmax(similarities).item()
                        curr_seg = torch.tensor([max(0, best_f/64 - 0.1), min(1.0, best_f/64 + 0.1)]).to(DEVICE)
                        for _ in range(settings.MAX_STEPS):
                            s_idx = max(0, min(63, int(curr_seg[0]*64)))
                            e_idx = max(s_idx+1, min(64, int(curr_seg[1]*64)))
                            local_v = v_f[i, s_idx:e_idx].mean(0)
                            st = torch.cat([local_v, v_f_global[i], t_f[i], curr_seg], dim=-1)
                            logits, _ = model(st)
                            act = torch.argmax(logits)
                            if act == 0: curr_seg[0] = max(0, curr_seg[0] - 0.04)
                            elif act == 1: curr_seg[0] = min(curr_seg[1] - 0.01, curr_seg[0] + 0.04)
                            elif act == 2: curr_seg[1] = max(curr_seg[0] + 0.01, curr_seg[1] - 0.04)
                            elif act == 3: curr_seg[1] = min(1.0, curr_seg[1] + 0.04)
                            if act == 4: break
                        val_iou_sum += calculate_iou(curr_seg, gt_n[i])

        avg_val_iou = val_iou_sum / len(val_dataset)
        print(f"✅ Kết thúc Epoch {epoch+1} | Val IoU: {avg_val_iou:.4f}")
        
        writer.add_scalar('Epoch/Val_IoU', avg_val_iou, epoch+1)
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), f"checkpoints/{TRAIN_MODE}_best.pth")
            print(f"⭐ Lưu Checkpoint mới: {TRAIN_MODE}_best.pth")
        
        scheduler.step(avg_val_iou)
    writer.close()

if __name__ == "__main__":
    train()