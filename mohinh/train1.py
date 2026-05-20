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

# ==========================================
# 🚩 CẤU HÌNH CHẾ ĐỘ CHẠY
# TRAIN_MODE = "B1" # CLIP Only (Supervised)
# TRAIN_MODE = "B2" # CLIP + ROI (Supervised)
TRAIN_MODE = "B3"   # Proposed: CLIP + ROI + RL
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = settings.BATCH_SIZE
EPOCHS = settings.EPOCHS
LEARNING_RATE = settings.LEARNING_RATE 
GAMMA = settings.GAMMA

# --- Mạng Regressor đơn giản cho Baseline 1 & 2 ---
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
            nn.Sigmoid() 
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
    writer = SummaryWriter(f'logs/VMR_{TRAIN_MODE}_Mission')

    print(f"--- 🚀 Đang khởi tạo Mode: {TRAIN_MODE} (GPU: {torch.cuda.get_device_name(0)}) ---")
    clip_model, _ = clip.load("ViT-L/14", device=DEVICE)
    clip_model.eval()
    scaler = torch.amp.GradScaler('cuda')

    # Khởi tạo mô hình tương ứng
    if TRAIN_MODE in ["B1", "B2"]:
        model = SupervisedBaseline(input_dim=settings.AGENT_INPUT_DIM).to(DEVICE)
        criterion = nn.MSELoss()
    else:
        model = ActorCriticAgent(input_dim=settings.AGENT_INPUT_DIM).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Nạp dataset
    train_dataset = NhangocDataset(label_file="data/train_annotations.jsonl", use_cache=True)
    val_dataset = NhangocDataset(label_file="data/val_annotations.jsonl", use_cache=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, 
                              num_workers=settings.NUM_WORKERS, pin_memory=settings.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

    best_iou = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss, total_train_iou = 0, 0
        
        for batch_idx, batch in enumerate(train_loader):
            videos = batch['video'].to(DEVICE, non_blocking=True)
            raw_queries = batch['query']
            gt_segments = batch['segment'].to(DEVICE, non_blocking=True)
            durations = batch['duration'].to(DEVICE, non_blocking=True)
            cleaned_queries = [clean_query(q) for q in raw_queries]

            with torch.amp.autocast('cuda'):
                b = videos.shape[0]
                with torch.no_grad():
                    video_input = videos.view(-1, 3, 224, 224)
                    v_feat = clip_model.encode_image(video_input).view(b, settings.NUM_FRAMES, -1).float()
                    text_tokens = clip.tokenize(cleaned_queries, truncate=True).to(DEVICE)
                    t_feat = clip_model.encode_text(text_tokens).float()
                    v_feat = F.normalize(v_feat, p=2, dim=-1)
                    t_feat = F.normalize(t_feat, p=2, dim=-1)

                v_feat_global = v_feat.mean(dim=1)
                gt_norm = gt_segments / durations.unsqueeze(1)

                # --- NHÁNH TRAIN B1 & B2 (SUPERVISED) ---
                if TRAIN_MODE in ["B1", "B2"]:
                    # State giả lập 2306 chiều để đồng bộ kiến trúc
                    state = torch.cat([v_feat_global, v_feat_global, t_feat, torch.zeros(b, 2).to(DEVICE)], dim=-1)
                    pred_boundaries = model(state)
                    loss = criterion(pred_boundaries, gt_norm)
                    
                    batch_iou_sum = 0
                    for i in range(b):
                        batch_iou_sum += calculate_iou(pred_boundaries[i], gt_norm[i])
                    avg_batch_iou = batch_iou_sum / b

                # --- NHÁNH TRAIN B3 (RL - ACTOR CRITIC) ---
                else:
                    batch_rl_loss = 0
                    batch_iou_sum = 0
                    for i in range(b):
                        # Khởi tạo vị trí dựa trên similarity cao nhất
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

                            # Cập nhật biên
                            step_sz = 0.04
                            if act == 0: curr_seg[0] = max(0, curr_seg[0] - step_sz)
                            elif act == 1: curr_seg[0] = min(curr_seg[1] - 0.01, curr_seg[0] + step_sz)
                            elif act == 2: curr_seg[1] = max(curr_seg[0] + 0.01, curr_seg[1] - step_sz)
                            elif act == 3: curr_seg[1] = min(1.0, curr_seg[1] + step_sz)

                            c_iou = calculate_iou(curr_seg, gt_norm[i])
                            reward = (c_iou - p_iou) * settings.REWARD_SCALE
                            if act == 4: # Action STOP
                                if c_iou >= 0.7: reward += settings.REWARD_BONUS_HIGH
                                elif c_iou >= 0.5: reward += 1.0
                            
                            advantage = (reward + GAMMA * val.detach()) - val
                            batch_rl_loss += -m.log_prob(act) * advantage.detach() + F.mse_loss(val, torch.tensor([reward + GAMMA * val.item()]).to(DEVICE))
                            
                            p_iou = c_iou
                            if act == 4: break
                        batch_iou_sum += p_iou
                    loss = batch_rl_loss / b
                    avg_batch_iou = batch_iou_sum / b

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            total_train_iou += avg_batch_iou
            
            if batch_idx % 20 == 0:
                print(f"[{TRAIN_MODE}] E{epoch+1} B{batch_idx} | Loss: {loss.item():.3f} | IoU: {avg_batch_iou:.4f}")

        # === VALIDATION EPOCH ===
        model.eval()
        val_iou_sum = 0
        with torch.no_grad():
            for val_batch in val_loader:
                v_v = val_batch['video'].to(DEVICE)
                q_v = [clean_query(q) for q in val_batch['query']]
                gt_v = val_batch['segment'].to(DEVICE)
                d_v = val_batch['duration'].to(DEVICE)
                
                # Logic Valid tương tự Train nhưng chọn argmax action
                # (Lược bớt để code ngắn gọn, Thái giữ nguyên logic Valid của B3 cũ)
                pass 

        avg_val_iou = val_iou_sum / len(val_dataset)
        print(f"✅ Epoch {epoch+1} Done | Val IoU: {avg_val_iou:.4f}")
        
        # Ghi log và Lưu Checkpoint
        writer.add_scalar('IoU/Val', avg_val_iou, epoch+1)
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), f"checkpoints/{TRAIN_MODE}_best.pth")
        
        torch.save(model.state_dict(), f"checkpoints/{TRAIN_MODE}_last.pth")
        scheduler.step(avg_val_iou)

    writer.close()

if __name__ == "__main__":
    train()