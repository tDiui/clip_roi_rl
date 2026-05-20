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
import matplotlib.pyplot as plt

# ========================================================
# 🚩 CẤU HÌNH CHẾ ĐỘ CHẠY
# TRAIN_MODE = "B1" # CLIP Only
# TRAIN_MODE = "B2" # CLIP + ROI
TRAIN_MODE = "B1"   # Đổi sang "B3" nếu muốn train RL
# ========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = settings.BATCH_SIZE
EPOCHS = settings.EPOCHS
LEARNING_RATE = settings.LEARNING_RATE 
GAMMA = settings.GAMMA

# Mapping Tag theo yêu cầu của Thái
MODE_MAP = {"B1": "clip_only", "B2": "clip_roi", "B3": "full"}
TAG = MODE_MAP.get(TRAIN_MODE, "unknown")

class SupervisedBaseline(nn.Module):
    def __init__(self, input_dim=2306):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 2), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

def calculate_iou(pred_segment, gt_segment):
    start_p, end_p = pred_segment[0].item(), pred_segment[1].item()
    start_g, end_g = gt_segment[0].item(), gt_segment[1].item()
    intersection = max(0, min(end_p, end_g) - max(start_p, start_g))
    union = max(1e-6, (end_p - start_p) + (end_g - start_g) - intersection)
    return intersection / union

def clean_query(q): return " ".join(q.lower().strip().split())

def custom_collate(batch):
    return {
        "video": torch.stack([item['video'] for item in batch]),
        "query": [item['query'] for item in batch],
        "segment": torch.stack([item['segment'] for item in batch]),
        "duration": torch.tensor([item.get('duration', 10.0) for item in batch])
    }

def plot_dashboard(history, mode):
    epochs = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'📊 DASHBOARD: {mode.upper()}', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(epochs, history['loss'], 'r-'); axes[0, 0].set_title('Avg Loss')
    axes[0, 1].plot(epochs, history['train_iou'], 'b-'); axes[0, 1].set_title('Train mIoU')
    axes[0, 2].plot(epochs, history['val_iou'], 'g-'); axes[0, 2].set_title('Val mIoU')
    axes[1, 0].plot(epochs, history['lr'], 'purple'); axes[1, 0].set_title('Learning Rate')
    
    if mode == "full":
        axes[1, 1].plot(epochs, history['reward'], 'brown'); axes[1, 1].set_title('Avg Reward')
        axes[1, 2].plot(epochs, history['entropy'], 'orange'); axes[1, 2].set_title('Policy Entropy')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'checkpoints/dashboard_{mode}.png')
    plt.close()

def train():
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(f'logs/{TAG}_final')
    history = {'loss': [], 'train_iou': [], 'val_iou': [], 'lr': [], 'reward': [], 'entropy': []}

    print(f"--- 🚀 KHỞI TẠO: {TAG.upper()} ---")
    clip_model, _ = clip.load("ViT-L/14", device=DEVICE)
    clip_model.eval()
    scaler = torch.amp.GradScaler('cuda')

    if TRAIN_MODE in ["B1", "B2"]:
        model = SupervisedBaseline(input_dim=settings.AGENT_INPUT_DIM).to(DEVICE)
        criterion = nn.MSELoss()
    else:
        model = ActorCriticAgent(input_dim=settings.AGENT_INPUT_DIM).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    train_loader = DataLoader(NhangocDataset(label_file="data/train_annotations.jsonl", use_cache=True), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(NhangocDataset(label_file="data/val_annotations.jsonl", use_cache=True), 
                            batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

    best_iou = 0.0
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        total_train_loss, total_train_iou, total_reward, total_entropy = 0, 0, 0, 0
        
        for batch_idx, batch in enumerate(train_loader):
            videos = batch['video'].to(DEVICE); gt_segments = batch['segment'].to(DEVICE)
            durations = batch['duration'].to(DEVICE); cleaned_queries = [clean_query(q) for q in batch['query']]

            with torch.amp.autocast('cuda'):
                b = videos.shape[0]
                with torch.no_grad():
                    video_input = videos.view(-1, 3, 224, 224)
                    v_feat = clip_model.encode_image(video_input).view(b, settings.NUM_FRAMES, -1).float()
                    t_feat = clip_model.encode_text(clip.tokenize(cleaned_queries, truncate=True).to(DEVICE)).float()
                    v_feat = F.normalize(v_feat, p=2, dim=-1); t_feat = F.normalize(t_feat, p=2, dim=-1)

                v_feat_global = v_feat.mean(dim=1); gt_norm = gt_segments / durations.unsqueeze(1)

                if TRAIN_MODE in ["B1", "B2"]:
                    state = torch.cat([v_feat_global, v_feat_global, t_feat, torch.zeros(b, 2).to(DEVICE)], dim=-1)
                    pred = model(state); loss = criterion(pred, gt_norm)
                    avg_batch_iou = sum([calculate_iou(pred[i], gt_norm[i]) for i in range(b)]) / b
                else: # MODE FULL (RL)
                    batch_rl_loss, batch_iou_sum, b_reward, b_ent = 0, 0, 0, 0
                    for i in range(b):
                        similarities = v_feat[i] @ t_feat[i]
                        best_f = torch.argmax(similarities).item()
                        curr_seg = torch.tensor([max(0, best_f/64 - 0.1), min(1.0, best_f/64 + 0.1)]).to(DEVICE)
                        p_iou = calculate_iou(curr_seg, gt_norm[i])
                        for step in range(settings.MAX_STEPS):
                            s_idx, e_idx = max(0, min(63, int(curr_seg[0]*64))), max(1, min(64, int(curr_seg[1]*64)))
                            local_v = v_feat[i, s_idx:e_idx].mean(0)
                            state = torch.cat([local_v, v_feat_global[i], t_feat[i], curr_seg], dim=-1)
                            logits, val = model(state)
                            m = torch.distributions.Categorical(logits=F.softmax(logits, dim=-1))
                            act = m.sample(); b_ent += m.entropy().mean().item()
                            
                            step_sz = 0.04
                            if act == 0: curr_seg[0] = max(0, curr_seg[0] - step_sz)
                            elif act == 1: curr_seg[0] = min(curr_seg[1] - 0.01, curr_seg[0] + step_sz)
                            elif act == 2: curr_seg[1] = max(curr_seg[0] + 0.01, curr_seg[1] - step_sz)
                            elif act == 3: curr_seg[1] = min(1.0, curr_seg[1] + step_sz)
                            
                            c_iou = calculate_iou(curr_seg, gt_norm[i])
                            reward = (c_iou - p_iou) * settings.REWARD_SCALE
                            if act == 4: reward += settings.REWARD_BONUS_HIGH if c_iou >= 0.7 else 1.0
                            b_reward += reward
                            advantage = (reward + GAMMA * val.detach()) - val
                            batch_rl_loss += -m.log_prob(act) * advantage.detach() + F.mse_loss(val, torch.tensor(reward + GAMMA * val.item()).to(DEVICE))
                            p_iou = c_iou
                            if act == 4: break
                        batch_iou_sum += p_iou
                    loss = batch_rl_loss / b; avg_batch_iou = batch_iou_sum / b
                    total_reward += b_reward / b; total_entropy += b_ent / (b * settings.MAX_STEPS)

            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_train_loss += loss.item(); total_train_iou += avg_batch_iou
            
            # --- 📊 TENSORBOARD BATCH (LIVE) ---
            g_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar(f'{TAG}/batch_train_loss', loss.item(), g_step)
            writer.add_scalar(f'{TAG}/batch_train_iou', avg_batch_iou, g_step)
            if batch_idx % 10 == 0: writer.flush()

        # === VALIDATION ===
        model.eval(); val_iou_sum = 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_v = v_batch['video'].to(DEVICE); q_v = [clean_query(q) for q in v_batch['query']]
                gt_v = v_batch['segment'].to(DEVICE); d_v = v_batch['duration'].to(DEVICE)
                b_v = v_v.shape[0]; v_f = clip_model.encode_image(v_v.view(-1, 3, 224, 224)).view(b_v, 64, -1).float()
                t_f = clip_model.encode_text(clip.tokenize(q_v, truncate=True).to(DEVICE)).float()
                v_f = F.normalize(v_f, dim=-1); t_f = F.normalize(t_f, dim=-1); v_fg = v_f.mean(1); gt_n = gt_v/d_v.unsqueeze(1)
                
                if TRAIN_MODE in ["B1", "B2"]:
                    st = torch.cat([v_fg, v_fg, t_f, torch.zeros(b_v, 2).to(DEVICE)], dim=-1)
                    pr = model(st)
                    for i in range(b_v): val_iou_sum += calculate_iou(pr[i], gt_n[i])
                else: # VALID RL
                    for i in range(b_v):
                        sim = v_f[i] @ t_f[i]; c_s = torch.tensor([max(0, torch.argmax(sim).item()/64 - 0.1), min(1.0, torch.argmax(sim).item()/64 + 0.1)]).to(DEVICE)
                        for _ in range(settings.MAX_STEPS):
                            s_i, e_i = max(0, min(63, int(c_s[0]*64))), max(1, min(64, int(c_s[1]*64)))
                            st = torch.cat([v_f[i, s_i:e_i].mean(0), v_fg[i], t_f[i], c_s], dim=-1)
                            act = torch.argmax(model(st)[0])
                            if act == 0: c_s[0] -= 0.04
                            elif act == 1: c_s[0] += 0.04
                            elif act == 2: c_s[1] -= 0.04
                            elif act == 3: c_s[1] += 0.04
                            if act == 4: break
                        val_iou_sum += calculate_iou(c_s, gt_n[i])

        # --- 📊 TENSORBOARD EPOCH ---
        avg_v_iou = val_iou_sum / len(val_loader.dataset)
        history['loss'].append(total_train_loss/len(train_loader)); history['train_iou'].append(total_train_iou/len(train_loader))
        history['val_iou'].append(avg_v_iou); history['lr'].append(optimizer.param_groups[0]['lr'])
        history['reward'].append(total_reward/len(train_loader)); history['entropy'].append(total_entropy/len(train_loader))
        
        writer.add_scalar(f'{TAG}/epoch_avg_train_loss', history['loss'][-1], epoch+1)
        writer.add_scalar(f'{TAG}/epoch_avg_val_iou', avg_v_iou, epoch+1)
        writer.flush(); scheduler.step(avg_v_iou)
        if avg_v_iou > best_iou: best_iou = avg_v_iou; torch.save(model.state_dict(), f"checkpoints/{TAG}_best.pth")
        print(f"✅ E{epoch+1} | Val IoU: {avg_v_iou:.4f}")

    plot_dashboard(history, TAG); writer.close()

if __name__ == "__main__": train()