import torch
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = settings.BATCH_SIZE
EPOCHS = settings.EPOCHS
LEARNING_RATE = settings.LEARNING_RATE 
GAMMA = settings.GAMMA

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
    writer = SummaryWriter('logs/vmr_final_mission')

    print(f"--- 🚀 Đang khởi tạo CLIP & Agent (Mục tiêu IoU 0.7) ---")
    clip_model, _ = clip.load("ViT-L/14", device=DEVICE)
    clip_model.eval()
    scaler = torch.amp.GradScaler('cuda')

    agent = ActorCriticAgent(input_dim=settings.AGENT_INPUT_DIM).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    START_EPOCH = 0
    list_of_files = glob.glob('checkpoints/*.pth')
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"--- 🔄 Đang nạp lại trí khôn từ: {latest_file} ---")
        checkpoint = torch.load(latest_file, map_location=DEVICE, weights_only=True)
        agent.load_state_dict(checkpoint)
        try:
            START_EPOCH = int(latest_file.split('_')[-1].split('.')[0])
        except:
            START_EPOCH = 0

    print(f"--- 🏁 Bắt đầu huấn luyện từ Epoch {START_EPOCH + 1} ---")
    
    train_dataset = NhangocDataset(label_file="data/train_annotations.jsonl", use_cache=True)
    val_dataset = NhangocDataset(label_file="data/val_annotations.jsonl", use_cache=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=custom_collate, 
        num_workers=settings.NUM_WORKERS, 
        pin_memory=settings.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=custom_collate, 
        num_workers=settings.NUM_WORKERS, 
        pin_memory=settings.PIN_MEMORY
    )

    best_iou = 0.0
    
    for epoch in range(START_EPOCH, EPOCHS):
        agent.train()
        total_train_loss = 0
        total_train_iou = 0
        
        for batch_idx, batch in enumerate(train_loader):
            videos = batch['video'].to(DEVICE, non_blocking=True)
            raw_queries = batch['query']
            gt_segments = batch['segment'].to(DEVICE, non_blocking=True)
            durations = batch['duration'].to(DEVICE, non_blocking=True)

            cleaned_queries = [clean_query(q) for q in raw_queries]
            if not all(len(q) > 0 for q in cleaned_queries): continue 

            with torch.amp.autocast('cuda'):
                b = videos.shape[0]
                with torch.no_grad():
                    video_input = videos.view(-1, 3, 224, 224)
                    v_feat = clip_model.encode_image(video_input).view(b, settings.NUM_FRAMES, -1).float()
                    text_tokens = clip.tokenize(cleaned_queries, truncate=True).to(DEVICE)
                    t_feat = clip_model.encode_text(text_tokens).float()

                    v_feat = F.normalize(v_feat, p=2, dim=-1)
                    t_feat = F.normalize(t_feat, p=2, dim=-1)

                v_feat_global = v_feat.max(dim=1)[0] 
                logits = (v_feat_global @ t_feat.t()) / 0.07
                labels = torch.arange(b).to(DEVICE)

                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.t(), labels)

                loss_alignment = (loss_i + loss_t) / 2

                batch_rl_loss = 0
                current_batch_iou_sum = 0
                
                for i in range(b):

                    # Tính độ tương đồng giữa từng frame và câu truy vấn để tìm điểm bắt đầu tốt nhất
                    with torch.no_grad():
                        # v_feat[i] shape: (NUM_FRAMES, 512), t_feat[i] shape: (512)
                        similarities = v_feat[i] @ t_feat[i] 
                        best_frame_idx = torch.argmax(similarities).item()
                        
                        # Khởi tạo cửa sổ 20% thời lượng video quanh frame tốt nhất
                        center = best_frame_idx / settings.NUM_FRAMES
                        start = max(0, center - 0.1)
                        end = min(1.0, center + 0.1)
                        curr_seg = torch.tensor([start, end]).to(DEVICE)

                    gt_norm = torch.tensor([
                        gt_segments[i][0] / durations[i], 
                        gt_segments[i][1] / durations[i]
                    ]).to(DEVICE)

                    p_iou = calculate_iou(curr_seg, gt_norm)
                    
                    for step in range(settings.MAX_STEPS):

                        # Lấy đặc trưng trung bình của đúng đoạn video đang xét (Local Feature)
                        s_idx = int(curr_seg[0] * settings.NUM_FRAMES)
                        e_idx = int(curr_seg[1] * settings.NUM_FRAMES)

                        s_idx = max(0, min(settings.NUM_FRAMES - 1, s_idx))
                        e_idx = max(s_idx + 1, min(settings.NUM_FRAMES, e_idx))

                        if e_idx <= s_idx:
                            local_v_feat = v_feat[i].mean(0)
                        else:
                            local_v_feat = v_feat[i, s_idx:e_idx].mean(0)

                        if torch.isnan(local_v_feat).any():
                            local_v_feat = v_feat[i].mean(0)


                        global_v_feat = v_feat[i].mean(0)

                        state = torch.cat([local_v_feat, global_v_feat, t_feat[i], curr_seg], dim=-1)
                        logits, val = agent(state)
                        logits = torch.clamp(logits, -10, 10)
                        m = torch.distributions.Categorical(logits=logits)
                        act = m.sample()
                        
                        # ✅ FIX 2: step size lớn hơn
                        step_size = 0.04

                        if act == 0: curr_seg[0] = max(0, curr_seg[0] - step_size)
                        elif act == 1: curr_seg[0] = min(curr_seg[1] - 0.01, curr_seg[0] + step_size)
                        elif act == 2: curr_seg[1] = max(curr_seg[0] + 0.01, curr_seg[1] - step_size)
                        elif act == 3: curr_seg[1] = min(1.0, curr_seg[1] + step_size)
                        
                        c_iou = calculate_iou(curr_seg, gt_norm)
                        
                        # ✅ FIX 3: reward scale
                        reward = (c_iou - p_iou) * settings.REWARD_SCALE * 1.5
                        reward -= settings.STEP_PENALTY
                        # thêm penalty độ dài
                        seg_len = curr_seg[1] - curr_seg[0]
                        gt_len = gt_norm[1] - gt_norm[0]
                        reward -= abs(seg_len - gt_len) * 2.0

                        if act == settings.STOP_ACTION_IDX:
                                if c_iou >= settings.IOU_THRESHOLD_HIGH:
                                    reward += settings.REWARD_BONUS_HIGH
                                elif c_iou >= settings.IOU_THRESHOLD_MID:
                                    reward += settings.REWARD_BONUS_MID
                                elif c_iou >= 0.5: reward += 1.0
                        
                        # ✅ FIX 4: advantage chuẩn
                        
                        target = reward + GAMMA * val.detach()
                        advantage = (target - val).clamp(-5, 5)

                        # ✅ FIX 5: value loss đúng
                        entropy = m.entropy().mean()
                        batch_rl_loss += (
                            -m.log_prob(act) * advantage.detach() + 
                            F.mse_loss(val, target) - 
                            0.01 * entropy  # 0.01 là trọng số entropy để ép Agent khám phá
                        )
                        
                        p_iou = c_iou
                        if act == settings.STOP_ACTION_IDX: break
                    
                    current_batch_iou_sum += p_iou

                total_batch_loss = (loss_alignment * 0.4) + (batch_rl_loss / b * 0.6)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            avg_batch_iou = current_batch_iou_sum / b
            total_train_iou += avg_batch_iou
            total_train_loss += total_batch_loss.item()
            
            # --- 2. GHI LOG BATCH (Nằm trong vòng lặp batch) ---
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Batch/Train_Loss', total_batch_loss.item(), global_step)
            writer.add_scalar('Batch/Train_IoU', avg_batch_iou, global_step)
            writer.flush() # Đẩy dữ liệu đi ngay

            if batch_idx % 10 == 0:
                print(f"E[{epoch+1}] B_{batch_idx} | Train Loss: {total_batch_loss.item():.2f} | IoU: {avg_batch_iou:.4f}")

                # --- 3. GHI LOG EPOCH (Phải nằm NGOÀI vòng lặp batch - Thụt lề sang trái 1 bậc) ---
            avg_train_iou = total_train_iou / len(train_loader)
            avg_train_loss = total_train_loss / len(train_loader)
            writer.add_scalar('Epoch/Avg_Train_IoU', avg_train_iou, epoch + 1)
            writer.add_scalar('Epoch/Avg_Train_Loss', avg_train_loss, epoch + 1)

        # ===== VALID =====
        agent.eval()
        val_iou_sum = 0
        print(f"--- 🧪 Đang đánh giá trên bộ Validation ({len(val_loader)} batches)... ---")
        
        with torch.no_grad():
            for val_batch in val_loader:
                v_val = val_batch['video'].to(DEVICE)
                q_val = [clean_query(q) for q in val_batch['query']]
                gt_val = val_batch['segment'].to(DEVICE)
                dur_val = val_batch['duration'].to(DEVICE)
                b_v = v_val.shape[0]

                v_f = clip_model.encode_image(v_val.view(-1, 3, 224, 224)).view(b_v, settings.NUM_FRAMES, -1).float()
                t_f = clip_model.encode_text(clip.tokenize(q_val, truncate=True).to(DEVICE)).float()
                v_f = F.normalize(v_f, dim=-1)
                t_f = F.normalize(t_f, dim=-1)

                for i in range(b_v):
                    similarities = v_f[i] @ t_f[i]
                    best_idx = torch.argmax(similarities).item()

                    center = best_idx / settings.NUM_FRAMES
                    start = max(0, center - 0.1)
                    end = min(1.0, center + 0.1)

                    curr_seg = torch.tensor([start, end]).to(DEVICE)
                    gt_n = torch.tensor([gt_val[i][0]/dur_val[i], gt_val[i][1]/dur_val[i]]).to(DEVICE)
                    
                    for _ in range(settings.MAX_STEPS):
                        s_idx = int(curr_seg[0] * settings.NUM_FRAMES)
                        e_idx = int(curr_seg[1] * settings.NUM_FRAMES)

                        s_idx = max(0, min(settings.NUM_FRAMES - 1, s_idx))
                        e_idx = max(s_idx + 1, min(settings.NUM_FRAMES, e_idx))

                        local_v = v_f[i, s_idx:e_idx].mean(0)

                        if torch.isnan(local_v).any():
                            local_v = v_f[i].mean(0)

                        global_v = v_f[i].mean(0)

                        st = torch.cat([local_v, global_v, t_f[i], curr_seg], dim=-1)
                        logits, val = agent(st)
                        logits = torch.clamp(logits, -10, 10)
                        act = torch.argmax(logits)
                        
                        step_size = 0.04
                        if act == 0: curr_seg[0] = max(0, curr_seg[0] - step_size)
                        elif act == 1: curr_seg[0] = min(curr_seg[1] - 0.01, curr_seg[0] + step_size)
                        elif act == 2: curr_seg[1] = max(curr_seg[0] + 0.01, curr_seg[1] - step_size)
                        elif act == 3: curr_seg[1] = min(1.0, curr_seg[1] + step_size)
                        if act == settings.STOP_ACTION_IDX: break
                    
                    val_iou_sum += calculate_iou(curr_seg, gt_n)

        avg_val_iou = val_iou_sum / len(val_dataset)
        writer.add_scalar('Epoch/Avg_Val_IoU', avg_val_iou, epoch + 1)

        print(f"=== Epoch {epoch+1} Hoàn tất | Train IoU: {total_train_iou/len(train_loader):.4f} | Val IoU: {avg_val_iou:.4f} ===")
        
        scheduler.step(avg_val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch + 1)

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(agent.state_dict(), "checkpoints/best_vmr_agent.pth")
            print(f"⭐ Kỷ lục mới trên bộ Validation: {best_iou:.4f}")

        torch.save(agent.state_dict(), f"checkpoints/vmr_agent_epoch_{epoch+1}.pth")

    writer.close()

if __name__ == "__main__":
    train()