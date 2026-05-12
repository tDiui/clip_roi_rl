# import torch
# import os
# import torch.backends.cudnn as cudnn

# # --- 1. CẤU HÌNH HỆ THỐNG ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Tối ưu hóa phần cứng dòng RTX 30 Series
# if DEVICE == "cuda":
#     cudnn.benchmark = True # Tự động chọn thuật toán tích chập nhanh nhất cho phần cứng
#     # Bật chế độ Tensor Core cho các phép tính ma trận lớn
#     torch.set_float32_matmul_precision('high') 

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# # --- 2. THAM SỐ HUẤN LUYỆN (Tận dụng 12GB VRAM) ---
# # Với CLIP ViT-L/14, 12GB có thể chịu được Batch 128. 
# # Nếu bị báo lỗi 'Out of Memory', hãy hạ xuống 96.
# BATCH_SIZE = 64 

# # Tăng NUM_WORKERS: Giảm nghẽn cổ chai khi CPU cắt video. 
# # Thường đặt bằng số luồng thực tế của CPU (8 hoặc 12).
# NUM_WORKERS = 0 

# # PIN_MEMORY: Tăng tốc độ truyền dữ liệu từ RAM lên thẳng GPU
# PIN_MEMORY = False 

# # Automatic Mixed Precision (AMP): Tăng tốc huấn luyện gấp 2-3 lần trên RTX 3060
# # bằng cách sử dụng số thực 16-bit (FP16) mà không làm mất độ chính xác.
# USE_AMP = True 

# LEARNING_RATE = 1e-6 # CLIP cần LR nhỏ để bảo tồn kiến thức [cite: 141]
# EPOCHS = 100 # Tăng số Epoch cho 4.900 nhãn để đạt mAP > 90% 

# # --- 3. KIẾN TRÚC MÔ HÌNH (Đồng bộ với CLIP ViT-L/14) ---
# HIDDEN_DIM = 768 # [cite: 16]
# AGENT_INPUT_DIM = 1538 # [cite: 45]
# NUM_FRAMES = 16 
# IMG_SIZE = (224, 224) # [cite: 93]

# # --- 4. THAM SỐ HỌC TĂNG CƯỜNG (RL) ---
# GAMMA = 0.99 
# MAX_STEPS = 15
# STOP_ACTION_IDX = 4 

# # Cấu hình cũ
# REWARD_POSITIVE = 1.0 
# REWARD_NEGATIVE = -1.0

# # --- THÊM MỚI: CHIẾN THUẬT PHÁ ĐỈNH (IOU 0.7) ---
# LEARNING_RATE = 1e-6       # Giữ mức thấp để tinh chỉnh sâu
# IOU_THRESHOLD_MID = 0.6    # Ngưỡng khuyến khích 1
# IOU_THRESHOLD_HIGH = 0.7   # Ngưỡng mục tiêu (Gold Standard)

# REWARD_BONUS_MID = 0.7     # Thưởng thêm khi vượt 0.6
# REWARD_BONUS_HIGH = 1.5    # Thưởng cực đậm khi vượt 0.7

# import torch
# import os
# import torch.backends.cudnn as cudnn

# # --- 1. SYSTEM ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# if DEVICE == "cuda":
#     cudnn.benchmark = True
#     torch.set_float32_matmul_precision('high')

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# # --- 2. TRAINING ---
# BATCH_SIZE = 64

# # ⚠️ Dataset bạn có xử lý video → nên bật lại
# NUM_WORKERS = 0        # (0 = rất chậm)
# PIN_MEMORY = False      # tăng tốc copy GPU

# USE_AMP = True

# # ⚠️ Bạn đang train RL → LR này đang QUÁ NHỎ
# LEARNING_RATE = 1e-4   # ✅ sửa từ 1e-6 → 1e-4

# EPOCHS = 100

# # --- 3. MODEL ---
# HIDDEN_DIM = 768

# # ⚠️ QUAN TRỌNG NHẤT (bạn vừa sửa state)
# # local(768) + global(768) + text(768) + segment(2)
# AGENT_INPUT_DIM = 2306   # ✅ FIX CHÍNH

# NUM_FRAMES = 16
# IMG_SIZE = (224, 224)

# # --- 4. RL ---
# GAMMA = 0.99
# MAX_STEPS = 15   # giảm nhẹ để học nhanh hơn
# STOP_ACTION_IDX = 4

# # --- 5. REWARD STRATEGY (đồng bộ với train) ---
# IOU_THRESHOLD_MID = 0.6
# IOU_THRESHOLD_HIGH = 0.7

# REWARD_SCALE = 20.0     # phải match với train
# STEP_PENALTY = 0.005    # ép agent dừng sớm

# REWARD_BONUS_MID = 3.0
# REWARD_BONUS_HIGH = 10.0

import torch
import os
import torch.backends.cudnn as cudnn

# --- 1. HỆ THỐNG (RTX 3060 12GB) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    cudnn.benchmark = True
    # Tối ưu cho kiến trúc Ampere của dòng 30 Series
    torch.set_float32_matmul_precision('high')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# --- 2. THAM SỐ HUẤN LUYỆN (QUAN TRỌNG) ---
BATCH_SIZE = 32         # Tốt nhất cho RL và 64 frames trên 12GB VRAM
EPOCHS = 100            # Đảm bảo có dòng này để train.py không lỗi
LEARNING_RATE = 1e-4    # LR chuẩn để Agent bứt phá khỏi mức 0.5
USE_AMP = True          # Dùng FP16 để train nhanh gấp đôi

# Tối ưu tốc độ nạp dữ liệu
NUM_WORKERS = 0         # Tận dụng CPU đa nhân của máy Thái
PIN_MEMORY = False       # Tăng tốc truyền dữ liệu RAM -> GPU

# --- 3. KIẾN TRÚC MÔ HÌNH (CLIP ViT-L/14) ---
HIDDEN_DIM = 768        # Kích thước vector của CLIP-L/14
# Feature: [Local(768) + Global(768) + Text(768) + Segment_Info(2)]
AGENT_INPUT_DIM = 2306  
NUM_FRAMES = 64         # Chìa khóa để đạt IoU 0.7
IMG_SIZE = (224, 224)

# --- 4. THAM SỐ RL & REWARD (CHIẾN THUẬT PHÁ ĐỈNH) ---
GAMMA = 0.99
MAX_STEPS = 30          # Tăng lên để Agent có đủ bước căn chỉnh 64 frames
STOP_ACTION_IDX = 4

# Các ngưỡng đánh giá
IOU_THRESHOLD_MID = 0.6
IOU_THRESHOLD_HIGH = 0.7

# Hệ thống phần thưởng "Jackpot"
REWARD_SCALE = 20.0
STEP_PENALTY = 0.01     # Phạt để Agent không đi lòng vòng
REWARD_BONUS_MID = 5.0  # Thưởng khi vượt 0.6
REWARD_BONUS_HIGH = 30.0 # Thưởng cực lớn khi vượt 0.7 (Phá đỉnh)