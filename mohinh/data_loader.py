import os
import json
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from config import settings

class NhangocDataset(Dataset):
    def __init__(self, label_file, transform=None, use_cache=True):
        self.data = []
        self.num_frames = settings.NUM_FRAMES
        self.use_cache = use_cache
        self.data_dir = "data"
        self.cache_dir = os.path.join(self.data_dir, "cached_tensors")
        os.makedirs(self.cache_dir, exist_ok=True)

        # --- BƯỚC MỚI: LẬP BẢN ĐỒ VIDEO (Xử lý lỗi đường dẫn sau khi Merge) ---
        print(f"--- 🔍 Đang quét vị trí video thực tế trong {self.data_dir}... ---")
        video_map = {}
        # Quét tất cả file .mp4 trong các thư mục con cam01, cam02...
        all_video_files = glob.glob(os.path.join(self.data_dir, "**", "*.mp4"), recursive=True)
        for v_path in all_video_files:
            # Lưu key là tên file (ví dụ: video_01.mp4), value là đường dẫn đầy đủ
            video_map[os.path.basename(v_path)] = v_path

        print(f"--- 📊 Đang nạp dữ liệu từ: {label_file} ---")
        
        if not os.path.exists(label_file):
            print(f"❌ LỖI: Không tìm thấy file nhãn {label_file}")
            return

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'image_path' in item:
                        # Lấy tên file gốc từ nhãn (ví dụ: cam01/v_1.mp4 -> v_1.mp4)
                        v_filename = os.path.basename(item['image_path'])
                        
                        # Truy vấn đường dẫn thực tế từ bản đồ video_map
                        full_v_path = video_map.get(v_filename)
                        
                        if full_v_path is None or not os.path.exists(full_v_path):
                            continue # Bỏ qua nếu vẫn không tìm thấy video

                        start, end = item.get('segment', [0.0, 0.0])
                        if start >= end:
                            continue 

                        # Tính duration (Vấn đề 4)
                        cap = cv2.VideoCapture(full_v_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        duration = total_f / fps if fps > 0 else 10.0
                        cap.release()

                        item['full_video_path'] = full_v_path
                        item['duration'] = duration
                        
                        # Tạo tên cache duy nhất dựa trên đường dẫn thực tế (tránh trùng tên cam khác nhau)
                        v_id = full_v_path.replace(os.sep, '_').replace(':', '').replace('.mp4', '')
                        # item['cache_file'] = os.path.join(self.cache_dir, f"{v_id}.pt")
                        item['cache_file'] = os.path.join(self.cache_dir, f"{v_id}_f{self.num_frames}.pt")
                        self.data.append(item)
                except:
                    continue
        
        print(f"✅ Đã nạp thành công: {len(self.data)} mẫu dữ liệu sạch.")
        
        # Chuẩn hóa ảnh cho CLIP
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(settings.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def _read_video_to_tensor(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: return torch.zeros((self.num_frames, 3, 224, 224))

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(Image.fromarray(frame))
                frames.append(frame)
            else:
                frames.append(torch.zeros((3, 224, 224)))
        cap.release()
        return torch.stack(frames)

    def __getitem__(self, idx):
        item = self.data[idx]
        cache_p = item['cache_file']

        if self.use_cache and os.path.exists(cache_p):
            video_tensor = torch.load(cache_p, weights_only=True)
        else:
            video_tensor = self._read_video_to_tensor(item['full_video_path'])
            if self.use_cache:
                torch.save(video_tensor, cache_p)

        return {
            "video": video_tensor,
            "query": item.get('query_vi', item.get('query', '')).strip(),
            "segment": torch.tensor(item.get('segment', [0.0, 1.0]), dtype=torch.float32),
            "duration": item['duration']
        }
