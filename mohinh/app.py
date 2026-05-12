import streamlit as st
import cv2
import torch
import clip
import numpy as np
from PIL import Image
import os
import pickle
import json
from deep_translator import GoogleTranslator
from ultralytics import YOLO

# --- 1. KHỞI TẠO MÔ HÌNH ---
st.set_page_config(page_title="CCTV AI Search - TDMU Project", layout="wide")

@st.cache_resource
def load_ai_cores():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mô hình CLIP trích xuất đặc trưng ngữ nghĩa [cite: 16]
    model, preprocess = clip.load("ViT-L/14", device=device)
    # Mô hình YOLOv8 phát hiện đối tượng [cite: 57, 130]
    yolo = YOLO('yolov8n.pt') 
    return model, preprocess, yolo, device

model, preprocess, yolo_model, device = load_ai_cores()

# --- 2. XỬ LÝ ROI ĐA CAMERA ---
def apply_roi_by_camera(frame, source_name, roi_path="roi.json"):
    try:
        with open(roi_path, 'r') as f:
            roi_all = json.load(f)
        
        name_lower = source_name.lower()
        cam_key = "cam01" if "cam01" in name_lower else "cam02"
        
        if cam_key not in roi_all: return frame
        roi_data = roi_all[cam_key]
        
        h, w = frame.shape[:2]
        scale_x, scale_y = w / roi_data["frame_w"], h / roi_data["frame_h"]
        points = np.array(roi_data["roi_polygon"], dtype=np.int32)
        points[:, 0] = (points[:, 0] * scale_x).astype(int)
        points[:, 1] = (points[:, 1] * scale_y).astype(int)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)
    except:
        return frame

# --- 3. TRÍCH XUẤT ĐẶC TRƯNG ---
def index_video(video_path, sampling_sec=1):
    video_id = os.path.basename(video_path).split('.')[0]
    cache_file = f"cache_{video_id}.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f: return pickle.load(f)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    
    feats, times = [], []
    p_bar = st.progress(0)
    status_text = st.empty()
    
    for s in range(0, total_sec, sampling_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ret, frame = cap.read()
        if not ret: break
        
        # Tiền xử lý ROI [cite: 120]
        roi_f = apply_roi_by_camera(frame, video_path)
        img = Image.fromarray(cv2.cvtColor(roi_f, cv2.COLOR_BGR2RGB))
        img_in = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            f_vec = model.encode_image(img_in).float()
            f_vec /= f_vec.norm(dim=-1, keepdim=True)
            feats.append(f_vec.cpu().numpy())
            times.append(s)
        
        if s % 100 == 0:
            p_bar.progress(s / total_sec)
            status_text.text(f"Đang phân tích video: {s}/{total_sec} giây...")

    cap.release()
    result = {"features": np.vstack(feats), "times": times}
    with open(cache_file, "wb") as f: pickle.dump(result, f)
    p_bar.empty()
    status_text.empty()
    return result

# --- 4. VẼ BOUNDING BOX MỤC TIÊU VÀ XUẤT CLIP ---
def render_result_clip(video_in, t_mark, score, output_name, query_en):
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    start_f = int(max(0, (t_mark - 1.5) * fps))
    end_f = int((t_mark + 5) * fps) 
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = None

    # Xác định danh sách ID lớp mục tiêu dựa trên từ khóa tiếng Anh [cite: 54, 165, 184]
    target_ids = []
    q_low = query_en.lower()
    if "car" in q_low or "vehicle" in q_low: target_ids.append(2)
    if "motorcycle" in q_low or "bike" in q_low: target_ids.append(3)
    if "bus" in q_low: target_ids.append(5)
    if "truck" in q_low: target_ids.append(7)
    if "person" in q_low or "man" in q_low or "woman" in q_low: target_ids.append(0)

    # Nếu không tìm thấy từ khóa đặc trưng, mặc định vẽ các loại xe cơ bản
    if not target_ids:
        target_ids = [2, 3, 5, 7]
    
    for _ in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret: break
        
        y_res = yolo_model(frame, conf=0.4, verbose=False)
        for r in y_res:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                # CHỈ VẼ BOX NẾU ĐỐI TƯỢNG THUỘC LỚP ĐANG TRUY VẤN [cite: 185, 194]
                if cls_id in target_ids:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"Target Match: {score:.2f}", (b[0], b[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(output_name, fourcc, fps, (w, h))
        writer.write(frame)
        
    if writer: writer.release()
    cap.release()

# --- 5. GIAO DIỆN ---
st.title("🎬 CCTV AI Search - Hệ thống Truy vấn Đa Camera")
col1, col2 = st.columns([1, 2])

if 'display_limit' not in st.session_state:
    st.session_state['display_limit'] = 5

with col1:
    st.subheader("⚙️ Cấu hình")
    v_file = st.file_uploader("Tải video", type=["mp4"])
    query_vi = st.text_input("Mô tả đối tượng:", "xe con màu đen")
    
    if st.button("🚀 Bắt đầu truy vấn") and v_file:
        st.session_state['display_limit'] = 5 
        with open(v_file.name, "wb") as f: f.write(v_file.getbuffer())
        
        q_en = GoogleTranslator(source='vi', target='en').translate(query_vi)
        st.info(f"AI đang tìm kiếm: {q_en}")
        
        # Lưu câu truy vấn tiếng Anh để dùng cho việc vẽ box
        st.session_state['current_query_en'] = q_en
        
        db = index_video(v_file.name)
        
        t_tokens = clip.tokenize([q_en]).to(device)
        with torch.no_grad():
            t_feat = model.encode_text(t_tokens).float()
            t_feat /= t_feat.norm(dim=-1, keepdim=True)
        
        sims = (torch.from_numpy(db["features"]).to(device) @ t_feat.T).cpu().numpy().flatten()
        
        top_results = []
        for idx in np.argsort(sims)[::-1]:
            t, sc = db["times"][idx], sims[idx]
            if not any(abs(t - res[0]) < 30 for res in top_results):
                top_results.append((t, sc))
            if len(top_results) == 20: break
        
        st.session_state['search_results'] = top_results
        st.session_state['v_path'] = v_file.name

    if st.button("🧹 Làm mới hệ thống"):
        for key in ['search_results', 'v_path', 'current_query_en']:
            if key in st.session_state: del st.session_state[key]
        st.session_state['display_limit'] = 5
        for f in os.listdir():
            if f.startswith("temp_res_") and f.endswith(".mp4"):
                os.remove(f)
        st.rerun()

with col2:
    st.subheader("🎯 Kết quả tìm kiếm")
    if 'search_results' in st.session_state:
        res_list = st.session_state['search_results']
        limit = st.session_state['display_limit']
        q_en_target = st.session_state.get('current_query_en', 'car')
        
        for i in range(min(len(res_list), limit)):
            t, sc = res_list[i]
            
            hours = int(t // 3600)
            minutes = int((t % 3600) // 60)
            seconds = int(t % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            with st.expander(f"Top {i+1} - Thời điểm: {time_str} (Score: {sc:.4f})"):
                out = f"temp_res_{i}.mp4"
                # TRUYỀN THÊM CÂU TRUY VẤN ĐỂ LỌC BOX
                render_result_clip(st.session_state['v_path'], t, sc, out, q_en_target)
                st.video(out)
        
        if limit < len(res_list):
            if st.button("➕ Xem thêm 10 kết quả"):
                st.session_state['display_limit'] += 10
                st.rerun()