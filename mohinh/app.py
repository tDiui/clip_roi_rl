import streamlit as st
import streamlit.elements.image as st_image
import importlib

# --- 🛠️ BƯỚC 0: MONKEY PATCH (Sửa lỗi cho Streamlit 1.57.0 - Phiên bản không gạch đỏ) ---
if not hasattr(st_image, "image_to_url"):
    try:
        # Nạp gián tiếp để tránh Pylance báo lỗi Missing Import
        runtime_mem = importlib.import_module("streamlit.runtime.memory_media_file_manager")
        get_instance = getattr(runtime_mem, "get_instance")
        
        def image_to_url_patch(data, width, height, clamp, channels, output_format, image_id):
            return get_instance().add(data, output_format, image_id)
        
        st_image.image_to_url = image_to_url_patch
    except Exception as e:
        st.error(f"Lỗi khởi tạo Canvas: {e}")
        
import cv2
import torch
import clip
import numpy as np
from PIL import Image
import os
import pickle
import json
import time
from deep_translator import GoogleTranslator
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas 

# --- 1. KHỞI TẠO MÔ HÌNH ---
st.set_page_config(page_title="CCTV AI Search - TDMU Project", layout="wide")

@st.cache_resource
def load_ai_cores():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mô hình CLIP trích xuất đặc trưng ngữ nghĩa
    model, preprocess = clip.load("ViT-L/14", device=device)
    # Mô hình YOLOv8 phát hiện đối tượng
    yolo = YOLO('yolov8n.pt') 
    return model, preprocess, yolo, device

model, preprocess, yolo_model, device = load_ai_cores()

# --- 2. XỬ LÝ ROI ĐA CAMERA ---
def apply_roi_by_camera(frame, source_name, roi_path="roi.json"):
    try:
        if not os.path.exists(roi_path):
            return frame
            
        with open(roi_path, 'r') as f:
            roi_all = json.load(f)
        
        name_lower = source_name.lower()
        cam_key = "cam01" if "cam01" in name_lower else "cam02"
        
        if cam_key not in roi_all:
            return frame
            
        roi_data = roi_all[cam_key]
        h, w = frame.shape[:2]
        
        # Tính toán tỷ lệ scale giữa ảnh vẽ và ảnh thực tế
        scale_x = w / roi_data["frame_w"]
        scale_y = h / roi_data["frame_h"]
        
        points = np.array(roi_data["roi_polygon"], dtype=np.int32)
        points[:, 0] = (points[:, 0] * scale_x).astype(int)
        points[:, 1] = (points[:, 1] * scale_y).astype(int)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)
    except:
        return frame

# --- 3. TRÍCH XUẤT ĐẶC TRƯNG (CÓ CƠ CHẾ CACHE THEO VIDEO THỰC TẾ) ---
def index_video(video_path, original_name, sampling_sec=1):
    video_id = os.path.basename(original_name).split('.')[0]
    cache_file = f"cache_{video_id}.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    
    feats, times = [], []
    p_bar = st.progress(0)
    status_text = st.empty()
    
    for s in range(0, total_sec, sampling_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        
        roi_f = apply_roi_by_camera(frame, original_name)
        img = Image.fromarray(cv2.cvtColor(roi_f, cv2.COLOR_BGR2RGB))
        img_in = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            f_vec = model.encode_image(img_in).float()
            f_vec /= f_vec.norm(dim=-1, keepdim=True)
            feats.append(f_vec.cpu().numpy())
            times.append(s)
        
        if s % 10 == 0:
            p_bar.progress(s / total_sec)
            status_text.text(f"Đang phân tích video: {s}/{total_sec} giây...")

    cap.release()
    result = {"features": np.vstack(feats), "times": times}
    
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
        
    p_bar.empty()
    status_text.empty()
    return result

# --- 4. TRÍCH XUẤT ẢNH TẠI THỜI ĐIỂM RÕ NHẤT VÀ VẼ BOX (ĐÃ SỬA LỖI VẼ NHẦM CLASS) ---
def render_result_image(video_in, peak_time, score, output_name, query_en):
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    target_frame = int(peak_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False
        
    # Phân loại chi tiết từng Class đối tượng dựa theo từ khóa để giới hạn cứng cho YOLO
    target_ids = []
    q_low = query_en.lower()
    
    # 1. Nhóm xe tải (Truck) - Lớp số 7 trong YOLO
    if "truck" in q_low or "lorry" in q_low or "xe tải" in q_low:
        target_ids.append(7)
        
    # 2. Nhóm xe con / Xe ô tô nói chung (Car / Vehicle) - Lớp số 2 trong YOLO
    if "car" in q_low or "vehicle" in q_low or "ô tô" in q_low or "xe con" in q_low:
        if "truck" not in q_low and "xe tải" not in q_low:
            target_ids.append(2)
            
    # 3. Nhóm xe máy / Xe đạp (Motorcycle / Bike) - Lớp số 3 trong YOLO
    if "motorcycle" in q_low or "bike" in q_low or "xe máy" in q_low:
        target_ids.append(3)
        
    # 4. Nhóm xe buýt / Xe khách (Bus) - Lớp số 5 trong YOLO
    if "bus" in q_low or "xe buýt" in q_low:
        target_ids.append(5)
        
    # 5. Nhóm người đi bộ (Person) - Lớp số 0 trong YOLO
    if "person" in q_low or "man" in q_low or "người" in q_low:
        target_ids.append(0)
        
    # Nếu câu lệnh chung chung không thuộc nhóm đặc thù nào thì mở hết các phương tiện giao thông
    if not target_ids:
        target_ids = [0, 2, 3, 5, 7]
    
    y_res = yolo_model(frame, conf=0.3, verbose=False)
    for r in y_res:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in target_ids:
                b = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                
    cv2.imwrite(output_name, frame)
    return True

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m}p{s}s"

# --- 5. GIAO DIỆN CHÍNH ---
st.title("🎬 CCTV AI Search - Hệ thống Truy vấn Đa Camera")
col1, col2 = st.columns([1, 2])

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 1

with col1:
    st.subheader("⚙️ Cấu hình")
    v_file = st.file_uploader("Tải video", type=["mp4"])
    
    if v_file:
        temp_path = "temp_preview.mp4"
        with open(temp_path, "wb") as f:
            f.write(v_file.getbuffer())
        
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            cap = cv2.VideoCapture(temp_path)
            ret, first_frame = cap.read()
            cap.release()
            
            if ret:
                with st.expander("📐 Vẽ vùng quan tâm (ROI) - Click để tạo đa giác", expanded=True):
                    st.write("Dùng chuột chấm điểm. Xong nhấn Lưu ROI.")
                    bg_img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
                    
                    canvas_w = 700
                    canvas_h = int(bg_img.height * (canvas_w / bg_img.width))
                    
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 255, 0, 0.3)",
                        stroke_width=2,
                        stroke_color="#00FF00",
                        background_image=bg_img.resize((canvas_w, canvas_h)),
                        update_streamlit=True,
                        height=canvas_h,
                        width=canvas_w,
                        drawing_mode="polygon",
                        key="roi_canvas",
                    )
                    
            if st.button("💾 Lưu ROI vào hệ thống"):
                if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                    obj = canvas_result.json_data["objects"][-1]
                    path = obj["path"]
                    pts = [[p[1], p[2]] for p in path if len(p) == 3]
                    
                    scale = bg_img.width / canvas_w
                    final_pts = [[int(x * scale), int(y * scale)] for x, y in pts]
                    
                    cam_k = "cam01" if "cam01" in v_file.name.lower() else "cam02"
                    
                    roi_data = {}
                    if os.path.exists("roi.json"):
                        with open("roi.json", "r", encoding="utf-8") as f:
                            try: roi_data = json.load(f)
                            except: roi_data = {}
                    
                    roi_data[cam_k] = {
                        "frame_w": bg_img.width,
                        "frame_h": bg_img.height,
                        "roi_polygon": final_pts
                    }
                    
                    with open("roi.json", "w", encoding="utf-8") as f:
                        json.dump(roi_data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno()) 
                    
                    st.success(f"🔥 Đã tạo file ROI thành công cho {cam_k}!")
                    st.balloons()

    query_vi = st.text_input("Mô tả đối tượng:", "xe con màu đen")
    
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("🚀 Bắt đầu truy vấn") and v_file:
            st.session_state['current_page'] = 1 
            
            q_en = GoogleTranslator(source='vi', target='en').translate(query_vi)
            st.info(f"AI đang tìm kiếm: {q_en}")
            st.session_state['current_query_en'] = q_en
            
            db = index_video(temp_path, v_file.name)
            
            t_tokens = clip.tokenize([q_en]).to(device)
            with torch.no_grad():
                t_feat = model.encode_text(t_tokens).float()
                t_feat /= t_feat.norm(dim=-1, keepdim=True)
            
            sims = (torch.from_numpy(db["features"]).to(device) @ t_feat.T).cpu().numpy().flatten()
            
            found_events = []
            visited = np.zeros_like(sims)
            
            # Thuật toán quét và triệt tiêu trùng lặp 2 lớp
            for _ in range(50): 
                idx = np.argmax(sims * (1 - visited))
                score = sims[idx]
                if score < 0.22: break
                
                threshold = score * 0.8
                start_idx = idx
                while start_idx > 0 and sims[start_idx-1] > threshold:
                    start_idx -= 1
                
                end_idx = idx
                while end_idx < len(sims)-1 and sims[end_idx+1] > threshold:
                    end_idx += 1
                
                # Triệt tiêu khoảng thời gian rộng bao trùm sự kiện (+/- 8 bước nhảy)
                clear_start = max(0, start_idx - 8)
                clear_end = min(len(sims), end_idx + 8)
                visited[clear_start : clear_end] = 1
                
                # Loại bỏ nếu mốc thời gian đỉnh (peak) nằm quá sát sự kiện cũ (dưới 5 giây)
                is_duplicate = False
                for event in found_events:
                    if abs(event["peak"] - db["times"][idx]) <= 5: 
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    found_events.append({
                        "start": db["times"][start_idx],
                        "end": db["times"][end_idx],
                        "peak": db["times"][idx], 
                        "score": score
                    })
                    
                if len(found_events) >= 25: 
                    break
            
            st.session_state['search_results'] = found_events
            st.session_state['v_path'] = temp_path
            st.rerun()

    with btn_col2:
        if st.button("🧹 Làm mới hệ thống"):
            for key in ['search_results', 'v_path', 'current_query_en', 'current_page']:
                if key in st.session_state:
                    del st.session_state[key]
            for f in os.listdir():
                if f.startswith("res_img_"):
                    os.remove(f)
            st.rerun()

# --- 6. GIAO DIỆN HIỂN THỊ KẾT QUẢ THEO ẢNH MẪU ĐỒ ÁN ---
with col2:
    st.subheader("🎯 Kết quả tìm kiếm")
    if 'search_results' in st.session_state:
        res_list = st.session_state['search_results']
        
        if len(res_list) == 0:
            st.warning("📭 Không tìm thấy kết quả phù hợp với mô tả.")
        else:
            items_per_page = 5
            total_pages = (len(res_list) + items_per_page - 1) // items_per_page
            
            p_c1, p_c2, p_c3 = st.columns([1, 2, 1])
            with p_c1:
                if st.button("⬅️ Trước") and st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
                    st.rerun()
            with p_c2:
                st.write(f"Trang {st.session_state['current_page']} / {total_pages}")
            with p_c3:
                if st.button("Sau ➡️") and st.session_state['current_page'] < total_pages:
                    st.session_state['current_page'] += 1
                    st.rerun()

            start_idx = (st.session_state['current_page'] - 1) * items_per_page
            end_idx = start_idx + items_per_page
            
            for i, res in enumerate(res_list[start_idx : end_idx]):
                s_t, e_t, p_t, sc = res['start'], res['end'], res['peak'], res['score']
                
                st.markdown(f"### 🏆 Top {start_idx + i + 1}")
                info_col, img_col = st.columns([1, 1])
                
                with info_col:
                    st.markdown(f"* **Thời gian bắt đầu - Kết thúc:** `{format_time(s_t)}` ➡️ `{format_time(e_t)}`")
                    st.markdown(f"* **Thời gian phát hiện:** `{format_time(p_t)}`")
                    st.markdown(f"* **Độ tín nhiệm:** `{sc:.2f}`")
                    
                with img_col:
                    out_img_name = f"res_img_p{st.session_state['current_page']}_{i}.jpg"
                    success = render_result_image(st.session_state['v_path'], p_t, sc, out_img_name, st.session_state['current_query_en'])
                    
                    if success and os.path.exists(out_img_name):
                        result_image = Image.open(out_img_name)
                        st.image(result_image, use_column_width=True)
                    else:
                        st.error("❌ Không thể trích xuất ảnh từ video.")
                st.markdown("---")