# import streamlit as st
# import cv2
# import torch
# import clip
# import numpy as np
# from PIL import Image
# import os
# import pickle
# import json
# from deep_translator import GoogleTranslator
# from ultralytics import YOLO
# from streamlit_drawable_canvas import st_canvas # --- THÊM MỚI ---

# # --- 1. KHỞI TẠO MÔ HÌNH ---
# st.set_page_config(page_title="CCTV AI Search - TDMU Project", layout="wide")

# @st.cache_resource
# def load_ai_cores():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-L/14", device=device)
#     yolo = YOLO('yolov8n.pt') 
#     return model, preprocess, yolo, device

# model, preprocess, yolo_model, device = load_ai_cores()

# # --- 2. XỬ LÝ ROI ĐA CAMERA (GIỮ NGUYÊN) ---
# def apply_roi_by_camera(frame, source_name, roi_path="roi.json"):
#     try:
#         with open(roi_path, 'r') as f:
#             roi_all = json.load(f)
        
#         name_lower = source_name.lower()
#         cam_key = "cam01" if "cam01" in name_lower else "cam02"
        
#         if cam_key not in roi_all: return frame
#         roi_data = roi_all[cam_key]
        
#         h, w = frame.shape[:2]
#         scale_x, scale_y = w / roi_data["frame_w"], h / roi_data["frame_h"]
#         points = np.array(roi_data["roi_polygon"], dtype=np.int32)
#         points[:, 0] = (points[:, 0] * scale_x).astype(int)
#         points[:, 1] = (points[:, 1] * scale_y).astype(int)
        
#         mask = np.zeros((h, w), dtype=np.uint8)
#         cv2.fillPoly(mask, [points], 255)
#         return cv2.bitwise_and(frame, frame, mask=mask)
#     except:
#         return frame

# # --- 3. TRÍCH XUẤT ĐẶC TRƯNG (GIỮ NGUYÊN) ---
# def index_video(video_path, sampling_sec=1):
#     video_id = os.path.basename(video_path).split('.')[0]
#     cache_file = f"cache_{video_id}.pkl"
    
#     if os.path.exists(cache_file):
#         with open(cache_file, "rb") as f: return pickle.load(f)
        
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25
#     total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    
#     feats, times = [], []
#     p_bar = st.progress(0)
#     status_text = st.empty()
    
#     for s in range(0, total_sec, sampling_sec):
#         cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
#         ret, frame = cap.read()
#         if not ret: break
        
#         roi_f = apply_roi_by_camera(frame, video_path)
#         img = Image.fromarray(cv2.cvtColor(roi_f, cv2.COLOR_BGR2RGB))
#         img_in = preprocess(img).unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             f_vec = model.encode_image(img_in).float()
#             f_vec /= f_vec.norm(dim=-1, keepdim=True)
#             feats.append(f_vec.cpu().numpy())
#             times.append(s)
        
#         if s % 100 == 0:
#             p_bar.progress(s / total_sec)
#             status_text.text(f"Đang phân tích video: {s}/{total_sec} giây...")

#     cap.release()
#     result = {"features": np.vstack(feats), "times": times}
#     with open(cache_file, "wb") as f: pickle.dump(result, f)
#     p_bar.empty()
#     status_text.empty()
#     return result

# # --- 4. VẼ BOUNDING BOX MỤC TIÊU VÀ XUẤT CLIP (GIỮ NGUYÊN) ---
# def render_result_clip(video_in, t_mark, score, output_name, query_en):
#     cap = cv2.VideoCapture(video_in)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25
#     start_f = int(max(0, (t_mark - 1.5) * fps))
#     end_f = int((t_mark + 5) * fps) 
    
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     writer = None

#     target_ids = []
#     q_low = query_en.lower()
#     if "car" in q_low or "vehicle" in q_low: target_ids.append(2)
#     if "motorcycle" in q_low or "bike" in q_low: target_ids.append(3)
#     if "bus" in q_low: target_ids.append(5)
#     if "truck" in q_low: target_ids.append(7)
#     if "person" in q_low or "man" in q_low or "woman" in q_low: target_ids.append(0)

#     if not target_ids:
#         target_ids = [2, 3, 5, 7]
    
#     for _ in range(start_f, end_f):
#         ret, frame = cap.read()
#         if not ret: break
        
#         y_res = yolo_model(frame, conf=0.4, verbose=False)
#         for r in y_res:
#             for box in r.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id in target_ids:
#                     b = box.xyxy[0].cpu().numpy().astype(int)
#                     cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
#                     cv2.putText(frame, f"Target Match: {score:.2f}", (b[0], b[1]-10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
#         if writer is None:
#             h, w = frame.shape[:2]
#             writer = cv2.VideoWriter(output_name, fourcc, fps, (w, h))
#         writer.write(frame)
        
#     if writer: writer.release()
#     cap.release()

# # --- 5. GIAO DIỆN ---
# st.title("🎬 CCTV AI Search - Hệ thống Truy vấn Đa Camera")
# col1, col2 = st.columns([1, 2])

# if 'display_limit' not in st.session_state:
#     st.session_state['display_limit'] = 5

# with col1:
#     st.subheader("⚙️ Cấu hình")
#     v_file = st.file_uploader("Tải video", type=["mp4"])
    
#     # --- [MỚI] CHỨC NĂNG VẼ ROI TRỰC TIẾP ---
#     if v_file:
#         with open("temp_preview.mp4", "wb") as f: f.write(v_file.getbuffer())
#         cap = cv2.VideoCapture("temp_preview.mp4")
#         ret, first_frame = cap.read()
#         cap.release()
        
#         if ret:
#             with st.expander("📐 Vẽ vùng quan tâm (ROI) - Click để tạo đa giác"):
#                 st.write("Click các điểm để tạo vùng xanh. Khi xong, nhấn nút Lưu.")
#                 bg_img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
                
#                 # Canvas cho phép vẽ đa giác
#                 canvas_result = st_canvas(
#                     fill_color="rgba(0, 255, 0, 0.3)",
#                     stroke_width=2,
#                     stroke_color="#00FF00",
#                     background_image=bg_img,
#                     update_streamlit=True,
#                     height=bg_img.height * (700 / bg_img.width), # Scale hiển thị
#                     width=700,
#                     drawing_mode="polygon",
#                     key="roi_canvas",
#                 )
                
#                 if st.button("💾 Lưu ROI vào hệ thống"):
#                     if canvas_result.json_data is not None:
#                         objects = canvas_result.json_data["objects"]
#                         if objects:
#                             # Lấy tọa độ từ canvas và scale ngược lại kích thước gốc
#                             poly = objects[-1] # Lấy đa giác cuối cùng
#                             path = poly["path"]
#                             pts = []
#                             # Chuyển đổi format path của canvas sang list [x,y]
#                             for p in path:
#                                 if len(p) == 3: pts.append([p[1], p[2]])
                            
#                             # Tính tỷ lệ scale giữa canvas (700px) và ảnh gốc
#                             scale = bg_img.width / 700
#                             final_pts = [[int(x * scale), int(y * scale)] for x, y in pts]
                            
#                             # Cập nhật roi.json theo logic camera của Thái
#                             name_l = v_file.name.lower()
#                             cam_k = "cam01" if "cam01" in name_l else "cam02"
                            
#                             roi_data = {}
#                             if os.path.exists("roi.json"):
#                                 with open("roi.json", "r") as f: roi_data = json.load(f)
                            
#                             roi_data[cam_k] = {
#                                 "frame_w": bg_img.width,
#                                 "frame_h": bg_img.height,
#                                 "roi_polygon": final_pts
#                             }
                            
#                             with open("roi.json", "w") as f: json.dump(roi_data, f, indent=2)
#                             st.success(f"Đã cập nhật ROI cho {cam_k}!")

#     query_vi = st.text_input("Mô tả đối tượng:", "xe con màu đen")
    
#     if st.button("🚀 Bắt đầu truy vấn") and v_file:
#         st.session_state['display_limit'] = 5 
#         with open(v_file.name, "wb") as f: f.write(v_file.getbuffer())
        
#         q_en = GoogleTranslator(source='vi', target='en').translate(query_vi)
#         st.info(f"AI đang tìm kiếm: {q_en}")
#         st.session_state['current_query_en'] = q_en
        
#         db = index_video(v_file.name)
        
#         t_tokens = clip.tokenize([q_en]).to(device)
#         with torch.no_grad():
#             t_feat = model.encode_text(t_tokens).float()
#             t_feat /= t_feat.norm(dim=-1, keepdim=True)
        
#         sims = (torch.from_numpy(db["features"]).to(device) @ t_feat.T).cpu().numpy().flatten()
        
#         top_results = []
#         for idx in np.argsort(sims)[::-1]:
#             t, sc = db["times"][idx], sims[idx]
#             if not any(abs(t - res[0]) < 30 for res in top_results):
#                 top_results.append((t, sc))
#             if len(top_results) == 20: break
        
#         st.session_state['search_results'] = top_results
#         st.session_state['v_path'] = v_file.name

#     if st.button("🧹 Làm mới hệ thống"):
#         for key in ['search_results', 'v_path', 'current_query_en']:
#             if key in st.session_state: del st.session_state[key]
#         st.session_state['display_limit'] = 5
#         for f in os.listdir():
#             if f.startswith("temp_res_") and f.endswith(".mp4"):
#                 os.remove(f)
#         st.rerun()

# with col2:
#     st.subheader("🎯 Kết quả tìm kiếm")
#     if 'search_results' in st.session_state:
#         res_list = st.session_state['search_results']
#         limit = st.session_state['display_limit']
#         q_en_target = st.session_state.get('current_query_en', 'car')
        
#         for i in range(min(len(res_list), limit)):
#             t, sc = res_list[i]
#             hours = int(t // 3600); minutes = int((t % 3600) // 60); seconds = int(t % 60)
#             time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
#             with st.expander(f"Top {i+1} - Thời điểm: {time_str} (Score: {sc:.4f})"):
#                 out = f"temp_res_{i}.mp4"
#                 render_result_clip(st.session_state['v_path'], t, sc, out, q_en_target)
#                 st.video(out)
        
#         if limit < len(res_list):
#             if st.button("➕ Xem thêm 10 kết quả"):
#                 st.session_state['display_limit'] += 10
#                 st.rerun()

# import streamlit as st
# import streamlit.elements.image as st_image
# import importlib

# # --- 🛠️ BƯỚC 0: MONKEY PATCH (Sửa lỗi cho Streamlit 1.57.0 - Phiên bản không gạch đỏ) ---
# if not hasattr(st_image, "image_to_url"):
#     try:
#         # Nạp gián tiếp để tránh Pylance báo lỗi Missing Import
#         runtime_mem = importlib.import_module("streamlit.runtime.memory_media_file_manager")
#         get_instance = getattr(runtime_mem, "get_instance")
        
#         def image_to_url_patch(data, width, height, clamp, channels, output_format, image_id):
#             return get_instance().add(data, output_format, image_id)
        
#         st_image.image_to_url = image_to_url_patch
#     except Exception as e:
#         st.error(f"Lỗi khởi tạo Canvas: {e}")
        
# import cv2
# import torch
# import clip
# import numpy as np
# from PIL import Image
# import os
# import pickle
# import json
# import time
# from deep_translator import GoogleTranslator
# from ultralytics import YOLO
# from streamlit_drawable_canvas import st_canvas 

# # --- 1. KHỞI TẠO MÔ HÌNH ---
# st.set_page_config(page_title="CCTV AI Search - TDMU Project", layout="wide")

# @st.cache_resource
# def load_ai_cores():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # Mô hình CLIP trích xuất đặc trưng ngữ nghĩa
#     model, preprocess = clip.load("ViT-L/14", device=device)
#     # Mô hình YOLOv8 phát hiện đối tượng
#     yolo = YOLO('yolov8n.pt') 
#     return model, preprocess, yolo, device

# model, preprocess, yolo_model, device = load_ai_cores()

# # --- 2. XỬ LÝ ROI ĐA CAMERA ---
# def apply_roi_by_camera(frame, source_name, roi_path="roi.json"):
#     try:
#         if not os.path.exists(roi_path):
#             return frame
            
#         with open(roi_path, 'r') as f:
#             roi_all = json.load(f)
        
#         name_lower = source_name.lower()
#         cam_key = "cam01" if "cam01" in name_lower else "cam02"
        
#         if cam_key not in roi_all:
#             return frame
            
#         roi_data = roi_all[cam_key]
#         h, w = frame.shape[:2]
        
#         # Tính toán tỷ lệ scale giữa ảnh vẽ và ảnh thực tế
#         scale_x = w / roi_data["frame_w"]
#         scale_y = h / roi_data["frame_h"]
        
#         points = np.array(roi_data["roi_polygon"], dtype=np.int32)
#         points[:, 0] = (points[:, 0] * scale_x).astype(int)
#         points[:, 1] = (points[:, 1] * scale_y).astype(int)
        
#         mask = np.zeros((h, w), dtype=np.uint8)
#         cv2.fillPoly(mask, [points], 255)
#         return cv2.bitwise_and(frame, frame, mask=mask)
#     except:
#         return frame

# # --- 3. TRÍCH XUẤT ĐẶC TRƯNG (CÓ CƠ CHẾ CACHE TIẾT KIỆM THỜI GIAN) ---
# def index_video(video_path, sampling_sec=1):
#     video_id = os.path.basename(video_path).split('.')[0]
#     cache_file = f"cache_{video_id}.pkl"
    
#     # Nếu đã có file cache, nạp lại ngay lập tức không cần quét video
#     if os.path.exists(cache_file):
#         with open(cache_file, "rb") as f:
#             return pickle.load(f)
        
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25
#     total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    
#     feats, times = [], []
#     p_bar = st.progress(0)
#     status_text = st.empty()
    
#     for s in range(0, total_sec, sampling_sec):
#         cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Áp dụng ROI trước khi đưa vào CLIP
#         roi_f = apply_roi_by_camera(frame, video_path)
#         img = Image.fromarray(cv2.cvtColor(roi_f, cv2.COLOR_BGR2RGB))
#         img_in = preprocess(img).unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             f_vec = model.encode_image(img_in).float()
#             f_vec /= f_vec.norm(dim=-1, keepdim=True)
#             feats.append(f_vec.cpu().numpy())
#             times.append(s)
        
#         if s % 10 == 0:
#             p_bar.progress(s / total_sec)
#             status_text.text(f"Đang phân tích video: {s}/{total_sec} giây...")

#     cap.release()
#     result = {"features": np.vstack(feats), "times": times}
    
#     # Lưu vào cache để lần sau nạp ngay
#     with open(cache_file, "wb") as f:
#         pickle.dump(result, f)
        
#     p_bar.empty()
#     status_text.empty()
#     return result

# # --- 4. VẼ BOX MỤC TIÊU VÀ XUẤT CLIP (Xử lý Start/End) ---
# def render_result_clip(video_in, s_time, e_time, score, output_name, query_en):
#     cap = cv2.VideoCapture(video_in)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
#     # Mở rộng biên 1 giây để clip không bị quá ngắn
#     start_f = int(max(0, (s_time - 1) * fps))
#     end_f = int((e_time + 1) * fps) 
    
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     writer = None

#     # Xác định đối tượng mục tiêu dựa trên từ khóa
#     target_ids = []
#     q_low = query_en.lower()
#     if "car" in q_low or "vehicle" in q_low: target_ids.append(2)
#     if "motorcycle" in q_low or "bike" in q_low: target_ids.append(3)
#     if "person" in q_low or "man" in q_low: target_ids.append(0)
#     if not target_ids: target_ids = [2, 3, 5, 7]
    
#     for _ in range(start_f, end_f):
#         ret, frame = cap.read()
#         if not ret: break
        
#         y_res = yolo_model(frame, conf=0.4, verbose=False)
#         for r in y_res:
#             for box in r.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id in target_ids:
#                     b = box.xyxy[0].cpu().numpy().astype(int)
#                     cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
#                     cv2.putText(frame, f"Match: {score:.2f}", (b[0], b[1]-10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
#         if writer is None:
#             h, w = frame.shape[:2]
#             writer = cv2.VideoWriter(output_name, fourcc, fps, (w, h))
#         writer.write(frame)
        
#     if writer: writer.release()
#     cap.release()

# def format_time(seconds):
#     h = int(seconds // 3600)
#     m = int((seconds % 3600) // 60)
#     s = int(seconds % 60)
#     return f"{h:02d}:{m:02d}:{s:02d}"

# # --- 5. GIAO DIỆN CHÍNH ---
# st.title("🎬 CCTV AI Search - Hệ thống Truy vấn Đa Camera")
# col1, col2 = st.columns([1, 2])

# # Khởi tạo trang hiện tại trong session state
# if 'current_page' not in st.session_state:
#     st.session_state['current_page'] = 1

# with col1:
#     st.subheader("⚙️ Cấu hình")
#     v_file = st.file_uploader("Tải video", type=["mp4"])
    
#     if v_file:
#         temp_path = "temp_preview.mp4"
#         with open(temp_path, "wb") as f:
#             f.write(v_file.getbuffer())
        
#         # Đảm bảo file được ghi xong trước khi mở
#         if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
#             cap = cv2.VideoCapture(temp_path)
#             ret, first_frame = cap.read()
#             cap.release()
            
#             if ret:
#                 with st.expander("📐 Vẽ vùng quan tâm (ROI) - Click để tạo đa giác", expanded=True):
#                     st.write("Dùng chuột chấm điểm. Xong nhấn Lưu ROI.")
#                     bg_img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
                    
#                     canvas_w = 700
#                     canvas_h = int(bg_img.height * (canvas_w / bg_img.width))
                    
#                     canvas_result = st_canvas(
#                         fill_color="rgba(0, 255, 0, 0.3)",
#                         stroke_width=2,
#                         stroke_color="#00FF00",
#                         background_image=bg_img.resize((canvas_w, canvas_h)),
#                         update_streamlit=True,
#                         height=canvas_h,
#                         width=canvas_w,
#                         drawing_mode="polygon",
#                         key="roi_canvas",
#                     )
                    
#             if st.button("💾 Lưu ROI vào hệ thống"):
#                         if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
#                             # Lấy tọa độ
#                             obj = canvas_result.json_data["objects"][-1]
#                             path = obj["path"]
#                             pts = [[p[1], p[2]] for p in path if len(p) == 3]
                            
#                             scale = bg_img.width / canvas_w
#                             final_pts = [[int(x * scale), int(y * scale)] for x, y in pts]
                            
#                             cam_k = "cam01" if "cam01" in v_file.name.lower() else "cam02"
                            
#                             # Ghi file trực tiếp và đóng luồng ngay
#                             roi_data = {}
#                             if os.path.exists("roi.json"):
#                                 with open("roi.json", "r", encoding="utf-8") as f:
#                                     try: roi_data = json.load(f)
#                                     except: roi_data = {}
                            
#                             roi_data[cam_k] = {
#                                 "frame_w": bg_img.width,
#                                 "frame_h": bg_img.height,
#                                 "roi_polygon": final_pts
#                             }
                            
#                             # Ép hệ thống ghi xuống ổ cứng (flush)
#                             with open("roi.json", "w", encoding="utf-8") as f:
#                                 json.dump(roi_data, f, indent=2)
#                                 f.flush()
#                                 os.fsync(f.fileno()) 
                            
#                             st.success(f"🔥 Đã tạo file ROI thành công cho {cam_k}!")
#                             st.balloons() # Hiện hiệu ứng bóng bay để ăn mừng

#     query_vi = st.text_input("Mô tả đối tượng:", "xe con màu đen")
    
#     # Hàng nút bấm chức năng
#     btn_col1, btn_col2 = st.columns(2)
    
#     with btn_col1:
#         if st.button("🚀 Bắt đầu truy vấn") and v_file:
#             st.session_state['current_page'] = 1 # Reset về trang 1
            
#             q_en = GoogleTranslator(source='vi', target='en').translate(query_vi)
#             st.info(f"AI đang tìm kiếm: {q_en}")
#             st.session_state['current_query_en'] = q_en
            
#             # Quét video (hoặc lấy từ cache .pkl)
#             db = index_video(temp_path)
            
#             # Tìm kiếm CLIP
#             t_tokens = clip.tokenize([q_en]).to(device)
#             with torch.no_grad():
#                 t_feat = model.encode_text(t_tokens).float()
#                 t_feat /= t_feat.norm(dim=-1, keepdim=True)
            
#             sims = (torch.from_numpy(db["features"]).to(device) @ t_feat.T).cpu().numpy().flatten()
            
#             # --- LOGIC TÌM KHOẢNG THỜI GIAN (START/END) ---
#             found_events = []
#             visited = np.zeros_like(sims)
            
#             for _ in range(25): # Lấy tối đa 25 sự kiện
#                 idx = np.argmax(sims * (1 - visited))
#                 score = sims[idx]
#                 if score < 0.22: break
                
#                 # Dò tìm biên trái và phải (ngưỡng 80% so với đỉnh)
#                 threshold = score * 0.8
#                 start_idx = idx
#                 while start_idx > 0 and sims[start_idx-1] > threshold:
#                     start_idx -= 1
                
#                 end_idx = idx
#                 while end_idx < len(sims)-1 and sims[end_idx+1] > threshold:
#                     end_idx += 1
                
#                 found_events.append({
#                     "start": db["times"][start_idx],
#                     "end": db["times"][end_idx],
#                     "score": score
#                 })
#                 # Đánh dấu vùng này đã xử lý để không trùng lặp
#                 visited[max(0, start_idx-3):min(len(sims), end_idx+3)] = 1
            
#             st.session_state['search_results'] = found_events
#             st.session_state['v_path'] = temp_path

#     with btn_col2:
#         if st.button("🧹 Làm mới hệ thống"):
#             # Xóa các biến lưu trữ
#             for key in ['search_results', 'v_path', 'current_query_en', 'current_page']:
#                 if key in st.session_state:
#                     del st.session_state[key]
#             # Xóa các clip kết quả cũ
#             for f in os.listdir():
#                 if f.startswith("temp_res_") or f.startswith("res_p"):
#                     os.remove(f)
#             st.rerun()

# with col2:
#     st.subheader("🎯 Kết quả tìm kiếm")
#     if 'search_results' in st.session_state:
#         res_list = st.session_state['search_results']
#         items_per_page = 5
#         total_pages = (len(res_list) + items_per_page - 1) // items_per_page
        
#         # Điều hướng phân trang
#         p_c1, p_c2, p_c3 = st.columns([1, 2, 1])
#         with p_c1:
#             if st.button("⬅️ Trước") and st.session_state['current_page'] > 1:
#                 st.session_state['current_page'] -= 1
#                 st.rerun()
#         with p_c2:
#             st.write(f"Trang {st.session_state['current_page']} / {total_pages}")
#         with p_c3:
#             if st.button("Sau ➡️") and st.session_state['current_page'] < total_pages:
#                 st.session_state['current_page'] += 1
#                 st.rerun()

#         # Hiển thị 5 kết quả của trang hiện tại
#         start_idx = (st.session_state['current_page'] - 1) * items_per_page
#         end_idx = start_idx + items_per_page
        
#         for i, res in enumerate(res_list[start_idx : end_idx]):
#             s_t, e_t, sc = res['start'], res['end'], res['score']
#             time_label = f"{format_time(s_t)} ➔ {format_time(e_t)}"
            
#             with st.expander(f"🎬 Kết quả {start_idx + i + 1} | {time_label} (Score: {sc:.4f})"):
#                 out_name = f"res_p{st.session_state['current_page']}_{i}.mp4"
#                 render_result_clip(st.session_state['v_path'], s_t, e_t, sc, out_name, st.session_state['current_query_en'])
#                 st.video(out_name)

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

# --- 3. TRÍCH XUẤT ĐẶC TRƯNG (SỬA LỖI ĐỔI TÊN CACHE THEO VIDEO THỰC TẾ) ---
def index_video(video_path, original_name, sampling_sec=1):
    # Sử dụng tên gốc của video (ví dụ: cam01_clip.mp4) thay vì temp_preview.mp4 để đặt tên cache riêng biệt
    video_id = os.path.basename(original_name).split('.')[0]
    cache_file = f"cache_{video_id}.pkl"
    
    # Nếu đã có file cache trùng với video gốc này, nạp lại ngay lập tức
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
        
        # Sử dụng original_name để hàm apply_roi_by_camera nhận diện đúng cam01/cam02
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
    
    # Lưu vào đúng file cache mang tên video thực tế
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
        
    p_bar.empty()
    status_text.empty()
    return result

# --- 4. VẼ BOX MỤC TIÊU VÀ XUẤT CLIP (Xử lý Start/End) ---
def render_result_clip(video_in, s_time, e_time, score, output_name, query_en):
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    # Mở rộng biên 1 giây để clip không bị quá ngắn
    start_f = int(max(0, (s_time - 1) * fps))
    end_f = int((e_time + 1) * fps) 
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = None

    # Xác định đối tượng mục tiêu dựa trên từ khóa
    target_ids = []
    q_low = query_en.lower()
    if "car" in q_low or "vehicle" in q_low: target_ids.append(2)
    if "motorcycle" in q_low or "bike" in q_low: target_ids.append(3)
    if "person" in q_low or "man" in q_low: target_ids.append(0)
    if not target_ids: target_ids = [2, 3, 5, 7]
    
    for _ in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret: break
        
        y_res = yolo_model(frame, conf=0.4, verbose=False)
        for r in y_res:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in target_ids:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"Match: {score:.2f}", (b[0], b[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(output_name, fourcc, fps, (w, h))
        writer.write(frame)
        
    if writer: writer.release()
    cap.release()

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# --- 5. GIAO DIỆN CHÍNH ---
st.title("🎬 CCTV AI Search - Hệ thống Truy vấn Đa Camera")
col1, col2 = st.columns([1, 2])

# Khởi tạo trang hiện tại trong session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 1

with col1:
    st.subheader("⚙️ Cấu hình")
    v_file = st.file_uploader("Tải video", type=["mp4"])
    
    if v_file:
        temp_path = "temp_preview.mp4"
        with open(temp_path, "wb") as f:
            f.write(v_file.getbuffer())
        
        # Đảm bảo file được ghi xong trước khi mở
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
                    # Lấy tọa độ
                    obj = canvas_result.json_data["objects"][-1]
                    path = obj["path"]
                    pts = [[p[1], p[2]] for p in path if len(p) == 3]
                    
                    scale = bg_img.width / canvas_w
                    final_pts = [[int(x * scale), int(y * scale)] for x, y in pts]
                    
                    cam_k = "cam01" if "cam01" in v_file.name.lower() else "cam02"
                    
                    # Ghi file trực tiếp và đóng luồng ngay
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
                    
                    # Ép hệ thống ghi xuống ổ cứng (flush)
                    with open("roi.json", "w", encoding="utf-8") as f:
                        json.dump(roi_data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno()) 
                    
                    st.success(f"🔥 Đã tạo file ROI thành công cho {cam_k}!")
                    st.balloons() # Hiện hiệu ứng bóng bay để ăn mừng

    query_vi = st.text_input("Mô tả đối tượng:", "xe con màu đen")
    
    # Hàng nút bấm chức năng
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("🚀 Bắt đầu truy vấn") and v_file:
            st.session_state['current_page'] = 1 # Reset về trang 1
            
            q_en = GoogleTranslator(source='vi', target='en').translate(query_vi)
            st.info(f"AI đang tìm kiếm: {q_en}")
            st.session_state['current_query_en'] = q_en
            
            # TRUYỀN THÊM v_file.name VÀO HÀM ĐỂ PHÂN BIỆT CACHE RIÊNG BIỆT
            db = index_video(temp_path, v_file.name)
            
            # Tìm kiếm CLIP
            t_tokens = clip.tokenize([q_en]).to(device)
            with torch.no_grad():
                t_feat = model.encode_text(t_tokens).float()
                t_feat /= t_feat.norm(dim=-1, keepdim=True)
            
            sims = (torch.from_numpy(db["features"]).to(device) @ t_feat.T).cpu().numpy().flatten()
            
            # --- LOGIC TÌM KHOẢNG THỜI GIAN (START/END) ---
            found_events = []
            visited = np.zeros_like(sims)
            
            for _ in range(25): # Lấy tối đa 25 sự kiện
                idx = np.argmax(sims * (1 - visited))
                score = sims[idx]
                if score < 0.22: break
                
                # Dò tìm biên trái và phải (ngưỡng 80% so với đỉnh)
                threshold = score * 0.8
                start_idx = idx
                while start_idx > 0 and sims[start_idx-1] > threshold:
                    start_idx -= 1
                
                end_idx = idx
                while end_idx < len(sims)-1 and sims[end_idx+1] > threshold:
                    end_idx += 1
                
                found_events.append({
                    "start": db["times"][start_idx],
                    "end": db["times"][end_idx],
                    "score": score
                })
                # Đánh dấu vùng này đã xử lý để không trùng lặp
                visited[max(0, start_idx-3):min(len(sims), end_idx+3)] = 1
            
            st.session_state['search_results'] = found_events
            st.session_state['v_path'] = temp_path

    with btn_col2:
        if st.button("🧹 Làm mới hệ thống"):
            # Xóa các biến lưu trữ
            for key in ['search_results', 'v_path', 'current_query_en', 'current_page']:
                if key in st.session_state:
                    del st.session_state[key]
            # Xóa các clip kết quả cũ
            for f in os.listdir():
                if f.startswith("temp_res_") or f.startswith("res_p"):
                    os.remove(f)
            st.rerun()

with col2:
    st.subheader("🎯 Kết quả tìm kiếm")
    if 'search_results' in st.session_state:
        res_list = st.session_state['search_results']
        items_per_page = 5
        total_pages = (len(res_list) + items_per_page - 1) // items_per_page
        
        # Điều hướng phân trang
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

        # Hiển thị 5 kết quả của trang hiện tại
        start_idx = (st.session_state['current_page'] - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        for i, res in enumerate(res_list[start_idx : end_idx]):
            s_t, e_t, sc = res['start'], res['end'], res['score']
            time_label = f"{format_time(s_t)} ➔ {format_time(e_t)}"
            
            with st.expander(f"🎬 Kết quả {start_idx + i + 1} | {time_label} (Score: {sc:.4f})"):
                out_name = f"res_p{st.session_state['current_page']}_{i}.mp4"
                
                # --- 🛠️ ĐOẠN CẢI TIẾN AN TOÀN: CHỐNG XUNG ĐỘT KHÓA FILE ---
                # Chỉ xuất video clip nếu file đó chưa tồn tại hoặc dung lượng rỗng (0 bytes)
                if not os.path.exists(out_name) or os.path.getsize(out_name) == 0:
                    render_result_clip(st.session_state['v_path'], s_t, e_t, sc, out_name, st.session_state['current_query_en'])
                
                # Đảm bảo file tồn tại và sẵn sàng trên ổ cứng rồi mới nạp vào Streamlit
                if os.path.exists(out_name) and os.path.getsize(out_name) > 0:
                    st.video(out_name)
                else:
                    st.error(f"❌ Không thể đọc hoặc xuất file kết quả: {out_name}")