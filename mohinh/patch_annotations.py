import json
import os

# --- CẤU HÌNH ĐƯỜNG DẪN FILE ---
# Thái kiểm tra và chỉnh lại đúng tên/đường dẫn file nhãn gốc trên máy của bạn nhé
FILE_NHAN_GOC = r"data/all_annotations.jsonl"
FILE_NHAN_SACH = r"data/train_ready_annotations.jsonl"

if not os.path.exists(FILE_NHAN_GOC):
    print(f"❌ Không tìm thấy file nhãn gốc tại đường dẫn: {FILE_NHAN_GOC}")
    print("👉 Thái hãy kiểm tra lại vị trí và tên file nhãn (.jsonl) của bạn nha!")
    exit()

ready_records = []
error_count = 0

print("--- 🔄 Bắt đầu tiến trình quy đổi tọa độ nhãn dữ liệu... ---")

with open(FILE_NHAN_GOC, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f, 1):
        if not line.strip(): 
            continue
        try:
            data = json.loads(line)
            
            # 1. Trích xuất và bóc tách tọa độ hộp biên cũ [x, y, w, h]
            bbox = data.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Thực hiện phép tính toán chuyển đổi sang ma trận tuyệt đối [xmin, ymin, xmax, ymax]
                xmin = int(x)
                ymin = int(y)
                xmax = int(x + w)
                ymax = int(y + h)
                
                # Cập nhật lại trường bbox thành tọa độ tuyệt đối mới
                data["bbox"] = [xmin, ymin, xmax, ymax]
                
                # 2. Tự động giả lập cấu trúc đa giác 4 góc (Polygon Mask) cho thuật toán nâng cấp
                data["polygon"] = [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]
                ]
            
            # 3. Đồng bộ hóa trường thời gian video cho hệ thống
            segment = data.get("segment", [0.0, 10.0])
            data["t_start"] = segment[0]
            data["t_end"] = segment[1]
            
            # 4. Bảo vệ và đồng bộ thông tin Ngữ nghĩa (Query) tránh lỗi KeyError
            # Nếu có query_vi thì lấy, nếu không có thì lấy query, hoặc mặc định chuỗi bối cảnh chung
            query_text = data.get("query_vi", data.get("query", "Phương tiện di chuyển trong khu vực CCTV.")).strip()
            data["query"] = query_text
            
            ready_records.append(data)
            
        except Exception as e:
            # Bỏ qua 2 dòng lỗi cấu trúc văn bản (do dấu xuống dòng Enter lỗi phát sinh)
            error_count += 1
            continue

# Tiến hành ghi toàn bộ dữ liệu sạch đã quy đổi ra file mới
with open(FILE_SACH_NAME := FILE_NHAN_SACH, 'w', encoding='utf-8') as f_out:
    for record in ready_records:
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\n" + "="*60)
print("✅ TIẾN TRÌNH HOÀN TẤT THÀNH CÔNG!")
print("="*60)
print(f"• Tổng số dòng nhãn sạch đã xử lý: {len(ready_records)} dòng")
if error_count > 0:
    print(f"• Số dòng lỗi cấu trúc văn bản đã loại bỏ an toàn: {error_count} dòng")
print(f"👉 File nhãn sạch sẵn sàng huấn luyện nằm tại: {FILE_SACH_NAME}")
print("="*60)