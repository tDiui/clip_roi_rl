import json
import pandas as pd

def export_coordinates_to_csv(jsonl_path, output_csv):
    results = []
    print(f"--- 🔍 Đang trích xuất tọa độ từ: {jsonl_path} ---")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            # Đọc toàn bộ nội dung để xử lý trường hợp JSON dính liền
            content = f.read()
            
            decoder = json.JSONDecoder()
            pos = 0
            count = 0
            
            while pos < len(content):
                # Nhảy qua khoảng trắng hoặc dấu phẩy thừa giữa các object
                while pos < len(content) and (content[pos].isspace() or content[pos] == ','):
                    pos += 1
                
                if pos >= len(content): break
                
                try:
                    # Bóc tách từng object JSON
                    data, pos = decoder.raw_decode(content, pos)
                    
                    # Lấy bbox: [x, y, w, h]
                    bbox = data.get('bbox', [0, 0, 0, 0])
                    
                    results.append({
                        "clip_id": data.get('clip_id'),
                        "class": data.get('class_name'),
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": bbox[2],
                        "h": bbox[3],
                        "query": data.get('query_vi')
                    })
                    count += 1
                except json.JSONDecodeError:
                    pos += 1 # Nhảy qua ký tự lỗi nếu có

        # Xuất ra CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✅ Thành công! Đã xuất {count} dòng tọa độ ra file: {output_csv}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

# Chạy lệnh
export_coordinates_to_csv("data/train_annotations.jsonl", "Toa_do_Toan_bo_Dataset.csv")