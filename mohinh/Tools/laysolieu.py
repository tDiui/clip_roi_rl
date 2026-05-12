import json
import pandas as pd

def analyze_dataset(file_path):
    # Khởi tạo từ điển thống kê
    stats = {
        "Phương tiện màu sắc": 0,
        "Hành vi giao thông": 0,
        "Đối tượng đơn lẻ": 0,
        "Nhiều đối tượng": 0,
        "Truy vấn phức hợp": 0
    }
    examples = {k: "" for k in stats.keys()}
    total_count = 0
    
    print(f"--- 🔍 Đang xử lý dữ liệu: {file_path} ---")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read() # Đọc toàn bộ nội dung file
            
            decoder = json.JSONDecoder()
            pos = 0
            
            # Vòng lặp bóc tách từng object JSON dính nhau
            while pos < len(content):
                # Bỏ qua khoảng trắng, dấu phẩy hoặc dòng mới thừa
                while pos < len(content) and (content[pos].isspace() or content[pos] == ','):
                    pos += 1
                
                if pos >= len(content): break
                
                try:
                    data, pos = decoder.raw_decode(content, pos)
                    total_count += 1
                    
                    # --- PHÂN LOẠI LOGIC ---
                    q = data.get('query_vi', '').lower()
                    
                    colors = ['màu', 'trắng', 'đen', 'đỏ', 'xanh', 'vàng', 'xám', 'tím']
                    actions = ['đi thẳng', 'rẽ trái', 'rẽ phải', 'quay đầu', 'dừng', 'vượt', 'chạy']
                    multi = ['hai', 'ba', 'nhiều', 'nhóm', 'đoàn', 'các xe']
                    
                    has_color = any(c in q for c in colors)
                    has_action = any(a in q for a in actions)
                    is_multi = any(m in q for m in q.split())

                    # Phân loại độc lập để các cột đều có số liệu (theo ý anh quản lý)
                    if has_color: 
                        stats["Phương tiện màu sắc"] += 1
                        if not examples["Phương tiện màu sắc"]: examples["Phương tiện màu sắc"] = q
                    
                    if has_action: 
                        stats["Hành vi giao thông"] += 1
                        if not examples["Hành vi giao thông"]: examples["Hành vi giao thông"] = q
                    
                    if is_multi: 
                        stats["Nhiều đối tượng"] += 1
                        if not examples["Nhiều đối tượng"]: examples["Nhiều đối tượng"] = q
                    
                    if has_color and has_action: 
                        stats["Truy vấn phức hợp"] += 1
                        if not examples["Truy vấn phức hợp"]: examples["Truy vấn phức hợp"] = q
                    
                    if not has_color and not has_action and not is_multi:
                        stats["Đối tượng đơn lẻ"] += 1
                        if not examples["Đối tượng đơn lẻ"]: examples["Đối tượng đơn lẻ"] = q
                        
                except json.JSONDecodeError:
                    pos += 1 # Nếu lỗi thì nhảy qua 1 ký tự để tìm tiếp

        # Tạo bảng và hiển thị
        df_data = [{"Loại truy vấn": k, "Số lượng mẫu": v, "Ví dụ": examples[k]} for k, v in stats.items()]
        df = pd.DataFrame(df_data)
        
        print("\n" + "="*80)
        print(f"Bảng 3.2. THỐNG KÊ DỮ LIỆU (Tổng cộng: {total_count} mẫu)")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Xuất file Excel
        df.to_excel("Bao_cao_Thong_ke_VMR.xlsx", index=False)
        print(f"\n✅ Đã xuất báo cáo ra file: Bao_cao_Thong_ke_VMR.xlsx")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    analyze_dataset("data/train_annotations.jsonl")