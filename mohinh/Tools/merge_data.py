import os
import glob

def merge_all_annotations(data_dir="data"):
    # Tìm tất cả file annotations.jsonl trong các thư mục con
    search_pattern = os.path.join(data_dir, "**", "annotations.jsonl")
    jsonl_files = glob.glob(search_pattern, recursive=True)
    
    merged_content = []
    print(f"--- 📂 Đang gom dữ liệu từ {len(jsonl_files)} camera... ---")

    for file_path in jsonl_files:
        # Bỏ qua chính file tổng nếu nó đã lỡ tồn tại
        if "all_annotations.jsonl" in file_path:
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            merged_content.extend(lines)
            
    # Lưu file tổng vào thư mục data
    output_path = os.path.join(data_dir, "all_annotations.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(merged_content)
        
    print(f"✅ Đã tạo file tổng hợp tại: {output_path}")
    print(f"📊 Tổng số nhãn: {len(merged_content)}")

if __name__ == "__main__":
    merge_all_annotations()