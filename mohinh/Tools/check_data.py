import os
import json
import glob

def check_orphan_labels(data_dir="data"):
    # Tìm tất cả các file annotations.jsonl trong các thư mục con (cam01, cam02...)
    search_pattern = os.path.join(data_dir, "**", "annotations.jsonl")
    jsonl_files = glob.glob(search_pattern, recursive=True)
    
    total_labels = 0
    valid_labels = 0
    missing_videos = []
    
    print(f"--- 🔍 Đang quét dữ liệu từ {len(jsonl_files)} file nhãn... ---")

    for file_path in jsonl_files:
        current_folder = os.path.dirname(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    total_labels += 1
                    
                    if 'image_path' in item:
                        # Ghép đường dẫn để kiểm tra file video thực tế
                        video_rel_path = item['image_path']
                        full_video_path = os.path.join(current_folder, video_rel_path)
                        
                        if os.path.exists(full_video_path):
                            valid_labels += 1
                        else:
                            missing_videos.append({
                                "file": file_path,
                                "line": line_num,
                                "video_missing": full_video_path
                            })
                except Exception as e:
                    print(f"❌ Lỗi đọc dòng {line_num} tại {file_path}: {e}")

    # --- BÁO CÁO KẾT QUẢ ---
    orphan_count = total_labels - valid_labels
    print("\n" + "="*50)
    print(f"📊 TỔNG KẾT DỮ LIỆU:")
    print(f"✅ Tổng số nhãn quét được: {total_labels}")
    print(f"🟢 Số nhãn có clip hợp lệ: {valid_labels}")
    print(f"🔴 Số nhãn 'mồ côi' (thiếu clip): {orphan_count}")
    
    if orphan_count > 0:
        print(f"\n⚠️ DANH SÁCH THIẾU (Top 5):")
        for i, m in enumerate(missing_videos[:5], 1):
            print(f"{i}. File: {m['file']} (Dòng {m['line']})")
            print(f"   ➜ Thiếu: {m['video_missing']}")
        
        # Gợi ý hành động
        if orphan_count > 5:
            print(f"... và {orphan_count - 5} mẫu khác.")
            
        print("\n💡 Lời khuyên: Thái nên xóa hoặc bổ sung clip cho các mẫu này")
        print("để tránh làm sai lệch tỉ lệ 80/10/10 khi train.")
    else:
        print("\n✨ Tuyệt vời! Dữ liệu của Thái hoàn toàn sạch sẽ.")
    print("="*50)

if __name__ == "__main__":
    check_orphan_labels()