import json
import random
import os

def split_vmr_dataset(input_file="data/all_annotations.jsonl", output_dir="data"):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Đảm bảo kết quả giống nhau mỗi lần chạy để đối chiếu
    random.seed(42) 
    random.shuffle(lines)
    
    total = len(lines)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    datasets = {
        "train": lines[:train_end],
        "val": lines[train_end:val_end],
        "test": lines[val_end:]
    }
    
    for name, content in datasets.items():
        file_path = os.path.join(output_dir, f"{name}_annotations.jsonl")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(content)
        print(f"✅ Đã tạo {name}_annotations.jsonl: {len(content)} mẫu")

if __name__ == "__main__":
    split_vmr_dataset()