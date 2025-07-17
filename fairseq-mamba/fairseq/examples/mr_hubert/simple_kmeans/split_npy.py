import os
import sys
import numpy as np

def main():
    tsv_path = sys.argv[1]
    label_path = sys.argv[2]
    output_dir = sys.argv[3]

    # 確保 output_dir 存在
    os.makedirs(output_dir, exist_ok=True)

    # 讀取檔案清單（從第二行開始）
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
    
    # 讀取標籤清單
    with open(label_path, 'r', encoding='utf-8') as f:
        label_lines = f.readlines()

    if len(lines) != len(label_lines):
        print(f"Error: Number of files ({len(lines)}) and labels ({len(label_lines)}) do not match.")
        sys.exit(1)

    # 一行一行處理
    for file_line, label_line in zip(lines, label_lines):
        file_path = file_line.strip()
        name = os.path.basename(file_path).split('.')[0]

        # 把label字串轉成整數list
        label_list = [int(x) for x in label_line.strip().split() if x.isdigit()]

        # 儲存成npy檔
        save_path = os.path.join(output_dir, f"{name}.npy")
        np.save(save_path, np.array(label_list, dtype=np.int32))

        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
