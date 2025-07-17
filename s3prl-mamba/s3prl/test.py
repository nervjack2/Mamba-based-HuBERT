import os
import torchaudio

file_path = "/livingrooms/nervjack2/dataset/LibriSpeech/train-clean-100/374/180298/374-180298-0021.flac"

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"File does not exist: {file_path}")
else:
    try:
        # 尝试加载音频文件
        info = torchaudio.info(file_path)
        print(f"File loaded successfully: {file_path}")
        print(f"Number of frames: {info.num_frames}")
    except Exception as e:
        # print(f"Error loading file: {file_path}")
        print(e)
