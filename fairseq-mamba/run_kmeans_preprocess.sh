##### Generate Dataset Manifest #####
python fairseq/examples/wav2vec/wav2vec_manifest.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/valid \
    --dest /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest --ext flac --valid-percent 0
mv /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/train.tsv /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/valid.tsv

python fairseq/examples/wav2vec/wav2vec_manifest.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/train \
    --dest /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest --ext flac --valid-percent 0

#### Hubert Preprocessing - Iter 1 #####
python fairseq/examples/hubert/simple_kmeans/dump_mfcc_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ valid 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc
python fairseq/examples/hubert/simple_kmeans/dump_mfcc_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ train 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc

python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc train 1 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc/kmeans.ckpt 100 --percent -1

python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc valid /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc
python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc train /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc

nshard=1
for rank in $(seq 0 $((nshard - 1))); do
  cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc/valid_${rank}_${nshard}.km
done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc/valid.km
for rank in $(seq 0 $((nshard - 1))); do
  cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc/train_${rank}_${nshard}.km
done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc/train.km

for x in $(seq 0 $((100 - 1))); do
  echo "$x 1"
done >> /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc/dict.km.txt

##### Hubert-Mamba-small Preprocessing - Iter 2 #####
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ valid  \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_mamba_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ train  \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_mamba_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1

# python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1 train 1 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1/kmeans.ckpt 500 --percent 0.1

# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1 valid /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1 train /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1

# nshard=1
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1/valid_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1/valid.km
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1/train_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1/train.km

# for x in $(seq 0 $((500 - 1))); do
#   echo "$x 1"
# done >> /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_small_iter1/dict.km.txt



##### Hubert-Mamba-base Preprocessing - Iter 2 #####
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ valid \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_mamba_base_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ train \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_mamba_base_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1

# python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1 train 1 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1/kmeans.ckpt 500 --percent 0.1

# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1 valid /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1 train /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1

# nshard=1
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1/valid_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1/valid.km
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1/train_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1/train.km

# for x in $(seq 0 $((500 - 1))); do
#   echo "$x 1"
# done >> /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba_base_iter1/dict.km.txt



##### Hubert-Trans-base Preprocessing - Iter 2 #####
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ valid \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_trans_base_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ train \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_trans_base_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1

# python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1 train 1 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1/kmeans.ckpt 500 --percent 0.1

# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1 valid /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1 train /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1

# nshard=1
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1/valid_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1/valid.km
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1/train_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1/train.km

# for x in $(seq 0 $((500 - 1))); do
#   echo "$x 1"
# done >> /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_base_iter1/dict.km.txt



##### Hubert-Trans-small Preprocessing - Iter 2 #####
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ valid \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_trans_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ train \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_trans_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1

# python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1 train 1 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1/kmeans.ckpt 500 --percent 0.1

# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1 valid /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1 train /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1

# nshard=1
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1/valid_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1/valid.km
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1/train_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1/train.km

# for x in $(seq 0 $((500 - 1))); do
#   echo "$x 1"
# done >> /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1/dict.km.txt



##### Hubert-Mamba+MLP-small Preprocessing - Iter 2 #####
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ valid \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_mamba+mlp_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ train \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_mamba+mlp_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1

# python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1 train 1 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1/kmeans.ckpt 500 --percent 0.1

# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1 valid /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1 train /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1

# nshard=1
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1/valid_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1/valid.km
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1/train_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1/train.km

# for x in $(seq 0 $((500 - 1))); do
#   echo "$x 1"
# done >> /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_mamba+mlp_small_iter1/dict.km.txt



##### Hubert-BiMamba-small Preprocessing - Iter 2 #####
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ valid \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_bimamba_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ train \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_bimamba_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1

# python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1 train 1 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1/kmeans.ckpt 500 --percent 0.1

# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1 valid /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1 train /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1

# nshard=1
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1/valid_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1/valid.km
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1/train_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1/train.km

# for x in $(seq 0 $((500 - 1))); do
#   echo "$x 1"
# done >> /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_small_iter1/dict.km.txt



#### Hubert-BiMamba+MLP-small Preprocessing - Iter 2 #####
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ valid \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_bimamba+mlp_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest/ train \
#     /work/hckuo145/MambaSpeechSSL/outputs/hubert_bimamba+mlp_small_iter1/checkpoints/checkpoint_best.pt 6 1 0                                \
#     /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1

# python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1 train 1 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1/kmeans.ckpt 500 --percent 0.1

# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1 valid /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1
# python fairseq/examples/hubert/simple_kmeans/dump_km_label.py /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1 train /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1/kmeans.ckpt 1 0 /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1

# nshard=1
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1/valid_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1/valid.km
# for rank in $(seq 0 $((nshard - 1))); do
#   cat /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1/train_${rank}_${nshard}.km
# done > /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1/train.km

# for x in $(seq 0 $((500 - 1))); do
#   echo "$x 1"
# done >> /work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba+mlp_small_iter1/dict.km.txt