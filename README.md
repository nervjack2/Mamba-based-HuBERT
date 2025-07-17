# Mamba-based HuBERT 

This is the official implementation of the paper [An Exploration of Mamba for Speech Self-Supervised Models](https://arxiv.org/abs/2506.12606).

## 🔥 Pre-training 
The script is set up for iteration 1 by default. If you proceed to iteration 2 after completing iteration 1, please uncomment the corresponding lines in the script.

1. Please run the following command to do preprocess.
```
cd fairseq-mamba
bash run_kmeans_preprocess.sh
```
2. Please run the following command for pre-trining.
```
cd fairseq-mamba
bash run_hubert_mamba_base.sh # Mamba Base
bash run_hubert_causal_trans_base.sh # Causal Transformer Base
bash run_hubert_trans_base.sh # Transformer Base
```

## 🎓 Fine-tuning 
Training Command
```
cd s3prl-mamba/s3prl/
bash finetune_s3prl_model.sh python3 DOWNSTREAM_NAME MODE EXP_NAME CONFIG_PATH CKPT_PTH
```

Fine-tuning for long-context ASR (Table 1 in the paper)
```
cd s3prl-mamba/s3prl/
bash finetune_s3prl_model.sh python3 asr_ted train hubert_trans_base_iter2 ./downstream/asr_ted/config_finetune.yaml hubert_trans_base_iter2.pt
```

Fine-tuning for Causal ASR (Table 2 in the paper)
```
cd s3prl-mamba/s3prl/
bash finetune_s3prl_model.sh python3 asr train causal_hubert_trans_base_iter2 ./downstream/asr/config_finetune.yaml causal_hubert_trans_base_iter2.pt
```

## 📚 Citation
If you find this codebase helpful, please consider citing our paper:
@article{lin2025exploration,
  title   = {An Exploration of Mamba for Speech Self‑Supervised Models},
  author  = {Tzu‑Quan Lin and Heng‑Cheng Kuo and Tzu‑Chieh Wei and Hsi‑Chun Cheng and Chun‑Wei Chen and Hsien‑Fu Hsiao and Yu Tsao and Hung‑yi Lee},
  journal = {arXiv preprint arXiv:2506.12606},
  year    = {2025}
}