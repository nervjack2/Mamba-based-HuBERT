#### Hubert-Causal-Trans-small Training - Iter 1 #####
/home/nervjack2/anaconda3/envs/mamba-speech-ssl/bin/python3 fairseq/fairseq_cli/hydra_train.py                                              \
    task.data=/groups/nervjack2/mamba-ssl/manifest/               \
    task.label_dir=/groups/nervjack2/mamba-ssl/label/mfcc/      \
    task.labels='["km"]'                                                               \
    model.label_rate=100                                                               \
    dataset.max_tokens=5600000                                                         \
    +optimization.update_freq='[2]'                                                    \
    --config-dir /home/nervjack2/mamba-speech-ssl/fairseq-mamba/fairseq/examples/hubert/config/pretrain  \
    --config-name hubert_causal_trans_base_iter1.yaml

### Hubert-Causal-Trans-small Training - Iter 2 #####
# /home/nervjack2/anaconda3/envs/mamba-speech-ssl/bin/python3 fairseq/fairseq_cli/hydra_train.py                                              \
#     task.data=/groups/nervjack2/mamba-ssl/manifest/               \
#     task.label_dir=/groups/nervjack2/mamba-ssl/feature/hubert_causal_trans_base_iter1/      \
#     task.labels='["km"]'                                                               \
#     model.label_rate=50                                                               \
#     dataset.max_tokens=2800000                                                         \
#     +optimization.update_freq='[4]'                                                    \
#     --config-dir /home/nervjack2/mamba-speech-ssl/fairseq-mamba/fairseq/examples/hubert/config/pretrain  \
#     --config-name hubert_causal_trans_base_iter2.yaml \