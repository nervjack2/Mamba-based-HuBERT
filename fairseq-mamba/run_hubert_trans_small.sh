#### Hubert-Trans-small Training - Iter 1 #####
python fairseq/fairseq_cli/hydra_train.py                                              \
    task.data=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest               \
    task.label_dir=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc      \
    task.labels='["km"]'                                                               \
    model.label_rate=100                                                               \
    dataset.max_tokens=5600000                                                         \
    +optimization.update_freq='[2]'                                                    \
    --config-dir /work/hckuo145/MambaSpeechSSL/fairseq/examples/hubert/config/pretrain \
    --config-name hubert_trans_small_iter1.yaml

### Hubert-Trans-small Training - Iter 2 #####
# python fairseq/fairseq_cli/hydra_train.py                                                             \
#     task.data=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest                              \
#     task.label_dir=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_trans_small_iter1 \
#     task.labels='["km"]'                                                                              \
#     model.label_rate=50                                                                               \
#     dataset.max_tokens=2800000                                                                        \
#     +optimization.update_freq='[4]'                                                                   \
#     --config-dir /work/hckuo145/MambaSpeechSSL/fairseq/examples/hubert/config/pretrain                \
#     --config-name hubert_trans_small_iter2.yaml