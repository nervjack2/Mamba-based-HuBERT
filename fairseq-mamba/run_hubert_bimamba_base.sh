##### Hubert-BiMamba-base Training - Iter 1 #####
# python fairseq/fairseq_cli/hydra_train.py                                                            \
#     task.data=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest                             \
#     task.label_dir=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/mfcc                    \
#     task.labels='["km"]'                                                                             \
#     model.label_rate=100                                                                             \
#     dataset.max_tokens=5600000                                                                       \
#     +optimization.update_freq='[2]'                                                                  \
#     --config-dir /work/hckuo145/MambaSpeechSSL/fairseq-mamba/fairseq/examples/hubert/config/pretrain \
#     --config-name hubert_bimamba_base_iter1.yaml

##### Hubert-BiMamba-base Training - Iter 2 -Init #####
# python fairseq/fairseq_cli/hydra_train.py                                                              \
#     task.data=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest                               \
#     task.label_dir=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_base_iter1 \
#     task.labels='["km"]'                                                                               \
#     model.label_rate=50                                                                                \
#     dataset.max_tokens=280000                                                                          \
#     distributed_training.distributed_world_size=1                                                      \
#     checkpoint.save_interval_updates=200                                                               \
#     +optimization.update_freq='[4]'                                                                    \
#     --config-dir /work/hckuo145/MambaSpeechSSL/fairseq-mamba/fairseq/examples/hubert/config/pretrain   \
#     --config-name hubert_bimamba_base_iter2.yaml

### Hubert-BiMamba-base Training - Iter 2 #####
python fairseq/fairseq_cli/hydra_train.py                                                                                                \
    --config-dir /work/hckuo145/MambaSpeechSSL/fairseq-mamba/fairseq/examples/hubert/config/pretrain                                     \
    --config-name hubert_bimamba_base_iter2.yaml                                                                                         \
    task.data=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/manifest                                                                 \
    task.label_dir=/work/hckuo145/MambaSpeechSSL/dataset/LibriSpeech/feature/hubert_bimamba_base_iter1                                   \
    task.labels='["km"]'                                                                                                                 \
    model.label_rate=50                                                                                                                  \
    dataset.max_tokens=1400000                                                                                                           \
    distributed_training.distributed_world_size=8                                                                                        \
    checkpoint.restore_file=/work/hckuo145/MambaSpeechSSL/fairseq-mamba/outputs/hubert_bimamba_base_iter2/checkpoints/checkpoint_init.pt \
    checkpoint.reset_optimizer=True                                                                                                      \
    checkpoint.reset_lr_scheduler=True                                                                                                   \
    checkpoint.reset_meters=True                                                                                                         \
    checkpoint.reset_dataloader=True    
