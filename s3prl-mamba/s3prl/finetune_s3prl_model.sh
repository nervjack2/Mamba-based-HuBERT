python_path=$1
downstream_name=$2
mode=$3 # train, evaluate, or resume
exp_name=$4
config_path=$5
upstream_ckpt=$6

if [ "$mode" = "resume" ]; then
    if [ "$downstream_name" = "asr" ]; then
        # Resume training on PR
        $python_path run_downstream.py -m train -e result/downstream/asr_finetune_$exp_name/
    else
        echo "Unknown downstream_name for resume."
    fi
elif [ "$mode" = "evaluate" ]; then
    if [ "$downstream_name" = "asr" ]; then
        # Resume training on PR
        $python_path run_downstream.py -m evaluate -e result/downstream/asr_finetune_$exp_name/dev-best.ckpt
    else
        echo "Unknown downstream_name for resume."
    fi
elif [ "$mode" = "train" ]; then
    if [ "$downstream_name" = "asr" ]; then
        # Fine-tune on Librispeech for ASR
        $python_path run_downstream.py -m train -u hubert_local \
            -d asr -c $config_path \
            -n asr_finetune_$exp_name -s last_hidden_state -k $upstream_ckpt --upstream_trainable
    elif [ "$downstream_name" = "asr_ted" ]; then
        # Fine-tune on Librispeech for ASR
        $python_path run_downstream.py -m train -u hubert_local \
            -d asr_ted -c $config_path \
            -n asr_ted_finetune_$exp_name -s last_hidden_state -k $upstream_ckpt --upstream_trainable
    else
        echo "Unknown downstream_name."
    fi
else 
    echo "Unknown mode."
fi