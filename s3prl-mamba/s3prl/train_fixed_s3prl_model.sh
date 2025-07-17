python_path=$1
upstream_name=$2
downstream_name=$3
mode=${4:-"train"}  # 預設為 "train"，也可以設置為 "resume" 或 "evaluate"
ckpt_pth=$5
exp_name=$6

# Need to change the path manually when running on different machine
voxceleb1=/livingrooms/nervjack2/voxceleb1/ # Needed by: Evaluation of ASV
quesst14Database=/groups/public/benchmark/quesst14Database/ # Needed by QbE
S3PRL_DIR=/home/nervjack2/s3prl-merge/s3prl/s3prl # Needed by QbE

if [ "$mode" = "resume" ]; then
    if [ "$downstream_name" = "pr" ]; then
        # Resume training on PR
        $python_path run_downstream.py -m train -e result/downstream/pr_$exp_name/
    elif [ "$downstream_name" = "sid" ]; then
        # Resume training on SID
        $python_path run_downstream.py -m train -e result/downstream/sid_$exp_name/
    elif [ "$downstream_name" = "er" ]; then
        # Resume training on ER
        for test_fold in fold1 fold2 fold3 fold4 fold5;
        do
            # Train frmo scratch if exp dir not exis
            if [ -d "result/downstream/er_${exp_name}_$test_fold/" ]; then
                $python_path run_downstream.py -m train -e result/downstream/er_${exp_name}_$test_fold/
            else
                $python_path run_downstream.py -m train -u $upstream_name \
                    -d emotion \
                    -n er_${exp_name}_$test_fold -k $ckpt_pth -c downstream/emotion/config.yaml \
                    -o "config.downstream_expert.datarc.test_fold='$test_fold'" --upstream_feature_normalize
            fi
        done
    elif [ "$downstream_name" = "ic" ]; then
        # Resume training on IC
        $python_path run_downstream.py -m train -e result/downstream/ic_$exp_name/
    elif [ "$downstream_name" = "asr" ]; then
        # Resume training on ASR
        $python_path run_downstream.py -m train -e result/downstream/asr_$exp_name/
    elif [ "$downstream_name" = "asv" ]; then
        # Resume training on ASV
        $python_path run_downstream.py -m train -e result/downstream/asv_$exp_name/
    elif [ "$downstream_name" = "ks" ]; then
        # Resume training on KS
        $python_path run_downstream.py -m train -e result/downstream/ks_$exp_name/
    elif [ "$downstream_name" = "sd" ]; then
        # Resume training on SD
        $python_path run_downstream.py -m train -e result/downstream/sd_$exp_name/
    elif [ "$downstream_name" = "sf" ]; then
        # Resume training on SF
        $python_path run_downstream.py -m train -e result/downstream/sf_$exp_name/
    else
        echo "Unknown downstream_name for resume."
    fi
elif [ "$mode" = "evaluate" ]; then
    if [ "$downstream_name" = "pr" ]; then
        # Evaluate on PR
        $python_path run_downstream.py -m evaluate -e result/downstream/pr_$exp_name/dev-best.ckpt
    elif [ "$downstream_name" = "sid" ]; then
        # Evaluate on SID
        $python_path run_downstream.py -m evaluate -e result/downstream/sid_$exp_name/dev-best.ckpt
    elif [ "$downstream_name" = "er" ]; then
        # Evaluate on ER
        for test_fold in fold1 fold2 fold3 fold4 fold5;
        do
            $python_path run_downstream.py -m evaluate -e result/downstream/er_${exp_name}_$test_fold/dev-best.ckpt
        done
    elif [ "$downstream_name" = "ic" ]; then
        # Evaluate on IC
        $python_path run_downstream.py -m evaluate -e result/downstream/ic_$exp_name/dev-best.ckpt
    elif [ "$downstream_name" = "asr" ]; then
        # Evaluate on ASR
        $python_path run_downstream.py -m evaluate -e result/downstream/asr_$exp_name/dev-clean-best.ckpt -t "test-clean"
    elif [ "$downstream_name" = "asv" ]; then
        # Evaluate on ASV
        ./downstream/sv_voxceleb1/test_expdir.sh result/downstream/asv_$exp_name/ $voxceleb1 $python_path
    elif [ "$downstream_name" = "ks" ]; then
        # Evaluate on KS
        $python_path run_downstream.py -m evaluate -e result/downstream/ks_$exp_name/dev-best.ckpt
    elif [ "$downstream_name" = "sd" ]; then
        # Evaluate on SD
        $python_path run_downstream.py -m evaluate -e result/downstream/sd_$exp_name/best-states-dev.ckpt
    elif [ "$downstream_name" = "sf" ]; then
        # Evaluate on SF
        $python_path run_downstream.py -m evaluate -e result/downstream/sf_$exp_name/dev-best.ckpt
    elif [ "$downstream_name" = "qbe" ]; then
        cd ${quesst14Database}/scoring
        for layer in {0..12}; do
            # dev
            ./score-TWV-Cnxe.sh ${S3PRL_DIR}/result/downstream/qbe_${exp_name}_${layer}_dev \
                groundtruth_quesst14_dev -10
            # test
            ./score-TWV-Cnxe.sh ${S3PRL_DIR}/result/downstream/qbe_${exp_name}_${layer}_test \
                groundtruth_quesst14_eval -10
        done
    else
        echo "Unknown downstream_name for evaluate."
    fi
elif [ "$mode" = "train" ]; then
    if [ "$downstream_name" = "pr" ]; then
        # Train on PR with upstream fixed
        $python_path run_downstream.py -m train -u $upstream_name \
            -d ctc -c downstream/ctc/libriphone.yaml \
            -n pr_$exp_name -k $ckpt_pth --upstream_feature_normalize
    elif [ "$downstream_name" = "sid" ]; then
        # Train on SID with upstream fixed
        $python_path run_downstream.py -m train -u $upstream_name \
            -d voxceleb1 \
           -n sid_$exp_name -k $ckpt_pth --upstream_feature_normalize
    elif [ "$downstream_name" = "er" ]; then
        # Train on ER with upstream fixed
        for test_fold in fold1 fold2 fold3 fold4 fold5;
        do
            $python_path run_downstream.py -m train -u $upstream_name \
                -d emotion \
                -n er_${exp_name}_$test_fold -k $ckpt_pth -c downstream/emotion/config.yaml \
                -o "config.downstream_expert.datarc.test_fold='$test_fold'" --upstream_feature_normalize
        done
    elif [ "$downstream_name" = "ic" ]; then
        # Train on IC with upstream fixed
        $python_path run_downstream.py -m train -u $upstream_name \
            -d fluent_commands \
            -n ic_$exp_name -k $ckpt_pth --upstream_feature_normalize
    elif [ "$downstream_name" = "asr" ]; then
        # Train on ASR with upstream fixed
        $python_path run_downstream.py -m train -u $upstream_name \
            -d asr \
            -n asr_$exp_name -k $ckpt_pth --upstream_feature_normalize
    elif [ "$downstream_name" = "asv" ]; then
        # Train on ASV with upstream fixed
        $python_path run_downstream.py -m train -u $upstream_name \
            -d sv_voxceleb1 \
            -n asv_$exp_name -k $ckpt_pth --upstream_feature_normalize 
    elif [ "$downstream_name" = "ks" ]; then
        # Train on KS with upstream fixed
        $python_path run_downstream.py -m train -u $upstream_name \
            -d speech_commands \
            -n ks_$exp_name -k $ckpt_pth --upstream_feature_normalize
    elif [ "$downstream_name" = "sd" ]; then
        # Train on SD with upstream fixed
        $python_path run_downstream.py -m train -u $upstream_name \
            -d diarization \
            -n sd_$exp_name -k $ckpt_pth --upstream_feature_normalize  
    elif [ "$downstream_name" = "sf" ]; then
        # Train on SF with upstream fixed
        $python_path run_downstream.py -m train -u $upstream_name \
            -d ctc \
            -n sf_$exp_name -k $ckpt_pth -c downstream/ctc/snips.yaml --upstream_feature_normalize               
    elif [ "$downstream_name" = "qbe" ]; then 
        # Run DTW for QbE
        dist_fn=cosine
        for layer in {0..12}; do
            # dev
            $python_path run_downstream.py -m evaluate -t "dev" -u $upstream_name -l ${layer} \
                -d quesst14_dtw -n qbe_${exp_name}_${layer}_dev \
                -k $ckpt_pth \
                -o config.downstream_expert.dtwrc.dist_method=$dist_fn --upstream_feature_normalize
            # test
            $python_path run_downstream.py -m evaluate -t "test" -u $upstream_name -l ${layer} \
                -d quesst14_dtw -n qbe_${exp_name}_${layer}_test \
                -k $ckpt_pth \
                -o config.downstream_expert.dtwrc.dist_method=$dist_fn --upstream_feature_normalize
        done
    else
        echo "Unknown downstream_name."
    fi
else 
    echo "Unknown mode."
fi