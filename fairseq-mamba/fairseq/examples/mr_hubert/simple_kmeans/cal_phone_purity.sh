
ckpt_pth=$1
feat_dir=$2
out_file=$3

for layer_idx in {1..12}
do
    mkdir -p $feat_dir

    /home/nervjack2/anaconda3/envs/mamba-speech-ssl/bin/python3 dump_hubert_feature.py \
        /home/nervjack2/fairseq_hubert_kmeans/tsv/ \
        libri-100 $ckpt_pth \
        $layer_idx 1 0 $feat_dir

    /home/nervjack2/anaconda3/envs/mamba-speech-ssl/bin/python3 dump_hubert_feature.py \
        /home/nervjack2/fairseq_hubert_kmeans/tsv/ \
        valid $ckpt_pth \
        $layer_idx 1 0 $feat_dir

    /home/nervjack2/anaconda3/envs/mamba-speech-ssl/bin/python3 learn_kmeans.py $feat_dir \
            libri-100 1 $feat_dir/kmeans.pt 100 --percent 1.0

    /home/nervjack2/anaconda3/envs/mamba-speech-ssl/bin/python3 dump_km_label.py $feat_dir \
            valid $feat_dir/kmeans.pt \
            1 0 $feat_dir

    /home/nervjack2/anaconda3/envs/mamba-speech-ssl/bin/python3 split_npy.py \
     /home/nervjack2/fairseq_hubert_kmeans/tsv/valid.tsv $feat_dir/valid_0_1.km \
     $feat_dir/cluster_npy/

    /home/nervjack2/anaconda3/envs/mamba-speech-ssl/bin/python3 cal_purity.py \
    $feat_dir/cluster_npy/ /home/nervjack2/important_script/phone_purity/dev-clean-phone/ \
    100 41 $out_file

    rm -rf $feat_dir
done