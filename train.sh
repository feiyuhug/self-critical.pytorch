#!/bin/bash
batch_size=60
iter_times=8
topic_num=500
gpu_num=1
#id=td_topic_num${topic_num}_bs${batch_size}_gpu${gpu_num}
old_id=td_newfeat_bs40_gpu4_rnn2048_2_seed342_epoch5_2
epoch=18
id=sc_bs${batch_size}_iters${iter_times}_${old_id}_epoch_${epoch}
#id=tmp
if [ ! -d "log/state_of_art/log_$id"  ]; then
    mkdir log/state_of_art/log_$id
fi

#--start_from log/log_td_topic_num500_bs100_gpu1_ftcnn --old_id td_topic_num500_bs100_gpu1_ftcnn \
CUDA_VISIBLE_DEVICES=5 python \
    -u train.py --id $id --caption_model topdown --topic_num $topic_num \
    --start_from ../neuraltalk2.pytorch/log/state_of_art/log_${old_id}/ \
    --old_id ${old_id}  \
    --self_critical_after 0 \
    --use_img 0 --use_topic 0 --use_fc 1\
    --rnn_size 2048 \
    --sample_weights ../neuraltalk2.pytorch/data/dataset/sample_weights_sqrt.npy \
    --input_json  ../neuraltalk2.pytorch/data/dataset/trainval_meta_v5k.json \
    --img_fold  ../neuraltalk2.pytorch/data/images --img_size 512 --img_csize 448 \
    --input_fc_dir  ../neuraltalk2.pytorch/data/dataset/train_val_feat448_log_${old_id}_fc \
    --input_att_dir  ../neuraltalk2.pytorch/data/dataset/train_val_feat448_log_${old_id}_att \
    --input_topic_h5 ../topic_model/classifier/log/log_warmup/prediction.h5 \
    --input_label_h5  ../neuraltalk2.pytorch/data/dataset/trainval_label.h5 \
    --cached_tokens trainval_ngrams-idxs \
    --batch_size $batch_size --iter_times $iter_times --gpu_num $gpu_num \
    --fix_rnn 0 --learning_rate 5e-5 --learning_rate_decay_start -1 \
    --checkpoint_path log/state_of_art/log_$id \
    --val_images_use 5000 --language_eval 1 --max_epochs -1  \
    2>&1 | tee log/state_of_art/log_$id/train.log





