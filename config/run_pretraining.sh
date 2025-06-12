#!/bin/bash

# Set paths
DATA_PATH="/data/biomedia1/ssl_resized_224_cropped_12_March"
DATA_PATH_CSV="/data/biomedia1/ssl_split_csv_files/ssl_split_csv_files/site_01_train_21_March.csv"
DATA_PATH_CSV_VAL="/data/biomedia1/ssl_split_csv_files/ssl_split_csv_files/site_01_val_21_March.csv"
DATA_PATH_CSV_TEST="/data/biomedia1/ssl_split_csv_files/ssl_split_csv_files/site_01_test_21_March.csv"
OUTPUT_DIR="/data/biomedia1/discovr/results/DISCOVR_PRETRAIN_SITE_01"

# Add code to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/data/biomedia1/discovr/code

# Run training
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 \
        --master_port 12000 /data/biomedia1/discovr/code/discovr/scripts/run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_path_csv ${DATA_PATH_CSV} \
        --data_path_val ${DATA_PATH_CSV_VAL} \
        --data_path_test ${DATA_PATH_CSV_TEST} \
        --mask_type tube \
        --loss_func SWAV \
        --run_name DISCOVR_PRETRAIN \
        --mask_ratio 0.9 \
        --target_type mlp \
        --model pretrain_videomae_base_patch16_224 \
        --input_size 112 \
        --decoder_depth 4 \
        --batch_size 16 \
        --num_frames 64 \
        --sampling_rate 3 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 400 \
        --output_dir ${OUTPUT_DIR} \
        --normlize_target True \
        --num_prototypes 3000 \
        --sinkhorn_eps 0.05 \
        --sinkhorn_iterations 10 \
        --augmentation multi_scale_crop \
        --tokenizer_type default \
        --num_workers 8 \
        --use_torchcodec \
        --dino_out_dim 16384 \
        --use_combined_dino_swav \
        --use_video_dino \
        --mask_type multi_local \
        --local_mask_ratio 0.9 \
        --global_mask_ratio 0.9 \
        --num_local_views 4 