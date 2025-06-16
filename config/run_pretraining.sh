#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate discovr

# Load CUDA
ml cuda/11.8

# Set paths
DATA_PATH="Path to Videos"
DATA_PATH_CSV="Path to Train CSV"
DATA_PATH_CSV_VAL="Path to Val CSV"
DATA_PATH_CSV_TEST="Path to Test CSV"
OUTPUT_DIR="Path to Output"

# Add code to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:Path to Code

# Run training
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 12000 scripts/run_mae_pretraining.py \
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
        --num_workers 4 \
        --use_torchcodec \
        --dino_out_dim 16384 \
        --use_combined_dino_swav \
        --use_video_dino \
        --mask_type multi_local \
        --local_mask_ratio 0.9 \
        --global_mask_ratio 0.9 \
        --num_local_views 4 