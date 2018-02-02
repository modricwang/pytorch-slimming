#!/usr/bin/env bash


MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
    --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=1 --partition=VIBackEnd1 \
    python main.py  \
    -shuffle \
    -train_record \
    -model vgg \
    -data_dir /mnt/lustre/share/wangyiru/car_attr_list/v3 \
    -save_path checkpoints \
    -color_classes 10 \
    -type_classes 7 \
    -n_epochs 15 \
    -learn_rate 1e-3 \
    -batch_size 64 \
    -workers 32 \
    -nGPU 4 \
2>&1 | tee train.log
