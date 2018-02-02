TASET=5pts
JOB_NAME=hg_${DATASETaa}

MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
    --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=slimming --partition=RL_DSK_Face \
    python main.py --refine pruned.pth.tar --epochs 80
