#!/bin/sh
#PBS -q i1accs
#PBS -l select=1:ncpus=64:mpiprocs=1:ompthreads=64
##PBS -l walltime=12:00:00
#PBS -N GCN

module purge
module load nvhpc-nompi/22.2_cuda11.6

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nequip3

wandb offline
nequip-train minimal.yaml
#nequip-deploy build --train-dir results/aspirin/minimal/ aspirin.pth

