#!/bin/bash
#BSUB -P <PROJECT_ID>
#BSUB -W 1
#BSUB -nnodes 1
#BSUB -o out_cc.txt -e err_cc.txt

module load python/3.8.10 gcc/9.3.0 cuda/11.4.0 openblas/0.3.15-omp

## "--smpiargs=-gpu" is for enabling GPU-Direct RDMA
## 4 GPUs
jsrun -n4 -a1 -g1 -c1 --smpiargs="-gpu" ./a.out