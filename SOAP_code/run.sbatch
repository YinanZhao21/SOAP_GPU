#!/bin/sh

#SBATCH --partition=debug-gpu
#SBATCH --time=00:15:00
#SBATCH --gpus=1

echo "I: full hostname: $(hostname -f)"

module load fosscuda/2020a
module load GSL/2.6
# if you need to know the allocated CUDA device, you can obtain it here:
echo "I: CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

python run_all_SOAP.py
echo "====="
