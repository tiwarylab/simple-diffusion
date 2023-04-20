#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=16gb
#SBATCH --job-name="train"
#SBATCH --output=sample.out

module purge
module load cuda/11.6.2/gcc
. "/home/lherron/scratch.tiwary-prj/miniconda/etc/profile.d/conda.sh"
conda activate ML

python -u "sample.py"
