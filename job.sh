#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/deepSVHN
#SBATCH --job-name="deepSVHN"
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out
#SBATCH --wait-all-nodes=1

PYTHON="/scratch/nas/4/norma/venv/bin/python"

$PYTHON experiments/experiment1_svhn_mpl.py