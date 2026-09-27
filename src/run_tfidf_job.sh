#!/bin/bash
#SBATCH --job-name=tfidf_index
#SBATCH --output=tfidf_index.out
#SBATCH --error=tfidf_index.err
#SBATCH --partition=cpu_prod_long
#SBATCH --time=2-00:00:00  # 2 jours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module purge
module load python/3.9

source ~/venv/bin/activate

cd ~/AI_matcher

PYTHONPATH=. python src/update_tfidf_index.py