#!/bin/bash
#SBATCH --job-name=prepare_corpus
#SBATCH --output=logs/prepare_corpus.out
#SBATCH --error=logs/prepare_corpus.err
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=1G

module load Anaconda3/2024.02-1
source activate aligner
python3 /users/ac4ma/Speech_Language_Internship/mfa_align/prepare_corpus.py