#!/bin/bash
#SBATCH --job-name=run_mfa
#SBATCH --output=logs/run_mfa.out
#SBATCH --error=logs/run_mfa.err
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4   
#SBATCH --mem=1G

module load Anaconda3/2024.02-1
source activate aligner
python3 /users/ac4ma/Speech_Language_Internship/mfa_align/run_mfa.py