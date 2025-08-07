#!/bin/bash
#SBATCH --job-name=mp3_to_wav
#SBATCH --output=logs/mp3_to_wav.out
#SBATCH --error=logs/mp3_to_wav.err
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=1G

module load Anaconda3/2024.02-1
source activate internship
python3 /users/ac4ma/Speech_Language_Internship/utils/mp3_to_wav.py