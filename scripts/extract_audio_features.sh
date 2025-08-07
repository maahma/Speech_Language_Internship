#!/bin/bash
#SBATCH --job-name=extract_audio_features
#SBATCH --output=logs/extract_audio_features.out
#SBATCH --error=logs/extract_audio_features.err
#SBATCH --ntasks=1
#SBATCH --time=00:50:00
#SBATCH --mem=1G

module load Anaconda3/2024.02-1
source activate internship
python3 /users/ac4ma/Speech_Language_Internship/feature_extraction/extract_audio_features.py