#!/bin/bash
#SBATCH --job-name=plot_speech_features
#SBATCH --output=logs/plot_speech_features.out
#SBATCH --error=logs/plot_speech_features.err
#SBATCH --ntasks=1
#SBATCH --time=00:50:00
#SBATCH --mem=1G

module load Anaconda3/2024.02-1
source activate internship
python3 /users/ac4ma/Speech_Language_Internship/feature_extraction/plot_speech_features.py