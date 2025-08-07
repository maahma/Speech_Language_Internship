#!/bin/bash
#SBATCH --job-name=extract_sbert_embeddings
#SBATCH --output=logs/extract_sbert_embeddings.out
#SBATCH --error=logs/extract_sbert_embeddings.err
#SBATCH --ntasks=1
#SBATCH --time=00:50:00
#SBATCH --mem=1G

module load Anaconda3/2024.02-1
source activate internship
python3 /users/ac4ma/Speech_Language_Internship/feature_extraction/extract_sbert_embeddings.py