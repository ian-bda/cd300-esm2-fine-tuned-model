#!/bin/bash
#SBATCH --job-name=step8_phylomorphospace_updated
#SBATCH --output=step8_phylomorphospace_updated_%j.out
#SBATCH --error=step8_phylomorphospace_updated_%j.err
#SBATCH --time=UNLIMITED
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load required modules
module load python/3.11
module load R/4.3.0

# Activate conda environment
source /home5/ibirchl/miniconda3/etc/profile.d/conda.sh
conda activate esm2_env

# Set working directory
cd /home5/ibirchl/Yoder_Lab/CD300_Project/Euarchontoglires/Fine-Tuned_ESM2_3B_Model

# Run Step 8 phylomorphospace analysis (updated)
echo "Starting Step 8: Phylomorphospace Analysis (Updated)"
echo "=================================================="

# Run the updated script
python scripts/step8_phylomorphospace_analysis_updated.py \
    --fasta_path data/euarchontoglires_cd300_sequences.fasta \
    --output_dir step8_phylomorphospace_analysis \
    --step4_dir step4_euarchontoglires_embeddings

echo "Step 8 phylomorphospace analysis (updated) completed!"
