#!/bin/bash
#SBATCH --job-name=enhanced_step5
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=enhanced_step5_%j.out
#SBATCH --error=enhanced_step5_%j.err

# Load modules and set up environment
export PATH="/home5/ibirchl/miniconda3/bin:$PATH"
source activate base

# Create output directory
mkdir -p step5_enhanced_novel_variant_detection

# Install required packages if needed
pip install scikit-learn matplotlib seaborn biopython

# Run enhanced novel variant detection
echo "Starting enhanced novel variant detection..."
python scripts/step5_enhanced_novel_variant_detection.py \
    --embeddings step4_euarchontoglires_embeddings/euarchontoglires_embeddings.npy \
    --predictions step4_euarchontoglires_embeddings/euarchontoglires_predictions.csv \
    --fasta data/Euarchontoglires_CD300s_complete.fasta \
    --output_dir step5_enhanced_novel_variant_detection

echo "Enhanced novel variant detection completed!"
echo "Results saved to: step5_enhanced_novel_variant_detection/"










