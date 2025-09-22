#!/bin/bash
#SBATCH --job-name=step7_esmfold_structure
#SBATCH --output=step7_esmfold_structure_%j.out
#SBATCH --error=step7_esmfold_structure_%j.err
#SBATCH --time=UNLIMITED
#SBATCH --partition=standard
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Set up environment
export PATH="/home5/ibirchl/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directory
mkdir -p step7_structure_analysis

# Install required packages
pip install transformers[esmfold] biopython plotly kaleido

# Run the script
python scripts/step7_esmfold_structure_analysis.py \
    --fasta_path data/Euarchontoglires_CD300s_complete.fasta \
    --output_dir step7_structure_analysis \
    --step4_dir step4_euarchontoglires_embeddings \
    --step5_dir step5_enhanced_novel_variant_detection \
    --device cpu \
    --max_sequences 351
