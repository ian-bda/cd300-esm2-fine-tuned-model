#!/bin/bash
#SBATCH --job-name=step6_local
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=UNLIMITED
#SBATCH --output=step6_interproscan_local_%j.out
#SBATCH --error=step6_interproscan_local_%j.err

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"

# Check if InterProScan is available
if [ ! -f "/home5/ibirchl/Bioinformatics_tools/interproscan-5.65-97.0/interproscan.sh" ]; then
    echo "ERROR: InterProScan not found at /home5/ibirchl/Bioinformatics_tools/interproscan-5.65-97.0"
    exit 1
fi

# Run the analysis
echo "Starting Step 6: InterProScan Domain Analysis (Local)"
echo "Using InterProScan at: /home5/ibirchl/Bioinformatics_tools/interproscan-5.65-97.0"

python scripts/step6_interproscan_local.py \
    --fasta data/Euarchontoglires_CD300s_complete.fasta \
    --interproscan /home5/ibirchl/Bioinformatics_tools/interproscan-5.65-97.0 \
    --output step6_interproscan_local \
    --skip_interproscan

echo "Step 6 completed!"
