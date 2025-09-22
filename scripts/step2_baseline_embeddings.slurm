#!/bin/bash
#SBATCH --job-name=cd300_step2_embeddings
#SBATCH --output=step2_baseline_embeddings/step2_%j.out
#SBATCH --error=step2_baseline_embeddings/step2_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=gpu

# Load any necessary modules (adjust based on your system)
# module load python/3.10
# module load cuda/11.8

# Note: Using base conda environment (cd300_esm2 not found)

# Set working directory
cd /home5/ibirchl/Yoder_Lab/CD300_Project/Euarchontoglires/Fine-Tuned_ESM2_3B_Model

# Create output directory if it doesn't exist
mkdir -p step2_baseline_embeddings

# Print job information
echo "=========================================="
echo "CD300 Step 2: Baseline Embeddings"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Time limit: $SLURM_TIMELIMIT"
echo "=========================================="

# Start timing
start_time=$(date +%s)

# Run the script
echo "Starting Step 2 baseline embeddings generation..."
python scripts/step2_baseline_embeddings.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Step 2 completed successfully!"
else
    echo "Step 2 failed with exit code $?"
    exit 1
fi

# End timing
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total runtime: ${duration} seconds ($(($duration / 60)) minutes)"

echo "Job completed at $(date)"
