#!/bin/bash
#SBATCH --job-name=step4_euarchontoglires_embeddings
#SBATCH --output=step4_%j.out
#SBATCH --error=step4_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu

# Set up environment
export PATH="/home5/ibirchl/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directory
mkdir -p step4_euarchontoglires_embeddings

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Check Python version
echo "Python version:"
python --version

# Check PyTorch version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Uninstall and reinstall compatible versions
echo "Installing compatible package versions..."
pip uninstall -y transformers accelerate tokenizers huggingface-hub
pip install huggingface-hub==0.17.3
pip install tokenizers==0.13.3
pip install transformers==4.30.2
pip install accelerate==0.20.3

# Install PyTorch 2.1.0 (stable version)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install tf-keras
pip install biopython

# Run the Euarchontoglires embeddings script
echo "Starting Euarchontoglires embeddings generation..."
python scripts/step4_euarchontoglires_embeddings.py \
    --fasta_path data/Euarchontoglires_CD300s_complete.fasta \
    --model_path step3_fine_tuned_model/best_model \
    --output_dir step4_euarchontoglires_embeddings \
    --batch_size 8 \
    --max_length 1022 \
    --num_classes 10

# Check output files
echo "Output files:"
ls -la step4_euarchontoglires_embeddings/

# Check embedding file size
echo "Embeddings file size:"
du -sh step4_euarchontoglires_embeddings/euarchontoglires_embeddings.npy

echo "End Time: $(date)"
echo "Job completed"
