#!/bin/bash
#SBATCH --job-name=step3_fine_tune_esm2_clusters
#SBATCH --output=step3_fine_tuned_model/step3_clusters_%j.out
#SBATCH --error=step3_fine_tuned_model/step3_clusters_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu

# Set up environment
export PATH="/home5/ibirchl/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# Run the fine-tuning script
echo "Starting ESM2 fine-tuning for sequence cluster classification..."
python scripts/step3_fine_tune_esm2_clusters.py \
    --data_path step1_data_preparation/cleaned_cd300_dataset.csv \
    --output_dir step3_fine_tuned_model \
    --num_epochs 20 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --max_length 1022

# Check output files
echo "Output files:"
ls -la step3_fine_tuned_model/

# Check model checkpoint size
echo "Model checkpoint size:"
du -sh step3_fine_tuned_model/best_model/checkpoint.pt

echo "End Time: $(date)"
echo "Job completed"
