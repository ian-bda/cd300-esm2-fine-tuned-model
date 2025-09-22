#!/usr/bin/env python3
"""
Step 3: Fine-Tune ESM2 650M for CD300 Sequence Cluster Classification

This script fine-tunes the ESM2 650M model to classify CD300 protein sequences
into their sequence clusters (from Step 1) rather than unreliable homology labels.

Key features:
- Uses sequence clusters from Step 1 as training labels
- Multi-class classification for CD300 sequence clusters
- Early stopping and learning rate scheduling
- GPU acceleration
- Comprehensive evaluation metrics
- Model checkpointing
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    EsmModel, EsmTokenizer
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step3_fine_tune_esm2_clusters.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CD300Dataset(Dataset):
    """Dataset class for CD300 protein sequences and their sequence clusters."""
    
    def __init__(self, sequences: List[str], labels: List[int], tokenizer, max_length: int = 1022):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CD300Classifier(nn.Module):
    """Custom classifier head for CD300 sequence cluster classification."""
    
    def __init__(self, esm_model, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.esm = esm_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(esm_model.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get ESM embeddings
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

def load_and_prepare_data(data_path: str) -> Tuple[List[str], List[int], Dict[int, str]]:
    """Load and prepare the CD300 dataset for training, using sequence clusters as labels."""
    logger.info(f"Loading data from {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} sequences")
    
    # Define Euarchontoglires orders
    euarchontoglires_orders = [
        'Primates', 'Rodentia', 'Lagomorpha', 'Scandentia', 'Dermoptera'
    ]
    
    # Remove Euarchontoglires sequences (already have them in Euarchontoglires_CD300s_complete.fasta)
    logger.info("Removing Euarchontoglires sequences for final validation...")
    euarchontoglires_mask = df['Order'].isin(euarchontoglires_orders)
    euarchontoglires_count = euarchontoglires_mask.sum()
    training_df = df[~euarchontoglires_mask].copy()
    
    logger.info(f"Euarchontoglires sequences removed: {euarchontoglires_count}")
    logger.info(f"Training sequences (non-Euarchontoglires): {len(training_df)}")
    
    # Use training data (non-Euarchontoglires) for model training
    df = training_df
    
    # Get unique sequence clusters and create label mapping
    unique_clusters = df['cluster_label'].unique()
    # Remove any NaN values
    unique_clusters = [c for c in unique_clusters if pd.notna(c)]
    
    label_to_cluster = {i: cluster for i, cluster in enumerate(sorted(unique_clusters))}
    cluster_to_label = {cluster: i for i, cluster in label_to_cluster.items()}
    
    logger.info(f"Found {len(unique_clusters)} unique sequence clusters in training data")
    logger.info("Cluster distribution:")
    for cluster, count in df['cluster_label'].value_counts().head(10).items():
        logger.info(f"  {cluster}: {count}")
    
    # Filter out clusters with very few samples (less than 3) for better training
    min_samples_per_class = 3
    cluster_counts = df['cluster_label'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_samples_per_class].index.tolist()
    
    if len(valid_clusters) < len(unique_clusters):
        logger.warning(f"Filtering out {len(unique_clusters) - len(valid_clusters)} clusters with < {min_samples_per_class} samples")
        logger.warning("Removed clusters:")
        for cluster in set(unique_clusters) - set(valid_clusters):
            count = cluster_counts[cluster]
            logger.warning(f"  {cluster}: {count} samples")
        
        # Filter dataframe to only include valid clusters
        df = df[df['cluster_label'].isin(valid_clusters)]
        
        # Recreate label mapping with filtered data
        unique_clusters = df['cluster_label'].unique()
        unique_clusters = [c for c in unique_clusters if pd.notna(c)]
        label_to_cluster = {i: cluster for i, cluster in enumerate(sorted(unique_clusters))}
        cluster_to_label = {cluster: i for i, cluster in label_to_cluster.items()}
        
        logger.info(f"After filtering: {len(unique_clusters)} clusters with ≥{min_samples_per_class} samples each")
    
    # Prepare sequences and labels
    sequences = df['Protein Sequence'].tolist()
    labels = [cluster_to_label[cluster] for cluster in df['cluster_label']]
    
    # Filter out sequences that are too long or contain invalid characters
    valid_sequences = []
    valid_labels = []
    
    for seq, label in zip(sequences, labels):
        if isinstance(seq, str) and len(seq) <= 1022 and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in seq):
            valid_sequences.append(seq)
            valid_labels.append(label)
    
    logger.info(f"After filtering: {len(valid_sequences)} valid sequences for training")
    
    return valid_sequences, valid_labels, label_to_cluster

def train_model(
    sequences: List[str],
    labels: List[int],
    label_to_cluster: Dict[int, str],
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 1022
):
    """Train the ESM2 model for CD300 sequence cluster classification."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info("Loading ESM2 tokenizer and model...")
    model_name = "facebook/esm2_t33_650M_UR50D"  # Using ESM2 650M model
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    
    # Load base ESM model (let GradScaler handle FP16)
    esm_model = EsmModel.from_pretrained(model_name)
    
    # Create custom classifier
    num_classes = len(label_to_cluster)
    model = CD300Classifier(esm_model, num_classes)
    
    model.to(device)
    
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Number of classes: {num_classes}")
    
    # Check for classes with insufficient samples for stratification
    from collections import Counter
    label_counts = Counter(labels)
    min_samples = min(label_counts.values())
    
    if min_samples < 2:
        logger.warning(f"Some classes have only {min_samples} sample(s). Cannot use stratified split.")
        logger.warning("Using random split instead of stratified split.")
        # Use random split without stratification
        train_sequences, val_sequences, train_labels, val_labels = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
    else:
        # Use stratified split
        train_sequences, val_sequences, train_labels, val_labels = train_test_split(
            sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )
    
    logger.info(f"Training set: {len(train_sequences)} sequences")
    logger.info(f"Validation set: {len(val_sequences)} sequences")
    
    # Create datasets
    train_dataset = CD300Dataset(train_sequences, train_labels, tokenizer, max_length)
    val_dataset = CD300Dataset(val_sequences, val_labels, tokenizer, max_length)
    
    # Create data loaders with smaller batch size for memory efficiency
    effective_batch_size = min(batch_size, 1)  # Force batch size to 1 for memory
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False, pin_memory=True)
    
    logger.info(f"Using effective batch size: {effective_batch_size} (requested: {batch_size})")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(outputs['logits'], dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
            
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}"
            })
            
            # Clear GPU cache periodically to prevent memory buildup
            if torch.cuda.is_available() and (len(train_pbar) % 100 == 0):
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Log metrics
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save model
            model_path = os.path.join(output_dir, 'best_model')
            os.makedirs(model_path, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'label_to_cluster': label_to_cluster,
                'num_classes': num_classes
            }, os.path.join(model_path, 'checkpoint.pt'))
            
            # Save tokenizer
            tokenizer.save_pretrained(model_path)
            
            logger.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Generate evaluation report
    generate_evaluation_report(
        all_predictions, all_labels, label_to_cluster, 
        train_losses, val_losses, val_accuracies, output_dir
    )
    
    return model, tokenizer

def generate_evaluation_report(
    predictions: List[int],
    true_labels: List[int],
    label_to_cluster: Dict[int, str],
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    output_dir: str
):
    """Generate comprehensive evaluation report."""
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Classification report
    class_names = [label_to_cluster[i] for i in range(len(label_to_cluster))]
    report = classification_report(
        true_labels, predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }
    
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confusion matrix separately
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - CD300 Sequence Cluster Classification')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('Actual Cluster')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info("=" * 50)
    
    # Top performing classes
    logger.info("Top 10 Performing Clusters:")
    class_f1_scores = [(class_name, report[class_name]['f1-score']) 
                      for class_name in class_names if class_name in report]
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (class_name, f1_score) in enumerate(class_f1_scores[:10]):
        logger.info(f"{i+1:2d}. {class_name}: {f1_score:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune ESM2 for CD300 sequence cluster classification')
    parser.add_argument('--data_path', type=str, 
                       default='step1_data_preparation/cleaned_cd300_dataset.csv',
                       help='Path to the cleaned CD300 dataset with sequence clusters')
    parser.add_argument('--output_dir', type=str, default='step3_fine_tuned_model_clusters',
                       help='Output directory for the fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate for training')
    parser.add_argument('--max_length', type=int, default=1022,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting ESM2 fine-tuning for CD300 sequence cluster classification")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    try:
        # Load and prepare data
        sequences, labels, label_to_cluster = load_and_prepare_data(args.data_path)
        
        # Train model
        model, tokenizer = train_model(
            sequences=sequences,
            labels=labels,
            label_to_cluster=label_to_cluster,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length
        )
        
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()

