#!/usr/bin/env python3
"""
Step 4: Generate embeddings for Euarchontoglires sequences using fine-tuned ESM2 model
This script loads the fine-tuned model from Step 3 and generates embeddings for
the completely unseen Euarchontoglires CD300 sequences.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import EsmTokenizer, EsmModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step4_euarchontoglires_embeddings.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class EuarchontogliresDataset(Dataset):
    """Dataset for Euarchontoglires sequences"""
    def __init__(self, sequences, tokenizer, max_length=1022):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Tokenize sequence
        tokens = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }

def load_fine_tuned_model(model_path, num_classes=10):
    """Load the fine-tuned ESM2 model and classifier"""
    logger.info(f"Loading fine-tuned model from {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(os.path.join(model_path, "checkpoint.pt"), map_location='cpu')
    
    # Load ESM2 model
    esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    # Create the complete model with same architecture as training
    model = CD300Classifier(esm_model, num_classes=num_classes)
    
    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model

def load_euarchontoglires_sequences(fasta_path):
    """Load Euarchontoglires sequences from FASTA file"""
    logger.info(f"Loading Euarchontoglires sequences from {fasta_path}")
    
    sequences = []
    sequence_ids = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        sequence_ids.append(record.id)
    
    logger.info(f"Loaded {len(sequences)} Euarchontoglires sequences")
    return sequences, sequence_ids

def generate_embeddings(model, dataloader, device):
    """Generate embeddings and predictions for Euarchontoglires sequences"""
    logger.info("Generating embeddings and predictions...")
    
    model.to(device)
    
    all_embeddings = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0:
                logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get ESM2 embeddings
            outputs = model.esm(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            
            # Get classifier predictions
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Store results
            all_embeddings.append(pooled_output.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    
    # Concatenate all results
    embeddings = np.vstack(all_embeddings)
    predictions = np.concatenate(all_predictions)
    probabilities = np.vstack(all_probabilities)
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Generated predictions shape: {predictions.shape}")
    
    return embeddings, predictions, probabilities

def create_visualizations(embeddings, predictions, sequence_ids, output_dir):
    """Create visualizations of Euarchontoglires embeddings"""
    logger.info("Creating visualizations...")
    
    # PCA visualization
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                         c=predictions, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Predicted Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Euarchontoglires CD300 Sequences - PCA Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'euarchontoglires_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                         c=predictions, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Predicted Cluster')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Euarchontoglires CD300 Sequences - t-SNE Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'euarchontoglires_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved")

def analyze_predictions(predictions, probabilities, sequence_ids, output_dir):
    """Analyze the prediction results"""
    logger.info("Analyzing predictions...")
    
    # Count predictions per cluster
    unique, counts = np.unique(predictions, return_counts=True)
    cluster_counts = dict(zip([f"Cluster_{i}" for i in unique], counts.tolist()))
    
    # Create prediction summary
    results_df = pd.DataFrame({
        'sequence_id': sequence_ids,
        'predicted_cluster': [f"Cluster_{p}" for p in predictions],
        'max_probability': np.max(probabilities, axis=1)
    })
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'euarchontoglires_predictions.csv'), index=False)
    
    # Create summary statistics
    summary = {
        'total_sequences': len(predictions),
        'cluster_distribution': cluster_counts,
        'mean_confidence': float(np.mean(np.max(probabilities, axis=1))),
        'min_confidence': float(np.min(np.max(probabilities, axis=1))),
        'max_confidence': float(np.max(np.max(probabilities, axis=1)))
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'euarchontoglires_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create cluster distribution plot
    plt.figure(figsize=(10, 6))
    clusters = list(cluster_counts.keys())
    counts = list(cluster_counts.values())
    plt.bar(clusters, counts)
    plt.xlabel('Predicted Cluster')
    plt.ylabel('Number of Sequences')
    plt.title('Euarchontoglires CD300 Sequences - Cluster Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'euarchontoglires_cluster_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return summary

def create_interactive_plots(embeddings, predictions, probabilities, sequence_ids, output_dir):
    """Create interactive HTML plots for Step 4 results"""
    logger.info("Creating interactive plots...")
    
    # Perform PCA and t-SNE
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    # Create combined interactive plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'PCA (PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})',
            't-SNE'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Create color palette
    unique_clusters = [f"Cluster_{p}" for p in np.unique(predictions)]
    colors = px.colors.qualitative.Set3[:len(unique_clusters)]
    color_map = dict(zip(unique_clusters, colors))
    
    # Add PCA plot
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = predictions == i
        cluster_data_pca = embeddings_pca[cluster_mask]
        cluster_sequence_ids = [sequence_ids[j] for j in range(len(sequence_ids)) if cluster_mask[j]]
        cluster_confidences = np.max(probabilities, axis=1)[cluster_mask]
        
        # Create hover text for PCA
        hover_text_pca = [
            f"<b>Sequence ID:</b> {seq_id}<br>"
            f"<b>Cluster:</b> {cluster}<br>"
            f"<b>Confidence:</b> {conf:.3f}<br>"
            f"<b>PC1:</b> {pc1:.3f}<br>"
            f"<b>PC2:</b> {pc2:.3f}"
            for seq_id, conf, pc1, pc2 in zip(
                cluster_sequence_ids,
                cluster_confidences,
                cluster_data_pca[:, 0],
                cluster_data_pca[:, 1]
            )
        ]
        
        fig.add_trace(go.Scatter(
            x=cluster_data_pca[:, 0],
            y=cluster_data_pca[:, 1],
            mode='markers',
            marker=dict(
                color=color_map[cluster],
                size=8,
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            name=cluster,
            hovertemplate='%{text}<extra></extra>',
            text=hover_text_pca,
            showlegend=True
        ), row=1, col=1)
    
    # Add t-SNE plot
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = predictions == i
        cluster_data_tsne = embeddings_tsne[cluster_mask]
        cluster_sequence_ids = [sequence_ids[j] for j in range(len(sequence_ids)) if cluster_mask[j]]
        cluster_confidences = np.max(probabilities, axis=1)[cluster_mask]
        
        # Create hover text for t-SNE
        hover_text_tsne = [
            f"<b>Sequence ID:</b> {seq_id}<br>"
            f"<b>Cluster:</b> {cluster}<br>"
            f"<b>Confidence:</b> {conf:.3f}<br>"
            f"<b>t-SNE 1:</b> {tsne1:.3f}<br>"
            f"<b>t-SNE 2:</b> {tsne2:.3f}"
            for seq_id, conf, tsne1, tsne2 in zip(
                cluster_sequence_ids,
                cluster_confidences,
                cluster_data_tsne[:, 0],
                cluster_data_tsne[:, 1]
            )
        ]
        
        fig.add_trace(go.Scatter(
            x=cluster_data_tsne[:, 0],
            y=cluster_data_tsne[:, 1],
            mode='markers',
            marker=dict(
                color=color_map[cluster],
                size=8,
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            name=cluster,
            hovertemplate='%{text}<extra></extra>',
            text=hover_text_tsne,
            showlegend=False
        ), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Euarchontoglires CD300 Sequences - Interactive PCA and t-SNE Comparison',
            'x': 0.5,
            'xanchor': 'center'
        },
        width=1400,
        height=700,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", row=1, col=1)
    fig.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", row=1, col=1)
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)
    
    # Save as HTML
    output_path = os.path.join(output_dir, 'interactive_combined_plot.html')
    fig.write_html(output_path)
    logger.info(f"Interactive combined plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for Euarchontoglires sequences')
    parser.add_argument('--fasta_path', type=str, 
                       default='data/Euarchontoglires_CD300s_complete.fasta',
                       help='Path to Euarchontoglires FASTA file')
    parser.add_argument('--model_path', type=str,
                       default='step3_fine_tuned_model/best_model',
                       help='Path to fine-tuned model')
    parser.add_argument('--output_dir', type=str,
                       default='step4_euarchontoglires_embeddings',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--max_length', type=int, default=1022,
                       help='Maximum sequence length')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of sequence clusters')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info("Loading ESM2 tokenizer...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    # Load fine-tuned model
    model = load_fine_tuned_model(args.model_path, args.num_classes)
    
    # Load Euarchontoglires sequences
    sequences, sequence_ids = load_euarchontoglires_sequences(args.fasta_path)
    
    # Create dataset and dataloader
    dataset = EuarchontogliresDataset(sequences, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Generate embeddings and predictions
    embeddings, predictions, probabilities = generate_embeddings(
        model, dataloader, device
    )
    
    # Create visualizations
    create_visualizations(embeddings, predictions, sequence_ids, args.output_dir)
    
    # Create interactive plots
    create_interactive_plots(embeddings, predictions, probabilities, sequence_ids, args.output_dir)
    
    # Analyze predictions
    summary = analyze_predictions(predictions, probabilities, sequence_ids, args.output_dir)
    
    # Save embeddings
    np.save(os.path.join(args.output_dir, 'euarchontoglires_embeddings.npy'), embeddings)
    np.save(os.path.join(args.output_dir, 'euarchontoglires_predictions.npy'), predictions)
    np.save(os.path.join(args.output_dir, 'euarchontoglires_probabilities.npy'), probabilities)
    
    logger.info("="*50)
    logger.info("STEP 4 COMPLETED SUCCESSFULLY")
    logger.info("="*50)
    logger.info(f"Total Euarchontoglires sequences processed: {len(sequences)}")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Cluster distribution: {summary['cluster_distribution']}")
    logger.info(f"Mean confidence: {summary['mean_confidence']:.3f}")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
