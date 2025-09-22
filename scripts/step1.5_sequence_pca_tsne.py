#!/usr/bin/env python3
"""
Simple PCA and t-SNE Analysis on Protein Sequences
==================================================

This script performs PCA and t-SNE analysis directly on protein sequences
without using ESM2 embeddings or ESMFold structures - just raw sequence data.

Author: AI Assistant
Date: 2025-01-11
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Bio import SeqIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SequenceAnalyzer:
    """Analyze protein sequences using PCA and t-SNE"""
    
    def __init__(self, fasta_path: str, output_dir: str):
        self.fasta_path = fasta_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_sequences(self) -> Tuple[List[str], List[str]]:
        """Load sequences from FASTA file"""
        logger.info(f"Loading sequences from {self.fasta_path}")
        
        sequences = []
        sequence_ids = []
        
        for record in SeqIO.parse(self.fasta_path, "fasta"):
            sequences.append(str(record.seq))
            sequence_ids.append(record.id)
        
        logger.info(f"Loaded {len(sequences)} sequences")
        return sequences, sequence_ids
    
    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """Encode protein sequences using simple amino acid composition"""
        logger.info("Encoding sequences using amino acid composition")
        
        # Amino acid alphabet
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        
        # Calculate amino acid composition for each sequence
        features = []
        for seq in sequences:
            # Count amino acids
            aa_counts = np.zeros(len(amino_acids))
            for aa in seq:
                if aa in aa_to_idx:
                    aa_counts[aa_to_idx[aa]] += 1
            
            # Normalize by sequence length
            if len(seq) > 0:
                aa_counts = aa_counts / len(seq)
            
            # Add additional features
            seq_features = list(aa_counts)
            
            # Add sequence length (normalized)
            seq_features.append(len(seq) / 1000.0)  # Normalize length
            
            # Add hydrophobicity (simplified)
            hydrophobic_aas = 'AILMFWV'
            hydrophobicity = sum(1 for aa in seq if aa in hydrophobic_aas) / len(seq) if len(seq) > 0 else 0
            seq_features.append(hydrophobicity)
            
            # Add charge
            positive_charge = sum(1 for aa in seq if aa in 'KR') / len(seq) if len(seq) > 0 else 0
            negative_charge = sum(1 for aa in seq if aa in 'DE') / len(seq) if len(seq) > 0 else 0
            net_charge = positive_charge - negative_charge
            seq_features.append(net_charge)
            
            # Add aromatic content
            aromatic = sum(1 for aa in seq if aa in 'FWY') / len(seq) if len(seq) > 0 else 0
            seq_features.append(aromatic)
            
            features.append(seq_features)
        
        features_array = np.array(features)
        logger.info(f"Created feature matrix: {features_array.shape}")
        return features_array
    
    def perform_pca(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform PCA analysis"""
        logger.info("Performing PCA analysis")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform PCA
        pca = PCA(n_components=min(50, features.shape[1]))
        pca_components = pca.fit_transform(features_scaled)
        
        # Get explained variance
        explained_variance = pca.explained_variance_ratio_
        
        logger.info(f"PCA explained variance (first 5 components): {explained_variance[:5]}")
        logger.info(f"Total variance explained by first 2 components: {explained_variance[:2].sum():.3f}")
        
        return pca_components, explained_variance
    
    def perform_tsne(self, features: np.ndarray, perplexity: int = 30) -> np.ndarray:
        """Perform t-SNE analysis"""
        logger.info(f"Performing t-SNE analysis (perplexity={perplexity})")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        tsne_components = tsne.fit_transform(features_scaled)
        
        logger.info("t-SNE analysis completed")
        return tsne_components
    
    def perform_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, int]:
        """Perform K-means clustering with optimal number of clusters"""
        logger.info("Performing K-means clustering")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Find optimal number of clusters using silhouette score
        silhouette_scores = []
        k_range = range(2, min(11, len(features)//10 + 1))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            logger.info(f"k={k}: silhouette score = {silhouette_avg:.3f}")
        
        # Choose optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Final clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        return cluster_labels, optimal_k
    
    def create_plots(self, pca_components: np.ndarray, tsne_components: np.ndarray, 
                    explained_variance: np.ndarray, clusters: np.ndarray, 
                    sequence_ids: List[str]) -> None:
        """Create PCA and t-SNE visualizations"""
        logger.info("Creating visualizations")
        
        # Create static PCA plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            pca_components[:, 0],
            pca_components[:, 1],
            c=clusters,
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title('Sequence PCA: CD300 Protein Sequences', fontsize=14, fontweight='bold')
        plt.xlabel(f'PC1 ({explained_variance[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_variance[1]:.1%} variance)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequence_pca_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create static t-SNE plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            tsne_components[:, 0],
            tsne_components[:, 1],
            c=clusters,
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title('Sequence t-SNE: CD300 Protein Sequences', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequence_tsne_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create combined HTML plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sequence PCA', 'Sequence t-SNE'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add PCA plot
        fig.add_trace(
            go.Scatter(
                x=pca_components[:, 0],
                y=pca_components[:, 1],
                mode='markers',
                marker=dict(
                    color=clusters,
                    colorscale='viridis',
                    size=8,
                    opacity=0.7
                ),
                text=sequence_ids,
                hovertemplate='<b>%{text}</b><br>' +
                             'PC1: %{x:.2f}<br>' +
                             'PC2: %{y:.2f}<br>' +
                             'Cluster: %{marker.color}<extra></extra>',
                name='PCA'
            ),
            row=1, col=1
        )
        
        # Add t-SNE plot
        fig.add_trace(
            go.Scatter(
                x=tsne_components[:, 0],
                y=tsne_components[:, 1],
                mode='markers',
                marker=dict(
                    color=clusters,
                    colorscale='viridis',
                    size=8,
                    opacity=0.7
                ),
                text=sequence_ids,
                hovertemplate='<b>%{text}</b><br>' +
                             't-SNE1: %{x:.2f}<br>' +
                             't-SNE2: %{y:.2f}<br>' +
                             'Cluster: %{marker.color}<extra></extra>',
                name='t-SNE'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Sequence Clustering Analysis: CD300 Protein Sequences',
            width=1600,
            height=600,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text=f'PC1 ({explained_variance[0]:.1%} variance)', row=1, col=1)
        fig.update_yaxes(title_text=f'PC2 ({explained_variance[1]:.1%} variance)', row=1, col=1)
        fig.update_xaxes(title_text='t-SNE Component 1', row=1, col=2)
        fig.update_yaxes(title_text='t-SNE Component 2', row=1, col=2)
        
        fig.write_html(self.output_dir / 'sequence_combined_plot.html')
        
        logger.info("All plots created successfully!")
    
    def save_results(self, pca_components: np.ndarray, tsne_components: np.ndarray,
                    explained_variance: np.ndarray, clusters: np.ndarray,
                    sequence_ids: List[str], optimal_k: int) -> None:
        """Save analysis results"""
        logger.info("Saving results")
        
        # Create results dictionary
        results = {
            'pca_components': pca_components.tolist(),
            'tsne_components': tsne_components.tolist(),
            'explained_variance': explained_variance.tolist(),
            'clusters': clusters.tolist(),
            'sequence_ids': sequence_ids,
            'optimal_clusters': optimal_k,
            'total_sequences': len(sequence_ids)
        }
        
        # Save to JSON
        with open(self.output_dir / 'sequence_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        report = f"""
# Sequence Analysis Results

## Summary
- **Total sequences analyzed**: {len(sequence_ids)}
- **Optimal number of clusters**: {optimal_k}
- **PCA variance explained (first 2 components)**: {explained_variance[:2].sum():.1%}

## Clustering Results
"""
        
        for i in range(optimal_k):
            cluster_count = np.sum(clusters == i)
            report += f"- **Cluster {i}**: {cluster_count} sequences ({cluster_count/len(sequence_ids)*100:.1f}%)\n"
        
        report += f"""
## Files Generated
- `sequence_pca_plot.png`: Static PCA visualization
- `sequence_tsne_plot.png`: Static t-SNE visualization  
- `sequence_combined_plot.html`: Interactive combined plot
- `sequence_analysis_results.json`: Complete analysis results

## Analysis Method
This analysis uses simple amino acid composition features:
- Amino acid frequencies (20 features)
- Sequence length (normalized)
- Hydrophobicity
- Net charge
- Aromatic content

No deep learning embeddings or structural predictions were used.
"""
        
        with open(self.output_dir / 'sequence_analysis_report.md', 'w') as f:
            f.write(report)
        
        logger.info("Results saved successfully!")

def main():
    parser = argparse.ArgumentParser(description='Perform PCA and t-SNE on protein sequences')
    parser.add_argument('--fasta_path', required=True, help='Path to FASTA file')
    parser.add_argument('--output_dir', default='sequence_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SequenceAnalyzer(args.fasta_path, args.output_dir)
    
    # Load sequences
    sequences, sequence_ids = analyzer.load_sequences()
    
    # Encode sequences
    features = analyzer.encode_sequences(sequences)
    
    # Perform PCA
    pca_components, explained_variance = analyzer.perform_pca(features)
    
    # Perform t-SNE
    tsne_components = analyzer.perform_tsne(features)
    
    # Perform clustering
    clusters, optimal_k = analyzer.perform_clustering(features)
    
    # Create visualizations
    analyzer.create_plots(pca_components, tsne_components, explained_variance, 
                         clusters, sequence_ids)
    
    # Save results
    analyzer.save_results(pca_components, tsne_components, explained_variance,
                         clusters, sequence_ids, optimal_k)
    
    logger.info("Sequence analysis completed successfully!")

if __name__ == "__main__":
    main()
