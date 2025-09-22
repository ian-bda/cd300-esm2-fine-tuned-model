#!/usr/bin/env python3
"""
Step 2: Baseline Embedding Generation and Visualization
======================================================

This script generates embeddings using pretrained ESM2 650M (no fine-tuning) and:
1. Generates embeddings for all CD300 sequences
2. Visualizes embeddings with UMAP/t-SNE
3. Colors points by sequence cluster labels and inspects clustering
4. Generates cluster overlap summary between embedding clusters and sequence clusters

Author: CD300 Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import warnings
from pathlib import Path
import time
import pickle
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up paths
DATA_DIR = Path("data")
STEP1_DIR = Path("step1_data_preparation")
RESULTS_DIR = Path("step2_baseline_embeddings")
MODELS_DIR = Path("models")

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_cleaned_dataset():
    """Load the cleaned CD300 dataset from Step 1."""
    logger.info("Loading cleaned CD300 dataset...")
    
    csv_path = STEP1_DIR / "cleaned_cd300_dataset.csv"
    df = pd.read_csv(csv_path)
    
    logger.info(f"Dataset loaded: {len(df)} sequences")
    logger.info(f"Sequence clusters: {df['cluster_label'].value_counts().to_dict()}")
    
    return df

def load_esm2_model():
    """Load the ESM2 650M model and tokenizer."""
    logger.info("Loading ESM2 650M model...")
    
    model_name = "facebook/esm2_t33_650M_UR50D"  # Using 650M model for memory efficiency
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        logger.info(f"ESM2 model loaded successfully: {model_name}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading ESM2 model: {e}")
        # Fallback to smaller model if needed
        model_name = "facebook/esm2_t6_8M_UR50D"
        logger.info(f"Trying fallback model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        return model, tokenizer

def generate_embeddings(model, tokenizer, sequences, batch_size=8):
    """Generate embeddings for protein sequences using ESM2."""
    logger.info(f"Generating embeddings for {len(sequences)} sequences...")
    
    embeddings = []
    sequence_ids = []
    
    # Process sequences in batches
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        batch_ids = list(range(i, min(i+batch_size, len(sequences))))
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
        
        try:
            # Tokenize sequences
            batch_tokens = tokenizer(batch_sequences, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=1022,  # ESM2 limit
                                   return_tensors="pt")
            
            batch_tokens = {k: v.to(device) for k, v in batch_tokens.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**batch_tokens)
                
                # Use mean pooling over sequence length (excluding special tokens)
                attention_mask = batch_tokens['attention_mask']
                sequence_output = outputs.last_hidden_state
                
                # Mean pooling
                masked_output = sequence_output * attention_mask.unsqueeze(-1)
                summed = torch.sum(masked_output, dim=1)
                counts = torch.sum(attention_mask, dim=1, keepdim=True)
                mean_pooled = summed / counts
                
                embeddings.extend(mean_pooled.cpu().numpy())
                sequence_ids.extend(batch_ids)
                
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add zero embeddings for failed sequences
            embedding_dim = model.config.hidden_size
            for _ in range(len(batch_sequences)):
                embeddings.append(np.zeros(embedding_dim))
                sequence_ids.append(batch_ids[len(embeddings)-1])
    
    embeddings = np.array(embeddings)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    
    return embeddings, sequence_ids

def perform_dimensionality_reduction(embeddings, method='umap', n_components=2):
    """Perform dimensionality reduction using UMAP or t-SNE."""
    logger.info(f"Performing {method.upper()} dimensionality reduction...")
    
    if method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, 
                           random_state=42, 
                           n_neighbors=15,
                           min_dist=0.1)
        reduced_embeddings = reducer.fit_transform(embeddings)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, 
                      random_state=42, 
                      perplexity=30,
                      n_iter=1000)
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Method must be 'umap' or 'tsne'")
    
    logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    return reduced_embeddings

def perform_clustering(embeddings, n_clusters=3):
    """Perform K-means clustering on embeddings."""
    logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    logger.info(f"Silhouette score: {silhouette_avg:.3f}")
    
    return cluster_labels, silhouette_avg

def create_visualizations(reduced_embeddings, labels, cluster_labels, functional_classes):
    """Create visualization plots for embeddings."""
    logger.info("Creating visualization plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('CD300 Protein Embeddings - Baseline ESM2 650M Analysis', fontsize=16, fontweight='bold')
    
    # Encode string labels to numeric values for coloring
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    encoded_clusters = le.fit_transform(cluster_labels)
    
    # Plot 1: UMAP colored by sequence cluster
    scatter1 = axes[0,0].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                                 c=encoded_labels, cmap='tab10', alpha=0.7, s=20)
    axes[0,0].set_title('UMAP Embeddings - Colored by Sequence Cluster', fontweight='bold')
    axes[0,0].set_xlabel('UMAP 1')
    axes[0,0].set_ylabel('UMAP 2')
    
    # Add legend for functional classifications
    unique_labels = np.unique(labels)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(i/len(unique_labels)), 
                                 markersize=10, label=func_class) 
                      for i, func_class in enumerate(unique_labels)]
    axes[0,0].legend(handles=legend_elements, title="Sequence Cluster", 
                     loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Plot 2: UMAP colored by embedding cluster assignment
    scatter2 = axes[0,1].scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                                 c=encoded_clusters, cmap='Set3', alpha=0.7, s=20)
    axes[0,1].set_title('UMAP Embeddings - Colored by Embedding Cluster Assignment', fontweight='bold')
    axes[0,1].set_xlabel('UMAP 1')
    axes[0,1].set_ylabel('UMAP 2')
    
    # Plot 3: t-SNE colored by sequence cluster
    tsne_embeddings = perform_dimensionality_reduction(reduced_embeddings, method='tsne', n_components=2)
    scatter3 = axes[1,0].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                 c=encoded_labels, cmap='tab10', alpha=0.7, s=20)
    axes[1,0].set_title('t-SNE Embeddings - Colored by Sequence Cluster', fontweight='bold')
    axes[1,0].set_xlabel('t-SNE 1')
    axes[1,0].set_ylabel('t-SNE 2')
    
    # Plot 4: t-SNE colored by embedding cluster assignment
    scatter4 = axes[1,1].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                 c=encoded_clusters, cmap='Set3', alpha=0.7, s=20)
    axes[1,1].set_title('t-SNE Embeddings - Colored by Embedding Cluster Assignment', fontweight='bold')
    axes[1,1].set_xlabel('t-SNE 1')
    axes[1,1].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = RESULTS_DIR / 'baseline_embeddings_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to: {plot_path}")
    
    plt.show()
    
    return tsne_embeddings

def analyze_cluster_overlap(cluster_labels, sequence_clusters):
    """Analyze overlap between embedding clusters and sequence clusters."""
    logger.info("Analyzing overlap between embedding clusters and sequence clusters...")
    
    # Create confusion matrix
    df_cluster = pd.DataFrame({
        'embedding_cluster': cluster_labels,
        'sequence_cluster': sequence_clusters
    })
    
    # Create cross-tabulation
    cluster_func_table = pd.crosstab(df_cluster['embedding_cluster'], df_cluster['sequence_cluster'])
    
    # Calculate cluster purity
    cluster_purity = {}
    for cluster in range(len(np.unique(cluster_labels))):
        cluster_mask = cluster_labels == cluster
        cluster_funcs = sequence_clusters[cluster_mask]
        if len(cluster_funcs) > 0:
            most_common_func = pd.Series(cluster_funcs).mode()[0]
            purity = (cluster_funcs == most_common_func).sum() / len(cluster_funcs)
            cluster_purity[cluster] = {
                'most_common_function': most_common_func,
                'purity': purity,
                'size': len(cluster_funcs)
            }
    
    # Calculate adjusted Rand index
    le = LabelEncoder()
    encoded_funcs = le.fit_transform(sequence_clusters)
    ari = adjusted_rand_score(encoded_funcs, cluster_labels)
    
    logger.info(f"Adjusted Rand Index: {ari:.3f}")
    
    # Save analysis results
    analysis_results = {
        'cluster_function_table': cluster_func_table,
        'cluster_purity': cluster_purity,
        'adjusted_rand_index': ari
    }
    
    # Save to file
    with open(RESULTS_DIR / 'cluster_analysis_results.pkl', 'wb') as f:
        pickle.dump(analysis_results, f)
    
    # Create summary report
    report_path = RESULTS_DIR / 'cluster_overlap_report.txt'
    with open(report_path, 'w') as f:
        f.write("CD300 Baseline Embeddings - Sequence vs Embedding Cluster Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Embedding Cluster vs Sequence Cluster Cross-Tabulation:\n")
        f.write("-" * 55 + "\n")
        f.write(cluster_func_table.to_string())
        f.write("\n\n")
        
        f.write("Embedding Cluster Purity Analysis:\n")
        f.write("-" * 35 + "\n")
        for cluster, info in cluster_purity.items():
            f.write(f"Embedding Cluster {cluster}:\n")
            f.write(f"  Most common sequence cluster: {info['most_common_function']}\n")
            f.write(f"  Purity: {info['purity']:.3f}\n")
            f.write(f"  Size: {info['size']}\n\n")
        
        f.write(f"Adjusted Rand Index: {ari:.3f}\n")
        f.write(f"Interpretation: {'Strong agreement' if ari > 0.7 else 'Moderate agreement' if ari > 0.3 else 'Weak agreement'}\n")
    
    logger.info(f"Cluster overlap report saved to: {report_path}")
    
    return analysis_results

def save_embeddings(embeddings, reduced_embeddings, tsne_embeddings, sequence_ids):
    """Save embeddings and reduced embeddings to files."""
    logger.info("Saving embeddings to files...")
    
    # Save full embeddings
    embeddings_path = RESULTS_DIR / 'baseline_embeddings_full.npy'
    np.save(embeddings_path, embeddings)
    logger.info(f"Full embeddings saved to: {embeddings_path}")
    
    # Save UMAP embeddings
    umap_path = RESULTS_DIR / 'baseline_embeddings_umap.npy'
    np.save(umap_path, reduced_embeddings)
    logger.info(f"UMAP embeddings saved to: {umap_path}")
    
    # Save t-SNE embeddings
    tsne_path = RESULTS_DIR / 'baseline_embeddings_tsne.npy'
    np.save(tsne_path, tsne_embeddings)
    logger.info(f"t-SNE embeddings saved to: {tsne_path}")
    
    # Save sequence IDs mapping
    ids_path = RESULTS_DIR / 'sequence_ids_mapping.npy'
    np.save(ids_path, sequence_ids)
    logger.info(f"Sequence IDs mapping saved to: {ids_path}")

def main():
    """Main execution function."""
    logger.info("CD300 Protein Analysis - Step 2: Baseline Embeddings")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load cleaned dataset
        df = load_cleaned_dataset()
        
        # Load ESM2 model
        model, tokenizer = load_esm2_model()
        
        # Prepare sequences and labels
        sequences = df['Protein Sequence'].tolist()
        cluster_labels_data = df['cluster_label'].tolist()
        
        # Generate embeddings
        embeddings, sequence_ids = generate_embeddings(model, tokenizer, sequences)
        
        # Perform UMAP dimensionality reduction
        umap_embeddings = perform_dimensionality_reduction(embeddings, method='umap')
        
        # Perform clustering
        cluster_labels, silhouette_score_val = perform_clustering(embeddings)
        
        # Create visualizations
        tsne_embeddings = create_visualizations(umap_embeddings, 
                                              np.array(cluster_labels_data), 
                                              cluster_labels, 
                                              np.array(cluster_labels_data))
        
        # Analyze cluster overlap
        analysis_results = analyze_cluster_overlap(cluster_labels, np.array(cluster_labels_data))
        
        # Save all embeddings
        save_embeddings(embeddings, umap_embeddings, tsne_embeddings, sequence_ids)
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info(f"\nStep 2 completed successfully in {elapsed_time:.2f} seconds!")
        logger.info(f"Results saved in: {RESULTS_DIR}")
        
        # Print key metrics
        logger.info(f"Silhouette Score: {silhouette_score_val:.3f}")
        logger.info(f"Adjusted Rand Index: {analysis_results['adjusted_rand_index']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
