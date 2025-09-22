#!/usr/bin/env python3
"""
Step 1: Data Preparation and Understanding
==========================================

This script analyzes the CD300 dataset to:
1. Verify label consistency and completeness
2. Ensure subtype granularity (CD300A, CD300B, CD300LG, etc.)
3. Flag missing or ambiguous subtype annotations
4. Generate data quality reports
5. Prepare cleaned datasets for training

Author: CD300 Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from pathlib import Path
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import numpy as np
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("step1_data_preparation")
RESULTS_DIR.mkdir(exist_ok=True)

def load_and_analyze_dataset():
    """Load the CD300 dataset and perform initial analysis."""
    print("Loading CD300 dataset...")
    
    # Load the CSV dataset
    csv_path = DATA_DIR / "CD300_NCBI_vertebrates_all.xlsx - vertebrate_NCBI_cd300_hits.csv"
    print(f"Looking for file: {csv_path}")
    if not csv_path.exists():
        print(f"File not found at {csv_path}")
        # Try alternative approach
        csv_files = list(DATA_DIR.glob("*.csv"))
        print(f"Available CSV files: {csv_files}")
        if csv_files:
            csv_path = csv_files[0]
            print(f"Using file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    return df

def perform_sequence_clustering(df):
    """Perform sequence clustering to discover natural groupings instead of relying on homology labels."""
    print("\n=== PERFORMING SEQUENCE CLUSTERING ===")
    
    # Create a copy for analysis
    df_analysis = df.copy()
    
    # Add sequence length column
    df_analysis['sequence_length'] = df_analysis['Protein Sequence'].str.len()
    
    # Generate sequence features using k-mer frequencies
    print("Generating sequence features using k-mer frequencies...")
    
    def generate_kmers(sequence, k=3):
        """Generate k-mers from a protein sequence."""
        if pd.isna(sequence):
            return []
        seq = str(sequence)
        return [seq[i:i+k] for i in range(len(seq)-k+1)]
    
    # Generate 3-mer features (trigrams)
    k = 3
    print(f"Generating {k}-mer features...")
    
    # Create k-mer frequency matrix
    sequences = df_analysis['Protein Sequence'].fillna('')
    kmers_list = [generate_kmers(seq, k) for seq in sequences]
    
    # Convert to k-mer frequency vectors
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer=lambda x: generate_kmers(x, k), lowercase=False)
    
    # Create feature matrix
    try:
        feature_matrix = vectorizer.fit_transform(sequences)
        print(f"Feature matrix shape: {feature_matrix.shape}")
    except Exception as e:
        print(f"Error creating feature matrix: {e}")
        # Fallback to simpler approach
        print("Using fallback approach with basic sequence features...")
        return df_analysis
    
    # Perform dimensionality reduction for visualization
    print("Performing PCA for dimensionality reduction...")
    pca = PCA(n_components=min(50, feature_matrix.shape[1]))
    features_pca = pca.fit_transform(feature_matrix.toarray())
    
    # Determine optimal number of clusters using silhouette score
    print("Determining optimal number of clusters...")
    silhouette_scores = []
    K_range = range(2, min(11, len(df_analysis) // 100 + 1))  # Reasonable range
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_pca)
        if len(set(cluster_labels)) > 1:  # At least 2 clusters
            score = silhouette_score(features_pca, cluster_labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    # Find optimal K
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")
    
    # Perform final clustering
    print(f"Performing K-means clustering with {optimal_k} clusters...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(features_pca)
    
    # Add cluster labels to dataframe
    df_analysis['sequence_cluster'] = cluster_labels
    df_analysis['cluster_label'] = [f'Cluster_{i}' for i in cluster_labels]
    
    # Analyze cluster characteristics
    print("\nCluster Analysis:")
    cluster_summary = df_analysis.groupby('cluster_label').agg({
        'sequence_length': ['count', 'mean', 'std'],
        'Order': 'nunique',
        'Family': 'nunique',
        'genus': 'nunique'
    }).round(2)
    
    print(cluster_summary)
    
    # Compare with homology labels (for reference, not for classification)
    print("\nComparing clusters with homology labels (for reference only):")
    homology_comparison = df_analysis.groupby(['cluster_label', 'Query CD300']).size().unstack(fill_value=0)
    print(homology_comparison.head(10))
    
    # Store clustering results
    df_analysis['pca_features'] = list(features_pca)
    
    return df_analysis, features_pca, optimal_k

def analyze_labels(df):
    """Analyze the sequence labels and functional classifications."""
    print("\n=== LABEL ANALYSIS ===")
    
    # Check sequence labeling status
    if 'Sequence Labeled?  (Yes/In Process or Blank for No)' in df.columns:
        print(f"Sequences labeled: {df['Sequence Labeled?  (Yes/In Process or Blank for No)'].value_counts()}")
    
    # Analyze functional classifications
    print(f"\nFunctional classifications:")
    func_counts = df['functional_classification'].value_counts()
    print(func_counts)
    
    # Check for missing values
    print(f"\nMissing functional classifications: {df['functional_classification'].isna().sum()}")
    
    # Analyze Query CD300 column
    print(f"\nQuery CD300 types:")
    query_counts = df['Query CD300'].value_counts()
    print(query_counts.head(10))
    
    return func_counts, query_counts

def analyze_sequences(df):
    """Analyze protein sequences for quality and characteristics."""
    print("\n=== SEQUENCE ANALYSIS ===")
    
    # Sequence length analysis
    df['sequence_length'] = df['sequence_length'].fillna(0)
    print(f"Sequence length statistics:")
    print(df['sequence_length'].describe())
    
    # Check for very short or very long sequences
    short_seqs = df[df['sequence_length'] < 50]
    long_seqs = df[df['sequence_length'] > 1000]
    
    print(f"\nVery short sequences (<50 aa): {len(short_seqs)}")
    print(f"Very long sequences (>1000 aa): {len(long_seqs)}")
    
    # Check for sequences with unusual characters
    def check_sequence_quality(seq):
        if pd.isna(seq):
            return "missing"
        seq_str = str(seq)
        if 'X' in seq_str:
            return "contains_X"
        if '*' in seq_str:
            return "contains_stop"
        if len(seq_str) < 50:
            return "very_short"
        if len(seq_str) > 1000:
            return "very_long"
        return "good"
    
    df['sequence_quality'] = df['Protein Sequence'].apply(check_sequence_quality)
    quality_counts = df['sequence_quality'].value_counts()
    print(f"\nSequence quality:")
    print(quality_counts)
    
    return df

def analyze_taxonomy(df):
    """Analyze taxonomic distribution of the dataset."""
    print("\n=== TAXONOMIC ANALYSIS ===")
    
    # Order distribution
    print("Orders represented:")
    order_counts = df['Order'].value_counts()
    print(order_counts.head(10))
    
    # Family distribution
    print(f"\nFamilies represented: {df['Family'].nunique()}")
    family_counts = df['Family'].value_counts()
    print(family_counts.head(10))
    
    # Genus distribution
    print(f"\nGenera represented: {df['genus'].nunique()}")
    genus_counts = df['genus'].value_counts()
    print(genus_counts.head(10))
    
    return order_counts, family_counts, genus_counts

def create_visualizations(df, features_pca, optimal_k, order_counts):
    """Create visualizations for the dataset analysis."""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CD300 Dataset Analysis - Sequence Clustering', fontsize=16, fontweight='bold')
    
    # 1. Sequence cluster distribution
    cluster_counts = df['cluster_label'].value_counts()
    axes[0, 0].pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Sequence Cluster Distribution\n(based on k-mer similarity)')
    
    # 2. Sequence length distribution
    axes[0, 1].hist(df['sequence_length'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Sequence Length (amino acids)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sequence Length Distribution')
    axes[0, 1].axvline(df['sequence_length'].median(), color='red', linestyle='--', 
                       label=f'Median: {df["sequence_length"].median():.0f}')
    axes[0, 1].legend()
    
    # 3. PCA visualization of clusters
    if features_pca is not None and len(features_pca) > 0:
        # Use first two PCA components for visualization
        pca_2d = features_pca[:, :2]
        cluster_labels = df['sequence_cluster']
        
        # Create scatter plot colored by cluster
        scatter = axes[1, 0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=cluster_labels, 
                                    cmap='tab10', alpha=0.6, s=20)
        axes[1, 0].set_xlabel('PCA Component 1')
        axes[1, 0].set_ylabel('PCA Component 2')
        axes[1, 0].set_title(f'PCA Visualization of {optimal_k} Sequence Clusters')
        
        # Add legend
        legend1 = axes[1, 0].legend(*scatter.legend_elements(), title="Clusters")
        axes[1, 0].add_artist(legend1)
    
    # 4. Top orders
    top_orders = order_counts.head(10)
    axes[1, 1].barh(range(len(top_orders)), top_orders.values)
    axes[1, 1].set_yticks(range(len(top_orders)))
    axes[1, 1].set_yticklabels(top_orders.index)
    axes[1, 1].set_xlabel('Number of Sequences')
    axes[1, 1].set_title('Top 10 Orders by Sequence Count')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {RESULTS_DIR / 'dataset_analysis.png'}")

def generate_data_quality_report(df):
    """Generate a comprehensive data quality report."""
    print("\n=== GENERATING DATA QUALITY REPORT ===")
    
    report = []
    report.append("CD300 Dataset Quality Report")
    report.append("=" * 50)
    report.append(f"Total sequences: {len(df)}")
    report.append(f"Total columns: {len(df.columns)}")
    report.append("")
    
    # Missing data analysis
    report.append("Missing Data Analysis:")
    report.append("-" * 25)
    for col in df.columns:
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        report.append(f"{col}: {missing} ({missing_pct:.1f}%)")
    report.append("")
    
    # Label completeness
    if 'Sequence Labeled?  (Yes/In Process or Blank for No)' in df.columns:
        labeled_seqs = df['Sequence Labeled?  (Yes/In Process or Blank for No)'].value_counts()
        report.append("Labeling Status:")
        report.append("-" * 20)
        for status, count in labeled_seqs.items():
            report.append(f"{status}: {count}")
        report.append("")
    
    # Sequence cluster summary
    cluster_counts = df['cluster_label'].value_counts()
    report.append("Sequence Clusters (based on k-mer similarity):")
    report.append("-" * 45)
    for cluster, count in cluster_counts.items():
        report.append(f"{cluster}: {count}")
    report.append("")
    
    # Sequence statistics
    report.append("Sequence Statistics:")
    report.append("-" * 20)
    report.append(f"Mean length: {df['sequence_length'].mean():.1f} aa")
    report.append(f"Median length: {df['sequence_length'].median():.1f} aa")
    report.append(f"Min length: {df['sequence_length'].min():.1f} aa")
    report.append(f"Max length: {df['sequence_length'].max():.1f} aa")
    report.append("")
    
    # Save report
    report_path = RESULTS_DIR / 'data_quality_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Data quality report saved to: {report_path}")
    
    # Print summary to console
    print('\n'.join(report))

def prepare_cleaned_dataset(df):
    """Prepare a cleaned dataset for training."""
    print("\n=== PREPARING CLEANED DATASET ===")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Remove rows with missing cluster labels
    df_clean = df_clean.dropna(subset=['cluster_label'])
    print(f"Removed {initial_count - len(df_clean)} rows with missing cluster labels")
    
    # Remove rows where subject_id doesn't contain "CMRF" or "CD300"
    before_filter = len(df_clean)
    df_clean = df_clean[
        df_clean['subject_id'].str.contains('CMRF|CD300', case=False, na=False)
    ]
    print(f"Removed {before_filter - len(df_clean)} rows without CMRF or CD300 in subject_id")
    
    # Remove very short sequences (<50 aa) as they may be fragments
    short_count = len(df_clean[df_clean['sequence_length'] < 50])
    df_clean = df_clean[df_clean['sequence_length'] >= 50]
    print(f"Removed {short_count} sequences shorter than 50 amino acids")
    
    # Remove sequences with X characters (ambiguous amino acids)
    x_count = len(df_clean[df_clean['Protein Sequence'].str.contains('X', na=False)])
    df_clean = df_clean[~df_clean['Protein Sequence'].str.contains('X', na=False)]
    print(f"Removed {x_count} sequences containing ambiguous amino acids (X)")
    
    # Check final dataset size
    print(f"Final cleaned dataset: {len(df_clean)} sequences")
    
    # Save cleaned dataset
    cleaned_path = RESULTS_DIR / 'cleaned_cd300_dataset.csv'
    df_clean.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataset saved to: {cleaned_path}")
    
    # Save a summary of the cleaned dataset
    summary_path = RESULTS_DIR / 'cleaned_dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Cleaned CD300 Dataset Summary\n")
        f.write(f"Total sequences: {len(df_clean)}\n")
        f.write(f"Sequence clusters:\n")
        for cluster, count in df_clean['cluster_label'].value_counts().items():
            f.write(f"  {cluster}: {count}\n")
        f.write(f"Orders represented: {df_clean['Order'].nunique()}\n")
        f.write(f"Families represented: {df_clean['Family'].nunique()}\n")
        f.write(f"Genera represented: {df_clean['genus'].nunique()}\n")
    
    print(f"Dataset summary saved to: {summary_path}")
    
    return df_clean

def main():
    """Main execution function."""
    print("CD300 Protein Analysis - Step 1: Data Preparation")
    print("=" * 60)
    
    # Load and analyze dataset
    df = load_and_analyze_dataset()
    
    # Perform sequence clustering to discover natural groupings
    df, features_pca, optimal_k = perform_sequence_clustering(df)
    
    # Perform analyses
    df = analyze_sequences(df)
    order_counts, family_counts, genus_counts = analyze_taxonomy(df)
    
    # Create visualizations
    create_visualizations(df, features_pca, optimal_k, order_counts)
    
    # Generate data quality report
    generate_data_quality_report(df)
    
    # Prepare cleaned dataset
    df_clean = prepare_cleaned_dataset(df)
    
    print("\n" + "=" * 60)
    print("Step 1 Complete! Data preparation finished successfully.")
    print(f"Results saved in: {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
