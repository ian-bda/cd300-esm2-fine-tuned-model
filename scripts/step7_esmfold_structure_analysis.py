#!/usr/bin/env python3
"""
Step 7: Multi-Modal Structure-Function Analysis
===============================================

This script performs comprehensive structural analysis of CD300 proteins using ESMFold:
1. Predict 3D structures for all Euarchontoglires CD300 sequences
2. Extract structural features (distances, secondary structure, surface properties)
3. Perform structural clustering analysis (PCA, t-SNE)
4. Cross-modal validation with sequence and functional clusters
5. Generate comprehensive multi-modal analysis report

Author: AI Assistant
Date: 2025-01-11
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# BioPython for sequence handling
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

# ESMFold for structure prediction
try:
    import torch
    from transformers import EsmForProteinFolding
    ESMFOLD_AVAILABLE = True
except ImportError:
    ESMFOLD_AVAILABLE = False
    print("Warning: ESMFold not available. Install with: pip install transformers[esmfold]")

# Scikit-learn for clustering and dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Plotly for interactive visualizations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step7_structure_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ESMFoldStructurePredictor:
    """ESMFold structure prediction and analysis"""
    
    def __init__(self, device='cuda', model_name='facebook/esmfold_v1'):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load ESMFold model"""
        if not ESMFOLD_AVAILABLE:
            raise ImportError("ESMFold not available. Please install transformers[esmfold]")
        
        logger.info(f"Loading ESMFold model: {self.model_name}")
        self.model = EsmForProteinFolding.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        
        # Load tokenizer separately
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        logger.info("ESMFold model and tokenizer loaded successfully")
        
    def predict_structure(self, sequence: str, sequence_id: str) -> Dict:
        """Predict 3D structure for a single sequence"""
        try:
            # Tokenize sequence
            tokenized = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Predict structure
            with torch.no_grad():
                output = self.model(**tokenized)
                
            # Extract coordinates and confidence scores
            # ESMFold returns [8, 1, L, 14, 3] - 8 models, 1 sequence, L residues, 14 atoms, 3 coords
            # We'll use the first model and extract CA coordinates (atom index 1)
            coords = output.positions[0, 0, :, 1, :].cpu().numpy()  # [L, 3] - CA coordinates
            confidence = output.plddt[0, 0, :].cpu().numpy()  # [L] - per-residue confidence
            
            # Calculate structure metrics
            ca_coords = coords  # CA coordinates are already extracted
            structure_metrics = self._calculate_structure_metrics(ca_coords, confidence)
            
            return {
                'sequence_id': sequence_id,
                'sequence': sequence,
                'coordinates': coords,
                'confidence': confidence,
                'ca_coordinates': ca_coords,
                'structure_metrics': structure_metrics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Structure prediction failed for {sequence_id}: {str(e)}")
            return {
                'sequence_id': sequence_id,
                'sequence': sequence,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_structure_metrics(self, ca_coords: np.ndarray, confidence: np.ndarray) -> Dict:
        """Calculate structural features from CA coordinates"""
        metrics = {}
        
        # Basic metrics
        metrics['length'] = len(ca_coords)
        metrics['mean_confidence'] = np.mean(confidence)
        metrics['min_confidence'] = np.min(confidence)
        metrics['max_confidence'] = np.max(confidence)
        
        # Distance matrix
        distances = np.linalg.norm(ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :], axis=2)
        metrics['max_distance'] = np.max(distances)
        metrics['mean_distance'] = np.mean(distances[distances > 0])
        
        # Radius of gyration
        center = np.mean(ca_coords, axis=0)
        rg_squared = np.mean(np.sum((ca_coords - center)**2, axis=1))
        metrics['radius_of_gyration'] = np.sqrt(rg_squared)
        
        # End-to-end distance
        metrics['end_to_end_distance'] = np.linalg.norm(ca_coords[-1] - ca_coords[0])
        
        # Compactness (end-to-end / radius of gyration)
        if metrics['radius_of_gyration'] > 0:
            metrics['compactness'] = metrics['end_to_end_distance'] / metrics['radius_of_gyration']
        else:
            metrics['compactness'] = 0
            
        return metrics

class StructuralFeatureExtractor:
    """Extract comprehensive structural features from 3D structures"""
    
    def __init__(self):
        self.features = {}
        
    def extract_features(self, structure_data: Dict) -> Dict:
        """Extract structural features from structure prediction"""
        if not structure_data['success']:
            return None
            
        features = {}
        ca_coords = structure_data['ca_coordinates']
        confidence = structure_data['confidence']
        
        # Basic geometric features
        features.update(self._extract_geometric_features(ca_coords))
        
        # Secondary structure features (simplified)
        features.update(self._extract_secondary_structure_features(ca_coords))
        
        # Surface and accessibility features
        features.update(self._extract_surface_features(ca_coords))
        
        # Confidence-based features
        features.update(self._extract_confidence_features(confidence))
        
        return features
    
    def _extract_geometric_features(self, ca_coords: np.ndarray) -> Dict:
        """Extract geometric features from CA coordinates"""
        features = {}
        
        # Calculate distances between consecutive residues
        consecutive_distances = np.linalg.norm(np.diff(ca_coords, axis=0), axis=1)
        features['mean_consecutive_distance'] = np.mean(consecutive_distances)
        features['std_consecutive_distance'] = np.std(consecutive_distances)
        
        # Calculate angles between consecutive residues
        if len(ca_coords) >= 3:
            vectors = np.diff(ca_coords, axis=0)
            angles = []
            for i in range(len(vectors) - 1):
                v1, v2 = vectors[i], vectors[i + 1]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
                angle = np.arccos(cos_angle)
                angles.append(angle)
            features['mean_bond_angle'] = np.mean(angles)
            features['std_bond_angle'] = np.std(angles)
        else:
            features['mean_bond_angle'] = 0
            features['std_bond_angle'] = 0
            
        return features
    
    def _extract_secondary_structure_features(self, ca_coords: np.ndarray) -> Dict:
        """Extract simplified secondary structure features"""
        features = {}
        
        # Calculate local curvature (simplified)
        if len(ca_coords) >= 3:
            curvatures = []
            for i in range(1, len(ca_coords) - 1):
                p1, p2, p3 = ca_coords[i-1], ca_coords[i], ca_coords[i+1]
                v1 = p2 - p1
                v2 = p3 - p2
                cross = np.cross(v1, v2)
                curvature = np.linalg.norm(cross) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                curvatures.append(curvature)
            features['mean_curvature'] = np.mean(curvatures)
            features['std_curvature'] = np.std(curvatures)
        else:
            features['mean_curvature'] = 0
            features['std_curvature'] = 0
            
        return features
    
    def _extract_surface_features(self, ca_coords: np.ndarray) -> Dict:
        """Extract surface accessibility features (simplified)"""
        features = {}
        
        # Calculate local density (simplified surface measure)
        if len(ca_coords) > 10:
            densities = []
            for i in range(len(ca_coords)):
                distances = np.linalg.norm(ca_coords - ca_coords[i], axis=1)
                # Count neighbors within 10Å
                neighbors = np.sum(distances < 10.0) - 1  # Exclude self
                densities.append(neighbors)
            features['mean_local_density'] = np.mean(densities)
            features['std_local_density'] = np.std(densities)
        else:
            features['mean_local_density'] = 0
            features['std_local_density'] = 0
            
        return features
    
    def _extract_confidence_features(self, confidence: np.ndarray) -> Dict:
        """Extract confidence-based features"""
        features = {}
        
        features['confidence_mean'] = np.mean(confidence)
        features['confidence_std'] = np.std(confidence)
        features['confidence_min'] = np.min(confidence)
        features['confidence_max'] = np.max(confidence)
        
        # Confidence distribution
        features['high_confidence_ratio'] = np.sum(confidence > 0.8) / len(confidence)
        features['low_confidence_ratio'] = np.sum(confidence < 0.5) / len(confidence)
        
        return features

class StructuralClusteringAnalyzer:
    """Analyze structural clustering and cross-modal validation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def perform_structural_clustering(self, features_df: pd.DataFrame) -> Dict:
        """Perform structural clustering analysis"""
        logger.info("Performing structural clustering analysis...")
        
        # Prepare features
        feature_columns = [col for col in features_df.columns if col not in ['sequence_id', 'sequence']]
        X = features_df[feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=min(50, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
        X_tsne = tsne.fit_transform(X_pca)
        
        # Determine optimal number of clusters
        silhouette_scores = []
        K_range = range(2, min(20, len(X)//2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_pca)
            silhouette_avg = silhouette_score(X_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # Final clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        structural_clusters = kmeans_final.fit_predict(X_pca)
        
        return {
            'pca_components': X_pca,
            'tsne_components': X_tsne,
            'structural_clusters': structural_clusters,
            'optimal_k': optimal_k,
            'silhouette_scores': silhouette_scores,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'feature_columns': feature_columns
        }
    
    def cross_modal_validation(self, structural_clusters: np.ndarray, 
                             sequence_clusters: np.ndarray,
                             functional_clusters: np.ndarray) -> Dict:
        """Compare structural clusters with sequence and functional clusters"""
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # Calculate alignment metrics
        structural_vs_sequence = adjusted_rand_score(structural_clusters, sequence_clusters)
        structural_vs_functional = adjusted_rand_score(structural_clusters, functional_clusters)
        sequence_vs_functional = adjusted_rand_score(sequence_clusters, functional_clusters)
        
        # NMI scores
        structural_vs_sequence_nmi = normalized_mutual_info_score(structural_clusters, sequence_clusters)
        structural_vs_functional_nmi = normalized_mutual_info_score(structural_clusters, functional_clusters)
        sequence_vs_functional_nmi = normalized_mutual_info_score(sequence_clusters, functional_clusters)
        
        return {
            'structural_vs_sequence_ari': structural_vs_sequence,
            'structural_vs_functional_ari': structural_vs_functional,
            'sequence_vs_functional_ari': sequence_vs_functional,
            'structural_vs_sequence_nmi': structural_vs_sequence_nmi,
            'structural_vs_functional_nmi': structural_vs_functional_nmi,
            'sequence_vs_functional_nmi': sequence_vs_functional_nmi
        }

class StructureVisualizer:
    """Create visualizations for structural analysis"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_structural_plots(self, clustering_results: Dict, features_df: pd.DataFrame) -> None:
        """Create structural clustering visualizations"""
        logger.info("Creating structural clustering visualizations...")
        
        # Static plots (matplotlib)
        self._create_pca_plot_static(clustering_results, features_df)
        self._create_tsne_plot_static(clustering_results, features_df)
        
        # Interactive HTML plot with both PCA and t-SNE side by side
        self._create_combined_plot_html(clustering_results, features_df)
        
    def _create_pca_plot_static(self, clustering_results: Dict, features_df: pd.DataFrame) -> None:
        """Create PCA visualization (static)"""
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(
            clustering_results['pca_components'][:, 0],
            clustering_results['pca_components'][:, 1],
            c=clustering_results['structural_clusters'],
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('Structural PCA: CD300 Protein Structures', fontsize=14, fontweight='bold')
        plt.xlabel(f'PC1 ({clustering_results["pca_explained_variance"][0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({clustering_results["pca_explained_variance"][1]:.1%} variance)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'structural_pca_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_tsne_plot_static(self, clustering_results: Dict, features_df: pd.DataFrame) -> None:
        """Create t-SNE visualization (static)"""
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(
            clustering_results['tsne_components'][:, 0],
            clustering_results['tsne_components'][:, 1],
            c=clustering_results['structural_clusters'],
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('Structural t-SNE: CD300 Protein Structures', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'structural_tsne_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_combined_plot_html(self, clustering_results: Dict, features_df: pd.DataFrame) -> None:
        """Create combined PCA and t-SNE visualization (HTML)"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots with 1 row and 2 columns
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Structural PCA', 'Structural t-SNE'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add PCA plot
        fig.add_trace(
            go.Scatter(
                x=clustering_results['pca_components'][:, 0],
                y=clustering_results['pca_components'][:, 1],
                mode='markers',
                marker=dict(
                    color=clustering_results['structural_clusters'],
                    colorscale='viridis',
                    size=8,
                    opacity=0.7
                ),
                text=features_df['sequence_id'],
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
                x=clustering_results['tsne_components'][:, 0],
                y=clustering_results['tsne_components'][:, 1],
                mode='markers',
                marker=dict(
                    color=clustering_results['structural_clusters'],
                    colorscale='viridis',
                    size=8,
                    opacity=0.7
                ),
                text=features_df['sequence_id'],
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
            title='Structural Clustering Analysis: CD300 Protein Structures',
            width=1600,
            height=600,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text=f'PC1 ({clustering_results["pca_explained_variance"][0]:.1%} variance)', row=1, col=1)
        fig.update_yaxes(title_text=f'PC2 ({clustering_results["pca_explained_variance"][1]:.1%} variance)', row=1, col=1)
        fig.update_xaxes(title_text='t-SNE Component 1', row=1, col=2)
        fig.update_yaxes(title_text='t-SNE Component 2', row=1, col=2)
        
        fig.write_html(self.output_dir / 'structural_combined_plot.html')
        

def load_sequences(fasta_path: str) -> List[Tuple[str, str]]:
    """Load sequences from FASTA file"""
    logger.info(f"Loading sequences from {fasta_path}")
    sequences = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append((record.id, str(record.seq)))
    
    logger.info(f"Loaded {len(sequences)} sequences")
    return sequences

def load_previous_results(step4_dir: str, step5_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load results from previous steps"""
    logger.info("Loading results from previous steps...")
    
    # Load Step 4 results (embeddings and predictions)
    step4_predictions = pd.read_csv(f"{step4_dir}/euarchontoglires_predictions.csv")
    
    # Load Step 5 results (enhanced analysis)
    step5_results = pd.read_csv(f"{step5_dir}/enhanced_analysis_results.csv")
    
    return step4_predictions, step5_results

def main():
    parser = argparse.ArgumentParser(description='Step 7: Multi-Modal Structure-Function Analysis')
    parser.add_argument('--fasta_path', type=str, required=True,
                       help='Path to Euarchontoglires CD300 sequences FASTA file')
    parser.add_argument('--output_dir', type=str, default='step7_structure_analysis',
                       help='Output directory for results')
    parser.add_argument('--step4_dir', type=str, default='step4_euarchontoglires_embeddings',
                       help='Directory containing Step 4 results')
    parser.add_argument('--step5_dir', type=str, default='step5_enhanced_novel_variant_detection',
                       help='Directory containing Step 5 results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for ESMFold (cuda/cpu)')
    parser.add_argument('--max_sequences', type=int, default=None,
                       help='Maximum number of sequences to process (for testing)')
    parser.add_argument('--skip_structure_prediction', action='store_true',
                       help='Skip structure prediction and use existing results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("STEP 7: MULTI-MODAL STRUCTURE-FUNCTION ANALYSIS")
    logger.info("="*80)
    
    # Load sequences
    sequences = load_sequences(args.fasta_path)
    if args.max_sequences:
        sequences = sequences[:args.max_sequences]
        logger.info(f"Limited to {len(sequences)} sequences for testing")
    
    # Load previous results
    try:
        step4_results, step5_results = load_previous_results(args.step4_dir, args.step5_dir)
        logger.info("Successfully loaded previous step results")
    except Exception as e:
        logger.warning(f"Could not load previous results: {e}")
        step4_results, step5_results = None, None
    
    # Initialize components
    predictor = ESMFoldStructurePredictor(device=args.device)
    feature_extractor = StructuralFeatureExtractor()
    clustering_analyzer = StructuralClusteringAnalyzer()
    visualizer = StructureVisualizer(args.output_dir)
    
    # Structure prediction
    if not args.skip_structure_prediction:
        logger.info("Starting ESMFold structure prediction...")
        predictor.load_model()
        
        structure_results = []
        for i, (seq_id, sequence) in enumerate(sequences):
            logger.info(f"Predicting structure {i+1}/{len(sequences)}: {seq_id}")
            result = predictor.predict_structure(sequence, seq_id)
            structure_results.append(result)
            
            # Save intermediate results
            if (i + 1) % 50 == 0:
                with open(output_dir / f'structure_results_batch_{i+1}.json', 'w') as f:
                    json.dump(structure_results[-50:], f, indent=2, default=str)
        
        # Save all structure results
        with open(output_dir / 'all_structure_results.json', 'w') as f:
            json.dump(structure_results, f, indent=2, default=str)
        
        logger.info("Structure prediction completed")
    else:
        logger.info("Loading existing structure results...")
        with open(output_dir / 'all_structure_results.json', 'r') as f:
            structure_results = json.load(f)
    
    # Extract structural features
    logger.info("Extracting structural features...")
    features_list = []
    successful_structures = []
    
    for result in structure_results:
        if result['success']:
            features = feature_extractor.extract_features(result)
            if features:
                features['sequence_id'] = result['sequence_id']
                features['sequence'] = result['sequence']
                features_list.append(features)
                successful_structures.append(result)
    
    if not features_list:
        logger.error("No structural features extracted! All structure predictions failed.")
        logger.error("Cannot proceed with structural clustering analysis.")
        return
    
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(output_dir / 'structural_features.csv', index=False)
    logger.info(f"Extracted features for {len(features_df)} structures")
    
    # Structural clustering analysis
    clustering_results = clustering_analyzer.perform_structural_clustering(features_df)
    
    # Cross-modal validation
    if step4_results is not None and step5_results is not None:
        logger.info("Performing cross-modal validation...")
        
        # Align sequences between datasets
        common_sequences = set(features_df['sequence_id']).intersection(
            set(step4_results['sequence_id']).intersection(set(step5_results['sequence_id']))
        )
        
        if len(common_sequences) > 10:  # Need sufficient overlap
            # Filter to common sequences
            features_subset = features_df[features_df['sequence_id'].isin(common_sequences)]
            step4_subset = step4_results[step4_results['sequence_id'].isin(common_sequences)]
            step5_subset = step5_results[step5_results['sequence_id'].isin(common_sequences)]
            
            # Get cluster assignments
            structural_clusters = clustering_results['structural_clusters'][:len(features_subset)]
            sequence_clusters = step4_subset['predicted_cluster'].values
            functional_clusters = step5_subset['cluster_label'].values if 'cluster_label' in step5_subset.columns else sequence_clusters
            
            # Cross-modal validation
            cross_modal_metrics = clustering_analyzer.cross_modal_validation(
                structural_clusters, sequence_clusters, functional_clusters
            )
            clustering_results['cross_modal_metrics'] = cross_modal_metrics
            
            logger.info("Cross-modal validation completed")
        else:
            logger.warning("Insufficient sequence overlap for cross-modal validation")
    
    # Create visualizations
    visualizer.create_structural_plots(clustering_results, features_df)
    
    # Generate comprehensive report
    logger.info("Generating comprehensive multi-modal analysis report...")
    
    report = {
        'summary': {
            'total_sequences': len(sequences),
            'successful_structures': len(successful_structures),
            'success_rate': len(successful_structures) / len(sequences),
            'structural_clusters': clustering_results['optimal_k'],
            'features_extracted': len(clustering_results['feature_columns'])
        },
        'clustering_results': clustering_results,
        'cross_modal_validation': clustering_results.get('cross_modal_metrics', {}),
        'structure_metrics': {
            'mean_confidence': np.mean([r['structure_metrics']['mean_confidence'] for r in successful_structures]),
            'mean_radius_of_gyration': np.mean([r['structure_metrics']['radius_of_gyration'] for r in successful_structures]),
            'mean_compactness': np.mean([r['structure_metrics']['compactness'] for r in successful_structures])
        }
    }
    
    # Save report
    with open(output_dir / 'multi_modal_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create markdown report
    with open(output_dir / 'multi_modal_analysis_report.md', 'w') as f:
        f.write("# Multi-Modal Structure-Function Analysis Report\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total Sequences**: {report['summary']['total_sequences']}\n")
        f.write(f"- **Successful Structures**: {report['summary']['successful_structures']}\n")
        f.write(f"- **Success Rate**: {report['summary']['success_rate']:.1%}\n")
        f.write(f"- **Structural Clusters**: {report['summary']['structural_clusters']}\n")
        f.write(f"- **Features Extracted**: {report['summary']['features_extracted']}\n\n")
        
        if 'cross_modal_metrics' in clustering_results:
            f.write("## Cross-Modal Validation\n\n")
            metrics = clustering_results['cross_modal_metrics']
            f.write(f"- **Structural vs Sequence ARI**: {metrics['structural_vs_sequence_ari']:.3f}\n")
            f.write(f"- **Structural vs Functional ARI**: {metrics['structural_vs_functional_ari']:.3f}\n")
            f.write(f"- **Sequence vs Functional ARI**: {metrics['sequence_vs_functional_ari']:.3f}\n\n")
        
        f.write("## Structure Metrics\n\n")
        f.write(f"- **Mean Confidence**: {report['structure_metrics']['mean_confidence']:.3f}\n")
        f.write(f"- **Mean Radius of Gyration**: {report['structure_metrics']['mean_radius_of_gyration']:.2f} Å\n")
        f.write(f"- **Mean Compactness**: {report['structure_metrics']['mean_compactness']:.3f}\n\n")
    
    logger.info("Step 7 completed successfully!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
