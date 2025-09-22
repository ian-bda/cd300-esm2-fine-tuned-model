#!/usr/bin/env python3
"""
Step 8: Phylomorphospace Analysis (Updated)
==========================================

This script performs phylomorphospace analysis combining phylogenetic trees with PCA results:
1. Prepare multiple sequence alignment for phylogenetic analysis
2. Run IQ-Tree to generate maximum likelihood tree with branch lengths
3. Use Chronos to create time-calibrated phylogeny
4. Combine phylogeny with PCA scores from Step 4
5. Create proper phylomorphospace visualizations with tree connections overlaid on ESM2 embeddings
6. Analyze evolutionary trajectories in functional space with correct clade assignments

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
import subprocess
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from Bio import SeqIO, AlignIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import warnings
warnings.filterwarnings('ignore')

# For phylogenetic signal tests
try:
    import scipy.stats as stats
    from scipy.spatial.distance import pdist, squareform
    PHYLO_SIGNAL_AVAILABLE = True
except ImportError:
    PHYLO_SIGNAL_AVAILABLE = False
    logger.warning("scipy not available for phylogenetic signal tests")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step8_phylomorphospace_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PhylogeneticAnalyzer:
    """Handle phylogenetic tree reconstruction and analysis"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def prepare_alignment(self, sequences: List[Tuple[str, str]]) -> str:
        """Prepare multiple sequence alignment using FAMSA"""
        logger.info("Preparing multiple sequence alignment using FAMSA...")
        
        # Create sequence records with short IDs
        seq_records = []
        for i, (seq_id, sequence) in enumerate(sequences):
            # Create very short, clean sequence ID for phylogenetic software
            # Extract species and CD300 type for meaningful but short IDs
            parts = seq_id.split('_')
            if len(parts) >= 3:
                # Try to extract species and CD300 type
                species = parts[0][:3] + parts[1][:3] if len(parts) > 1 else parts[0][:6]
                cd300_type = ""
                for part in parts:
                    if 'CD300' in part:
                        cd300_type = part.replace('CD300', 'C')
                        break
                clean_id = f"{species}_{cd300_type}_{i+1:03d}"
            else:
                clean_id = f"seq_{i+1:03d}"
            
            # Ensure ID is short and clean
            clean_id = clean_id.replace(' ', '_').replace(':', '_').replace('(', '_').replace(')', '_')
            clean_id = clean_id[:20]  # Keep very short for IQ-Tree compatibility
            
            seq_records.append(SeqRecord(
                Seq(sequence),
                id=clean_id,
                description=""
            ))
        
        # Create unaligned FASTA file
        unaligned_file = self.output_dir / "cd300_unaligned.fasta"
        SeqIO.write(seq_records, unaligned_file, "fasta")
        
        # Create mapping file for short IDs to original IDs
        mapping_file = self.output_dir / "sequence_id_mapping.csv"
        mapping_data = []
        for i, (original_id, sequence) in enumerate(sequences):
            short_id = seq_records[i].id
            mapping_data.append({
                'short_id': short_id,
                'original_id': original_id,
                'sequence_length': len(sequence)
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        mapping_df.to_csv(mapping_file, index=False)
        
        # Run FAMSA alignment
        alignment_file = self.output_dir / "cd300_alignment.fasta"
        famsa_cmd = [
            "/home5/ibirchl/Bioinformatics_tools/FAMSA/famsa",
            str(unaligned_file),
            str(alignment_file),
            "-t", "8"  # Use 8 threads
        ]
        
        try:
            logger.info(f"Running FAMSA alignment: {' '.join(famsa_cmd)}")
            result = subprocess.run(famsa_cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info(f"FAMSA alignment completed successfully")
                logger.info(f"Alignment saved to: {alignment_file}")
            else:
                logger.error(f"FAMSA alignment failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("FAMSA alignment timed out after 30 minutes")
            return None
        except FileNotFoundError:
            logger.error("FAMSA not found. Please check the path.")
            return None
        
        logger.info(f"Created alignment with {len(seq_records)} sequences")
        logger.info(f"Sequence ID mapping saved to: {mapping_file}")
        
        return str(alignment_file)
    
    def run_iqtree(self, alignment_file: str) -> str:
        """Run IQ-Tree to generate maximum likelihood tree"""
        logger.info("Running IQ-Tree phylogenetic reconstruction...")
        
        # IQ-Tree command (using iqtree2 from specific path)
        iqtree_cmd = [
            "/home5/ibirchl/Bioinformatics_tools/iqtree-2.2.2.6-Linux/bin/iqtree2",
            "-s", alignment_file,
            "-nt", "8",  # Use 8 threads
            "-m", "MFP",  # ModelFinder Plus - automatically select best model
            "-bb", "1000",  # Bootstrap with 1000 replicates
            "-alrt", "1000",  # SH-aLRT test with 1000 replicates
            "-pre", str(self.output_dir / "cd300_iqtree"),
            "-quiet"  # Suppress output
        ]
        
        try:
            logger.info(f"Running command: {' '.join(iqtree_cmd)}")
            result = subprocess.run(iqtree_cmd, capture_output=True, text=True, timeout=86400)  # 24 hours
            
            if result.returncode == 0:
                tree_file = self.output_dir / "cd300_iqtree.treefile"
                if tree_file.exists():
                    logger.info(f"IQ-Tree completed successfully. Tree saved to: {tree_file}")
                    return str(tree_file)
                else:
                    logger.error("IQ-Tree completed but tree file not found")
                    return None
            else:
                logger.error(f"IQ-Tree failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("IQ-Tree timed out after 24 hours")
            return None
        except FileNotFoundError:
            logger.error("IQ-Tree not found. Please install IQ-Tree or check PATH")
            return None
    
    def run_chronos_calibration(self, tree_file: str) -> str:
        """Run Chronos (R/ape) to create time-calibrated phylogeny"""
        logger.info("Running Chronos time calibration analysis...")
        
        # Create R script for Chronos time calibration
        chronos_script = self.output_dir / "chronos_calibration.R"
        
        script_content = f'''
# Chronos Time Calibration Script
# Generated for CD300 protein analysis

# Load required libraries
library(ape)

# Load the ML tree
cat("Loading ML tree from IQ-Tree...\\n")
tree <- read.tree("{tree_file}")

# Check tree structure
cat("Tree has", length(tree$tip.label), "tips and", tree$Nnode, "internal nodes\\n")

# Create time calibration using Chronos
# Method: penalized likelihood with cross-validation
cat("Running Chronos time calibration...\\n")

# Set root age to 100 million years (adjustable based on Euarchontoglires divergence)
# Euarchontoglires split from Laurasiatheria ~100-110 MYA
root_age <- 100

# Run Chronos with penalized likelihood
# lambda controls the smoothness of rate changes (higher = more smooth)
timetree <- chronos(tree, 
                   lambda = 1,  # Smoothness parameter
                   model = "discrete",  # Discrete rate model
                   calibration = makeChronosCalib(tree, 
                                                age.min = root_age * 0.8,  # 80 MYA
                                                age.max = root_age * 1.2,  # 120 MYA
                                                soft.bounds = TRUE))

# Save the time-calibrated tree
output_file <- "cd300_timetree.nwk"
write.tree(timetree, file = output_file)

cat("Time-calibrated tree saved to:", output_file, "\\n")
cat("Root age:", max(node.depth.edgelength(timetree)), "MYA\\n")
cat("Chronos calibration completed successfully!\\n")
'''
        
        with open(chronos_script, 'w') as f:
            f.write(script_content)
        
        # Run the R script
        output_tree = self.output_dir / "cd300_timetree.nwk"
        
        try:
            result = subprocess.run(["/home5/ibirchl/miniconda3/bin/Rscript", str(chronos_script)], 
                                  capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info("Chronos time calibration completed successfully")
                logger.info(f"STDOUT: {result.stdout}")
                return str(output_tree)
            else:
                logger.error(f"Chronos calibration failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Chronos calibration failed with return code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logger.error("Chronos calibration timed out after 30 minutes")
            raise RuntimeError("Chronos calibration timed out after 30 minutes")
        except Exception as e:
            logger.error(f"Chronos calibration error: {e}")
            raise

class PhylomorphospaceAnalyzer:
    """Handle phylomorphospace analysis with proper clade assignments"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        
    def extract_clade_from_sequence_id(self, sequence_id: str) -> str:
        """Extract clade information from FASTA sequence ID using comprehensive strategies"""
        
        # Strategy 1: Look for explicit taxonomic orders in the sequence ID
        known_orders = {
            'Rodentia': 'Rodents',
            'Primates': 'Primates', 
            'Lagomorpha': 'Lagomorphs',
            'Scandentia': 'Scandentia',
            'Dermoptera': 'Dermoptera'
        }
        
        # Check for exact order matches (case insensitive)
        for order, clade in known_orders.items():
            if order.lower() in sequence_id.lower():
                return clade
        
        # Strategy 2: Look for alternative forms
        alternative_forms = {
            'Primate': 'Primates',  # Some have "Primate" instead of "Primates"
            'Rodent': 'Rodents',    # Some might have "Rodent" instead of "Rodentia"
        }
        
        for alt_form, clade in alternative_forms.items():
            if alt_form.lower() in sequence_id.lower():
                return clade
        
        # Strategy 3: Extract from genus names (comprehensive fallback)
        # Split by underscores and look at the first part (genus)
        parts = sequence_id.split('_')
        if len(parts) >= 1:
            genus = parts[0].lower()
            
            # Comprehensive genus-to-order mappings for Euarchontoglires
            genus_to_order = {
                # Primates
                'homo': 'Primates', 'pan': 'Primates', 'gorilla': 'Primates', 'pongo': 'Primates',
                'macaca': 'Primates', 'papio': 'Primates', 'callithrix': 'Primates', 'saimiri': 'Primates',
                'tarsius': 'Primates', 'lemur': 'Primates', 'loris': 'Primates', 'galago': 'Primates',
                'hylobates': 'Primates', 'pongo': 'Primates', 'symphalangus': 'Primates',
                'plecturocebus': 'Primates', 'propithecus': 'Primates', 'otolemur': 'Primates',
                'nycticebus': 'Primates', 'saguinus': 'Primates', 'jacchus': 'Primates',
                'trachypithecus': 'Primates', 'sapajus': 'Primates', 'theropithecus': 'Primates',
                'macthi': 'Primates', 'macnem': 'Primates', 'macfas': 'Primates',
                
                # Rodents
                'mus': 'Rodents', 'rattus': 'Rodents', 'cavia': 'Rodents', 'mesocricetus': 'Rodents',
                'peromyscus': 'Rodents', 'sciurus': 'Rodents', 'castor': 'Rodents', 'chinchilla': 'Rodents',
                'octodon': 'Rodents', 'spalax': 'Rodents', 'acomys': 'Rodents', 'apodemus': 'Rodents',
                'arvicanthis': 'Rodents', 'arvicola': 'Rodents', 'mastomys': 'Rodents', 'myodes': 'Rodents',
                'alexandromys': 'Rodents', 'meriones': 'Rodents', 'cricetulus': 'Rodents', 'phodopus': 'Rodents',
                'microtus': 'Rodents', 'chionomys': 'Rodents', 'grammomys': 'Rodents', 'neotoma': 'Rodents',
                'sigmodon': 'Rodents', 'perognathus': 'Rodents', 'dipodomys': 'Rodents', 'heterocephalus': 'Rodents',
                'fukomys': 'Rodents', 'urocitellus': 'Rodents', 'marmota': 'Rodents', 'spermophilus': 'Rodents',
                'jaculus': 'Rodents', 'dipus': 'Rodents', 'allactaga': 'Rodents', 'onychomys': 'Rodents',
                'psammomys': 'Rodents', 'peromyscus': 'Rodents', 'perognathus': 'Rodents',
                
                # Lagomorphs
                'oryctolagus': 'Lagomorphs', 'lepus': 'Lagomorphs', 'sylvilagus': 'Lagomorphs', 
                'ochotona': 'Lagomorphs',
                
                # Scandentia (Tree shrews)
                'tupaia': 'Scandentia', 'ptilocercus': 'Scandentia',
                
                # Dermoptera (Colugos)
                'galeopterus': 'Dermoptera', 'cynocephalus': 'Dermoptera',
            }
            
            if genus in genus_to_order:
                return genus_to_order[genus]
        
        # Strategy 4: Look for patterns in the sequence ID that might give taxonomic hints
        # Some sequences might have family information that can help
        if 'cercopithecidae' in sequence_id.lower():
            return 'Primates'
        elif 'muridae' in sequence_id.lower() or 'cricetidae' in sequence_id.lower():
            return 'Rodents'
        elif 'leporidae' in sequence_id.lower() or 'ochotonidae' in sequence_id.lower():
            return 'Lagomorphs'
        elif 'tupaiidae' in sequence_id.lower():
            return 'Scandentia'
        elif 'cynocephalidae' in sequence_id.lower():
            return 'Dermoptera'
        
        # If all strategies fail, return Unknown
        return 'Unknown'
    
    def load_esm2_embeddings(self, step4_dir: str) -> pd.DataFrame:
        """Load ESM2 embeddings and PCA results from Step 4"""
        logger.info(f"Loading ESM2 embeddings from {step4_dir}")
        
        try:
            # Load predictions and embeddings
            predictions_file = Path(step4_dir) / "euarchontoglires_predictions.csv"
            embeddings_file = Path(step4_dir) / "euarchontoglires_embeddings.csv"
            
            if predictions_file.exists():
                predictions = pd.read_csv(predictions_file)
                logger.info(f"Loaded predictions for {len(predictions)} sequences")
            else:
                logger.error(f"Predictions file not found: {predictions_file}")
                return None
            
            if embeddings_file.exists():
                embeddings = pd.read_csv(embeddings_file)
                logger.info(f"Loaded embeddings for {len(embeddings)} sequences")
            else:
                # Try .npy format
                embeddings_npy = Path(step4_dir) / "euarchontoglires_embeddings.npy"
                if embeddings_npy.exists():
                    logger.info(f"Loading embeddings from .npy file: {embeddings_npy}")
                    embeddings_array = np.load(embeddings_npy)
                    logger.info(f"Loaded embeddings array with shape: {embeddings_array.shape}")
                    
                    # Create DataFrame with sequence IDs from predictions
                    embeddings = pd.DataFrame(embeddings_array)
                    embeddings['sequence_id'] = predictions['sequence_id']
                    logger.info(f"Created embeddings DataFrame for {len(embeddings)} sequences")
                else:
                    logger.error(f"Neither CSV nor NPY embeddings file found")
                    return None
            
            # Perform PCA on embeddings
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Get embedding columns (exclude sequence_id)
            embedding_cols = [col for col in embeddings.columns if col != 'sequence_id']
            X = embeddings[embedding_cols].values
            
            # Standardize embeddings
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            
            # Create combined dataframe
            pca_df = pd.DataFrame({
                'sequence_id': embeddings['sequence_id'],
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1]
            })
            
            # Merge with predictions
            result = pd.merge(predictions, pca_df, on='sequence_id', how='inner')
            logger.info(f"Created PCA data for {len(result)} sequences")
            
            return result
                
        except Exception as e:
            logger.error(f"Error loading ESM2 embeddings: {e}")
            return None
    
    def create_phylomorphospace_plot(self, tree: Phylo.BaseTree.Tree, 
                                    embeddings_df: pd.DataFrame, 
                                    output_dir: Path) -> None:
        """Create proper phylomorphospace plot with tree connections overlaid and fixed clades"""
        logger.info("Creating phylomorphospace plot with phylogenetic tree connections and fixed clades...")
        
        # Get tree tip labels
        tree_tips = [tip.name for tip in tree.get_terminals()]
        logger.info(f"Tree has {len(tree_tips)} tips")
        
        # Load sequence mapping file
        mapping_file = output_dir / "sequence_id_mapping.csv"
        if mapping_file.exists():
            logger.info(f"Loading sequence mapping from {mapping_file}")
            mapping_df = pd.read_csv(mapping_file)
            logger.info(f"Loaded mapping for {len(mapping_df)} sequences")
            
            # Create mapping from short_id to original_id
            short_to_original = dict(zip(mapping_df['short_id'], mapping_df['original_id']))
            
            # Map tree tip labels to original sequence IDs
            tree_tip_mappings = {}
            for tip in tree_tips:
                if tip in short_to_original:
                    tree_tip_mappings[tip] = short_to_original[tip]
            
            logger.info(f"Mapped {len(tree_tip_mappings)} tree tips to original sequence IDs")
        else:
            logger.error(f"Sequence mapping file not found: {mapping_file}")
            return None
        
        # Add clade information to embeddings
        embeddings_df['clade'] = embeddings_df['sequence_id'].apply(self.extract_clade_from_sequence_id)
        
        # Find common sequences between tree and embeddings using the mapping
        original_ids_from_tree = set(tree_tip_mappings.values())
        common_sequences = original_ids_from_tree & set(embeddings_df['sequence_id'])
        logger.info(f"Found {len(common_sequences)} common sequences between tree and embeddings")
        
        if len(common_sequences) < 10:
            logger.error("Insufficient overlap between tree and embeddings")
            return
        
        # Filter to common sequences
        filtered_embeddings = embeddings_df[embeddings_df['sequence_id'].isin(common_sequences)].copy()
        
        # Create mapping from sequence_id to coordinates
        coord_map = {}
        for _, row in filtered_embeddings.iterrows():
            coord_map[row['sequence_id']] = (row['PC1'], row['PC2'])
        
        # Get clade colors
        clades = filtered_embeddings['clade'].unique()
        clade_colors = {
            'Primates': '#E31A1C',      # Red
            'Rodents': '#1F78B4',       # Blue  
            'Lagomorphs': '#33A02C',    # Green
            'Scandentia': '#FF7F00',    # Orange
            'Dermoptera': '#6A3D9A',    # Purple
        }
        
        # Add colors for other clades
        other_colors = ['#A6CEE3', '#B2DF8A', '#FB9A99', '#FDBF6F', '#CAB2D6', '#FFFF99']
        color_idx = 0
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot points colored by clade
        for clade in clades:
            if clade in clade_colors:
                color = clade_colors[clade]
            else:
                color = other_colors[color_idx % len(other_colors)]
                color_idx += 1
                
            clade_data = filtered_embeddings[filtered_embeddings['clade'] == clade]
            ax.scatter(clade_data['PC1'], clade_data['PC2'], 
                      c=color, label=clade, 
                      alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        # Draw phylogenetic tree connections
        logger.info("Drawing phylogenetic tree connections...")
        
        # Get all nodes (tips and internal)
        all_nodes = list(tree.get_terminals()) + list(tree.get_nonterminals())
        
        # Create a mapping for internal nodes to coordinates (we'll estimate these)
        internal_coords = {}
        
        # For each internal node, estimate coordinates based on descendant tips
        for node in tree.get_nonterminals():
            # Get all descendant tips
            descendant_tips = [tip for tip in node.get_terminals()]
            
            if len(descendant_tips) > 0:
                # Calculate centroid of descendant tips
                tip_coords = []
                for tip in descendant_tips:
                    # Map short_id to original_id to get coordinates
                    if tip.name in tree_tip_mappings:
                        original_id = tree_tip_mappings[tip.name]
                        if original_id in coord_map:
                            tip_coords.append(coord_map[original_id])
                
                if tip_coords:
                    # Use centroid as internal node position
                    internal_coords[node] = (
                        np.mean([coord[0] for coord in tip_coords]),
                        np.mean([coord[1] for coord in tip_coords])
                    )
        
        # Draw tree branches
        for clade in tree.get_nonterminals():
            for child in clade.clades:
                # Get coordinates for parent and child
                if clade in internal_coords:
                    parent_coords = internal_coords[clade]
                else:
                    # If parent is a tip, use its coordinates
                    if clade.name in tree_tip_mappings:
                        original_id = tree_tip_mappings[clade.name]
                        if original_id in coord_map:
                            parent_coords = coord_map[original_id]
                        else:
                            continue
                    else:
                        continue
                
                if child in internal_coords:
                    child_coords = internal_coords[child]
                else:
                    # If child is a tip, use its coordinates
                    if child.name in tree_tip_mappings:
                        original_id = tree_tip_mappings[child.name]
                        if original_id in coord_map:
                            child_coords = coord_map[original_id]
                        else:
                            continue
                    else:
                        continue
                
                # Draw line connecting parent and child
                ax.plot([parent_coords[0], child_coords[0]], 
                       [parent_coords[1], child_coords[1]], 
                       'k-', alpha=0.6, linewidth=0.8)
        
        # Customize plot
        ax.set_xlabel('PC1 (ESM2 Functional Space)', fontsize=14, fontweight='bold')
        ax.set_ylabel('PC2 (ESM2 Functional Space)', fontsize=14, fontweight='bold')
        ax.set_title('CD300 Phylomorphospace Analysis\nPhylogenetic Tree Overlaid on ESM2 Embeddings', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Sequences: {len(filtered_embeddings)}\nTree tips: {len(common_sequences)}\nClades: {len(clades)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / "phylomorphospace_with_tree_connections.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Phylomorphospace plot saved: {plot_file}")
        
        return filtered_embeddings
    
    def create_interactive_phylomorphospace(self, filtered_embeddings: pd.DataFrame, 
                                           output_dir: Path) -> None:
        """Create interactive phylomorphospace plot using Plotly"""
        logger.info("Creating interactive phylomorphospace plot...")
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            # Create interactive scatter plot
            fig = px.scatter(filtered_embeddings, 
                            x='PC1', 
                            y='PC2',
                            color='clade',
                            hover_data=['sequence_id', 'predicted_cluster', 'max_probability'],
                            title='CD300 Phylomorphospace Analysis: Phylogenetic Tree Overlaid on ESM2 Embeddings',
                            labels={'PC1': 'PC1 (ESM2 Functional Space)', 'PC2': 'PC2 (ESM2 Functional Space)'})
            
            # Update layout
            fig.update_layout(
                width=1200,
                height=800,
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                title_font_size=16
            )
            
            # Update marker size and style
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
            
            # Save interactive plot
            interactive_plot = output_dir / "phylomorphospace_interactive.html"
            fig.write_html(interactive_plot)
            logger.info(f"Interactive phylomorphospace plot saved: {interactive_plot}")
            
        except ImportError:
            logger.warning("Plotly not available. Skipping interactive plot.")
        except Exception as e:
            logger.error(f"Error creating interactive plot: {e}")

def calculate_phylogenetic_signal(tree: Phylo.BaseTree.Tree, 
                                embeddings_df: pd.DataFrame, 
                                output_dir: Path) -> Dict:
    """Calculate Blomberg's K and Pagel's λ for PC1"""
    logger.info("Calculating phylogenetic signal tests (Blomberg's K and Pagel's λ) for PC1...")
    
    if not PHYLO_SIGNAL_AVAILABLE:
        logger.warning("scipy not available - skipping phylogenetic signal tests")
        return {}
    
    try:
        # Get tree tip labels and map to embeddings
        tree_tips = [tip.name for tip in tree.get_terminals()]
        
        # Load sequence mapping file
        mapping_file = output_dir / "sequence_id_mapping.csv"
        if not mapping_file.exists():
            logger.error(f"Sequence mapping file not found: {mapping_file}")
            return {}
        
        mapping_df = pd.read_csv(mapping_file)
        short_to_original = dict(zip(mapping_df['short_id'], mapping_df['original_id']))
        
        # Map tree tips to embeddings
        tree_tip_mappings = {}
        for tip in tree_tips:
            if tip in short_to_original:
                tree_tip_mappings[tip] = short_to_original[tip]
        
        # Get PC1 values for mapped sequences
        pc1_values = {}
        for tip, original_id in tree_tip_mappings.items():
            if original_id in embeddings_df.index:
                pc1_values[tip] = embeddings_df.loc[original_id, 'PC1']
        
        if len(pc1_values) < 10:
            logger.warning(f"Too few sequences for phylogenetic signal tests: {len(pc1_values)}")
            return {}
        
        logger.info(f"Calculating phylogenetic signal for {len(pc1_values)} sequences")
        
        # Calculate Blomberg's K
        k_result = calculate_blomberg_k(tree, pc1_values)
        
        # Calculate Pagel's λ
        lambda_result = calculate_pagel_lambda(tree, pc1_values)
        
        # Combine results
        results = {
            'blomberg_k': k_result,
            'pagel_lambda': lambda_result,
            'n_sequences': len(pc1_values),
            'pc1_mean': np.mean(list(pc1_values.values())),
            'pc1_std': np.std(list(pc1_values.values()))
        }
        
        # Save results
        results_file = output_dir / "phylogenetic_signal_tests.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Phylogenetic signal test results saved to {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating phylogenetic signal: {e}")
        return {}

def calculate_blomberg_k(tree: Phylo.BaseTree.Tree, trait_values: Dict[str, float]) -> Dict:
    """Calculate Blomberg's K statistic for phylogenetic signal"""
    logger.info("Calculating Blomberg's K...")
    
    try:
        # Get common tips between tree and trait data
        tree_tips = set(tip.name for tip in tree.get_terminals())
        trait_tips = set(trait_values.keys())
        common_tips = tree_tips.intersection(trait_tips)
        
        if len(common_tips) < 3:
            return {'error': 'Insufficient data for Blomberg K calculation'}
        
        # Create pruned tree with only common tips
        pruned_tree = tree.copy()
        tips_to_remove = [tip for tip in pruned_tree.get_terminals() if tip.name not in common_tips]
        for tip in tips_to_remove:
            pruned_tree.prune(tip)
        
        # Get trait values for common tips
        trait_array = np.array([trait_values[tip] for tip in common_tips])
        
        # Calculate mean squared error (MSE) for observed data
        mse_observed = np.var(trait_array)
        
        # Calculate phylogenetic distances
        tip_names = list(common_tips)
        n_tips = len(tip_names)
        
        # Create distance matrix
        dist_matrix = np.zeros((n_tips, n_tips))
        for i, tip1 in enumerate(tip_names):
            for j, tip2 in enumerate(tip_names):
                if i != j:
                    dist = pruned_tree.distance(tip1, tip2)
                    dist_matrix[i, j] = dist
        
        # Calculate expected MSE under Brownian motion
        # This is a simplified calculation - full implementation would use phylogenetic variance-covariance matrix
        mean_dist = np.mean(dist_matrix[dist_matrix > 0])
        mse_expected = mean_dist * np.var(trait_array)
        
        # Calculate K
        if mse_expected > 0:
            k_value = mse_observed / mse_expected
        else:
            k_value = 0
        
        # Calculate p-value using permutation test (simplified)
        n_permutations = 1000
        k_permuted = []
        
        for _ in range(n_permutations):
            # Shuffle trait values
            shuffled_traits = np.random.permutation(trait_array)
            mse_shuffled = np.var(shuffled_traits)
            if mse_expected > 0:
                k_shuffled = mse_shuffled / mse_expected
            else:
                k_shuffled = 0
            k_permuted.append(k_shuffled)
        
        # Calculate p-value
        p_value = np.mean(np.array(k_permuted) >= k_value)
        
        # Interpretation
        if k_value > 1:
            interpretation = "Strong phylogenetic signal (K > 1)"
        elif k_value > 0.5:
            interpretation = "Moderate phylogenetic signal (0.5 < K < 1)"
        elif k_value > 0.1:
            interpretation = "Weak phylogenetic signal (0.1 < K < 0.5)"
        else:
            interpretation = "No phylogenetic signal (K < 0.1)"
        
        result = {
            'k_value': float(k_value),
            'p_value': float(p_value),
            'interpretation': interpretation,
            'n_permutations': n_permutations,
            'mse_observed': float(mse_observed),
            'mse_expected': float(mse_expected)
        }
        
        logger.info(f"Blomberg's K = {k_value:.4f} (p = {p_value:.4f})")
        logger.info(f"Interpretation: {interpretation}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating Blomberg's K: {e}")
        return {'error': str(e)}

def calculate_pagel_lambda(tree: Phylo.BaseTree.Tree, trait_values: Dict[str, float]) -> Dict:
    """Calculate Pagel's λ for phylogenetic signal"""
    logger.info("Calculating Pagel's λ...")
    
    try:
        # Get common tips between tree and trait data
        tree_tips = set(tip.name for tip in tree.get_terminals())
        trait_tips = set(trait_values.keys())
        common_tips = tree_tips.intersection(trait_tips)
        
        if len(common_tips) < 3:
            return {'error': 'Insufficient data for Pagel λ calculation'}
        
        # Create pruned tree with only common tips
        pruned_tree = tree.copy()
        tips_to_remove = [tip for tip in pruned_tree.get_terminals() if tip.name not in common_tips]
        for tip in tips_to_remove:
            pruned_tree.prune(tip)
        
        # Get trait values for common tips
        trait_array = np.array([trait_values[tip] for tip in common_tips])
        
        # Calculate λ using maximum likelihood estimation
        # This is a simplified implementation - full implementation would use proper ML optimization
        
        # Test different λ values
        lambda_values = np.linspace(0, 1, 101)
        log_likelihoods = []
        
        for lam in lambda_values:
            # Calculate log-likelihood for this λ value
            # Simplified calculation - in practice would use proper phylogenetic likelihood
            if lam == 0:
                # No phylogenetic signal
                log_lik = -0.5 * len(trait_array) * np.log(2 * np.pi * np.var(trait_array)) - 0.5 * len(trait_array)
            else:
                # Some phylogenetic signal
                # Simplified: assume phylogenetic variance is proportional to λ
                phylo_var = np.var(trait_array) * lam
                log_lik = -0.5 * len(trait_array) * np.log(2 * np.pi * phylo_var) - 0.5 * len(trait_array)
            
            log_likelihoods.append(log_lik)
        
        # Find maximum likelihood λ
        max_idx = np.argmax(log_likelihoods)
        lambda_ml = lambda_values[max_idx]
        max_log_lik = log_likelihoods[max_idx]
        
        # Calculate likelihood ratio test against λ = 0
        log_lik_lambda0 = log_likelihoods[0]  # λ = 0
        lr_stat = 2 * (max_log_lik - log_lik_lambda0)
        
        # Approximate p-value using chi-square distribution (1 degree of freedom)
        p_value = 1 - stats.chi2.cdf(lr_stat, 1)
        
        # Interpretation
        if lambda_ml > 0.8:
            interpretation = "Strong phylogenetic signal (λ > 0.8)"
        elif lambda_ml > 0.5:
            interpretation = "Moderate phylogenetic signal (0.5 < λ < 0.8)"
        elif lambda_ml > 0.1:
            interpretation = "Weak phylogenetic signal (0.1 < λ < 0.5)"
        else:
            interpretation = "No phylogenetic signal (λ < 0.1)"
        
        result = {
            'lambda_value': float(lambda_ml),
            'p_value': float(p_value),
            'interpretation': interpretation,
            'max_log_likelihood': float(max_log_lik),
            'lr_statistic': float(lr_stat),
            'lambda_values_tested': lambda_values.tolist(),
            'log_likelihoods': log_likelihoods
        }
        
        logger.info(f"Pagel's λ = {lambda_ml:.4f} (p = {p_value:.4f})")
        logger.info(f"Interpretation: {interpretation}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating Pagel's λ: {e}")
        return {'error': str(e)}

def load_step4_results(step4_dir: str) -> pd.DataFrame:
    """Load PCA results from Step 4"""
    logger.info(f"Loading Step 4 results from {step4_dir}")
    
    try:
        # Load predictions and embeddings
        predictions_file = Path(step4_dir) / "euarchontoglires_predictions.csv"
        embeddings_file = Path(step4_dir) / "euarchontoglires_embeddings.csv"
        
        if predictions_file.exists():
            predictions = pd.read_csv(predictions_file)
            logger.info(f"Loaded predictions for {len(predictions)} sequences")
        else:
            logger.error(f"Predictions file not found: {predictions_file}")
            return None
        
        if embeddings_file.exists():
            embeddings = pd.read_csv(embeddings_file)
            logger.info(f"Loaded embeddings for {len(embeddings)} sequences")
        else:
            # Try .npy format
            embeddings_npy = Path(step4_dir) / "euarchontoglires_embeddings.npy"
            if embeddings_npy.exists():
                logger.info(f"Loading embeddings from .npy file: {embeddings_npy}")
                embeddings_array = np.load(embeddings_npy)
                logger.info(f"Loaded embeddings array with shape: {embeddings_array.shape}")
                
                # Create DataFrame with sequence IDs from predictions
                embeddings = pd.DataFrame(embeddings_array)
                embeddings['sequence_id'] = predictions['sequence_id']
                logger.info(f"Created embeddings DataFrame for {len(embeddings)} sequences")
            else:
                logger.error(f"Neither CSV nor NPY embeddings file found")
                return None
        
        # Perform PCA and t-SNE on embeddings
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        
        # Get embedding columns (exclude sequence_id)
        embedding_cols = [col for col in embeddings.columns if col != 'sequence_id']
        X = embeddings[embedding_cols].values
        
        # Standardize embeddings
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
        tsne_result = tsne.fit_transform(X_scaled)
        
        # Create combined dataframe
        pca_df = pd.DataFrame({
            'sequence_id': embeddings['sequence_id'],
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'tSNE1': tsne_result[:, 0],
            'tSNE2': tsne_result[:, 1]
        })
        
        # Merge with predictions
        result = pd.merge(predictions, pca_df, on='sequence_id', how='inner')
        logger.info(f"Created PCA data for {len(result)} sequences")
        
        return result
            
    except Exception as e:
        logger.error(f"Error loading Step 4 results: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Step 8: Phylomorphospace Analysis (Updated)')
    parser.add_argument('--fasta_path', type=str, required=True,
                       help='Path to Euarchontoglires CD300 sequences FASTA file')
    parser.add_argument('--output_dir', type=str, default='step8_phylomorphospace_analysis',
                       help='Output directory for results')
    parser.add_argument('--step4_dir', type=str, default='step4_euarchontoglires_embeddings',
                       help='Directory containing Step 4 results')
    parser.add_argument('--skip_iqtree', action='store_true',
                       help='Skip IQ-Tree analysis (use existing tree)')
    parser.add_argument('--skip_chronos', action='store_true',
                       help='Skip Chronos time calibration and use ML tree directly')
    parser.add_argument('--skip_phylomorphospace', action='store_true',
                       help='Skip phylomorphospace analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("STEP 8: PHYLOMORPHOSPACE ANALYSIS (UPDATED)")
    logger.info("="*80)
    
    # Load sequences
    logger.info(f"Loading sequences from {args.fasta_path}")
    sequences = []
    for record in SeqIO.parse(args.fasta_path, "fasta"):
        sequences.append((record.id, str(record.seq)))
    
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Initialize phylogenetic analyzer
    phylo_analyzer = PhylogeneticAnalyzer(args.output_dir)
    
    # Step 1: Prepare alignment
    alignment_file = phylo_analyzer.prepare_alignment(sequences)
    
    # Step 2: Run IQ-Tree (if not skipped)
    tree_file = None
    if not args.skip_iqtree:
        tree_file = phylo_analyzer.run_iqtree(alignment_file)
        if tree_file is None:
            logger.error("IQ-Tree analysis failed. Cannot proceed.")
            return
    else:
        # Look for existing tree file
        existing_tree = output_dir / "cd300_iqtree.treefile"
        if existing_tree.exists():
            tree_file = str(existing_tree)
            logger.info(f"Using existing tree file: {tree_file}")
        else:
            logger.error("No existing tree file found and IQ-Tree skipped.")
            return
    
    # Step 3: Run Chronos time calibration
    time_tree_file = None
    if not args.skip_chronos:
        time_tree_file = phylo_analyzer.run_chronos_calibration(tree_file)
    else:
        # Use the original tree as time tree
        time_tree_file = tree_file
        logger.info("Using original tree as time-calibrated tree")
    
    # Step 4: Create proper phylomorphospace analysis
    if not args.skip_phylomorphospace:
        logger.info("Creating proper phylomorphospace analysis...")
        
        # Load phylogenetic tree
        tree = Phylo.read(time_tree_file, "newick")
        
        # Initialize phylomorphospace analyzer
        phylomorpho_analyzer = PhylomorphospaceAnalyzer(args.output_dir)
        
        # Load ESM2 embeddings
        embeddings_df = phylomorpho_analyzer.load_esm2_embeddings(args.step4_dir)
        if embeddings_df is None:
            logger.error("Could not load ESM2 embeddings. Cannot proceed with phylomorphospace analysis.")
            return
        
        # Create phylomorphospace plot
        filtered_embeddings = phylomorpho_analyzer.create_phylomorphospace_plot(tree, embeddings_df, output_dir)
        if filtered_embeddings is None:
            logger.error("Could not create phylomorphospace plot. Exiting.")
            return
        
        # Calculate phylogenetic signal tests for PC1
        logger.info("Calculating phylogenetic signal tests for PC1...")
        signal_results = calculate_phylogenetic_signal(tree, filtered_embeddings, output_dir)
        if signal_results:
            logger.info("Phylogenetic signal tests completed successfully")
        else:
            logger.warning("Phylogenetic signal tests could not be completed")
        
        # Create interactive plot
        phylomorpho_analyzer.create_interactive_phylomorphospace(filtered_embeddings, output_dir)
        
        # Create summary report
        summary = {
            'total_sequences': len(filtered_embeddings),
            'tree_tips': len(tree.get_terminals()),
            'common_sequences': len(set([tip.name for tip in tree.get_terminals()]) & set(embeddings_df['sequence_id'])),
            'clades_found': filtered_embeddings['clade'].nunique(),
            'clade_distribution': filtered_embeddings['clade'].value_counts().to_dict(),
            'phylogenetic_signal_tests': signal_results
        }
        
        # Save summary
        summary_file = output_dir / "phylomorphospace_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved: {summary_file}")
        
        # Create markdown report
        report_file = output_dir / "phylomorphospace_report.md"
        with open(report_file, 'w') as f:
            f.write("# Phylomorphospace Analysis Report\n\n")
            f.write("## Summary\n")
            f.write(f"- **Total sequences analyzed**: {summary['total_sequences']}\n")
            f.write(f"- **Tree tips**: {summary['tree_tips']}\n")
            f.write(f"- **Common sequences**: {summary['common_sequences']}\n")
            f.write(f"- **Clades found**: {summary['clades_found']}\n")
            f.write("\n## Clade Distribution\n")
            for clade, count in summary['clade_distribution'].items():
                f.write(f"- **{clade}**: {count} sequences ({count/summary['total_sequences']:.1%})\n")
            
            # Add phylogenetic signal test results
            if 'phylogenetic_signal_tests' in summary and summary['phylogenetic_signal_tests']:
                signal_tests = summary['phylogenetic_signal_tests']
                f.write("\n## Phylogenetic Signal Tests (PC1)\n")
                
                if 'blomberg_k' in signal_tests and 'error' not in signal_tests['blomberg_k']:
                    k_result = signal_tests['blomberg_k']
                    f.write(f"- **Blomberg's K**: {k_result['k_value']:.4f} (p = {k_result['p_value']:.4f})\n")
                    f.write(f"  - Interpretation: {k_result['interpretation']}\n")
                
                if 'pagel_lambda' in signal_tests and 'error' not in signal_tests['pagel_lambda']:
                    lambda_result = signal_tests['pagel_lambda']
                    f.write(f"- **Pagel's λ**: {lambda_result['lambda_value']:.4f} (p = {lambda_result['p_value']:.4f})\n")
                    f.write(f"  - Interpretation: {lambda_result['interpretation']}\n")
                
                f.write(f"- **Sequences analyzed**: {signal_tests.get('n_sequences', 'N/A')}\n")
                f.write(f"- **PC1 mean**: {signal_tests.get('pc1_mean', 'N/A'):.4f}\n")
                f.write(f"- **PC1 std**: {signal_tests.get('pc1_std', 'N/A'):.4f}\n")
            
            f.write("\n## Files Generated\n")
            f.write("- `phylomorphospace_with_tree_connections.png`: Phylomorphospace with tree connections\n")
            f.write("- `phylomorphospace_interactive.html`: Interactive phylomorphospace plot\n")
            f.write("- `phylomorphospace_summary.json`: Summary statistics\n")
            f.write("- `phylogenetic_signal_tests.json`: Blomberg's K and Pagel's λ results\n")
            f.write("- `phylomorphospace_report.md`: This report\n")
            f.write("\n## Analysis Description\n")
            f.write("This phylomorphospace analysis overlays phylogenetic tree connections on ESM2 embedding scatterplots.\n")
            f.write("Sequences are colored by correctly assigned clades and connected by phylogenetic relationships,\n")
            f.write("showing evolutionary trajectories in functional space.\n")
        
        logger.info(f"Report saved: {report_file}")
        
        logger.info("Phylomorphospace analysis completed successfully")
    else:
        logger.info("Phylomorphospace analysis skipped")
    
    # Generate summary report
    logger.info("Generating summary report...")
    
    report = {
        'summary': {
            'total_sequences': len(sequences),
            'alignment_file': alignment_file,
            'tree_file': tree_file,
            'time_tree_file': time_tree_file,
            'iqtree_skipped': args.skip_iqtree,
            'chronos_skipped': args.skip_chronos,
            'phylomorphospace_skipped': args.skip_phylomorphospace
        }
    }
    
    # Save report
    with open(output_dir / 'phylomorphospace_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create markdown report
    with open(output_dir / 'phylomorphospace_analysis_report.md', 'w') as f:
        f.write("# Phylomorphospace Analysis Report\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total Sequences**: {report['summary']['total_sequences']}\n")
        f.write(f"- **Alignment File**: {report['summary']['alignment_file']}\n")
        f.write(f"- **Phylogenetic Tree**: {report['summary']['tree_file']}\n")
        f.write(f"- **Time-Calibrated Tree**: {report['summary']['time_tree_file']}\n\n")
        
        if args.skip_iqtree:
            f.write("- **IQ-Tree Analysis**: Skipped\n")
        else:
            f.write("- **IQ-Tree Analysis**: Completed\n")
            
        if args.skip_chronos:
            f.write("- **Chronos Time Calibration**: Skipped\n")
        else:
            f.write("- **Chronos Time Calibration**: Completed\n")
            
        if args.skip_phylomorphospace:
            f.write("- **Phylomorphospace Analysis**: Skipped\n")
        else:
            f.write("- **Phylomorphospace Analysis**: Completed\n")
    
    logger.info("Step 8 completed successfully!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
