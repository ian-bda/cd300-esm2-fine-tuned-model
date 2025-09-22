#!/usr/bin/env python3
"""
Step 6: InterProScan Domain Analysis (Local Version - Updated)
============================================================

This script performs comprehensive domain analysis using local InterProScan:
1. Run InterProScan on all Euarchontoglires CD300 sequences
2. Parse domain predictions and functional sites
3. Generate mutation effects using fine-tuned ESM2 model
4. Create comprehensive domain and mutation analysis report with statistical significance testing

Author: AI Assistant
Date: 2025-01-11
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step6_interproscan_local.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocalInterProScanAnalyzer:
    """Analyze protein domains using local InterProScan"""
    
    def __init__(self, interproscan_path: str, output_dir: str):
        self.interproscan_path = Path(interproscan_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Verify InterProScan installation
        if not self.interproscan_path.exists():
            raise FileNotFoundError(f"InterProScan not found at {interproscan_path}")
        
        self.interproscan_cmd = self.interproscan_path / "interproscan.sh"
        if not self.interproscan_cmd.exists():
            raise FileNotFoundError(f"InterProScan script not found at {self.interproscan_cmd}")
    
    def run_interproscan(self, fasta_path: str) -> str:
        """Run InterProScan on FASTA file"""
        logger.info(f"Running InterProScan on {fasta_path}")
        
        output_file = self.output_dir / "interproscan_results.tsv"
        
        # InterProScan command - using the working format from your other project
        cmd = [
            str(self.interproscan_cmd),
            "-i", fasta_path,
            "-f", "tsv",
            "-o", str(output_file),
            "-appl", "SMART",
            "-cpu", "8"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)  # 24 hour timeout
            
            if result.returncode != 0:
                logger.error(f"InterProScan failed: {result.stderr}")
                return None
            
            logger.info(f"InterProScan completed successfully")
            return str(output_file)
            
        except subprocess.TimeoutExpired:
            logger.error("InterProScan timed out after 24 hours")
            return None
        except Exception as e:
            logger.error(f"Error running InterProScan: {e}")
            return None
    
    def parse_interproscan_results(self, tsv_file: str) -> Dict:
        """Parse InterProScan TSV results"""
        logger.info(f"Parsing InterProScan results from {tsv_file}")
        
        domains = defaultdict(list)
        
        try:
            with open(tsv_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 12:
                            protein_id = parts[0]
                            domain_id = parts[4]
                            domain_name = parts[5]
                            start = int(parts[6])
                            end = int(parts[7])
                            
                            domains[protein_id].append({
                                'domain_id': domain_id,
                                'domain_name': domain_name,
                                'start': start,
                                'end': end
                            })
            
            logger.info(f"Parsed domains for {len(domains)} proteins")
            return dict(domains)
            
        except Exception as e:
            logger.error(f"Error parsing InterProScan results: {e}")
            return {}
    
    def get_domain_at_position(self, protein_id: str, position: int, domains: Dict) -> Dict:
        """Get domain information for a specific position"""
        if protein_id not in domains:
            return {
                'domain': 'Other',
                'domain_type': 'other',
                'domain_description': 'Non-domain region',
                'domain_start': 0,
                'domain_end': 0
            }
        
        for domain in domains[protein_id]:
            if domain['start'] <= position <= domain['end']:
                return {
                    'domain': domain['domain_name'],
                    'domain_type': 'SMART',
                    'domain_description': f"SMART domain: {domain['domain_name']}",
                    'domain_start': domain['start'],
                    'domain_end': domain['end']
                }
        
        return {
            'domain': 'Other',
            'domain_type': 'other',
            'domain_description': 'Non-domain region',
            'domain_start': 0,
            'domain_end': 0
        }

class MutationEffectPredictor:
    """Predict mutation effects using fine-tuned ESM2 model"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the fine-tuned ESM2 model"""
        logger.info(f"Loading fine-tuned ESM2 model from {self.model_path}")
        
        try:
            import torch
            from transformers import EsmForSequenceClassification, EsmTokenizer
            
            # Load model and tokenizer
            self.model = EsmForSequenceClassification.from_pretrained(str(self.model_path))
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_sequence_class(self, sequence: str) -> Tuple[int, float]:
        """Predict class and confidence for a sequence"""
        if self.model is None or self.tokenizer is None:
            return 0, 0.0
        
        try:
            import torch
            
            # Tokenize sequence
            inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"Error predicting sequence class: {e}")
            return 0, 0.0
    
    def generate_mutations(self, sequence: str, max_mutations_per_position: int = 1) -> List[Tuple[int, str, str]]:
        """Generate mutations for a sequence"""
        mutations = []
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        for pos in range(len(sequence)):
            original_aa = sequence[pos]
            position_mutations = 0
            
            for new_aa in amino_acids:
                if new_aa != original_aa and position_mutations < max_mutations_per_position:
                    mutations.append((pos + 1, original_aa, new_aa))  # 1-indexed positions
                    position_mutations += 1
        
        return mutations
    
    def predict_mutation_effects(self, sequences: Dict[str, str], domains: Dict, 
                                max_mutations_per_position: int = 1) -> List[Dict]:
        """Predict effects of mutations on all sequences"""
        logger.info("Predicting mutation effects for all sequences...")
        
        if not self.load_model():
            logger.error("Could not load model. Cannot predict mutation effects.")
            return []
        
        results = []
        total_sequences = len(sequences)
        
        for i, (seq_id, sequence) in enumerate(sequences.items()):
            if i % 50 == 0:
                logger.info(f"Processing sequence {i+1}/{total_sequences}: {seq_id}")
            
            # Get original prediction
            original_class, original_confidence = self.predict_sequence_class(sequence)
            
            # Generate mutations
            mutations = self.generate_mutations(sequence, max_mutations_per_position)
            
            for position, original_aa, mutated_aa in mutations:
                try:
                    # Create mutated sequence
                    mutated_sequence = sequence[:position-1] + mutated_aa + sequence[position:]
                    
                    # Get prediction for mutated sequence
                    mutated_class, mutated_confidence = self.predict_sequence_class(mutated_sequence)
                    
                    # Get domain information
                    domain_info = self.get_domain_at_position(seq_id, position, domains)
                    
                    # Calculate effect metrics
                    prediction_change = (original_class != mutated_class)
                    confidence_change = mutated_confidence - original_confidence
                    stability_score = abs(confidence_change)
                    effect_magnitude = abs(confidence_change)
                    
                    # Determine effect type
                    if effect_magnitude > 0.1:
                        effect_type = 'deleterious'
                    elif effect_magnitude > 0.05:
                        effect_type = 'moderate'
                    else:
                        effect_type = 'neutral'
                    
                    results.append({
                        'position': position,
                        'original_aa': original_aa,
                        'mutated_aa': mutated_aa,
                        'domain': domain_info['domain'],
                        'domain_type': domain_info['domain_type'],
                        'domain_description': domain_info['domain_description'],
                        'domain_start': domain_info['domain_start'],
                        'domain_end': domain_info['domain_end'],
                        'original_prediction': original_class,
                        'mutated_prediction': mutated_class,
                        'prediction_change': prediction_change,
                        'confidence_change': confidence_change,
                        'stability_score': stability_score,
                        'effect_magnitude': effect_magnitude,
                        'effect_type': effect_type,
                        'sequence_id': seq_id
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing mutation {position}{original_aa}->{mutated_aa} in {seq_id}: {e}")
                    results.append({
                        'position': position,
                        'original_aa': original_aa,
                        'mutated_aa': mutated_aa,
                        'domain': 'Error',
                        'domain_type': 'error',
                        'domain_description': 'Error processing mutation',
                        'domain_start': 0,
                        'domain_end': 0,
                        'original_prediction': 0,
                        'mutated_prediction': 0,
                        'prediction_change': False,
                        'confidence_change': 0.0,
                        'stability_score': 0.0,
                        'effect_magnitude': 0.0,
                        'effect_type': 'error',
                        'error': str(e)
                    })
        
        return results

def calculate_significance_threshold(mutation_results: List[Dict], alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate proper statistical significance threshold using percentile-based approach"""
    logger.info("Calculating proper statistical significance threshold using percentile-based approach...")
    
    # Extract all effect magnitudes
    effects = [result['effect_magnitude'] for result in mutation_results if 'effect_magnitude' in result]
    
    if len(effects) < 10:
        logger.warning("Insufficient data for significance testing")
        return 0.01, 0.01
    
    # Convert to numpy array
    effects_array = np.array(effects)
    
    # Calculate basic statistics
    mean_effect = np.mean(effects_array)
    std_effect = np.std(effects_array)
    max_effect = np.max(effects_array)
    
    logger.info(f"Effect magnitude statistics:")
    logger.info(f"  Mean: {mean_effect:.6f}")
    logger.info(f"  Std: {std_effect:.6f}")
    logger.info(f"  Min: {np.min(effects_array):.6f}")
    logger.info(f"  Max: {max_effect:.6f}")
    logger.info(f"  Range: {max_effect - np.min(effects_array):.6f}")
    
    # Calculate percentile-based thresholds (more appropriate for this data)
    percentile_95 = np.percentile(np.abs(effects_array), 95)
    percentile_99 = np.percentile(np.abs(effects_array), 99)
    percentile_99_5 = np.percentile(np.abs(effects_array), 99.5)
    
    logger.info(f"Percentile-based thresholds:")
    logger.info(f"  95th percentile: {percentile_95:.6f} ({percentile_95/max_effect*100:.1f}% of max)")
    logger.info(f"  99th percentile: {percentile_99:.6f} ({percentile_99/max_effect*100:.1f}% of max)")
    logger.info(f"  99.5th percentile: {percentile_99_5:.6f} ({percentile_99_5/max_effect*100:.1f}% of max)")
    
    # Use 95th and 99th percentiles as moderate and strong effect thresholds
    moderate_effect_threshold = percentile_95
    strong_effect_threshold = percentile_99
    
    # One-sample t-test against zero
    t_stat, p_value = stats.ttest_1samp(effects_array, 0)
    logger.info(f"One-sample t-test against zero: t={t_stat:.3f}, p={p_value:.3e}")
    
    # Count mutations above thresholds
    moderate_count = np.sum(np.abs(effects_array) >= moderate_effect_threshold)
    strong_count = np.sum(np.abs(effects_array) >= strong_effect_threshold)
    
    logger.info(f"Mutations above thresholds:")
    logger.info(f"  Moderate effects (95th percentile): {moderate_count} ({moderate_count/len(effects)*100:.1f}%)")
    logger.info(f"  Strong effects (99th percentile): {strong_count} ({strong_count/len(effects)*100:.1f}%)")
    
    return moderate_effect_threshold, strong_effect_threshold

def create_mutation_effect_plot(mutation_results: List[Dict], output_dir: Path, moderate_threshold: float, strong_threshold: float):
    """Create scatterplot of mutation effects by position and domain with significance testing"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger.info("Creating mutation effect scatterplot with significance testing...")
    
    # Prepare data for plotting
    positions = []
    effects = []
    domain_types = []
    original_aas = []
    mutated_aas = []
    sequence_names = []
    is_moderate_effect = []
    is_strong_effect = []
    
    for result in mutation_results:
        if 'error' not in result:
            # Get sequence ID - use full header name for display
            if 'sequence_id' in result:
                seq_id = result['sequence_id']
                # Use the full sequence ID as the display name
                display_id = seq_id
            else:
                display_id = "Unknown"
            
            positions.append(result['position'])
            effects.append(result['effect_magnitude'])
            domain_types.append(result['domain_type'])
            original_aas.append(result['original_aa'])
            mutated_aas.append(result['mutated_aa'])
            sequence_names.append(display_id)
            
            # Check effect sizes
            effect_mag = abs(result['effect_magnitude'])
            is_moderate_effect.append(effect_mag >= moderate_threshold)
            is_strong_effect.append(effect_mag >= strong_threshold)
    
    if not positions:
        logger.warning("No mutation data to plot")
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        'position': positions,
        'effect_magnitude': effects,
        'domain_type': domain_types,
        'original_aa': original_aas,
        'mutated_aa': mutated_aas,
        'sequence_name': sequence_names,
        'is_moderate_effect': is_moderate_effect,
        'is_strong_effect': is_strong_effect
    })
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Define colors - make green less neon
    domain_colors = {
        'SMART': '#6A3D9A',      # Dark purple for IG domains
        'other': '#90EE90',      # Light green (less neon)
        'Other': '#90EE90'       # Light green (less neon)
    }
    
    # Plot points by domain type
    for domain_type in df['domain_type'].unique():
        domain_data = df[df['domain_type'] == domain_type]
        color = domain_colors.get(domain_type, '#90EE90')
        
        plt.scatter(domain_data['position'], domain_data['effect_magnitude'], 
                   c=color, label=f'IG domains' if domain_type == 'SMART' else 'Other',
                   alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
    
    # Add significance threshold lines
    # Moderate effect threshold (95th percentile)
    plt.axhline(y=moderate_threshold, color='red', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'Moderate effect threshold (±{moderate_threshold:.4f})')
    plt.axhline(y=-moderate_threshold, color='red', linestyle='--', 
                linewidth=2, alpha=0.8)
    
    # Strong effect threshold (99th percentile)
    plt.axhline(y=strong_threshold, color='blue', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'Strong effect threshold (±{strong_threshold:.4f})')
    plt.axhline(y=-strong_threshold, color='blue', linestyle='--', 
                linewidth=2, alpha=0.8)
    
    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Customize plot
    plt.xlabel('Amino Acid Position', fontsize=14, fontweight='bold')
    plt.ylabel('Mutation Effect Magnitude', fontsize=14, fontweight='bold')
    plt.title('CD300 Mutation Effects by InterProScan Domain\n(Red dashed lines indicate statistical significance)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    total_mutations = len(df)
    moderate_effects = df['is_moderate_effect'].sum()
    strong_effects = df['is_strong_effect'].sum()
    percent_moderate = (moderate_effects / total_mutations) * 100
    percent_strong = (strong_effects / total_mutations) * 100
    
    # Calculate additional statistics
    mean_effect = df['effect_magnitude'].mean()
    std_effect = df['effect_magnitude'].std()
    
    stats_text = (f'Total mutations: {total_mutations}\n'
                 f'Moderate effects: {moderate_effects} ({percent_moderate:.1f}%)\n'
                 f'Strong effects: {strong_effects} ({percent_strong:.1f}%)\n'
                 f'Mean effect: {mean_effect:.4f}\n'
                 f'Std effect: {std_effect:.4f}\n'
                 f'Moderate threshold: ±{moderate_threshold:.4f}\n'
                 f'Strong threshold: ±{strong_threshold:.4f}')
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'mutation_effects_by_domain.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Mutation effect plot saved to {plot_file}")

def create_interactive_mutation_plots(mutation_results: List[Dict], output_dir: Path, moderate_threshold: float, strong_threshold: float):
    """Create interactive HTML plots for Step 6 mutation effects with all mutations included"""
    logger.info("Creating interactive mutation effect plots...")
    
    # Prepare data - include ALL mutations, not just non-neutral ones
    positions = []
    effects = []
    domain_types = []
    original_aas = []
    mutated_aas = []
    effect_types = []
    sequence_names = []
    is_moderate_effect = []
    is_strong_effect = []
    
    for result in mutation_results:
        if 'error' not in result:
            # Get sequence ID - use full header name for display
            if 'sequence_id' in result:
                seq_id = result['sequence_id']
                # Use the full sequence ID as the display name
                display_id = seq_id
            else:
                display_id = "Unknown"
            
            positions.append(result['position'])
            effects.append(result['effect_magnitude'])
            domain_types.append(result['domain_type'])
            original_aas.append(result['original_aa'])
            mutated_aas.append(result['mutated_aa'])
            effect_types.append(result['effect_type'])
            sequence_names.append(display_id)
            
            # Check effect sizes
            effect_mag = abs(result['effect_magnitude'])
            is_moderate_effect.append(effect_mag >= moderate_threshold)
            is_strong_effect.append(effect_mag >= strong_threshold)
    
    if not positions:
        logger.warning("No mutation data to plot")
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        'position': positions,
        'effect_magnitude': effects,
        'domain_type': domain_types,
        'original_aa': original_aas,
        'mutated_aa': mutated_aas,
        'effect_type': effect_types,
        'sequence_name': sequence_names,
        'is_moderate_effect': is_moderate_effect,
        'is_strong_effect': is_strong_effect
    })
    
    # Create interactive scatter plot
    fig = go.Figure()
    
    # Define colors - make green less neon
    domain_colors = {
        'SMART': '#6A3D9A',      # Dark purple for IG domains
        'other': '#90EE90',      # Light green (less neon)
        'Other': '#90EE90'       # Light green (less neon)
    }
    
    # Add traces for each domain type
    for domain_type in df['domain_type'].unique():
        domain_data = df[df['domain_type'] == domain_type]
        color = domain_colors.get(domain_type, '#90EE90')
        label = 'IG domains' if domain_type == 'SMART' else 'Other'
        
        fig.add_trace(go.Scatter(
            x=domain_data['position'],
            y=domain_data['effect_magnitude'],
            mode='markers',
            marker=dict(
                color=color,
                size=8,
                line=dict(width=1, color='black')
            ),
            name=label,
            hovertemplate='<b>%{text}</b><br>' +
                         'Position: %{x}<br>' +
                         'Effect: %{y:.6f}<br>' +
                         'Domain: %{customdata[0]}<br>' +
                         'Moderate Effect: %{customdata[1]}<br>' +
                         'Strong Effect: %{customdata[2]}<extra></extra>',
            text=[f"Sequence: {row['sequence_name']}<br>Mutation: {row['original_aa']}→{row['mutated_aa']}" 
                  for _, row in domain_data.iterrows()],
            customdata=list(zip(domain_data['domain_type'], 
                              ['Yes' if med else 'No' for med in domain_data['is_moderate_effect']],
                              ['Yes' if strong else 'No' for strong in domain_data['is_strong_effect']]))
        ))
    
    # Add significance threshold lines
    fig.add_hline(y=moderate_threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Moderate effect threshold (+{moderate_threshold:.4f})")
    fig.add_hline(y=-moderate_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Moderate effect threshold (-{moderate_threshold:.4f})")
    
    # Add strong effect threshold
    fig.add_hline(y=strong_threshold, line_dash="dash", line_color="blue", 
                  annotation_text=f"Strong effect threshold (+{strong_threshold:.4f})")
    fig.add_hline(y=-strong_threshold, line_dash="dash", line_color="blue",
                  annotation_text=f"Strong effect threshold (-{strong_threshold:.4f})")
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title='CD300 Mutation Effects by InterProScan Domain (Interactive)',
        xaxis_title='Amino Acid Position',
        yaxis_title='Mutation Effect Magnitude',
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        title_font_size=16
    )
    
    # Save interactive plot
    interactive_plot = output_dir / "mutation_effects_interactive.html"
    fig.write_html(interactive_plot)
    logger.info(f"Interactive plot saved: {interactive_plot}")

def main():
    parser = argparse.ArgumentParser(description='Step 6: InterProScan Domain Analysis (Local Version - Updated)')
    parser.add_argument('--fasta_path', type=str, required=True,
                       help='Path to Euarchontoglires CD300 sequences FASTA file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned ESM2 model directory')
    parser.add_argument('--interproscan_path', type=str, 
                       default='/home5/ibirchl/Bioinformatics_tools/interproscan-5.65-97.0',
                       help='Path to InterProScan installation')
    parser.add_argument('--output_dir', type=str, default='step6_interproscan_local',
                       help='Output directory for results')
    parser.add_argument('--skip_interproscan', action='store_true',
                       help='Skip InterProScan analysis (use existing results)')
    parser.add_argument('--max_mutations_per_position', type=int, default=1,
                       help='Maximum mutations per position (default: 1)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("STEP 6: INTERPROSCAN DOMAIN ANALYSIS (LOCAL VERSION - UPDATED)")
    logger.info("="*80)
    
    # Load sequences
    logger.info(f"Loading sequences from {args.fasta_path}")
    sequences = {}
    for record in SeqIO.parse(args.fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Initialize InterProScan analyzer
    interproscan_analyzer = LocalInterProScanAnalyzer(args.interproscan_path, args.output_dir)
    
    # Run InterProScan (if not skipped)
    domains = {}
    if not args.skip_interproscan:
        tsv_file = interproscan_analyzer.run_interproscan(args.fasta_path)
        if tsv_file:
            domains = interproscan_analyzer.parse_interproscan_results(tsv_file)
        else:
            logger.error("InterProScan failed. Cannot proceed.")
            return
    else:
        # Load existing results
        tsv_file = output_dir / "interproscan_results.tsv"
        if tsv_file.exists():
            domains = interproscan_analyzer.parse_interproscan_results(str(tsv_file))
            logger.info("Loaded existing InterProScan results")
        else:
            logger.error("No existing InterProScan results found and analysis skipped.")
            return
    
    # Initialize mutation effect predictor
    mutation_predictor = MutationEffectPredictor(args.model_path)
    
    # Predict mutation effects
    mutation_results = mutation_predictor.predict_mutation_effects(
        sequences, domains, args.max_mutations_per_position
    )
    
    # Save mutation results
    results_file = output_dir / "mutation_results.json"
    with open(results_file, 'w') as f:
        json.dump(mutation_results, f, indent=2)
    logger.info(f"Mutation results saved to {results_file}")
    
    # Calculate significance thresholds
    moderate_threshold, strong_threshold = calculate_significance_threshold(mutation_results)
    
    # Create plots with significance testing
    create_mutation_effect_plot(mutation_results, output_dir, moderate_threshold, strong_threshold)
    create_interactive_mutation_plots(mutation_results, output_dir, moderate_threshold, strong_threshold)
    
    # Create summary report
    total_mutations = len(mutation_results)
    moderate_count = sum(1 for r in mutation_results 
                        if 'error' not in r and abs(r['effect_magnitude']) >= moderate_threshold)
    strong_count = sum(1 for r in mutation_results 
                      if 'error' not in r and abs(r['effect_magnitude']) >= strong_threshold)
    
    summary = {
        'total_mutations': total_mutations,
        'moderate_effects': moderate_count,
        'strong_effects': strong_count,
        'percent_moderate': (moderate_count / total_mutations) * 100,
        'percent_strong': (strong_count / total_mutations) * 100,
        'moderate_threshold': moderate_threshold,
        'strong_threshold': strong_threshold,
        'domain_distribution': {}
    }
    
    # Count by domain type
    domain_counts = {}
    for result in mutation_results:
        if 'error' not in result:
            domain_type = result['domain_type']
            domain_counts[domain_type] = domain_counts.get(domain_type, 0) + 1
    
    summary['domain_distribution'] = domain_counts
    
    # Save summary
    summary_file = output_dir / "mutation_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_file}")
    
    # Create markdown report
    report_file = output_dir / "mutation_analysis_report.md"
    with open(report_file, 'w') as f:
        f.write("# Mutation Effect Analysis Report\n\n")
        f.write("## Summary\n")
        f.write(f"- **Total mutations analyzed**: {summary['total_mutations']}\n")
        f.write(f"- **Moderate effects (95th percentile)**: {summary['moderate_effects']} ({summary['percent_moderate']:.1f}%)\n")
        f.write(f"- **Strong effects (99th percentile)**: {summary['strong_effects']} ({summary['percent_strong']:.1f}%)\n")
        f.write(f"- **Moderate effect threshold**: ±{summary['moderate_threshold']:.4f}\n")
        f.write(f"- **Strong effect threshold**: ±{summary['strong_threshold']:.4f}\n")
        f.write("\n## Domain Distribution\n")
        for domain_type, count in summary['domain_distribution'].items():
            f.write(f"- **{domain_type}**: {count} mutations\n")
        f.write("\n## Files Generated\n")
        f.write("- `mutation_effects_by_domain.png`: Static plot with significance testing\n")
        f.write("- `mutation_effects_interactive.html`: Interactive plot with hover information\n")
        f.write("- `mutation_results.json`: Complete mutation effect data\n")
        f.write("- `mutation_analysis_summary.json`: Summary statistics\n")
        f.write("- `mutation_analysis_report.md`: This report\n")
        f.write("\n## Analysis Description\n")
        f.write("This analysis predicts mutation effects using the fine-tuned ESM2 model and identifies\n")
        f.write("statistically significant effects using 95th percentile threshold and Cohen's d effect size.\n")
        f.write("Red dashed lines in the plots indicate the significance threshold.\n")
    
    logger.info(f"Report saved: {report_file}")
    
    logger.info("="*80)
    logger.info("STEP 6 COMPLETED SUCCESSFULLY!")
    logger.info(f"Total mutations: {total_mutations}")
    logger.info(f"Moderate effects: {moderate_count} ({summary['percent_moderate']:.1f}%)")
    logger.info(f"Strong effects: {strong_count} ({summary['percent_strong']:.1f}%)")
    logger.info(f"Moderate threshold: ±{moderate_threshold:.4f}")
    logger.info(f"Strong threshold: ±{strong_threshold:.4f}")
    logger.info("="*80)

if __name__ == "__main__":
    from Bio import SeqIO
    main()
