# CD300 Protein Analysis Using Fine-Tuned ESM2 650M Model

## Project Overview

This project is a comprehensive multi-modal analysis pipeline that uses state-of-the-art protein language models and bioinformatics tools to discover, characterize, and understand CD300 protein diversity across vertebrate evolution. The pipeline combines:

- **Deep learning-based sequence analysis** using fine-tuned ESM2 650M models
- **Functional domain prediction** with InterProScan and mutation effect analysis
- **3D structure prediction** using ESMFold for structural characterization
- **Phylogenetic analysis** with phylomorphospace visualization and evolutionary signal testing
- **Cross-modal validation** integrating sequence, functional, and structural perspectives
- **Novel variant discovery** through rigorous clustering and outlier detection methods

The analysis focuses on the Euarchontoglires superorder (primates, rodents, lagomorphs) as a test case for discovering previously unannotated CD300 protein variants and understanding their functional properties, evolutionary relationships, and structural characteristics.

### Key Questions Addressed

1. **Can ESM2 650M generalize across evolutionary distances?** How well does a model trained on diverse vertebrates perform on completely unseen Euarchontoglires sequences?
2. **Are there novel CD300 variants in Euarchontoglires?** What previously unannotated or unknown CD300 subtypes exist in this superorder?
3. **How do novel variants differ functionally?** Are novel CD300 variants more or less sensitive to mutations compared to known subtypes?
4. **What are the structural implications?** How do 3D structures of novel variants compare to known CD300 subtypes?

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: A100 40GB or equivalent)
- SLURM job scheduler (for cluster execution)
- Java (for InterProScan)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Hugging Face cache (models will auto-download on first use):
```bash
# Set cache directory (optional - defaults to ~/.cache/huggingface/)
export HF_HOME=/path/to/your/cache/directory

# Verify cache setup
python -c "from transformers import AutoTokenizer; print('Cache location:', AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D').cache_dir)"

# Models will be automatically downloaded when first used in the pipeline
# ESM2 650M model: ~1.2GB (used in Steps 2, 3, 4, 6)
# ESMFold model: ~2.8GB (used in Step 7)
```

## Pipeline Overview

The analysis consists of 8 main steps (Step 9 is in development):

### Step 1: Data Preparation and Sequence Clustering
- Load and validate CD300 protein sequence datasets
- Generate sequence features using k-mer frequency analysis
- Apply unsupervised clustering to discover natural sequence groupings
- Clean and prepare dataset for downstream analysis

**Outputs:** Cleaned CSV with cluster assignments, cluster analysis reports

### Step 1.5: Simple Sequence Composition Analysis (Baseline)
- Establish baseline clustering using traditional bioinformatics methods
- Analyze protein sequences using amino acid composition features
- Provides baseline to evaluate deep learning improvements

**Outputs:** Static PCA/t-SNE plots, interactive HTML plots, baseline analysis report

### Step 2: Baseline Embedding Generation
- Generate embeddings using pretrained ESM2 650M (no fine-tuning)
- Visualize embeddings with PCA/t-SNE
- Color points by subtype labels and inspect clustering

**Outputs:** Embedding matrices, visualization plots, cluster overlap summary

### Step 3: Fine-Tune ESM2 650M for CD300 Classification
- Remove Euarchontoglires sequences from training data
- Split remaining data into 80% training and 20% validation sets
- Fine-tune ESM2 on sequence clusters (not homology annotations)
- Add classification head with multi-class output
- Use early stopping and GPU acceleration

**Outputs:** Fine-tuned model weights, training logs, evaluation reports

### Step 4: Euarchontoglires Embedding Generation and Classification
- Apply trained model to completely unseen Euarchontoglires sequences
- Generate embeddings and classification predictions
- Create interactive visualizations

**Outputs:** Embedding matrices, prediction results, interactive HTML plots

### Step 5: Enhanced Novel Variant Detection
- Identify structural outliers in embedding space
- Apply DBSCAN clustering to find novel sequence groups
- Perform comprehensive validation with multiple metrics
- Distinguish technical artifacts from biological clustering

**Outputs:** Enhanced analysis report, validation metrics, 16-panel visualizations

### Step 6: Mutation Effect Prediction with Domain Analysis
- Use InterProScan to identify functional domains
- Predict mutation effects using classification-based approach
- Map effects to functional domains
- Generate interactive scatterplots

**Outputs:** Domain predictions, mutation effect analysis, interactive HTML plots

### Step 7: Multi-Modal Structure-Function Analysis
- Generate 3D structures using ESMFold
- Extract structural features and perform clustering
- Cross-validate with sequence and functional clusters
- Create comprehensive multi-modal visualizations

**Outputs:** 3D structures (PDB), structural features, cross-modal analysis

### Step 8: Phylomorphospace Analysis
- Generate phylogenetic trees using IQ-Tree2 maximum likelihood reconstruction
- Create time-calibrated phylogenies with relative time scaling
- Overlay phylogenetic tree connections on ESM2 embedding space
- Perform phylogenetic signal tests (Blomberg's K and Pagel's λ)
- Color points by taxonomic clades (primates, rodents, lagomorphs)
- Generate interactive visualizations with evolutionary trajectories

**Outputs:** Phylogenetic trees, phylomorphospace plots, phylogenetic signal statistics, interactive HTML visualizations

## Usage

### Running Individual Steps

Each step can be run independently using SLURM scripts:

```bash
# Step 1: Data preparation
sbatch scripts/step1_data_preparation.slurm

# Step 1.5: Baseline sequence analysis
sbatch scripts/sequence_pca_tsne.slurm

# Step 2: Baseline embeddings
sbatch scripts/step2_baseline_embeddings.slurm

# Step 3: Fine-tuning
sbatch scripts/step3_fine_tune_esm2_clusters.slurm

# Step 4: Euarchontoglires embeddings
sbatch scripts/step4_euarchontoglires_embeddings.slurm

# Step 5: Novel variant detection
sbatch scripts/step5_enhanced_novel_variant_detection.slurm

# Step 6: Mutation effects with InterProScan
sbatch scripts/step6_interproscan_local.slurm

# Step 7: Structure analysis
sbatch scripts/step7_esmfold_structure_analysis.slurm

# Step 8: Phylomorphospace analysis
sbatch scripts/step8_phylomorphospace_analysis.slurm
```

### Running Complete Pipeline

To run the entire pipeline sequentially:

```bash
# Run all steps in order
for step in {1..8}; do
    echo "Running Step $step..."
    sbatch scripts/step${step}_*.slurm
    # Wait for completion before next step
done
```

## Adapting for Different Clades/Immune Gene Families

This pipeline can be adapted for analyzing other immune gene families or different taxonomic clades. Here's how to modify the approach:

### For Different Immune Gene Families and/or Clades:

1. **Data Preparation**:
   - Replace CD300 sequences with your target gene family (e.g., KIR, LILR, NKG2D)
   - Update sequence filtering criteria in `step1_data_preparation.py`
   - Modify clustering parameters based on your gene family's sequence diversity

2. **Training Data Management**:
   - Ensure your target clade is completely excluded from training data
   - Balance training data across remaining clades to avoid bias
   - Validate that no sequences from your target clade appear in training/validation sets

### Key Considerations
- **Sequence Diversity**: Adjust clustering parameters based on your gene family's sequence diversity
- **Functional Domains**: Update domain prediction and analysis for your protein's characteristic domains
- **Evolutionary Timescale**: Modify phylogenetic analysis parameters for your clade's divergence times
- **Computational Resources**: Adjust memory and GPU requirements based on your dataset size
- **Validation Metrics**: Update clustering and classification metrics for your specific research questions

## Project Structure

```
cd300-esm2-fine-tuned-model/
├── data/                          # Input data files
│   ├── Euarchontoglires_CD300s_complete.fasta
│   └── all_orders_CD300s.csv
├── scripts/                       # Analysis scripts
│   ├── step1_data_preparation.py
│   ├── step2_baseline_embeddings.py
│   ├── step3_fine_tune_esm2_clusters.py
│   ├── step4_euarchontoglires_embeddings.py
│   ├── step5_enhanced_novel_variant_detection.py
│   ├── step6_interproscan_local.py
│   ├── step7_esmfold_structure_analysis.py
│   ├── step8_phylomorphospace_analysis.py
│   └── *.slurm                    # SLURM job scripts
├── step1_data_preparation/        # Step 1 outputs
├── step1_5_sequence_analysis/     # Step 1.5 outputs
├── step2_baseline_embeddings/     # Step 2 outputs
├── step3_fine_tuned_model/        # Step 3 outputs
├── step4_euarchontoglires_embeddings/  # Step 4 outputs
├── step5_novel_variant_detection/ # Step 5 outputs
├── step6_interproscan_local/      # Step 6 outputs
├── step7_structure_analysis/      # Step 7 outputs
├── step8_phylomorphospace_analysis/  # Step 8 outputs
├── requirements.txt               # Python dependencies
├── projectoverview.md            # Detailed project documentation
└── README.md                     # This file
```

## Key Features

### Novel Variant Discovery
- **Rigorous validation**: Multiple clustering approaches and metrics
- **Cross-modal analysis**: Sequence, functional, and structural perspectives
- **Outlier detection**: Identifies truly novel variants vs. technical artifacts

### Mutation Effect Prediction
- **Classification-based approach**: Uses functional subtype changes, not embedding distances
- **Domain-specific analysis**: Maps effects to InterProScan functional domains
- **Interactive visualizations**: Hover-enabled plots for detailed exploration

### Multi-Modal Integration
- **Sequence clustering**: Traditional and deep learning approaches
- **Structural prediction**: ESMFold 3D structure generation
- **Cross-modal validation**: Compares sequence, functional, and structural clusters

### Interactive Visualizations
- **HTML plots**: Hover-enabled interactive plots for all major analyses
- **Combined views**: Side-by-side PCA and t-SNE comparisons
- **Domain mapping**: Color-coded mutation effects by functional domains

## Results Summary

### Key Findings
- **351 Euarchontoglires sequences** analyzed across 7 steps
- **Novel CD300 variants** identified through rigorous validation
- **Cross-modal consistency** between sequence, functional, and structural clustering
- **Domain-specific mutation effects** mapped to functional regions

### Performance Metrics
- **Model accuracy**: High performance on held-out test sets
- **Clustering quality**: Strong silhouette scores and cluster separation
- **Cross-modal alignment**: Significant correlation between different analysis modalities

## Dependencies

### Core Requirements
```txt
torch>=2.0
transformers>=4.35
fair-esm
scikit-learn
umap-learn
hdbscan
matplotlib
seaborn
biopython
plotly
kaleido
```

### External Tools
- **InterProScan**: Domain prediction and functional annotation
- **IQ-Tree**: Phylogenetic tree reconstruction
- **ESMFold**: 3D structure prediction
- **FAMSA**: Multiple sequence alignment

### Hardware Requirements
- **GPU**: CUDA-compatible (A100 40GB recommended for ESMFold)
- **Memory**: 32GB+ RAM for large sequence datasets
- **Storage**: 100GB+ for intermediate files and results

## Acknowledgments

- **ESM2 model**: Facebook AI Research
- **ESMFold**: Meta AI Research
- **InterProScan**: European Bioinformatics Institute
- **Hugging Face**: Transformers library and model hosting


