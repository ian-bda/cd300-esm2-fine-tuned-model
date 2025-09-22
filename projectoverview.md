# Project Overview: CD300 Protein Analysis Using Fine-Tuned ESM2 650M Model

## Model Background: ESM2 650M
- **Parameters**: 650 million
- **Architecture**: 33 layers, 20 attention heads
- **Input capacity**: Sequences up to length 1022 amino acids
- **Capabilities**: Embedding generation, mutation effect prediction, subtype classification

---

## Project Goals
- **Discover novel CD300 variants** in Euarchontoglires using rigorous AI-based validation.
- **Fine-tune ESM2 650M** on diverse vertebrate data (excluding Euarchontoglires) for CD300 subtype classification.
- **Validate model generalization** by testing on completely unseen Euarchontoglires sequences.
- **Characterize functional properties** of novel variants through mutation effect prediction and 3D structure modeling.
- **Establish framework** for discovering novel protein variants across other vertebrate clades using the same approach.

---

## Key Questions
1. **Can ESM2 650M generalize across evolutionary distances?** How well does a model trained on diverse vertebrates perform on completely unseen Euarchontoglires sequences?
2. **Are there novel CD300 variants in Euarchontoglires?** What previously unannotated or unknown CD300 subtypes exist in this superorder?
3. **How do novel variants differ functionally?** Are novel CD300 variants more or less sensitive to mutations compared to known subtypes?
4. **What are the structural implications?** How do 3D structures of novel variants compare to known CD300 subtypes?
5. **Which residues drive subtype specificity?** What amino acid motifs and residues are most important for distinguishing CD300 subtypes?
6. **Can this approach be extended?** How well does this validation framework work for discovering novel variants in other vertebrate clades?

---

## Step-by-Step Workflow

### Step 1: Data Preparation and Sequence Clustering
- Load and validate CD300 protein sequence datasets.
- Generate sequence features using k-mer frequency analysis.
- Apply unsupervised clustering to discover natural sequence groupings.
- Compare discovered clusters with homology labels for reference.
- Clean and prepare dataset for downstream analysis.

**Outputs:**
- Cleaned CSV file with sequence cluster assignments.
- Cluster analysis reports and visualizations.
- Dataset quality assessment and summary statistics.
---

### Step 1.5: Simple Sequence Composition Analysis (Baseline)
- **Purpose**: Establish baseline clustering using traditional bioinformatics methods before applying deep learning.
- **Method**: Analyze protein sequences using simple amino acid composition features.
- **Features used**:
  - Amino acid frequencies (20 features)
  - Sequence length (normalized)
  - Hydrophobicity (AILMFWV content)
  - Net charge (positive - negative)
  - Aromatic content (FWY content)
- **Analysis**: PCA, t-SNE, and K-means clustering on composition features.
- **Comparison**: Provides baseline to evaluate how much deep learning (ESM2) improves clustering.

**Outputs:**
- Static PCA and t-SNE plots (`sequence_pca_plot.png`, `sequence_tsne_plot.png`)
- Interactive combined HTML plot (`sequence_combined_plot.html`)
- Clustering results and analysis report (`sequence_analysis_report.md`)
- Complete results in JSON format (`sequence_analysis_results.json`)

**Key Findings:**
- **351 sequences analyzed** using 24 composition features
- **Optimal clustering**: 2 clusters (37% vs 63% distribution)
- **PCA variance**: 30.5% explained by first 2 components
- **Baseline comparison**: Shows that simple composition finds fewer clusters than deep learning methods

---

### Step 2: Baseline Embedding Generation and Visualization
- Generate embeddings using pretrained ESM2 650M (no fine-tuning).
- Visualize embeddings with UMAP/t-SNE.
- Color points by subtype labels and inspect clustering.


**Outputs:**
- Embedding matrices (`.npy` or `.h5`).
- Visualization plots (`.png`).
- Cluster overlap summary.

---

### Step 3: Fine-Tune ESM2 650M for CD300 Sequence Cluster Classification
- **Data preparation**: Start with CSV of all sequences from lots of orders (large diverse dataset).
- **Clade removal**: Remove Euarchontoglires sequences → place in separate file for final validation.
- **Data splitting**: Split remaining data into 80% training / 20% test.
- **Model training**: Fine-tune ESM2 on training data (80% of non-Euarchontoglires sequences).
- **Use sequence clusters as labels**: Train on sequence clusters from Step 1 (e.g., Cluster_1, Cluster_2) instead of unreliable homology annotations.
- Add classification head with multi-class output (one class per sequence cluster).
- Use early stopping, LR scheduling, and GPU acceleration.
- Evaluate with accuracy, macro/weighted F1, ROC/PR curves.

**Data Flow:**
```
All Orders Data → Remove Euarchontoglires → Split 80/20 → Train ESM2 on Sequence Clusters
                                                           ↓
Final Validation: Apply trained model to Euarchontoglires (completely unseen)
```

**Outputs:**
- Fine-tuned model weights + tokenizer (trained on sequence clusters).
- Training logs with per-epoch metrics.
- Confusion matrices and evaluation reports (cluster-based classification).
- Model performance on test set (20% of non-Euarchontoglires).

---

### Step 4: Euarchontoglires Embedding Generation and Classification
- **Final validation on Euarchontoglires**: Apply trained model to completely unseen Euarchontoglires sequences.
- **Generate embeddings**: Extract high-dimensional embeddings from fine-tuned ESM2 model.
- **Classification predictions**: Predict sequence cluster assignments for Euarchontoglires sequences.
- **Visualization**: Create PCA and t-SNE plots of embeddings colored by predicted clusters.
- **True external validation**: Euarchontoglires sequences never seen during training or testing.

**Data Flow:**
```
Euarchontoglires (completely unseen) → Fine-tuned ESM2 → Embeddings + Predictions
```

**Outputs:**
- Embedding matrices (`.npy`) for Euarchontoglires sequences.
- Prediction results (`.csv`) with cluster assignments and confidence scores.
- Visualization plots (`.png`) showing embedding distributions.
- Summary statistics and cluster distribution analysis.

---

### Step 5: Enhanced Novel Variant Detection and Clustering Interpretation
- **Outlier detection**: Identify sequences that are structural outliers in embedding space.
- **Low confidence analysis**: Analyze sequences with low prediction confidence.
- **Spatial clustering**: Apply DBSCAN clustering to identify novel sequence groups.
- **Taxonomic analysis**: Examine distribution of novel variants across species.
- **Cluster annotation comparison**: Overlay known CD300 subtypes with cluster labels to assess alignment.
- **Cluster metrics calculation**: Calculate silhouette scores, cluster purity, and within/between cluster distances.
- **Technical vs biological clustering**: Use metadata to distinguish evolutionary relatedness from technical artifacts.
- **Overlap investigation**: Examine sequences in overlapping cluster regions for hybrids or intermediates.
- **Novel outlier validation**: Identify unique, well-separated groups as evidence of novel biology.

**Practical Validation Steps:**
- Compare clusters to known CD300 subtypes and lineage information
- Calculate comprehensive clustering metrics (silhouette, purity, distances)
- Check for technical artifacts vs biological clustering using species metadata
- Investigate cluster overlap regions for potential hybrids or annotation errors
- Validate results through multiple clustering approaches (DBSCAN, K-means)

**Data Flow:**
```
Euarchontoglires Embeddings → Multi-method Analysis → Comprehensive Validation
```

**Outputs:**
- Enhanced novel variants analysis with 16-panel visualization.
- Comprehensive validation report with clustering interpretation.
- Technical vs biological clustering assessment.
- Novel outlier identification with confidence scores.
- Detailed cluster annotation analysis and recommendations.

---

### Step 6: Mutation Effect Prediction with InterProScan Domain Analysis
- **InterProScan domain prediction**: Use local InterProScan to identify functional domains in CD300 sequences.
- **Classification-based mutation effects**: Use fine-tuned ESM2 model to predict mutation effects based on **functional classification changes**, not embedding distances.
- **Mutation effect methodology**: 
  - **Original sequence** → ESM2 model → CD300 subtype prediction (e.g., "CD300A", "CD300B")
  - **Mutated sequence** → ESM2 model → CD300 subtype prediction
  - **Effect measurement**: Changes in predicted CD300 subtype and model confidence
- **Effect types detected**:
  - **Functional_Change**: Mutation changes predicted CD300 subtype (e.g., CD300A → CD300B)
  - **Destabilizing**: Mutation reduces model confidence (confidence drop > 0.1)
  - **Stabilizing**: Mutation increases model confidence (confidence increase > 0.1)
  - **Neutral**: No significant change in prediction or confidence
- **Domain-specific analysis**: Map mutation effects to InterProScan-predicted functional domains.
- **Scatterplot visualization**: X-axis = amino acid position, Y-axis = mutation effect magnitude, colored by domain type.

**Mutation Effect Algorithm:**
```python
# Get original and mutated predictions from fine-tuned ESM2 model
original_prediction = model(original_sequence)    # e.g., Cluster_3 (CD300A-like)
original_confidence = max(softmax(logits))        # e.g., 0.85
mutated_prediction = model(mutated_sequence)      # e.g., Cluster_1 (CD300B-like)  
mutated_confidence = max(softmax(logits))         # e.g., 0.72

# Calculate functional impact
prediction_change = (original_prediction != mutated_prediction)  # True/False
confidence_change = mutated_confidence - original_confidence     # -0.13
effect_magnitude = abs(confidence_change)                       # 0.13
```

**Data Flow:**
```
CD300 Sequences → InterProScan → Domain Predictions
CD300 Sequences → Fine-tuned ESM2 → Classification-based Mutation Effects
Domain Predictions + Mutation Effects → Domain-specific Effect Mapping
```

**Statistical Analysis:**
- **Percentile-based significance thresholds**: 95th percentile (moderate effects) and 99th percentile (strong effects)
- **Effect magnitude distribution**: Mean 0.000738, Std 0.001073, Range 0.027746
- **Significance testing**: One-sample t-test against zero (t=182.721, p<0.001)
- **Effect classification**: 5.0% moderate effects, 1.0% strong effects
- **Domain-specific analysis**: IG domains vs. other regions with statistical comparison

**Outputs:**
- InterProScan domain predictions (`.tsv`) with functional annotations.
- Mutation effect predictions (`.json`) with classification-based impact scores.
- Scatterplot visualizations (`.png`) showing mutation effects by position and domain.
- Interactive HTML plots with full sequence names and statistical thresholds.
- Domain-specific mutation analysis with functional interpretation.
- Statistical significance report with percentile-based thresholds.

---

### Step 7: Multi-Modal Structure-Function Analysis
- **ESMFold structure prediction**: Generate 3D structures for all 351 Euarchontoglires CD300 sequences using ESMFold.
- **Structural feature extraction**: Extract structural features including distance matrices, secondary structure elements, domain boundaries, and surface properties.
- **Structural clustering analysis**: Perform PCA and t-SNE on structural features to identify structural clusters and relationships.
- **Cross-modal validation**: Compare structural clusters with sequence clusters (Step 1) and functional clusters (Steps 4-5) for comprehensive validation.
- **Multi-modal insights**: Integrate sequence, functional, and structural perspectives to provide complete CD300 characterization.

**Data Flow:**
```
All 351 Euarchontoglires Sequences → ESMFold → 3D Structures
3D Structures → Structural Feature Extraction → Structural PCA/t-SNE
Structural Clusters ↔ Sequence Clusters ↔ Functional Clusters → Cross-Modal Analysis
```

**Outputs:**
- Predicted 3D structures (`.pdb`/`.cif`) for all 351 sequences.
- Structural feature matrices and distance calculations.
- Structural PCA and t-SNE visualizations with interactive plots.
- Cross-modal clustering comparison analysis (sequence vs. functional vs. structural).
- Comprehensive multi-modal CD300 characterization report.

---

### Step 8: Phylomorphospace Analysis
- **Data preparation**: Load Step 4 embeddings and generate PCA/t-SNE coordinates for 351 Euarchontoglires sequences.
- **Sequence alignment**: Use FAMSA to create multiple sequence alignment with short, clean sequence IDs for phylogenetic software compatibility.
- **Phylogenetic reconstruction**: Run IQ-Tree2 on aligned sequences to generate maximum likelihood tree with branch lengths.
- **Time calibration**: Use R/Chronos to create time-calibrated phylogeny with penalized likelihood method.
- **Phylomorphospace analysis**: Use R/phytools to combine phylogeny with PCA/t-SNE coordinates, showing evolutionary trajectories in functional space.
- **Interactive visualization**: Generate both static PDF and interactive HTML plots with hover functionality.

**Data Flow:**
```
Step 4 Embeddings → PCA/t-SNE → Short Sequence IDs → FAMSA Alignment
FAMSA Alignment → IQ-Tree2 → ML Tree → MEGA RelTime → Time Tree
Time Tree + PCA/t-SNE → R/phytools → Phylomorphospace Plots
```

**Statistical Analysis:**
- **Phylogenetic signal tests**: Blomberg's K and Pagel's λ on PC1 values
- **Blomberg's K**: Measures phylogenetic signal strength (K=1.0 indicates strong signal)
- **Pagel's λ**: Measures phylogenetic dependence (λ=1.0 indicates complete phylogenetic dependence)
- **Clade assignment**: Robust taxonomic mapping using multiple strategies (direct matching, genus-based mapping, family patterns)
- **Tree-embedding integration**: Proper mapping between phylogenetic tree tips and ESM2 embedding coordinates

**Outputs:**
- Multiple sequence alignment (`.fasta`) and sequence ID mapping (`.csv`).
- Maximum likelihood phylogenetic tree (`.treefile`) with branch lengths.
- Time-calibrated phylogeny (`.nwk`) with relative time scaling.
- Static phylomorphospace plots (`.png`) with tree connections overlaid on ESM2 embeddings.
- Interactive phylomorphospace visualizations (`.html`) with hover details and clade information.
- Phylogenetic signal test results (`.json`) with Blomberg's K and Pagel's λ statistics.
- Comprehensive analysis summary and evolutionary trajectory report.

---

### Step 9: Interpretation and Reporting
- Generate attention maps from fine-tuned model to highlight cluster-specific motifs.
- Cross-reference predictions with UniProt and literature.
- Summarize findings in a comprehensive report.

**Outputs:**
- Attention heatmaps.
- Interpretability reports.
- Functional and evolutionary insights.

---

## Training Workflow:

Training Data (80%) → Fine-tune ESM2 650M on Sequence Clusters
     ↓
Test Data (20%) → Extract embeddings → Validate model performance
     ↓
External Dataset (Euarchontoglires) → Generate embeddings → Novel variant detection

---

## Visualization and Output Summary

| Task | Visualization | Output Files |
|------|---------------|--------------|
| Sequence Clustering | Cluster distribution pie chart, PCA scatter plot | Cluster CSV, cluster plots, PNG |
| Baseline Embeddings | UMAP/t-SNE plots | Embedding matrices, PNG/PDF plots |
| Fine-tuning Metrics | Loss/Accuracy curves | Training logs, JSON/CSV summaries |
| Euarchontoglires Embeddings | Interactive PCA/t-SNE plots, cluster distribution | Embedding matrices, prediction CSV, HTML plots |
| Enhanced Novel Variant Detection | 16-panel validation plots, clustering metrics | Enhanced analysis report, validation metrics, PNG |
| Mutation Effects | Interactive domain scatterplots, heatmaps | Mutation tables, HTML plots, PNGs |
| Multi-Modal Structure Analysis | Structural PCA/t-SNE, cross-modal comparisons | PDB/mmCIF files, structural features, HTML plots |
| Phylomorphospace Analysis | Evolutionary trajectories in PCA space, time-calibrated phylogeny | ML tree, time tree, phylomorphospace plots, HTML visualizations |
| Interpretability | Attention maps | Interpretability reports |

---

## Environment & Dependencies
- **Core**: Python 3.10+, PyTorch 2.0+, Hugging Face Transformers, fair-esm
- **ESM2 650M Model**: ~1.2GB, 650 million parameters, 33 layers
- **Visualization**: matplotlib, seaborn, umap-learn, scikit-learn, plotly, kaleido
- **Clustering**: hdbscan, scipy
- **Structure**: ESMFold, biopython, PyMOL (optional)
- **Interactive plots**: plotly, kaleido
- **Phylogenetics**: IQ-Tree, MEGA, R with phytools package
- GPU strongly recommended (A100 40GB or equivalent) for ESMFold structure prediction.

Example: `requirements.txt`  
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

---

## Evaluation Metrics
- **Classification**: Accuracy, macro/weighted F1, per-class ROC/PR curves.
- **Clustering**: Silhouette score, adjusted Rand index (if ground truth available).
- **Mutation Prediction**: ΔΔG-like stability scores, ranking by predicted impact.

---

## Risks and Challenges
- **Sequence length limit (1022 aa)** may truncate longer isoforms.
- **Cluster imbalance** (some CD300 sequence clusters may have few examples).
- **Computational cost**: fine-tuning a 650M parameter model is resource-intensive but manageable.
- **Unreliable homology labels**: Original dataset contains potentially incorrect homology-based annotations that we avoid by using sequence clusters.

---

## Future Directions
- Extend analysis to **pan-vertebrate datasets** beyond Euarchontoglires.
- Integrate **single-cell RNA-seq expression** of CD300 sequence clusters.
- Wet-lab validation of **predicted novel variants** (e.g., CRISPR knock-in).
- Functional studies of **mutation hotspots** in immune signaling.

---

## Deliverables
- `README.md` for setup and instructions.
- Workflow diagram (flowchart of pipeline).
- `projectoverview.md` (this file).
- Final report summarizing sequence cluster classification, novel variants, mutation predictions, and structural insights.
