# MSF-Mamba
Description
Drug--miRNA association prediction is essential for understanding therapeutic mechanisms and identifying regulatory biomarkers. Existing computational approaches rarely incorporate miRNA secondary-structure information into drug--miRNA association modeling frameworks. To address these challenges, we propose MSF-Mamba, a unified multimodal framework for drug--miRNA association prediction. MSF-Mamba encodes drug SMILES strings, molecular graphs, miRNA sequences, and miRNA secondary-structure features into modality-specific embeddings and further integrates them into a task-oriented multimodal representation. A Multi-Scale Mamba architecture is designed with local, global, and convolutional branches to capture short-range motifs, contextual dependencies, and fine-grained compositional features with linear computational complexity. Furthermore, a Calibration--Interaction Attention Module is introduced to decouple intra-modality feature calibration from inter-modality interaction learning. Specifically, Differential Attention is used for intra-modality feature calibration by reducing redundant signals and preserving informative differences, whereas Cross-Attention learns association-relevant interactions between drug and miRNA representations. 


# Availability

Datasets and source code are available at: https://github.com/lweiwei0918-maker/MSF-Mamba

# File Description

+ `data/`
  * `drug_id_smiles.xlsx` contains drug ID (from DrugBank) and SMILES;
  * `miRNA_drug_matrix.xlsx` contains known miRNA-drug association;
  * `miRNA_sequences.xlsx` contains miRNA ID (from miRBase) and sequences.

# Local Running
python training.py

# Environment
Before running, please make sure the following packages are installed in your Python environment. We strongly recommend using a virtual environment.

Core Dependencies

- python = 3.9  
- torch = 2.3.0+cu118  
- torch-geometric = 2.5.3  
- torch-scatter = 2.1.2+pt23cu118  
- mamba-ssm = 2.2.2  
- rdkit = 2024.3.5  
- transformers = 4.46.3  

Basic Data Handling & Utilities

- pandas >= 2.0.0  
- numpy >= 1.24.0  
- scikit-learn >= 1.3.0  
- tqdm >= 4.67.0  
- matplotlib >= 3.7.0  
- seaborn >= 0.13.0