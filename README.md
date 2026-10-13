# MSF-Mamba
Description
Drug–miRNA association prediction is important for understanding post-transcriptional regulatory mechanisms and identifying potential  therapeutic biomarkers. However, traditional biological experiments are time consuming and costly, making it necessary to develop efficient computational  approaches for drug-miRNA association prediction. To address these challenges, we propose MSF-Mamba, a multimodal multi-scale fusion framework, which integrates sequence and structure information of drugs and miRNAs. Specifically, we propose a multi-scale Mamba to capture long range contextual dependencies of SMILES strings and miRNA sequences. In addition,  the calibration-interaction attention module explicitly decouples intra-modality feature refinement from cross-modality interaction learning, producing task specific fused representations that enhance drug–miRNA association prediction. Experimental results on a benchmark dataset demonstrate that MSF-Mamba outperforms the compared methods, achieving AUC and AUPR values of 0.9458 and 0.9383 under 10-fold cross-validation.


# Availability

The datasets and source code are freely available at: https://github.com/zsflw/MSF-Mamba

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
