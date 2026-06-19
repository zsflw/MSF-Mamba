# MSF-Mamba
Description
Drug–miRNA association prediction plays a critical role in elucidating therapeutic mechanisms and identifying regulatory biomarkers. However, existing computational approaches remain limited in effectively capturing complex biological characteristics, including insufficient modeling of miRNA structural features, difficulties in learning long-range dependencies in biological sequences, and challenges in integrating heterogeneous multimodal data.We proposes MSF-Mamba, a unified multimodal framework for drug–miRNA association prediction. The framework applies a Mamba-based sequence modeling architecture (a state space model) to this task, enabling efficient and scalable long-sequence modeling. Drug SMILES sequences, molecular graph representations, miRNA primary sequences, and structure features are integrated into a unified representation space, allowing comprehensive characterization of both chemical and biological properties. A multi-scale sequence modeling mechanism captures both local and global dependency patterns, and a calibration–interaction attention module aligns heterogeneous feature distributions and enhances cross-modal interactions.Extensive experiments on a benchmark dataset comprising 1,578 miRNAs, 156 drugs, and 8,720 validated associations demonstrate that MSF-Mamba consistently outperforms state-of-the-art methods. The model achieves an AUC of 0.9458 and a PR-AUC of 0.9383 under 10-fold cross-validation, showing robust and consistent improvements over competitive baselines. Ablation studies further confirm the contribution of each component, and case studies validate the biological relevance of the predicted associations.MSF-Mamba provides an effective and scalable solution for drug–miRNA association prediction, facilitating accurate modeling of sequence–structure relationships and offering valuable insights into underlying regulatory mechanisms.


# Availability

Datasets and source code are available at: https://github.com/zsflw/MSF-Mamba

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
