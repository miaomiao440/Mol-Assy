# Mol-Assy: A Deep Learning Framework for Conditional-Imbalance-Aware Molecular Pair Prediction

Mol-Assy is a deep learning framework designed for molecular pair prediction under **conditional imbalance**, enabling robust modeling in scenarios where standardized benchmarks are lacking, data acquisition is difficult, samples are extremely sparse, and reporting bias is significant.  
Binary small-molecule co-assembly prediction is presented as a representative application.

---

## Overview

###Task Definition

In many molecular pair prediction tasks, the model takes **two molecules** as input and predicts whether they satisfy a specific relationship, interaction, or cooperative behavior. Such problems are widely encountered in molecular interaction modeling, combination drug design, and molecular pairing screening.

A key characteristic of these tasks is that the candidate pair space grows **combinatorially** with the size of the molecular library. As a result, exhaustive experimental screening is typically costly and time-consuming. Meanwhile, real-world datasets are often characterized by **severe class imbalance and data sparsity**, which further complicates model development.

In this work, **binary small-molecule co-assembly prediction** is used as a representative instance of this class of problems. Specifically, given two small molecules, the model aims to predict whether they can form a stable binary co-assembly system. Similar to other molecular pair prediction tasks, this problem also suffers from an enormous combinatorial search space, making large-scale experimental screening impractical. This motivates the development of a **generalizable and reproducible computational framework**.

---

### Computational Challenges

Compared with conventional molecular classification tasks, binary molecular pair prediction faces more complex real-world constraints:

#### (1) Lack of standardized benchmarks and unified evaluation protocols

This research area has long lacked standardized benchmark datasets, unified evaluation protocols, and rigorously reproducible validation frameworks. As a result, it is difficult to perform systematic and fair comparisons across different methods.

For **binary small-molecule co-assembly prediction**, this issue is particularly pronounced. The research foundation is significantly less mature than that of well-established molecular property prediction tasks, and the data and evaluation systems required for data-driven modeling remain insufficient.

#### (2) Difficult data acquisition and strong literature bias

Existing studies tend to focus on molecular systems with relatively simple structures and high experimental success rates, while novel or structurally complex molecules are underrepresented.

Moreover, the literature almost exclusively reports successful co-assembly cases, whereas failed experiments are rarely documented. This leads to datasets that are **extremely sparse, skewed, and unrepresentative**, fundamentally limiting the model’s ability to learn generalizable patterns.

#### (3) Fine-grained conditional imbalance and graph over-smoothing

Beyond data distribution issues, graph neural networks (GNNs) typically suffer from **over-smoothing**, which weakens feature discriminability in rare conditional domains.

This phenomenon further exacerbates overfitting toward high-frequency samples, making the model more likely to misclassify **rare, complex, or novel molecular pairs** as negative samples. At the same time, the presence of **fine-grained conditional imbalance** introduces an additional layer of difficulty beyond conventional global class imbalance.

---

### Method Overview

Mol-Assy employs a **molecular graph encoding module** to learn structural representations of input molecules, mapping atoms, chemical bonds, and local topological information into embeddings suitable for downstream prediction.

In addition, a **cross-molecular interaction modeling module** is introduced to explicitly capture key interaction patterns between the two molecules.

To address the fine-grained conditional imbalance commonly observed in this task, Mol-Assy introduces a **condition-adaptive weighted objective**, which adaptively adjusts the training contribution of samples from sparse conditional domains.

Furthermore, Mol-Assy incorporates an **over-smoothing-resistant graph representation learning design**, aiming to preserve discriminative features of rare samples during long-range information propagation. This design improves the robustness and generalization ability of the model under complex and sparse data distributions.

---


## Repository Structure

```text
Mol-Assy/
│  data.py                     # Data processing scripts
│  features.py                 # Molecular feature construction and encoding
│  julei.py                    # Clustering-related analysis scripts
│  loaddataset_new2.py         # Data loading and dataset construction
│  long.py                     # Chain-length or statistical analysis scripts
│  main_tiaocanloss1.py        # Main training entry point
│  MPNN.py                     # Molecular graph message passing network implementation
│  preprocess_new.py           # Data preprocessing script
│  test.py                     # Model testing and evaluation
│  weight.py                   # Conditional weighting / sample weight implementation
│  requirements.txt            # Python dependencies
│  README.md                   # Project documentation
│
├─data/
│      all-c1_pairs.xlsx                    # Raw / curated molecular pair data
│      all_smiles.xlsx                      # Molecular SMILES information
│      fine_data.py                         # Fine-grained data processing script
│      train_val1_s.xlsx                    # Data split file (example)
│      train_val1_weights.xlsx              # Sample weight file (example)
│      train_val1_weight_analysis.xlsx      # Weight analysis results (example)
│      train_val1_zs.xlsx                   # Auxiliary split file (example)
│      train_val4_s.xlsx
│      train_val4_weights.xlsx
│      train_val4_weight_analysis.xlsx
│      train_val4_zs.xlsx
│
├─datasets/
│      dataset.xlsx                         # Full dataset
│      all_smiles.xlsx                      # Molecular SMILES information
│      Fingerprints.npy                     # Molecular fingerprint features
│      butina_cluster_results.xlsx          # Butina clustering results
│
├─models/
│  │  loss.py                   # Loss function implementation
│  └─__pycache__/               # Python cache files
│
├─tb/                           # TensorBoard logs and training visualizations
│
└─__pycache__/                  # Python cache files
```
---

## Dependencies and Setup

Please ensure that `conda` is installed on your system.  
We recommend using a CUDA-compatible environment (e.g., `cu116`) for GPU acceleration.

To set up the `Mol-Assy` environment, run the following commands:

```bash
conda create -n mol-assy python=3.10
conda activate mol-assy
pip install -r requirements.txt
```
---

## Dataset

###Dataset Motivation

Binary small-molecule co-assembly prediction has long lacked standardized benchmarks. Data sources in this field are highly fragmented and subject to significant literature bias: existing studies tend to focus on structurally simple molecules with higher success rates, while failed cases are rarely systematically reported.

As a result, the dataset itself becomes a critical foundation for computational research in this task.

---

###Dataset Construction

To improve data coverage from fragmented literature sources, we introduce a literature information extraction pipeline based on a fine-tuned **DeepSeek** model.

First, relevant academic papers and patents are collected and preprocessed. The fine-tuned model is then used to extract structured molecular pair information from abstracts and full texts. All automatically extracted samples are manually verified one by one before being included in the final dataset.

It should be noted that this pipeline belongs to the **benchmark construction stage** and is not part of the Mol-Assy prediction model itself.

The Mol-Assy dataset is constructed through a combination of manual curation and literature-based information extraction, resulting in a total of **2,706 binary small-molecule pairs**.

---

###Dataset Statistics

The key statistics of the dataset are summarized below:

| Statistic | Value |
|---|---:|
| Total samples | 2,706 |
| Positive samples | 587 |
| Negative samples | 2,119 |
| Conditional molecules with only 1 pair | 91.4% |
| Conditional molecules with ≤ 5 samples | 92.9% |
| Conditional molecules with no positive samples | 19.9% |
| Conditional molecules with only 1 positive sample | 73.0% |

---

### Evaluation Scenarios

To systematically evaluate model performance, Mol-Assy defines two benchmark scenarios:

- **S1**: Globally balanced, but still conditionally imbalanced  
- **S2**: Both globally imbalanced and conditionally imbalanced  

Both scenarios are evaluated using **5-fold cross-validation**.

Therefore, the Mol-Assy dataset serves not only as input data but also as an integral part of the overall benchmark design.

---

## Training

**Note:** Before training Mol-Assy, please complete data preprocessing and ensure that the data paths, split file paths, and output directories in the configuration are correctly specified.

### Data Preprocessing

```bash
python preprocess_new.py
```
Training from Scratch
After preprocessing, run the training script:
```
python main_tiaocanloss1.py
```
By default, Mol-Assy is trained using the Adam optimizer, with predefined settings for the initial learning rate, weight decay, and batch size. The training process also incorporates dropout, DropEdge, and a learning rate scheduler to improve convergence stability and generalization performance.
The trained model checkpoints and log files will be saved to the output directory specified in the configuration file.

## Evaluation and Reproducibility

Mol-Assy is evaluated under two benchmark scenarios:

- **S1**: Globally balanced but conditionally imbalanced  
- **S2**: Both globally and conditionally imbalanced  

Both scenarios adopt **5-fold cross-validation** and are evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- AUC-PR  

### Run Evaluation

```bash
python test.py
```
Reproducing Paper Results
To reproduce the main results reported in the paper, we recommend running the configurations for:
Scenario S1 
Scenario S2 
Comparative experiments 
Ablation studies 
while keeping the same data splits and random seeds as described in the paper.
The reported performance of Mol-Assy includes:
S1: F1 = 0.882 
S2: F1 = 0.814, AUC-PR = 0.842, Accuracy = 0.926 
Mol-Assy consistently outperforms the unweighted model and multiple baseline methods across several evaluation metrics.
Output
Evaluation results are saved by default and include:
Per-fold evaluation metrics 
Aggregated results across folds 
Visualization outputs 
These results can be directly used for reproducing the paper results and conducting comparative analysis.

