# LSST Astronomical Time-Series Classification

This repository contains the code, data splits, and model architectures for classifying irregular, highly imbalanced astronomical time-series data using the unblinded PLAsTiCC dataset. 

To handle the extreme sparsity and intermittency of the astronomical light-curves, this project benchmarks advanced sequential deep learning architectures - specifically Temporal Fusion Transformers (TFT) and State Space Models (SSM) against traditional baseline models.

---

## Repository Navigation

The repository is structured to balance a highly standardized evaluation pipeline with complete architectural freedom for each model development.

### 1. Literature & Background (`/literature_survey/`)
This folder contains the foundational research guiding our modeling decisions. It currently houses 3 core research papers detailing the mathematical formulations of State Space Models, Transformer attention mechanisms applied to time-series, and the physical characteristics of the PLAsTiCC dataset.

### 2. The Shared Pipeline (`/00_Shared_Pipeline/` & `/data_splits/`)
To ensure all models are trained on the exact same universe and evaluated fairly, the starting and finishing lines are strictly standardized.
* **Exploratory Data Analysis (`00_data_exploration_EDA.ipynb`):** Global analysis of the 3.5M object universe, confirming dataset alignment, intermittency ($\Delta t$), and extreme class imbalance.
* **Data Stratification (`01_stratified_data_split.ipynb`):** The logic used to extract a statistically representative sample of the universe for deep learning.
* **The "Source of Truth" (`/data_splits/`):** Contains `train_ids.csv` (100k objects), `val_ids.csv` (15k objects), and `test_ids.csv`. **All individual models must load these exact IDs.**
* **The Finish Line (`metrics.py`):** The shared scoring script containing custom mathematical implementations for Weighted Log-Loss (Kaggle standard), Macro F1, Macro PR-AUC, and the Brier Score. 

### 3. Independent Model Architectures
The project is split into three independent environments:
* `/TFT/` (Temporal Fusion Transformer)
* `/SSM/` (State Space Model)
* `/SSM_MLP/` (State Space Model + Multi-Layer Perceptron)

**Delegated Responsibility:** Each model folder:
* **Preprocessing:** Custom flux normalization, metadata imputation, and feature engineering ($\Delta t$ calculations).
* **DataLoaders:** Custom sequence padding, masking, and tensor formatting.
* **Hyperparameters:** Batch sizes, learning rates, and optimizer choices.
* *Constraint:* All models must generate their final evaluation scores by importing and executing the shared `metrics.py` script.

### 4. Project Management
* `project_tracker.xlsx`: The central ledger for task delegation, phase deadlines, and milestone tracking.
* `Final_Project_Report.pdf`: The academic write-up and architectural comparison.
* `requirements.txt`: The required Python environments and library versions for reproducibility.

---

## Standard Operating Procedure (SOP)

### Compute & Kaggle Workflow
Because the raw unblinded dataset exceeds 22 GB, local training is not practical for most machines.
1. **Develop Locally:** Writing the `.py` scripts (architectures, datasets) locally in VS Code.
2. **Train on Kaggle:** Uploading the model's specific folder to a Kaggle Notebook. Read the massive Kaggle datasets natively using the `BASE_DIR = '/kaggle/input/...'` path, but filter the rows using the lightweight IDs stored in our `/data_splits/` folder.
3. **Export Weights:** Once Kaggle finishes the 12-hour GPU execution, download your `.pth` model weights and final prediction arrays.

### Git Branching Strategy

To prevent merge conflicts and protect the integrity of the shared pipeline, we adhere to a strict branching workflow:
* **`main` Branch:** The production-ready branch. Code here must be error-free and fully executed.
* **`dev` Branch:** The active integration branch. All feature branches merge here first.
* **Your Workflow:** 1. Pull the latest `dev` branch (`git pull origin dev`).
  2. Create a new branch for your specific task (`git checkout -b feature/tft-dataloader`).
  3. Commit your changes locally.
  4. Push to your feature branch and open a Pull Request into `dev` for team review.
