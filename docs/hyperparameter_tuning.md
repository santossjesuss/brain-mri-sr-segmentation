# 🎯 Hyperparameter Tuning
 
Before running the full experiments, a targeted hyperparameter study was performed on the two baseline segmentation models (**LR Seg** and **HR Seg**). The goal of this preliminary phase was to maximize the performance of each network independently, establishing a fair benchmark: the **lower bound** (LR) and the **theoretical ceiling** (HR) for all subsequent multi-stage pipeline evaluations.
 
---
 
## 🧪 Experimental Setup
 
Four configurations were evaluated: a baseline (V1) and three variants. A grid search over these configurations analyzed the effect of the **initial learning rate**, the **Dice loss weight** and the **Cross-Entropy (CE) loss weight**. Invariant hyperparameters (batch size and total epochs) were held constant across all trials.
 
### Configuration Versions
 
| Hyperparameter | Baseline (V1) | Version 2 | Version 3 | Version 4 |
|---|:---:|:---:|:---:|:---:|
| Total Epochs | 100 | 100 | 100 | 100 |
| Batch Size | 16 | 16 | 16 | 16 |
| Initial Learning Rate ($\eta$) | $1 \times 10^{-4}$ | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| Dice Loss Weight | 0.5 | 0.5 | 0.7 | 0.7 |
| Cross-Entropy Weight | 0.5 | 0.5 | 0.3 | 0.3 |
 
### Grid Search Matrix
 
| Loss Weights (Dice / CE) | $\eta = 1 \times 10^{-4}$ | $\eta = 3 \times 10^{-4}$ |
|---|:---:|:---:|
| **Balanced (0.5 / 0.5)** | V1 (Baseline) | V2 |
| **Dice-Weighted (0.7 / 0.3)** | V4 | V3 |
 
By isolating and calibrating these key optimization parameters before introducing multi-stage architectures, all subsequent comparisons start from the best configuration found, or one that closely approximates it.
 
---
 
## 📈 Results
 
Arrows ($\uparrow$) indicate that higher values correspond to better performance for every metric. **Bold** marks the best value per column.
 
### Training Metrics
 
| Version | Model | Dice Score $\uparrow$ | IoU $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ |
|---|---|:---:|:---:|:---:|:---:|
| **Baseline (V1)** | LR Seg | 0.7725 | 0.6293 | 0.8485 | 0.7090 |
| | HR Seg | 0.7979 | 0.6638 | 0.8007 | **0.7952** |
| **Version 2** | LR Seg | 0.7860 | 0.6475 | 0.8471 | **0.7332** |
| | HR Seg | 0.8067 | 0.6761 | 0.8561 | 0.7627 |
| **Version 3** | LR Seg | **0.7938** | **0.6581** | **0.8656** | 0.7329 |
| | HR Seg | **0.8069** | **0.6763** | **0.8625** | 0.7580 |
| **Version 4** | LR Seg | 0.7731 | 0.6302 | 0.8503 | 0.7088 |
| | HR Seg | 0.8010 | 0.6681 | 0.8513 | 0.7564 |
 
### Testing Metrics
 
| Version | Model | Dice Score $\uparrow$ | IoU $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ |
|---|---|:---:|:---:|:---:|:---:|
| **Baseline (V1)** | LR Seg | 0.6738 | 0.5080 | **0.7447** | 0.6152 |
| | HR Seg | **0.7150** | **0.5564** | 0.6981 | **0.7327** |
| **Version 2** | LR Seg | 0.6899 | 0.5266 | 0.7223 | **0.6603** |
| | HR Seg | 0.7072 | 0.5470 | 0.7029 | 0.7115 |
| **Version 3** | LR Seg | **0.6948** | **0.5323** | 0.7373 | 0.6569 |
| | HR Seg | 0.7097 | 0.5501 | 0.7201 | 0.6997 |
| **Version 4** | LR Seg | 0.6735 | 0.5078 | 0.7217 | 0.6313 |
| | HR Seg | 0.7120 | 0.5529 | **0.7389** | 0.6870 |
 
---
 
## 🔍 Analysis
 
The results show a distinct, **resolution-dependent** response to each parameter set:
 
- **Low-resolution inputs** benefited significantly from a higher learning rate ($\eta = 3 \times 10^{-4}$) combined with stronger region-based loss weighting (0.7 Dice / 0.3 CE). Larger step sizes prevented the optimizer from stalling on blurred low-resolution features.
- **High-resolution inputs** performed best under a conservative step size ($\eta = 1 \times 10^{-4}$) and balanced loss weighting (0.5 Dice / 0.5 CE).
In this domain, **Recall was prioritized over Precision** to minimize false negatives and capture as many true lesion pixels as possible.
 
---
 
## ✅ Selected Configurations
 
| Model / Pipeline | Selected Version | Rationale |
|---|:---:|---|
| Standalone **LR Seg** baseline | **Version 3** | Highest Dice and IoU for low-resolution inputs, with strong Recall. |
| **HR Seg** and all **multi-stage pipelines** | **Version 1** | Best Dice, IoU and Recall for high-resolution inputs. |
 
Optimizing each setup individually establishes a rigorous benchmark in which every model operates at peak capacity. As a result, any performance gains obtained by including super-resolution reflect genuine architectural improvements rather than better tuning.
 