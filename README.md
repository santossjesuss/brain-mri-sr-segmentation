# Enhancement of Brain Segmentation in MRI Scans
> **Improving segmentation performance using Super-Resolution & Segmentation pipelines for Multiple Sclerosis patients in brain MRI scans**

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch)](https://github.com/pytorch/pytorch/tree/v2.9.1)

---
### 📌 Overview

* **Author:** [Jesús Santos Barba](https://www.linkedin.com/in/jesús-santos-215706315)
* **Context:** Final Degree Project (TFG) at the [University of Málaga (UMA)](https://www.uma.es)
* **Domain:** Deep Learning · Medical Image Processing · Brain MRI Segmentation

<!-- A deep learning project focused on improving Multiple Sclerosis (MS) lesion segmentation performance on brain MRI scans through the use of multi-model pipelines. 

*Developed by [Jesús Santos Barba](https://www.linkedin.com/in/jesús-santos-215706315) as part of my TFG (Final Degree Project) at the University of Malaga.* -->

## 📖 Abstract
Automated brain segmentation in MRI poses significant challenges, such as inter-patient morphological variability, the presence of imaging artifacts, and low contrast with adjacent tissues. 

In this work, a set of pipelines combining deep neural networks for SR reconstruction and segmentation is presented, comparing them against a baseline segmentation neural network. The results of this baseline are established as a foundation to seek improvements through the pipeline processing of the proposed models. We integrate, within the pipelines, an RCAN-type SR model for reconstruction, a 2D U-Net for segmentation, and the same U-Net model for the baseline segmentation. 

The loss function employed in the segmentation is an average of Dice and CE, in the SR it is $L_1$, and in the pipelines, segmentation losses and singularly a multi-objective loss are chosen, the latter being an average of the SR and segmentation losses. 

We evaluate on brain planes using Dice, IoU, Precision, and Recall metrics. The system manages to improve the brain segmentation results in Multiple Sclerosis patients in MRI images.


## 📊 Key Findings & Results

> **Core Takeaway:** 

<details open>
<summary>
    <b>1. Training performance metrics</b>
</summary>
<br>

| Exp | Pipeline | Dice | IoU | Precision | Recall |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | LR Seg (Baseline) | 0.7796 | 0.6389 | 0.8412 | 0.7264 |
| **--** | HR Seg | 0.8027 | 0.6705 | 0.8623 | 0.7509 |
| **3** | Frozen SR $\rightarrow$ Trainable Seg | 0.7922 | 0.6559 | 0.8689 | 0.7279 |
| **4** | Trainable SR $\rightarrow$ Frozen Seg | 0.8004 | 0.6673 | 0.8610 | 0.7479 |
| **5** | Joint E2E | 0.7874 | 0.6494 | 0.8533 | 0.7310 |
| **6** | Joint Combined (Joint Loss) | 0.8019 | 0.6694 | **0.8798** | 0.7367 |
| **7** | Sequential Joint (SR First) | 0.7964 | 0.6617 | 0.8291 | **0.7662** |
| **8** | Sequential Joint (Seg First) | **0.8030** | **0.6709** | 0.8613 | 0.7521 |

</details>

<details open>
<summary>
    <b>2. Testing performance metrics (On Unseen Data)</b>
</summary>
<br>

| Exp | Pipeline | Dice | IoU | Precision | Recall |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | LR Seg (Baseline) | 0.6947 | 0.5323 | 0.7210 | 0.6704 |
| **--** | HR Seg | 0.7104 | 0.5508 | 0.7303 | 0.6914 |
| **2** | Frozen SR $\rightarrow$ Frozen Seg | 0.7077 | 0.5476 | 0.7165 | 0.6991 |
| **3** | Frozen SR $\rightarrow$ Trainable Seg | 0.7050 | 0.5444 | 0.7187 | 0.6917 |
| **4** | Trainable SR $\rightarrow$ Frozen Seg | 0.7109 | 0.5514 | 0.7308 | 0.6920 |
| **5** | Joint E2E | 0.7011 | 0.5398 | 0.7275 | 0.6767 |
| **6** | Joint Combined (Joint Loss) | 0.7062 | 0.5458 | **0.7533** | 0.6646 |
| **7** | Sequential Joint (SR First) | **0.7114** | **0.5521** | 0.7144 | **0.7085** |
| **8** | Sequential Joint (Seg First) | 0.7087 | 0.5488 | 0.7202 | 0.6976 |

</details>

<details open>
<summary>
    <b>3. Relative Performance Gains (%) vs. Baseline</b>
</summary>
<br>

| Exp | Pipeline | $\Delta$ Dice (%) | $\Delta$ IoU (%) | $\Delta$ Precision (%) | $\Delta$ Recall (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | LR Seg (Baseline) | 0.00 | 0.00 | 0.00 | 0.00 |
| **--** | HR Seg | +1.57 | +1.85 | +0.93 | +2.10 |
| **2** | Frozen SR $\rightarrow$ Frozen Seg | +1.30 | +1.53 | -0.45 | +2.87 |
| **3** | Frozen SR $\rightarrow$ Trainable Seg | +1.03 | +1.21 | -0.23 | +2.13 |
| **4** | Trainable SR $\rightarrow$ Frozen Seg | +1.62 | +1.91 | +0.98 | +2.16 |
| **5** | Joint E2E | +0.64 | +0.75 | +0.65 | +0.63 |
| **6** | Joint Combined (Joint Loss) | +1.15 | +1.35 | **+3.23** | -0.58 |
| **7** | Sequential Joint (SR First) | **+1.67** | **+1.98** | -0.66 | **+3.81** |
| **8** | Sequential Joint (Seg First) | +1.40 | +1.65 | -0.08 | +2.72 |

</details>


## 📖 Documentation & Deep Dives

- [🏗️ Pipeline Architectures & Experiments](docs/pipelines.md): Detailed breakdown of all 8 training configurations and joint loss functions.
<!-- - [⚙️ Setup & Training Guide](docs/setup.md): Instructions for data preprocessing, environment setup, and running experiments. -->


## 🏛️ Academic Context
This project was developed as a Final Degree Project (TFG) completed at the [University of Malaga (UMA)](https://www.uma.es).