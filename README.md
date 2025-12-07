# 🔬 CytoDINO: Bone Marrow Cell Classification model based on DINOv3

A computationally efficient approach to bone marrow cell classification achieving **state-of-the-art results** on the MLL dataset by fine-tuning DINOv3 with LoRA adapters.

## Highlights

- **SOTA Performance**: 88.2% Weighted F1, 76.5% Macro F1 — outperforming DAGDNet* and DinoBloom.
- **Efficient Fine-tuning**: LoRA injection enables training on consumer GPUs with only ~5-8% trainable parameters.
- **Biologically-Informed Loss**: Custom hierarchical label smoothing respecting cell lineage relationships.
- **Clinical Safety**: Critical error penalty to minimize dangerous misclassifications (e.g., blast to normal).

## Model Comparison on MLL 21-Class Bone Marrow Dataset

| Model | Year | Weighted F1 | Macro F1 | Accuracy | Weighted Precision | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| ResNeXt-50 (Matek et al.) | 2021 | — | 59.0% | — | — | Original baseline on MLL dataset |
| Siamese Network | 2022 | ~81.0% | — | 84.0% | — | Triplet loss; validation set |
| HematoNet (CoAtNet) | 2022 | ~86.0% | — | — | — | Used 17 of 21 classes |
| DAGDNet | 2023 | 87.8% | 71.5% | 88.1% | 88.1% | Dual Attention Gates DenseNet |
| DinoBloom-L | 2024 | 84.9% | — | 85.0% | — | Foundation model (linear probe) |
| DinoBloom-G | 2024 | 84.9% | — | 85.0% | — | Foundation model (linear probe) |
| ESRT | 2025 | 76.1% | — | 75.6% | — | Embedding-Space Re-sampling; test set |
| **CytoDINO (Ours)** | **2025** | **88.2%** | **76.5%** | **88.2%** | **88.3%** | LoRA fine-tuned DINOv3 |

> **Note:** CytoDINO achieves **SOTA Macro F1 (76.5%)**, a **+5.0%** improvement over DAGDNet (71.5%), indicating superior performance on minority classes. Some models report validation metrics only - our results are on an unseen test set.

### Dataset Reference

The MLL dataset (Matek et al., 2021) contains 171,374 expert-annotated single-cell images from 945 patients across 21 morphological classes.

- Matek et al. (2021). [Highly accurate differentiation of bone marrow cell morphologies using deep neural networks](https://doi.org/10.1182/blood.2020010568). *Blood* 138(20):1917-1927

## Quick Start

```bash
# Clone repository
git clone https://github.com/azizbekmuminoff/CytoDINO
cd CytoDINO

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py
