# 🔬 CytoDINO: Bone Marrow Cell Classification model based on DINOv3

A computationally efficient approach to bone marrow cell classification achieving **state-of-the-art results** on the MLL dataset by fine-tuning DINOv3 with LoRA adapters.

## Highlights

- **SOTA Performance**: 88.2% Weighted F1, 76.5% Macro F1 — outperforming DAGDNet* and DinoBloom.
- **Efficient Fine-tuning**: LoRA injection enables training on consumer GPUs with only ~5-8% trainable parameters.
- **Biologically-Informed Loss**: Custom hierarchical label smoothing respecting cell lineage relationships.
- **Clinical Safety**: Critical error penalty to minimize dangerous misclassifications (e.g., blast to normal).

## Summary Metrics

| Metric | DenseNet | DenseNet* | DAGDNet | DAGDNet* | DinoBloom-L | DinoBloom-G | CytoDINO (Proposed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Weighted Precision | 84.9% | 87.4% | 85.6% | 88.1% | — | — | 88.3% | 
| Weighted F1 | 84.5% | 87.1% | 84.9% | 87.8% | 84.9% | 84.9% | **88.2%** |
| Macro F1 | 55.8% | 64.0% | 55.3% | 71.5% | — | — | **76.5%** |
| Accuracy | — | — | — | — | 85.0% | 85.0% | **88.2%** |

> **Note:** Our models achieve state-of-the-art Macro F1 scores (76.5%), representing a **+5.0%** improvement over DAGDNet* (71.5%), indicating superior performance on minority classes.

## Quick Start

```bash
# Clone repository
git clone https://github.com/azizbekmuminoff/CytoDINO
cd CytoDINO

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py
