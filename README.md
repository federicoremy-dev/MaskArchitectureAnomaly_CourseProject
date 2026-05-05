# Semantic and Anomaly Segmentation with Mask Architectures

This repository contains the codebase and experiments for the course project on **Semantic and Anomaly Segmentation**. The project explores the evolution of segmentation models (from pixel-based to mask-based architectures) and evaluates various post-hoc methods for anomaly detection in urban driving scenes.

📄 **[Insert Link to the Final PDF Report Here]**

---

## 📌 Project Overview

The project is structured into three main phases, moving from theoretical understanding to practical implementation and evaluation of state-of-the-art segmentation models (ERFNet, DINOv2, EoMT).

### **Phase A: Theoretical Study (Steps 1-3)**
A comprehensive review of the evolution of segmentation architectures:
*   **Semantic Segmentation:** Understanding pixel-level classification and encoder-decoder structures (ERFNet).
*   **Panoptic Segmentation:** Unifying semantic (stuff) and instance (things) segmentation.
*   **Mask Architectures:** Tracing the shift from per-pixel paradigms to mask classification (`MaskFormer` → `Mask2Former` → `EoMT`), leveraging large-scale self-supervised ViT pre-training (`DINOv2`).

### **Phase B: Segmentation Experimentation (Steps 4-5)**
Practical evaluation and fine-tuning of the **Encoder-only Mask Transformer (EoMT)**:
*   **Step 4 - Model Comparison:** Qualitative and quantitative (mIoU) evaluation of two EoMT checkpoints trained on COCO (panoptic) and Cityscapes (semantic). This includes a custom class-mapping strategy to align COCO classes with Cityscapes train IDs.
*   **Step 5 - Fine-Tuning:** Leveraging COCO's visual priors by fine-tuning the COCO-pretrained EoMT model on the Cityscapes dataset, experimenting with head-only adaptation and backbone unfreezing.

### **Phase C: Anomaly Segmentation (Steps 6-8)**
Application of post-hoc anomaly detection methods to identify Out-of-Distribution (OoD) objects:
*   **Post-Hoc Methods Evaluated:** Maximum Softmax Probability (MSP), MaxLogit, Max Entropy, and Rejected by All (RbA).
*   **Step 7 - Pixel-Based Baseline:** Evaluating post-hoc methods on the **ERFNet** architecture.
*   **Step 8 - Mask-Based Evaluation:** Adapting the evaluation pipeline to extract pixel-wise confidence scores from EoMT mask outputs. The methods are evaluated across multiple EoMT checkpoints, including Temperature Scaling optimizations.

---

## 📊 Datasets

The experiments utilize the following datasets:
*   **Cityscapes:** Used for semantic segmentation evaluation and fine-tuning (requires `gtFine` and `leftImg8bit`).
*   **COCO:** Used implicitly via the pre-trained EoMT checkpoint.
*   **Anomaly Validation Datasets:** SMIYC (RA-21, RO-21), Fishyscapes (L&F, Static), and Road Anomaly.

---

## 🚀 How to Run the Code

The codebase is designed to be executed in **Google Colab** to leverage GPU acceleration (and AMP for memory efficiency). 

1. Clone this repository to your local machine or directly in your Colab environment:
   ```bash
   git clone https://github.com/[Your-Username]/[Repo-Name].git
   ```
2. Navigate to the `notebooks/` (or `src/`) directory where the `.ipynb` files are stored.
3. Open the desired notebook in Google Colab.
4. Follow the instructions within the first cell of each notebook to mount your Google Drive, set up the environment (`pip install -r requirements.txt`), and download the required weights/datasets.

---

## 📈 Key Deliverables

*   **Qualitative Results:** Side-by-side visual comparisons of Semantic vs. Panoptic predictions.
*   **mIoU Table:** Quantitative comparison of the base Cityscapes model, base COCO model, and our Fine-tuned model.
*   **Anomaly Detection Metrics:** Extensive evaluation tables reporting **AuPRC** and **FPR95** across 5 datasets, comparing ERFNet and multiple EoMT configurations.

---

## 👥 Team & Contributions

*   **[Name 1]**: Theoretical study (Steps 1-3), Anomaly segmentation background (Step 6), and Final Report formulation.
*   **[Name 2]**: Model comparison pipeline, evaluation logic, and COCO-Cityscapes class mapping strategy (Step 4).
*   **[Name 3]**: Fine-tuning implementation, AMP optimization, and LoRA experimentation (Step 5).
*   **[Name 4]**: ERFNet anomaly baselines (Step 7), EoMT post-hoc adaptation, RbA implementation, and Temperature Scaling (Step 8).

---

## 📚 References

The project builds upon the following key papers and architectures:

| Reference | Topic / Architecture |
| :--- | :--- |
| **[10]** | ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation |
| **[11]** | Panoptic Segmentation (Kirillov et al.) |
| **[3, 4]** | MaskFormer & Mask2Former |
| **[5]** | DINOv2: Learning Robust Visual Features without Supervision |
| **[6]** | EoMT (Encoder-only Mask Transformer) |
| **[1, 2]** | SMIYC (SegmentMeIfYouCan) & Fishyscapes Benchmarks |
| **[7, 8]** | RbA (Rejected by All) & Scaling Out-of-Distribution Detection (MaxLogit) |
```