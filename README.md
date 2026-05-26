# Comprehensive Road Scene Understanding for Autonomous Driving

**FAIMDL 2026 — Group 16**

This repository contains the code and experiments for the course project on semantic and anomaly segmentation for autonomous driving. The project explores the progression from pixel-based (ERFNet) to mask-based architectures (EoMT) and evaluates post-hoc anomaly detection methods on driving-scene benchmarks.

📄 **Report:** [PDF](link-to-report) <!-- Replace with actual link after submission -->

---

## Repository Structure

```
├── notebooks/
│   ├── Step4_EoMT_Comparison.ipynb      # EoMT-CS vs EoMT-COCO evaluation
│   ├── faimdl-project-step-5.ipynb      # Fine-tuning EoMT-COCO on Cityscapes
│   ├── step7_erfnet_baselines.ipynb     # ERFNet anomaly baselines (MSP, MaxLogit, MaxEntropy)
│   └── step8_eomt_anomaly.ipynb         # EoMT anomaly eval + RbA + temp scaling
├── eval/                                 # ERFNet evaluation scripts (from base repo)
│   ├── erfnet.py                         # ERFNet model definition
│   ├── evalAnomaly.py                    # MaxLogit reference implementation
│   └── ...
├── eomt/                                 # EoMT codebase
│   ├── configs/                          # Model configurations
│   ├── models/                           # EoMT model definition
│   └── inference.ipynb                   # EoMT inference demo
├── trained_models/                       # ERFNet pretrained weights
│   ├── erfnet_pretrained.pth
│   └── erfnet_encoder_pretrained.pth.tar
└── README.md
```

---

## How to Reproduce

### Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- GPU with ≥8 GB VRAM (T4/P100 or better)

### Datasets

1. **Cityscapes** (Steps 4–5): download from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) (`gtFine` + `leftImg8bit`)
2. **Anomaly Validation Datasets** (Steps 7–8): `Anomaly_Validation_Datasets.zip` from the course Drive folder. Contains:
   - `RoadAnomaly21/` — SMIYC RA-21 (10 images)
   - `RoadObsticle21/` — SMIYC RO-21 (30 images)
   - `FS_LostFound_full/` — FishyScapes Lost & Found (100 images)
   - `fs_static/` — FishyScapes Static (30 images)
   - `RoadAnomaly/` — Road Anomaly (60 images)

### Running the Notebooks

Each notebook is self-contained and designed to run on **Kaggle** with GPU acceleration.

| Notebook | Step | Platform | GPU needed |
|----------|------|----------|------------|
| `Step4_EoMT_Comparison.ipynb` | 4 | Kaggle/Colab | Yes |
| `faimdl-project-step-5.ipynb` | 5 | Kaggle/Colab | Yes (AMP) |
| `step7_erfnet_baselines.ipynb` | 7 | Kaggle | Optional (CPU works) |
| `step8_eomt_anomaly.ipynb` | 8 | Kaggle | Yes |

**Steps:**
1. Upload the relevant dataset as a Kaggle Dataset
2. Open the notebook and click **Add Input** to attach the dataset
3. Set Accelerator to **GPU T4** (or P100) in notebook settings
4. Run all cells — paths are configured at the top of each notebook

### EoMT Checkpoints

Download from the [EoMT model zoo](https://github.com/tkerssies/EoMT) (or course Drive):
- `eomt_cityscapes_semantic.pth` — trained on Cityscapes (19 classes)
- `eomt_coco_panoptic.pth` — trained on COCO (133 classes)

---

## Results Summary

### Semantic Segmentation (Cityscapes val)

| Model | mIoU |
|-------|------|
| EoMT-CS | 81.7% |
| EoMT-COCO (mapped) | 48.5% |
| EoMT-FT (head-only, 5 ep.) | 71.8% |

### Anomaly Segmentation — Pixel-based Baseline (ERFNet)

| Method | RA-21 AuPRC | RO-21 AuPRC | FS L&F AuPRC | FS Static AuPRC |
|--------|-------------|-------------|--------------|-----------------|
| MSP | 15.8 | 1.4 | 0.5 | 1.3 |
| Max Logit | 15.6 | 2.2 | 0.4 | 1.3 |
| Max Entropy | 15.1 | 1.3 | 0.5 | 1.2 |

---

## Team — Group 16

| Member | Contribution |
|--------|-------------|
| Federico Remy | Steps 4, 6, 7, 8 — Designed the COCO-to-Cityscapes class mapping and unified evaluation pipeline for semantic, instance, and panoptic segmentation; studied anomaly segmentation task and post-hoc methods; implemented pixel-based anomaly baselines (ERFNet + MSP, MaxLogit, MaxEntropy); implemented mask-based anomaly baselines (EoMT + MSP, MaxLogit, MaxEntropy, RbA) with temperature scaling; wrote and structured the final report |
| Jokebed Queen Lusambo | Steps 5, 7, 8 — Fine-tuned the COCO-trained EoMT on the Cityscapes training set; contributed to pixel-based and mask-based anomaly evaluation |
| Sphoorthi Madutha | Steps 1, 2, 3 — Studied semantic segmentation and ERFNet; studied instance and panoptic segmentation; reviewed mask architecture literature (MaskFormer, Mask2Former, EoMT/DINOv2) |
| Ali Ghorbanpour | Steps 1, 2, 3 — Studied semantic segmentation and ERFNet; studied instance and panoptic segmentation; reviewed mask architecture literature (MaskFormer, Mask2Former, EoMT/DINOv2) |

---

## References

- Romera et al., *ERFNet*, IEEE T-ITS 2018
- Kirillov et al., *Panoptic Segmentation*, CVPR 2019
- Cheng et al., *MaskFormer*, NeurIPS 2021
- Cheng et al., *Mask2Former*, CVPR 2022
- Kerssies et al., *EoMT*, CVPR 2025
- Oquab et al., *DINOv2*, TMLR 2023
- Nayal et al., *RbA*, ICCV 2023
- Hendrycks et al., *Scaling OoD Detection*, ICML 2022
- Chan et al., *SegmentMeIfYouCan*, NeurIPS 2021
- Blum et al., *FishyScapes*, ICCVW 2019
