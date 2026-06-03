# Comprehensive Road Scene Understanding for Autonomous Driving

**FAIMDL 2026 — Group 16**

This repository contains the code and experiments for the course project on semantic and anomaly segmentation for autonomous driving. The project explores the progression from pixel-based (ERFNet) to mask-based architectures (EoMT) and evaluates post-hoc anomaly detection methods on driving-scene benchmarks.


---

## Repository Structure

```
├── notebooks/
│   ├── Step4_EoMT_Comparison.ipynb          # EoMT-CS vs EoMT-COCO evaluation
│   ├── Step5_finetune_EoMT.ipynb            # Fine-tuning EoMT-COCO on Cityscapes
│   ├── Step7_ERFnet_Anomaly_Baseline.ipynb  # ERFNet anomaly baselines (MSP, MaxLogit, MaxEntropy)
│   └── Step8_EoMT_Anomaly_Baseline.ipynb   # EoMT anomaly eval + RbA + temp scaling
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
   - `RoadObstacle21/` — SMIYC RO-21 (30 images)
   - `FS_LostFound_full/` — FishyScapes Lost & Found (100 images)
   - `fs_static/` — FishyScapes Static (30 images)
   - `RoadAnomaly/` — Road Anomaly (60 images)

### Running the Notebooks

Each notebook is self-contained and designed to run on **Kaggle** with GPU acceleration.

| Notebook | Step | Platform | GPU needed |
|----------|------|----------|------------|
| `Step4_EoMT_Comparison.ipynb` | 4 | Kaggle/Colab | Yes |
| `Step5_finetune_EoMT.ipynb` | 5 | Kaggle/Colab | Yes (AMP) |
| `Step7_ERFnet_Anomaly_Baseline.ipynb` | 7 | Kaggle | Optional (CPU works) |
| `Step8_EoMT_Anomaly_Baseline.ipynb` | 8 | Kaggle | Yes |

**Steps:**
1. Upload the relevant dataset as a Kaggle Dataset
2. Open the notebook and click **Add Input** to attach the dataset
3. Set Accelerator to **GPU T4** (or P100) in notebook settings
4. Run all cells — paths are configured at the top of each notebook

### EoMT Checkpoints

Download from the [EoMT model zoo](https://github.com/tkerssies/EoMT) (or course Drive):
- `eomt_cityscapes.bin` — trained on Cityscapes (19 classes)
- `eomt_coco.bin` — trained on COCO (133 classes)
- `eomt_ft_head_epoch04.ckpt` — fine-tuned head on Cityscapes (19 classes)



## Team — Group 16

| Member | Contribution |
|--------|-------------|
| Federico Remy | Steps 4, 6, 7, 8 — Designed the COCO-to-Cityscapes class mapping and unified evaluation pipeline for semantic, instance, and panoptic segmentation; studied anomaly segmentation task and post-hoc methods; implemented pixel-based anomaly baselines (ERFNet + MSP, MaxLogit, MaxEntropy); implemented mask-based anomaly baselines (EoMT + MSP, MaxLogit, MaxEntropy, RbA) with temperature scaling; wrote and structured the final report |
| Jokebed Queen Lusambo | Steps 5, 7, 8 — Fine-tuned the COCO-trained EoMT on the Cityscapes training set; contributed to pixel-based and mask-based anomaly evaluation |
| Sphoorthi Madutha | Steps 1, 2, 3 — Studied semantic segmentation and ERFNet; studied instance and panoptic segmentation; reviewed mask architecture literature (MaskFormer, Mask2Former, EoMT/DINOv2); wrote and structured the final report|
| Ali Ghorbanpour | Steps 1, 2, 3 — Studied semantic segmentation and ERFNet; studied instance and panoptic segmentation; reviewed mask architecture literature (MaskFormer, Mask2Former, EoMT/DINOv2) |

---

## Key Results

### Semantic Segmentation — mIoU on Cityscapes val (Step 4–5)

| Model | mIoU (%) |
|-------|----------|
| EoMT-Cityscapes | 81.7 |
| EoMT-COCO (mapped) | 48.5 |
| EoMT-FT (COCO → Cityscapes) | 71.8 |

### Anomaly Segmentation — AuPRC (%) ↑ / FPR@95 (%) ↓

**ERFNet (pixel-based) — Step 7**

| Method | RA-21 | RO-21 | FS L&F | FS Static | Road Anomaly |
|--------|-------|-------|--------|-----------|--------------|
| MSP | 15.8 / 89.7 | 1.4 / 54.0 | 0.5 / 73.3 | 1.3 / 91.8 | 10.4 / 89.5 |
| MaxLogit | 15.6 / 88.4 | 2.2 / 56.2 | 0.4 / 71.8 | 1.3 / 86.8 | 10.2 / 87.1 |
| MaxEntropy | 15.1 / 89.8 | 1.3 / 54.1 | 0.5 / 73.4 | 1.2 / 92.0 | 10.0 / 89.7 |

**EoMT (mask-based) — Step 8 — best per-checkpoint results**

| Model | Method | RA-21 | RO-21 | FS L&F | FS Static | Road Anomaly |
|-------|--------|-------|-------|--------|-----------|--------------|
| EoMT-COCO | RbA | **37.7** / **47.7** | **26.8** / **37.0** | 0.5 / 56.7 | 2.5 / 64.9 | **22.6** / **53.3** |
| EoMT-CS | RbA | 33.8 / 61.0 | 15.5 / 46.5 | 0.9 / 55.5 | 2.3 / 69.8 | 18.4 / 61.6 |
| EoMT-FT | MaxLogit | 34.6 / 69.8 | 11.2 / 71.7 | **1.1** / 54.0 | **4.6** / **60.8** | 21.6 / 71.0 |

> RbA on EoMT-COCO achieves the strongest anomaly detection overall, with up to 2.4× the AuPRC of the best ERFNet baseline on RA-21. The fine-tuned model (EoMT-FT) offers the best trade-off on FS Static. Full results including temperature scaling are in the report.
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
