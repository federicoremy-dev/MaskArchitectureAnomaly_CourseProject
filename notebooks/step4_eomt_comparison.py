# -*- coding: utf-8 -*-
"""
Step 4 — EoMT COCO vs Cityscapes comparison
============================================
PREREQUISITES:
  1. pip install -r eomt/requirements.txt
     pip install huggingface_hub
  2. Cityscapes dataset: download leftImg8bit_trainvaltest.zip and
     gtFine_trainvaltest.zip from cityscapes-dataset.com, extract to DATA_PATH.
  3. Checkpoints: download eomt_cityscapes.bin and eomt_coco.bin from the
     course Drive and update CKPT_CITYSCAPES / CKPT_COCO below.

Run from the repo root or directly from VSCode — paths are resolved automatically.
"""

import os
import sys

# Add eomt/ to sys.path and set CWD so relative config paths resolve correctly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EOMT_DIR  = os.path.join(REPO_ROOT, "eomt")
sys.path.insert(0, EOMT_DIR)
os.chdir(EOMT_DIR)

import yaml
import importlib
import warnings
import numpy as np
import torch
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from lightning import seed_everything
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

seed_everything(0, verbose=False)

# =============================================================================
# >>> UPDATE THESE PATHS <<<
# =============================================================================
DATA_PATH       = "/path/to/cityscapes"           # folder with leftImg8bit/ and gtFine/
CKPT_CITYSCAPES = "/path/to/eomt_cityscapes.bin"
CKPT_COCO       = "/path/to/eomt_coco.bin"
DEVICE          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR      = "./step4_outputs"
N_QUALITATIVE   = 5                               # number of images for qualitative viz
# =============================================================================

_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

Path(OUTPUT_DIR).mkdir(exist_ok=True)

CONFIG_CS   = "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
CONFIG_COCO = "configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml"


# =============================================================================
# PART 1: Load the two models
# =============================================================================

def load_eomt_model(config_path, checkpoint_path=None, download_from_hf=False):
    """
    Load an EoMT model using the repo infrastructure.
    If download_from_hf=True, weights are downloaded from HuggingFace Hub;
    otherwise checkpoint_path is used.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_module_cls = getattr(importlib.import_module(data_module_name), class_name)
    data_kwargs = config["data"].get("init_args", {})

    data = data_module_cls(
        path=DATA_PATH,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **data_kwargs,
    ).setup()

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = encoder_cfg["class_path"].rsplit(".", 1)
    encoder = getattr(importlib.import_module(enc_mod), enc_cls)(
        img_size=data.img_size, **encoder_cfg.get("init_args", {})
    )

    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = getattr(importlib.import_module(net_mod), net_cls)(
        masked_attn_enabled=False,
        num_classes=data.num_classes,
        encoder=encoder,
        **net_kwargs,
    )

    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    lit_class = getattr(importlib.import_module(lit_mod), lit_cls)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in data_kwargs:
        model_kwargs["stuff_classes"] = data_kwargs["stuff_classes"]

    model = lit_class(
        img_size=data.img_size,
        num_classes=data.num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(DEVICE)

    if download_from_hf:
        name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")
        print(f"  Downloading weights from HuggingFace: tue-mps/{name} ...")
        ckpt_path = hf_hub_download(repo_id=f"tue-mps/{name}", filename="pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    elif checkpoint_path:
        print(f"  Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    return model, data


def load_eomt_model_no_data(config_path, checkpoint_path, img_size=(640, 640), num_classes=133):
    """Load EoMT COCO without requiring the COCO dataset."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = encoder_cfg["class_path"].rsplit(".", 1)
    encoder = getattr(importlib.import_module(enc_mod), enc_cls)(
        img_size=img_size, **encoder_cfg.get("init_args", {}))

    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = getattr(importlib.import_module(net_mod), net_cls)(
        masked_attn_enabled=False, num_classes=num_classes, encoder=encoder, **net_kwargs)

    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    lit_class = getattr(importlib.import_module(lit_mod), lit_cls)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    data_kwargs = config["data"].get("init_args", {})
    if "stuff_classes" in data_kwargs:
        model_kwargs["stuff_classes"] = data_kwargs["stuff_classes"]

    model = lit_class(
        img_size=img_size, num_classes=num_classes, network=network, **model_kwargs,
    ).eval().to(DEVICE)

    print(f"  Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model


print("=" * 60)
print("Loading Cityscapes model...")
print("=" * 60)
warnings.filterwarnings("ignore")
model_cs, data_cs = load_eomt_model(CONFIG_CS, checkpoint_path=CKPT_CITYSCAPES)

print("\n" + "=" * 60)
print("Loading COCO model...")
print("=" * 60)
model_coco = load_eomt_model_no_data(CONFIG_COCO, CKPT_COCO)


# =============================================================================
# PART 2: Inference functions
# =============================================================================

IGNORE_INDEX = 255


def infer_semantic(model, img):
    """Semantic inference. Returns pred_array (H, W) with class IDs."""
    with torch.no_grad(), autocast(dtype=_DTYPE, device_type=DEVICE.type):
        imgs = [img.to(DEVICE)]
        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
        crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_per_layer[-1])
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        preds = logits[0].argmax(0).cpu().numpy()
    return preds


def infer_panoptic(model, img):
    """Panoptic inference. Returns sem_pred, inst_pred."""
    with torch.no_grad(), autocast(dtype=_DTYPE, device_type=DEVICE.type):
        imgs = [img.to(DEVICE)]
        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
        mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)
        preds = model.to_per_pixel_preds_panoptic(
            mask_logits, class_logits_per_layer[-1],
            model.stuff_classes, model.mask_thresh, model.overlap_thresh,
        )[0].cpu().numpy()
    sem_pred  = preds[..., 0]   # (H, W) class IDs 0-132
    inst_pred = preds[..., 1]   # (H, W) instance IDs
    return sem_pred, inst_pred


# =============================================================================
# PART 3: Mapping COCO model IDs (0-132) → Cityscapes train IDs (0-18)
# =============================================================================

from datasets.coco_panoptic import CLASS_MAPPING

INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

COCO_CAT_TO_CITYSCAPES = {
    # THINGS
    1: 11,    # person
    2: 18,    # bicycle
    3: 13,    # car
    4: 17,    # motorcycle
    6: 15,    # bus
    7: 16,    # train
    8: 14,    # truck
    10: 6,    # traffic light
    13: 7,    # stop sign → traffic sign
    # STUFF
    149: 0,   # road
    191: 1,   # pavement → sidewalk
    197: 2,   # building
    181: 3,   # wall-other
    199: 3,   # wall-concrete
    144: 4,   # fence
    156: 10,  # sky
    187: 10,  # sky-other
    184: 8,   # tree-merged → vegetation
    166: 8,   # grass
    193: 8,   # plant-other
    107: 8,   # bush
    154: 8,   # flower
    177: 8,   # leaves
    138: 9,   # dirt → terrain
    168: 9,   # gravel
    175: 9,   # hill
    186: 9,   # mountain
    171: 9,   # ground-other
    176: 2,   # house → building
}

EOMT_COCO_TO_CS = {}
for model_id in range(133):
    coco_cat = INV_CLASS_MAPPING.get(model_id)
    if coco_cat and coco_cat in COCO_CAT_TO_CITYSCAPES:
        EOMT_COCO_TO_CS[model_id] = COCO_CAT_TO_CITYSCAPES[coco_cat]

print("\n--- COCO → Cityscapes mapping (model_id → CS train_id) ---")
CS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole",
            "traffic light", "traffic sign", "vegetation", "terrain",
            "sky", "person", "rider", "car", "truck", "bus",
            "train", "motorcycle", "bicycle"]
for mid, csid in sorted(EOMT_COCO_TO_CS.items()):
    coco_cat = INV_CLASS_MAPPING[mid]
    print(f"  model_id={mid:3d}  (COCO cat {coco_cat:3d}) → CS {csid:2d} ({CS_NAMES[csid]})")


def map_coco_pred_to_cityscapes(sem_pred_coco):
    """Remap COCO prediction (IDs 0-132) to Cityscapes space (0-18), void=255."""
    result = np.full_like(sem_pred_coco, IGNORE_INDEX, dtype=np.int64)
    for model_id, cs_id in EOMT_COCO_TO_CS.items():
        result[sem_pred_coco == model_id] = cs_id
    return result


# =============================================================================
# PART 4: Qualitative visualization
# =============================================================================

CS_PALETTE = np.array([
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
    [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
    [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
    [0,60,100],[0,80,100],[0,0,230],[119,11,32],
], dtype=np.uint8)


def colorize_pred(pred, palette):
    """Class map (H, W) → RGB image."""
    h, w = pred.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(len(palette)):
        img[pred == cid] = palette[cid]
    return img


def draw_panoptic_with_borders(sem, inst):
    """Color panoptic prediction with black borders between instances."""
    h, w = sem.shape
    all_ids = np.unique(sem)
    mapping = {}
    for i, s in enumerate(all_ids):
        if s == -1 or s >= 133:
            mapping[s] = np.array([0, 0, 0])
        else:
            mapping[s] = (np.array(plt.cm.hsv(i / max(len(all_ids), 1))[:3]) * 255).astype(np.uint8)

    out = np.zeros((h, w, 3), dtype=np.uint8)
    for s in all_ids:
        out[sem == s] = mapping[s]

    combined = sem.astype(np.int64) * 100000 + inst.astype(np.int64)
    border = np.zeros((h, w), dtype=bool)
    border[1:]     |= combined[1:]    != combined[:-1]
    border[:-1]    |= combined[1:]    != combined[:-1]
    border[:, 1:]  |= combined[:, 1:] != combined[:, :-1]
    border[:, :-1] |= combined[:, 1:] != combined[:, :-1]
    out[border] = 0
    return out


print("\n" + "=" * 60)
print("PART A: Qualitative visualization")
print("=" * 60)

val_dataset = data_cs.val_dataloader().dataset
indices = np.linspace(0, len(val_dataset) - 1, N_QUALITATIVE, dtype=int)

for i, idx in enumerate(indices):
    img, target = val_dataset[idx]
    print(f"  Image {i+1}/{N_QUALITATIVE} (idx={idx})...")

    pred_cs = infer_semantic(model_cs, img)
    sem_coco, inst_coco = infer_panoptic(model_coco, img)

    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    axes[0].imshow(img_np)
    axes[0].set_title("Original image", fontsize=14)
    axes[1].imshow(colorize_pred(pred_cs, CS_PALETTE))
    axes[1].set_title("EoMT Cityscapes\n(Semantic Segmentation)", fontsize=14)
    axes[2].imshow(draw_panoptic_with_borders(sem_coco, inst_coco))
    axes[2].set_title("EoMT COCO\n(Panoptic Segmentation)", fontsize=14)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/qualitative_{i}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# =============================================================================
# PART 5: Quantitative mIoU evaluation
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Quantitative mIoU evaluation (500 images)")
print("=" * 60)

NUM_CLASSES = 19
conf_cs   = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
conf_coco = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)


def update_confusion(conf, pred, gt, num_classes=19):
    """Update confusion matrix, ignoring pixels where gt==IGNORE_INDEX."""
    valid = gt != IGNORE_INDEX
    pred_v = pred[valid]
    gt_v   = gt[valid]
    mask_valid_pred = pred_v < num_classes
    pred_vv = pred_v[mask_valid_pred]
    gt_vv   = gt_v[mask_valid_pred]
    if len(gt_vv) > 0:
        idx = gt_vv * num_classes + pred_vv
        counts = np.bincount(idx, minlength=num_classes * num_classes)
        conf += counts.reshape(num_classes, num_classes)


for idx in tqdm(range(len(val_dataset)), desc="Evaluating"):
    img, target = val_dataset[idx]
    gt = model_cs.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].numpy()

    pred_cs = infer_semantic(model_cs, img)
    update_confusion(conf_cs, pred_cs.astype(np.int64), gt.astype(np.int64))

    sem_coco = infer_semantic(model_coco, img)
    pred_coco_mapped = map_coco_pred_to_cityscapes(sem_coco)
    update_confusion(conf_coco, pred_coco_mapped, gt.astype(np.int64))


def compute_miou(conf):
    ious = []
    for c in range(conf.shape[0]):
        tp    = conf[c, c]
        fp    = conf[:, c].sum() - tp
        fn    = conf[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
    return np.array(ious), np.nanmean(ious)


iou_cs,   miou_cs   = compute_miou(conf_cs)
iou_coco, miou_coco = compute_miou(conf_coco)

print("\n" + "=" * 70)
print("mIoU RESULTS on Cityscapes val (500 images)")
print("=" * 70)
print(f"\n{'Class':<20} {'EoMT Cityscapes':>16} {'EoMT COCO':>16}")
print("-" * 55)
for i, name in enumerate(CS_NAMES):
    ic = f"{iou_cs[i]*100:.1f}%"   if not np.isnan(iou_cs[i])   else "N/A"
    ik = f"{iou_coco[i]*100:.1f}%" if not np.isnan(iou_coco[i]) else "N/A"
    print(f"{name:<20} {ic:>16} {ik:>16}")
print("-" * 55)
print(f"{'mIoU':<20} {miou_cs*100:>15.1f}% {miou_coco*100:>15.1f}%")
print("=" * 70)

csv_path = f"{OUTPUT_DIR}/miou_results.csv"
with open(csv_path, "w") as f:
    f.write("class,eomt_cityscapes_iou,eomt_coco_iou\n")
    for i, name in enumerate(CS_NAMES):
        f.write(f"{name},{iou_cs[i]:.4f},{iou_coco[i]:.4f}\n")
    f.write(f"mIoU,{miou_cs:.4f},{miou_coco:.4f}\n")
print(f"\nResults saved to: {csv_path}")
print(f"Qualitative images in: {OUTPUT_DIR}/qualitative_*.png")


# =============================================================================
# DIAGNOSTICS: COCO prediction distribution per Cityscapes GT class
# =============================================================================

from collections import Counter

img, target = val_dataset[0]
sem_coco = infer_semantic(model_coco, img)
gt = model_cs.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].numpy()

for cs_class, cs_name in [(0,"road"), (2,"building"), (8,"vegetation"), (9,"terrain"), (10,"sky"), (4,"fence")]:
    mask = gt == cs_class
    if mask.sum() == 0:
        continue
    coco_ids = sem_coco[mask]
    top = Counter(coco_ids.flatten().tolist()).most_common(5)
    print(f"\nGT={cs_name}: total pixels={mask.sum()}")
    for mid, count in top:
        coco_cat = INV_CLASS_MAPPING.get(int(mid), "?")
        pct = count / mask.sum() * 100
        print(f"  model_id={mid} (COCO cat {coco_cat}): {count} px ({pct:.1f}%)")

for cs_class, cs_name in [(4, "fence"), (9, "terrain")]:
    mask = gt == cs_class
    if mask.sum() == 0:
        print(f"\nGT={cs_name}: no pixels in this image, try next one")
        continue
    coco_ids = sem_coco[mask]
    top = Counter(coco_ids.flatten().tolist()).most_common(5)
    print(f"\nGT={cs_name}: total pixels={mask.sum()}")
    for mid, count in top:
        coco_cat = INV_CLASS_MAPPING.get(int(mid), "?")
        pct = count / mask.sum() * 100
        print(f"  model_id={mid} (COCO cat {coco_cat}): {count} px ({pct:.1f}%)")

for test_idx in [100, 200, 300, 400]:
    img, target = val_dataset[test_idx]
    sem_coco = infer_semantic(model_coco, img)
    gt = model_cs.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].numpy()
    mask = gt == 9  # terrain
    if mask.sum() > 0:
        coco_ids = sem_coco[mask]
        top = Counter(coco_ids.flatten().tolist()).most_common(5)
        print(f"\nidx={test_idx}, GT=terrain: total pixels={mask.sum()}")
        for mid, count in top:
            coco_cat = INV_CLASS_MAPPING.get(int(mid), "?")
            pct = count / mask.sum() * 100
            print(f"  model_id={mid} (COCO cat {coco_cat}): {count} px ({pct:.1f}%)")
        break
    else:
        print(f"idx={test_idx}: no terrain pixels")
