# Australian Wildlife Detection & ADAS — Assessment Overview

This repository contains two assessment tasks for the Australian wildlife computer vision project.

---

## Weights

| File | Model | Location |
|---|---|---|
| `best.pt` | YOLOv11m (recommended) | `finalWeight/best.pt` |
| `checkpoint_best_ema.pth` | RF-DETR Medium | `w8/finalWeight/checkpoint_best_ema.pth` |

---

## Assessment Task 2 — Wildlife Detector Demo (`w8/`)

Side-by-side comparison of **YOLOv11m** vs **RF-DETR Medium** on 10 Australian wildlife species.

- **Demo app:** `w8demo.py` — Streamlit app, run with `streamlit run w8demo.py`
- **Training notebook:** `test.ipynb` in the `w8/` folder — shows full training, evaluation, and comparison
- **Trained on:** NVIDIA RTX 5090
- **Tested on:** Apple M4 Pro (MPS) — the demo auto-detects CUDA → MPS → CPU
- **Recommended VRAM:** 8 GB or more

```bash
streamlit run w8demo.py
```

---

## Assessment Task 3 — Wildlife-Aware ADAS Demo (`w12/`)

Pits a custom ADAS pipeline (YOLO + Depth Pro + Gemma reasoning) against a stock **Alpamayo-1.5-10B** baseline on road images. The batch analysis script was used internally to generate our results across the test set.

- **Demo app:** `app_compare.py` — Streamlit app, run with `streamlit run app_compare.py`
- **Batch script:** `batch_analysis.py` — used to generate results across the full test set
- **Trained & tested on:** NVIDIA DGX Spark (128 GB unified memory)
- **Recommended VRAM:** 80 GB or more

```bash
streamlit run app_compare.py
```

---

## Detected Species (both tasks)

| Common Name | Scientific Name |
|---|---|
| Emu | *Dromaius novaehollandiae* |
| Eastern Grey Kangaroo | *Macropus giganteus* |
| Koala | *Phascolarctos cinereus* |
| Swamp Wallaby | *Wallabia bicolor* |
| Red Fox | *Vulpes vulpes* |
| Platypus | *Ornithorhynchus anatinus* |
| Common Wombat | *Vombatus ursinus* |
| Spotted-tailed Quoll | *Dasyurus maculatus* |
| Short-beaked Echidna | *Tachyglossus aculeatus* |
| Southern Cassowary | *Casuarius casuarius* |

---

## Hardware Summary

| Task | Hardware | Recommended VRAM |
|---|---|---|
| Task 2 — `w8demo.py` | Trained: RTX 5090 · Tested: Apple M4 Pro | **8 GB+** |
| Task 3 — `app_compare.py` | NVIDIA DGX Spark (128 GB unified memory) | **80 GB+** |

> Task 3 runs Depth Pro + Alpamayo-1.5-10B + Gemma 3 12B simultaneously — the primary VRAM bottleneck. On smaller GPUs, load models sequentially or quantise to 4-bit.
