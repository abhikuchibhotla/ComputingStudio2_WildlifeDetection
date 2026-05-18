# Australian Wildlife-Aware ADAS vs Stock Alpamayo

> **Class demonstration only. Not a certified ADAS or autonomous driving system.**  
> Tested on **NVIDIA DGX Spark** · 128 GB VRAM / unified memory · Recommended minimum: **80 GB VRAM**

A Streamlit research demo that pits a custom wildlife-aware ADAS pipeline against a stock **Alpamayo-1.5-10B** multimodal baseline on identical road images. Built to show that a composed pipeline — specialist detector + monocular depth + VLM reasoning — outperforms a general-purpose driving VLM on Australian wildlife hazard detection.

These models may be difficult to run on lower-memory consumer GPUs.

If the full local stack cannot run on your hardware, the following components may be replaced with:

- Hugging Face-hosted inference endpoints

- smaller quantised local models

- alternative multimodal reasoning models

---

## Pipeline

```
Road image
    │
    ├──► YOLO (fine-tuned, Aussie animals)      → bounding boxes + species labels
    ├──► Depth Pro (monocular depth)            → per-animal distance in metres
    ├──► Risk heuristic                         → Near / Medium / Far → score
    └──► Gemma 3 12B                            → structured scene reasoning
    
    vs.
    
    Alpamayo-1.5-10B (stock VLM)               → free-form driving response
```

---

## Models

| Role | Model |
|---|---|
| Wildlife detector | `yolo_aussie_animals.pt` — custom YOLO v8 fine-tune |
| Depth estimation | `apple/DepthPro` |
| Scene reasoning | `google/gemma-3-12b-it` |
| Baseline VLM | `nvidia/Alpamayo-1.5-10B` |

---

## Supported Species

| Common Name | Scientific Name |
|---|---|
| Emu | *Dromaius novaehollandiae* |
| Eastern Grey Kangaroo | *Macropus giganteus* |
| Koala | *Phascolarctos cinereus* |
| Swamp Wallaby | *Wallabia bicolor* |
| Red Fox | *Vulpes vulpes* |
| Platypus | *Ornithorhynchus anatinus* |
| Common Wombat | *Vombatus ursinus* |
| Spotted-tail Quoll | *Dasyurus maculatus* |
| Short-beaked Echidna | *Tachyglossus aculeatus* |
| Southern Cassowary | *Casuarius casuarius* |

---

## Risk Scoring

| Depth | Range | Risk | Score |
|---|---|---|---|
| ≤ 10 m | Near | High | 0.95 |
| ≤ 25 m | Medium | Medium | 0.65 |
| > 25 m | Far | Monitor | 0.35 |
| No depth data | Near (fallback) | High | 0.90 |

---

## Requirements

```bash
pip install streamlit ultralytics torch torchvision pillow opencv-python \
            psutil timm transformers accelerate
```

---

## Usage

```bash
streamlit run app_compare.py
```

Upload a road image. The app will run YOLO → Depth Pro → risk scoring → Gemma reasoning → Alpamayo baseline, then display both outputs side by side.

---

## Hardware

Tested on **NVIDIA DGX Spark** (128 GB unified memory). Running Depth Pro + Alpamayo-1.5-10B + Gemma 3 12B simultaneously requires at least **80 GB VRAM**. On smaller GPUs, load models sequentially or quantise to 4-bit.
