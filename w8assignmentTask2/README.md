# Australian Wildlife Detector — YOLOv11m vs RF-DETR Medium

A computer vision project that trains and compares two object detection models — **YOLOv11m** and **RF-DETR Medium** — on a dataset of 10 Australian wildlife species. Includes a Streamlit demo app for side-by-side inference.

---

## Detected Species

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

## Project Structure

```
.
├── optimizedDataset/
│   ├── converted_for_yolo/        # YOLO-format dataset (train/valid/test splits)
│   │   └── data.yaml
│   └── converted_for_rf_detr/     # RF-DETR-format dataset
├── finalWeight/
│   ├── best.pt                    # Trained YOLOv11m weights
│   └── checkpoint_best_ema.pth    # Trained RF-DETR Medium weights
├── test.ipynb                     # Training, evaluation & comparison notebook
├── w8demo.py                      # Streamlit demo app
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset contains **7,000 images** split across three sets:

| Split | Images | Labels |
|---|---|---|
| Train | 5,600 | 5,600 |
| Validation | 700 | 700 |
| Test | 700 | 700 |

Images are provided in both YOLO format (`.txt` labels) and RF-DETR format. Platypus images are excluded from the Streamlit demo grid due to rarity in the test set.

---

## Model Performance

Per-species results on the test set (confidence threshold = 0.50):

| Species | YOLO mAP@50 | YOLO F1 | RF-DETR AP@50:95 | RF-DETR F1 |
|---|---|---|---|---|
| Emu | 0.851 | 0.851 | 0.873 | 0.918 |
| Kangaroo | 0.943 | 0.951 | 0.897 | 0.951 |
| Koala | 0.884 | 0.926 | 0.767 | 0.873 |
| Wallaby | 0.825 | 0.885 | 0.791 | 0.882 |
| Fox | 0.901 | 0.937 | 0.801 | 0.939 |
| Platypus | 0.837 | 0.861 | 0.600 | 0.822 |
| Wombat | 0.936 | 0.913 | 0.908 | 0.944 |
| Quoll | 0.891 | 0.897 | 0.673 | 0.806 |
| Echidna | 0.784 | 0.867 | 0.691 | 0.864 |
| Cassowary | 0.877 | 0.891 | 0.688 | 0.854 |

**YOLO** achieves higher mAP@50 on most classes. **RF-DETR** generally shows stronger precision but lower AP at the stricter 50:95 IoU threshold.

---

## Setup

### Requirements

- Python 3.12+
- CUDA GPU (recommended), Apple Silicon MPS, or CPU

### Install

```bash
pip install -r requirements.txt
```

### Training (notebook)

Open `test.ipynb` and update the path constants at the top of the config cell:

```python
DATA_YAML   = 'path/to/converted_for_yolo/data.yaml'
DATASET_DIR = 'path/to/converted_for_rf_detr/'
OUTPUT_DIR  = 'path/to/output/'
```

Then run all cells. Training is set to 50 epochs, batch size 16, image size 640.

---

## Streamlit Demo

```bash
streamlit run w8demo.py
```

The app will:
- Auto-detect your hardware (CUDA → MPS → CPU) and show a badge in the top-right corner
- Display a 4-image grid of random test images, each run through both models side by side
- Let you upload your own images for inference
- Randomise the test image selection with the **Randomise** button

Make sure `finalWeight/best.pt` and `finalWeight/checkpoint_best_ema.pth` are present before launching.

---

## Hardware Notes

| Backend | Notes |
|---|---|
| CUDA | Fastest; recommended for training and inference |
| MPS (Apple Silicon) | Supported for inference via `w8demo.py` |
| CPU | Fallback; inference will be slow |
