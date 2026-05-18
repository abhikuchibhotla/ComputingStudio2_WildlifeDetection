from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_mpl_dir = PROJECT_ROOT / ".streamlit-cache" / "matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRMedium
from ultralytics import YOLO


warnings.filterwarnings("ignore", message="Error fetching version info.*")

st.set_page_config(layout="wide")

CONF_THRESH = 0.5
TEST_IMAGES = PROJECT_ROOT / "optimizedDataset" / "converted_for_yolo" / "test" / "images"
YOLO_WEIGHTS = PROJECT_ROOT / "finalWeight" / "best.pt"
RFDETR_WEIGHTS = PROJECT_ROOT / "finalWeight" / "checkpoint_best_ema.pth"
CLASS_NAMES = [
    "Dromaius novaehollandiae",
    "Macropus giganteus",
    "Phascolarctos cinereus",
    "Wallabia bicolor",
    "Vulpes vulpes",
    "Ornithorhynchus anatinus",
    "Vombatus ursinus",
    "Dasyurus maculatus",
    "Tachyglossus aculeatus",
    "Casuarius casuarius",
]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MPS_DETECTED = torch.backends.mps.is_built()
RFDETR_BOX_COLOR = (255, 30, 30)


@st.cache_resource
def load_yolo_model() -> YOLO:
    return YOLO(str(YOLO_WEIGHTS))


@st.cache_resource
def load_rfdetr_model() -> RFDETRMedium:
    return RFDETRMedium(
        pretrain_weights=str(RFDETR_WEIGHTS),
        device=DEVICE,
        num_classes=len(CLASS_NAMES),
    )


def render_device_badge() -> None:
    badge_text = "MPS detected" if MPS_DETECTED else "CPU detected"
    badge_color = "#1f7a4d" if MPS_DETECTED else "#8a6d1f"
    st.markdown(
        f"""
        <style>
        .device-badge {{
            position: fixed;
            top: 0.75rem;
            right: 1rem;
            z-index: 9999;
            background: {badge_color};
            color: white;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 700;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        }}
        </style>
        <div class="device-badge">{badge_text}</div>
        """,
        unsafe_allow_html=True,
    )


def class_color(class_id: int) -> tuple[int, int, int]:
    return (
        (67 + class_id * 37) % 255,
        (129 + class_id * 59) % 255,
        (201 + class_id * 83) % 255,
    )


def get_label_font(image: Image.Image) -> ImageFont.ImageFont:
    font_size = max(22, image.width // 28)
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def get_test_images() -> list[Path]:
    image_paths: list[Path] = []
    for path in sorted(TEST_IMAGES.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "Ornithorhynchus anatinus" in path.name:
            continue
        image_paths.append(path)
    return image_paths


def sample_images(count: int) -> list[Path]:
    candidates = get_test_images()
    if len(candidates) < count:
        raise RuntimeError(f"Need at least {count} filtered images in {TEST_IMAGES}.")
    seed = st.session_state.get("random_seed", 12345)
    rng = random.Random(seed)
    return rng.sample(candidates, count)


def draw_rfdetr_boxes(image: Image.Image, class_ids: np.ndarray, confidences: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = get_label_font(canvas)
    line_width = max(5, image.width // 140)

    for class_id, confidence, box in zip(class_ids, confidences, boxes):
        cid = int(class_id)
        x1, y1, x2, y2 = [float(value) for value in box]
        label = f"{CLASS_NAMES[cid]} {float(confidence):.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=RFDETR_BOX_COLOR, width=line_width)

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        pad_x = max(8, image.width // 160)
        pad_y = max(6, image.height // 180)
        text_top = max(0.0, y1 - text_height - (pad_y * 2) - 4)
        draw.rectangle(
            (x1, text_top, x1 + text_width + (pad_x * 2), text_top + text_height + (pad_y * 2)),
            fill=RFDETR_BOX_COLOR,
        )
        draw.text((x1 + pad_x, text_top + pad_y), label, fill="white", font=font)

    return np.array(canvas)


def summarize_detections(class_ids: np.ndarray, confidences: np.ndarray) -> str:
    if len(class_ids) == 0:
        return f"No detections >= {CONF_THRESH:.2f}"

    parts = []
    for class_id, confidence in zip(class_ids[:3], confidences[:3]):
        cid = int(class_id)
        parts.append(f"{CLASS_NAMES[cid]} {float(confidence):.2f}")
    summary = " | ".join(parts)
    if len(class_ids) > 3:
        summary += f" | +{len(class_ids) - 3} more"
    return summary


def run_yolo_on_image(model: YOLO, image: Image.Image | str) -> tuple[np.ndarray, str]:
    result = model.predict(image, conf=CONF_THRESH, device=DEVICE, verbose=False)[0]
    annotated = result.plot()
    if result.boxes is None or len(result.boxes) == 0:
        summary = f"No detections >= {CONF_THRESH:.2f}"
    else:
        summary = summarize_detections(
            class_ids=result.boxes.cls.detach().cpu().numpy(),
            confidences=result.boxes.conf.detach().cpu().numpy(),
        )
    return annotated[:, :, ::-1], summary


def run_rfdetr_on_image(model: RFDETRMedium, image: Image.Image) -> tuple[np.ndarray, str]:
    detections = model.predict(image, threshold=CONF_THRESH)
    if len(detections) == 0:
        return np.array(image.convert("RGB")), f"No detections >= {CONF_THRESH:.2f}"

    annotated = draw_rfdetr_boxes(
        image=image,
        class_ids=detections.class_id,
        confidences=detections.confidence,
        boxes=detections.xyxy,
    )
    summary = summarize_detections(detections.class_id, detections.confidence)
    return annotated, summary


def render_test_grid(yolo_model: YOLO, rfdetr_model: RFDETRMedium, image_paths: list[Path]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))

    for row_index, image_path in enumerate(image_paths):
        pil_image = Image.open(image_path).convert("RGB")
        yolo_image, yolo_summary = run_yolo_on_image(yolo_model, str(image_path))
        rfdetr_image, rfdetr_summary = run_rfdetr_on_image(rfdetr_model, pil_image)

        axes[row_index, 0].imshow(yolo_image)
        axes[row_index, 0].set_title(f"YOLO\n{image_path.name}\n{yolo_summary}", fontsize=11)
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(rfdetr_image)
        axes[row_index, 1].set_title(f"RF-DETR\n{image_path.name}\n{rfdetr_summary}", fontsize=11)
        axes[row_index, 1].axis("off")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig, clear_figure=True)


def render_upload_results(yolo_model: YOLO, rfdetr_model: RFDETRMedium, uploaded_files: list) -> None:
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        pil_image = Image.open(uploaded_file).convert("RGB")
        yolo_image, yolo_summary = run_yolo_on_image(yolo_model, pil_image)
        rfdetr_image, rfdetr_summary = run_rfdetr_on_image(rfdetr_model, pil_image)

        st.markdown(f"### Uploaded image {index}: `{uploaded_file.name}`")
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown(f"**YOLO**  \nConfidence threshold: `{CONF_THRESH:.2f}`")
            st.image(yolo_image, use_container_width=True)
            st.caption(yolo_summary)

        with right_col:
            st.markdown(f"**RF-DETR Medium**  \nConfidence threshold: `{CONF_THRESH:.2f}`")
            st.image(rfdetr_image, use_container_width=True)
            st.caption(rfdetr_summary)


def render_upload_comparison(yolo_model: YOLO, rfdetr_model: RFDETRMedium) -> None:
    st.title("Test your own image")
    st.caption("Upload one or more images and both models will compare them side by side.")

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    uploaded_files = st.file_uploader(
        "Upload JPG or PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"upload_batch_{st.session_state['uploader_key']}",
    )

    if uploaded_files:
        render_upload_results(yolo_model, rfdetr_model, uploaded_files)

    button_col1, button_col2 = st.columns([1, 1])
    with button_col1:
        if st.button("Upload more images"):
            st.session_state["uploader_key"] += 1
            st.rerun()
    with button_col2:
        if uploaded_files:
            st.caption("Use the button above to reset the uploader and add another batch.")


def main() -> None:
    render_device_badge()
    if MPS_DETECTED:
        st.caption("Device detected: `mps`")
    else:
        st.caption("Device detected: `cpu`")
    st.caption(f"Active runtime device: `{DEVICE}`")
    st.caption(f"Test image folder: `{TEST_IMAGES}`")

    if "random_seed" not in st.session_state:
        st.session_state["random_seed"] = random.randint(0, 10_000_000)

    yolo_model = load_yolo_model()
    rfdetr_model = load_rfdetr_model()

    st.title("Test set detections — YOLOv11m vs RF-DETR Medium")
    st.caption("Each row shows the same image twice so you can compare YOLO on the left and RF-DETR on the right.")
    if st.button("Randomise"):
        st.session_state["random_seed"] = random.randint(0, 10_000_000)

    comparison_images = sample_images(4)
    render_test_grid(yolo_model, rfdetr_model, comparison_images)
    render_upload_comparison(yolo_model, rfdetr_model)


if __name__ == "__main__":
    main()
