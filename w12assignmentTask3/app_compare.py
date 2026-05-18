import base64, io, sys, time, requests, psutil, cv2
import numpy as np
import streamlit as st
import torch
import timm
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torchvision.transforms.functional as TF

sys.path.append("/workspace/alpamayo-test/depth-pro/src")
import depth_pro

YOLO_PATH = "/workspace/alpamayo-test/models/yolo_aussie_animals.pt"
ALPAMAYO_PATH = "/models/Alpamayo-1.5-10B"
OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"

SPECIES_MAP = {
    "Dromaius novaehollandiae": "Emu",
    "Macropus giganteus": "Eastern Grey Kangaroo",
    "Phascolarctos cinereus": "Koala",
    "Wallabia bicolor": "Swamp Wallaby",
    "Vulpes vulpes": "Red Fox",
    "Ornithorhynchus anatinus": "Platypus",
    "Vombatus ursinus": "Common Wombat",
    "Dasyurus maculatus": "Spotted-tail Quoll",
    "Tachyglossus aculeatus": "Short-beaked Echidna",
    "Casuarius casuarius": "Southern Cassowary",
}

st.set_page_config(page_title="Wildlife ADAS vs Alpamayo", layout="wide")
st.title("Australian Wildlife-Aware ADAS vs Stock Alpamayo")
st.caption("University research demo only. Not a certified ADAS or autonomous driving system.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Torch device", "cuda" if torch.cuda.is_available() else "cpu")
m2.metric("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
m3.metric("CUDA allocated", f"{torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "0 GB")
m4.metric("RAM used", f"{psutil.virtual_memory().percent}%")

prompt = st.text_area(
    "Prompt",
    "Analyze the driving scene. Identify wildlife hazards, estimate risk, and recommend whether the vehicle should continue, slow down, or brake.",
    height=110,
)

uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_PATH)

@st.cache_resource
def load_depth():
    model, transform = depth_pro.create_model_and_transforms()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), transform

@st.cache_resource
def load_alpamayo():
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    model = Alpamayo1_5.from_pretrained(
        ALPAMAYO_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    return model, processor, helper

def draw_boxes(image, results, names):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        scientific = names[cls_id]
        common = SPECIES_MAP.get(scientific, scientific)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({"common": common, "confidence": conf, "bbox": [x1, y1, x2, y2]})
        label = f"{common} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=7)
        tb = draw.textbbox((x1, y1), label)
        draw.rectangle([tb[0], tb[1]-6, tb[2]+10, tb[3]+6], fill="red")
        draw.text((x1+5, y1), label, fill="white")
    return img, detections

def depth_heatmap(image, detections):
    model, transform = load_depth()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_np = np.array(image.convert("RGB"))

    try:
        inp = transform(img_np).unsqueeze(0).to(device)
    except Exception:
        inp = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model.infer(inp)
        depth = pred["depth"].detach().float().cpu().numpy().squeeze()

    lo, hi = np.percentile(depth, [2, 98])
    depth_clip = np.clip(depth, lo, hi)
    depth_norm = (depth_clip - lo) / (hi - lo + 1e-8)
    depth_norm = 1.0 - depth_norm
    heat = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat_img = Image.fromarray(heat).resize(image.size)

    draw = ImageDraw.Draw(heat_img)
    ow, oh = image.size
    dh, dw = depth.shape

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=7)
        draw.text((x1 + 5, max(0, y1 - 24)), d["common"], fill="white")

        sx1, sx2 = int(x1 / ow * dw), int(x2 / ow * dw)
        sy1, sy2 = int(y1 / oh * dh), int(y2 / oh * dh)
        crop = depth[sy1:sy2, sx1:sx2]
        if crop.size:
            d["depth_m"] = float(np.median(crop))

    return heat_img

def risk(image, detections):
    if not detections:
        return "Not in frame", "Low", 0.00, "No wildlife detected."

    with_depth = [d for d in detections if "depth_m" in d]
    if with_depth:
        best = min(with_depth, key=lambda d: d["depth_m"])
        dm = best["depth_m"]
        if dm <= 10:
            return "Near", "High", 0.95, f"{best['common']} appears close at ~{dm:.1f}m. Brake now / prepare to stop."
        if dm <= 25:
            return "Medium", "Medium", 0.65, f"{best['common']} is at ~{dm:.1f}m. Slow down and prepare to brake."
        return "Far", "Monitor", 0.35, f"{best['common']} is visible at ~{dm:.1f}m. Monitor carefully."

    return "Near", "High", 0.90, f"{detections[0]['common']} detected. Brake now / prepare to stop."

def call_gemma(image, detections, range_label, risk_level, risk_score, action):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    det_text = "No YOLO wildlife detections."
    if detections:
        det_text = "\n".join(
            [f"- {d['common']}, confidence {d['confidence']:.2f}, bbox {d['bbox']}, estimated depth {d.get('depth_m', 'unknown')}" for d in detections]
        )

    gemma_prompt = f"""
You are an ADAS driving-safety reasoning assistant.

Use the image for scene context, YOLO for animal identity, and Depth Pro for estimated distance/risk.
Use common animal names only. Do not mention scientific names. Do not invent animals.
Write with both numbers and a natural Alpamayo-style explanation.

User prompt:
{prompt}

YOLO detections:
{det_text}

Depth/risk:
- Range: {range_label}
- Risk level: {risk_level}
- Risk score: {risk_score:.2f} out of 1.00
- Recommended action: {action}

Write in two parts:

1. Key measurements:
- Detected animal:
- YOLO confidence:
- Estimated range/distance:
- Risk score:
- Recommended action:

2. Driving scene reasoning:
Write one detailed natural paragraph describing the road/environment, the animal, the estimated distance, why it is unsafe or safe, and what the driver should do.
"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": gemma_prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.25, "num_predict": 420},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=240)
    r.raise_for_status()
    return r.json().get("response", "")

def run_alpamayo(image, question):
    model, processor, helper = load_alpamayo()
    frames = TF.to_tensor(image).unsqueeze(0)
    camera_indices = torch.tensor([0])
    messages = helper.create_vqa_message(frames, question=question, camera_indices=camera_indices)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device({"tokenized_data": inputs}, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        extra = model.generate_text(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_samples=1,
            max_generation_length=256,
        )
    return extra["answer"][0]

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    yolo = load_yolo()

    with st.spinner("Running YOLO..."):
        results = yolo(image, conf=0.20)
        boxed, detections = draw_boxes(image, results, yolo.names)

    with st.spinner("Running Depth Pro..."):
        heatmap = depth_heatmap(image, detections)

    range_label, risk_level, risk_score, action = risk(image, detections)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Our Solution — YOLO Wildlife Detection")
        st.image(boxed, width="stretch")
    with c2:
        st.subheader("Stock Alpamayo — Same Input")
        st.image(image, width="stretch")

    st.subheader("Depth Pro Map + YOLO Overlay")
    st.image(heatmap, width="stretch")

    a, b, c = st.columns(3)
    a.metric("Estimated range", range_label)
    b.metric("Risk level", risk_level)
    c.metric("Risk score", f"{risk_score:.2f}")
    st.write(action)

    st.subheader("Reasoning Outputs")
    progress = st.progress(0, text="Starting reasoning pipeline...")

    progress.progress(10, text="Running Gemma...")
    try:
        our_response = call_gemma(image, detections, range_label, risk_level, risk_score, action)
    except Exception as e:
        our_response = f"Gemma failed: {e}"
    progress.progress(50, text="Gemma complete. Running Alpamayo...")

    try:
        alpamayo_response = run_alpamayo(image, prompt)
    except Exception as e:
        alpamayo_response = f"Alpamayo failed: {e}"
    progress.progress(100, text="Complete.")

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("Our Gemma + YOLO + Depth Response")
        st.text_area("Our response", our_response, height=620)
    with r2:
        st.subheader("Stock Alpamayo Response")
        st.text_area("Alpamayo response", alpamayo_response, height=620)
else:
    st.info("Upload an image to begin.")
