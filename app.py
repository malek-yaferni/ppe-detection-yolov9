import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path
import tempfile
import sys
import os


sys.path.append(os.getcwd())

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

# ─── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide"
)

# ─── Load model (cached) ───────────────────────────────────
@st.cache_resource
def load_model():
    device = select_device('0')
    model = DetectMultiBackend(
        'runs/train/ppe_done/weights/best.pt',
        device=device
    )
    return model, device

# ─── Detection function ────────────────────────────────────
def detect_image(img_bgr, model, device, conf=0.25):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (480, 480))
    img_tensor = torch.from_numpy(img).permute(2,0,1).float().div(255).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf, 0.45)

    names = model.names
    annotator = Annotator(img.copy(), line_width=2)
    detected = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf_score, cls in det:
                label = f"{names[int(cls)]} {conf_score:.2f}"
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
                detected.append(names[int(cls)])

    return annotator.result(), detected

# ─── UI ────────────────────────────────────────────────────
st.title("🦺 PPE Detection System")
st.markdown("Detect **Helmet**, **Vest**, and **Head** in images and videos.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Mode", ["📷 Image", "🎥 Video"])
    confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    st.markdown("---")
    st.markdown("**Classes detected:**")
    st.markdown("🪖 Helmet")
    st.markdown("🦺 Vest")
    st.markdown("👤 Head")

# Load model
with st.spinner("Loading PPE model..."):
    model, device = load_model()
st.success("✅ Model loaded!")

# ─── Image mode ────────────────────────────────────────────
if mode == "📷 Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if st.button("🚀 Run Detection"):
            with st.spinner("Detecting..."):
                result_img, detected = detect_image(img, model, device, confidence)

            with col2:
                st.subheader("Result")
                st.image(result_img)

            # Alerts
            st.markdown("---")
            st.subheader("🚨 PPE Status")
            a, b, c = st.columns(3)
            with a:
                if "Helmet" in detected:
                    st.success("✅ Helmet detected")
                else:
                    st.error("❌ No Helmet!")
            with b:
                if "Vest" in detected:
                    st.success("✅ Vest detected")
                else:
                    st.error("❌ No Vest!")
            with c:
                if "Head" in detected:
                    st.warning("⚠️ Unprotected head")
                else:
                    st.info("👤 No head detected")

# ─── Video mode ────────────────────────────────────────────
elif mode == "🎥 Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded and st.button("🚀 Run Detection"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())

        cap = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stframe = st.empty()
        progress = st.progress(0)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_frame, _ = detect_image(frame, model, device, confidence)
            stframe.image(result_frame, channels="RGB", use_container_width=True)
            frame_count += 1
            progress.progress(min(frame_count / total, 1.0))

        cap.release()
        st.success("✅ Video processing complete!")
