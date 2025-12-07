import os
from collections import Counter

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# ------------------ CONFIG ------------------

# Local path on your laptop
LOCAL_MODEL_PATH = r"C:\Users\asus\OneDrive\Desktop\yolo deploy\best.pt"
# Path used on Streamlit Cloud (best.pt in same folder as app.py)
CLOUD_MODEL_PATH = "best.pt"

# Automatically pick local path if it exists, else use cloud path
MODEL_PATH = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else CLOUD_MODEL_PATH

CONFIDENCE = 0.25
IOU = 0.45

st.set_page_config(
    page_title="CircuitGuard – PCB Defect Detection",
    page_icon="🛡️",
    layout="wide"
)

# ------------------ CUSTOM STYLING ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #f8fbff;
        font-family: 'Poppins', sans-serif;
        color: #102a43;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5ff 0%, #e7fff7 100%);
        border-right: 1px solid #d0e2ff;
    }

    h1, h2, h3 {
        font-weight: 600;
        color: #13406b;
    }

    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.25rem;
        border: none;
        font-weight: 500;
        background: #85c5ff;
    }

    .stButton>button:hover {
        background: #63b1ff;
    }

    .upload-box {
        border-radius: 18px;
        border: 1px dashed #a3c9ff;
        padding: 1.5rem;
        background: #ffffff;
    }

    .metric-card {
        border-radius: 18px;
        padding: 1rem 1.25rem;
        background: #ffffff;
        border: 1px solid #dbeafe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ MODEL LOADING & INFERENCE ------------------
@st.cache_resource
def load_model(path: str):
    """Load YOLO model once and cache it."""
    return YOLO(path)


def run_inference(model, image):
    """Run detection and return plotted image + raw result."""
    results = model.predict(image, conf=CONFIDENCE, iou=IOU)
    r = results[0]
    plotted = r.plot()  # BGR numpy array
    plotted = plotted[:, :, ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(plotted)
    return pil_img, r


def get_class_counts(result, class_names):
    """Return a dict: {class_name: count} for one result."""
    if len(result.boxes) == 0:
        return {}
    cls_indices = result.boxes.cls.tolist()
    labels = [class_names[int(i)] for i in cls_indices]
    counts = Counter(labels)
    return dict(counts)


# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.subheader("Model configuration")
    st.write("**Active model path:**")
    st.code(MODEL_PATH, language="text")

    st.markdown("----")
    st.subheader("Model performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("mAP@50", "0.9823")
        st.metric("Precision", "0.9714")
    with col2:
        st.metric("mAP@50-95", "0.5598")
        st.metric("Recall", "0.9765")

# ------------------ MAIN LAYOUT ------------------
st.title("CircuitGuard – PCB Defect Detection")

# Top model performance card (same numbers as sidebar)
top_metric_col = st.columns(4)
with top_metric_col[0]:
    st.metric("mAP@50", "0.9823")
with top_metric_col[1]:
    st.metric("mAP@50-95", "0.5598")
with top_metric_col[2]:
    st.metric("Precision", "0.9714")
with top_metric_col[3]:
    st.metric("Recall", "0.9765")

st.markdown(
    """
    Detect and highlight **PCB defects** such as missing hole, mouse bite,
    open circuit, short, spur and spurious copper using a YOLO-based deep learning model.
    """
)

st.markdown("### Upload PCB Images")

with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more PCB images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    try:
        model = load_model(MODEL_PATH)
        class_names = model.names  # dict: {id: name}
    except Exception as e:
        st.error(f"Error loading model from `{MODEL_PATH}`: {e}")
    else:
        for file in uploaded_files:
            st.markdown(f"#### 📷 {file.name}")
            img = Image.open(file).convert("RGB")

            with st.spinner("Running detection..."):
                plotted_img, result = run_inference(model, img)

            # Show detection image
            st.image(plotted_img, caption="Detections", use_container_width=True)

            # Summary + per-image bar chart
            if len(result.boxes) == 0:
                st.success("No defects detected in this image.")
            else:
                st.info(f"Detected **{len(result.boxes)}** potential defect(s).")

                counts = get_class_counts(result, class_names)
                if counts:
                    df = pd.DataFrame(
                        {"Defect Type": list(counts.keys()),
                         "Count": list(counts.values())}
                    ).set_index("Defect Type")

                    st.markdown("**Defect distribution for this image:**")
                    st.bar_chart(df)

            st.markdown("---")
else:
    st.info("Upload one or more PCB images to start detection.")
