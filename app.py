import io
from typing import List, Dict

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from ultralytics import YOLO


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "C:\\Users\\asus\\OneDrive\\Desktop\\yolo by ultralytics\\runs\\detect\\train14\\weights\\best.pt"  # <-- change if needed
CLASS_NAMES = [
    # Put your classes in correct order as in your model
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "spur",
    "spurious_copper",
]

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model once and cache it for all users."""
    model = YOLO(model_path)
    return model


def run_inference(model, image: Image.Image, conf: float = 0.25, iou: float = 0.45):
    """
    Run YOLO inference on a PIL image and return:
    - annotated image (np.ndarray)
    - detections DataFrame
    - defect summary dict
    """
    # Convert PIL to numpy
    img_array = np.array(image)

    results = model.predict(
        source=img_array,
        conf=conf,
        iou=iou,
        verbose=False
    )

    if len(results) == 0:
        return img_array, pd.DataFrame(), {}

    r = results[0]

    # Annotated image
    annotated = r.plot()  # BGR np.ndarray

    # Boxes & details
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, pd.DataFrame(), {}

    data = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])

        # xyxy format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        data.append(
            {
                "class_id": cls_id,
                "defect_type": class_name,
                "confidence": round(conf_score, 3),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )

    df = pd.DataFrame(data)

    # Summary: count per class
    summary = (
        df.groupby("defect_type")["defect_type"]
        .count()
        .rename("count")
        .to_dict()
    )

    return annotated, df, summary


def rgb_from_bgr(img: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB for Streamlit."""
    return img[:, :, ::-1]


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="CircuitGuard - PCB Defect Detector",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ CircuitGuard – PCB Defect Detection")
    st.write(
        "Upload PCB images and CircuitGuard will detect and highlight defects such as "
        "**missing components, solder bridges, misalignment, scratches**, and more."
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")

    st.sidebar.subheader("Model Configuration")
    st.sidebar.write(f"**Model file:** `{MODEL_PATH}`")

    conf_thres = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Lower = more detections (including low confidence), higher = only very confident detections.",
    )

    iou_thres = st.sidebar.slider(
        "IoU threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for non-max suppression.",
    )

    st.sidebar.subheader("Model Performance")
    #st.sidebar.write("These values come from your test results. Update them in code or make them dynamic:")
    col_acc1, col_acc2 = st.sidebar.columns(2)
    with col_acc1:
        st.metric("mAP@50", "0.9823")  # <-- replace with your real values
        st.metric("Precision", "0.9714")
    with col_acc2:
        st.metric("mAP@50-95", "0.5598")
        st.metric("Recall", "0.9765")

    st.sidebar.markdown("---")
    #st.sidebar.caption("Tip: commit only the final `best.pt` model and not all training runs.")

    # File uploader
    st.subheader("📤 Upload PCB Images")
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload PCB images to see defect detection results.")
        return

    # Load the model once
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model from `{MODEL_PATH}`: {e}")
        st.stop()

    # Process each uploaded image
    all_summaries = []

    for file in uploaded_files:
        st.markdown(f"### 🔍 File: `{file.name}`")
        col1, col2 = st.columns([2, 1])

        # Read image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        with col1:
            st.caption("Original Image")
            st.image(image, use_column_width=True)

        # Run inference
        annotated, df_dets, summary = run_inference(
            model=model,
            image=image,
            conf=conf_thres,
            iou=iou_thres,
        )

        with col1:
            st.caption("Detected Defects")
            st.image(rgb_from_bgr(annotated), use_column_width=True)

        with col2:
            st.caption("Defect Summary")
            if summary:
                sum_df = pd.DataFrame(
                    [
                        {"Defect type": k, "Count": v}
                        for k, v in summary.items()
                    ]
                ).sort_values("Count", ascending=False)

                st.table(sum_df)

                st.bar_chart(
                    sum_df.set_index("Defect type")["Count"],
                    use_container_width=True,
                )

                total_defects = sum(summary.values())
                all_summaries.append(
                    {
                        "file": file.name,
                        "total_defects": total_defects,
                        **{f"{k}_count": v for k, v in summary.items()},
                    }
                )

            else:
                st.info("No defects detected above the selected confidence threshold.")

        # Detailed detection table
        with st.expander(f"📋 Detailed detections for `{file.name}`"):
            if df_dets.empty:
                st.write("No detections.")
            else:
                st.dataframe(df_dets, use_container_width=True)

        st.markdown("---")

    # Combined summary across images
    if all_summaries:
        st.subheader("📊 Overall Summary (All Uploaded Images)")
        overall_df = pd.DataFrame(all_summaries)
        st.dataframe(overall_df, use_container_width=True)


if __name__ == "__main__":
    main()
