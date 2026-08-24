import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time
import os
import pandas as pd

st.set_page_config(
    page_title="Intelligent Road Pothole Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Intelligent Road Pothole Detection System")
st.write("Automated road inspection using YOLOv8 Computer Vision")
st.divider()

model_path = "best.pt"

if not os.path.exists(model_path):
    st.warning("⚠️ Trained model is not available yet.")
    st.info(
        "Please make sure best.pt exists inside "
        "runs/detect/pothole_detection/weights/"
    )
    st.stop()

model = YOLO(model_path)

st.sidebar.header("⚙️ Inspection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.20,
    step=0.05
)

st.sidebar.write("Model: YOLOv8n")
st.sidebar.write("Maximum images: 100")

st.subheader("📁 Upload Road Images")

uploaded_files = st.file_uploader(
    "Select multiple road images for inspection (Maximum 100)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    if len(uploaded_files) > 100:
        st.error("❌ You can upload a maximum of 100 images.")
        st.stop()

    st.success(
        f"✅ {len(uploaded_files)} road image(s) uploaded successfully."
    )

    total_potholes = 0
    images_with_potholes = 0
    images_without_potholes = 0

    all_confidences = []
    results_table = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):

        status_text.write(
            f"🔍 Processing image {index + 1} of {len(uploaded_files)}: "
            f"{uploaded_file.name}"
        )

        image = Image.open(uploaded_file).convert("RGB")

        start_time = time.time()

        results = model.predict(
            image,
            conf=confidence,
            verbose=False
        )

        processing_time = time.time() - start_time

        boxes = results[0].boxes

        pothole_count = len(boxes)

        if pothole_count > 0:

            confidence_values = boxes.conf.cpu().numpy()

            highest_confidence = max(confidence_values) * 100

            average_confidence = (
                sum(confidence_values) /
                len(confidence_values)
            ) * 100

            all_confidences.extend(confidence_values)

            total_potholes += pothole_count
            images_with_potholes += 1

        else:

            highest_confidence = 0
            average_confidence = 0

            images_without_potholes += 1

        detected_image = results[0].plot()

        st.divider()

        st.subheader(
            f"🔍 Inspection {index + 1}: {uploaded_file.name}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write("### Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.write("### Detection Result")

            st.image(
                detected_image,
                channels="BGR",
                use_container_width=True
            )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Potholes",
                pothole_count
            )

        with result_col2:

            if pothole_count > 0:

                st.metric(
                    "Confidence",
                    f"{highest_confidence:.2f}%"
                )

            else:

                st.metric(
                    "Confidence",
                    "—"
                )

        with result_col3:

            st.metric(
                "Processing Time",
                f"{processing_time:.2f} sec"
            )

        if pothole_count > 0:

            st.warning(
                f"⚠️ {pothole_count} pothole(s) detected"
            )

        else:

            st.success(
                "✅ No potholes detected"
            )

        results_table.append({
            "Image": uploaded_file.name,
            "Potholes Detected": pothole_count,
            "Confidence": (
                f"{highest_confidence:.2f}%"
                if pothole_count > 0
                else "—"
            ),
            "Processing Time": f"{processing_time:.2f} sec",
            "Status": (
                "Pothole Detected"
                if pothole_count > 0
                else "No Pothole"
            )
        })

        progress_bar.progress(
            (index + 1) / len(uploaded_files)
        )

    status_text.success(
        "✅ All images have been processed."
    )

    st.divider()

    st.header("📊 Overall Road Inspection Summary")

    total_images = len(uploaded_files)

    if all_confidences:

        overall_confidence = (
            sum(all_confidences) /
            len(all_confidences)
        ) * 100

    else:

        overall_confidence = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Images Inspected",
            total_images
        )

    with col2:

        st.metric(
            "Images With Potholes",
            images_with_potholes
        )

    with col3:

        st.metric(
            "Total Potholes",
            total_potholes
        )

    with col4:

        st.metric(
            "Average Confidence",
            f"{overall_confidence:.2f}%"
        )

    st.subheader("📋 Inspection Report")

    dataframe = pd.DataFrame(results_table)

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True
    )