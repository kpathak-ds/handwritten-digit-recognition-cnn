import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import easyocr
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="✍️",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #64748b;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    padding: 25px;
    border-radius: 18px;
    color: white;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.15);

    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: scale(1.03);
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">✍️ Handwritten Digit Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered digit recognition using CNN, TensorFlow, Streamlit, OpenCV, and OCR</div>', unsafe_allow_html=True)
st.info(
    "🚀 AI-powered system capable of recognizing handwritten digits from images, camera input, videos, drawing canvas, and OCR-based text recognition."
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "CNN")

with col2:
    st.metric("Dataset", "MNIST")

with col3:
    st.metric("Classes", "0-9")

with col4:
    st.metric("Accuracy", "98.7%")

@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model("models/digit_model.h5")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"])

model = load_digit_model()
ocr_reader = load_ocr_reader()

def preprocess_image(image):
    image = image.convert("L")
    img = np.array(image)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    # If image has white background and black digit, invert it
    if np.mean(img) > 127:
        _, img = cv2.threshold(
            img,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, img = cv2.threshold(
            img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]

    img = cv2.resize(img, (20, 20))

    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[4:24, 4:24] = img

    final_img = final_img / 255.0
    final_img = final_img.reshape(1, 28, 28, 1)

    return final_img
def preprocess_video_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    gray = cv2.resize(gray, (28, 28))
    gray = gray / 255.0
    gray = gray.reshape(1, 28, 28, 1)

    return gray

def predict_digit(img):
    prediction = model.predict(img, verbose=0)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)
    return digit, confidence, prediction

def show_prediction_graph(prediction):
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])
    ax.set_title("Prediction Probability")
    ax.set_xlabel("Digits")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

menu = st.sidebar.radio(
    "Choose Input Type",
    [
        "Upload Image",
        "Camera Capture",
        "Upload Video",
        "Drawing Canvas",
        "Multi-Digit OCR"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Project Features")
st.sidebar.write("✅ Image Prediction")
st.sidebar.write("✅ Camera Capture")
st.sidebar.write("✅ Video Processing")
st.sidebar.write("✅ Drawing Canvas")
st.sidebar.write("✅ Multi-Digit OCR")
st.sidebar.markdown("---")
st.sidebar.info("Built by Kanhaiya Pathak")

if menu == "Upload Image":

    st.subheader("📤 Upload Handwritten Digit Image")

    st.markdown("### ⚙️ How It Works")
    st.write("1️⃣ Upload a digit image")
    st.write("2️⃣ Image is preprocessed using OpenCV")
    st.write("3️⃣ CNN model predicts the digit")
    st.write("4️⃣ Confidence score and graph are displayed")

    uploaded_file = st.file_uploader(
        "Upload PNG/JPG/JPEG image",
        type=["png", "jpg", "jpeg"]
    )

    st.markdown("### 🛠 Technology Stack")
    st.write("Python • TensorFlow • CNN • Streamlit • OpenCV • EasyOCR • NumPy • Matplotlib")

    st.markdown("### 📊 Project Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Images", "60,000")

    with col2:
        st.metric("Test Images", "10,000")

    with col3:
        st.metric("Prediction Classes", "10")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        img = preprocess_image(image)

        with st.spinner("🔍 Analyzing digit..."):
            digit, confidence, prediction = predict_digit(img)

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h2>Predicted Digit</h2>
                        <h1>{digit}</h1>
                        <h3>Confidence: {confidence:.2f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                show_prediction_graph(prediction)

elif menu == "Camera Capture":

    st.subheader("📷 Capture Handwritten Digit Photo")

    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Camera Image", use_container_width=True)

        img = preprocess_image(image)

        with st.spinner("🔍 Analyzing digit..."):
            digit, confidence, prediction = predict_digit(img)

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h2>Predicted Digit</h2>
                        <h1>{digit}</h1>
                        <h3>Confidence: {confidence:.2f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                show_prediction_graph(prediction)

elif menu == "Upload Video":

    st.subheader("🎥 Upload Video for Digit Prediction")

    video_file = st.file_uploader(
        "Upload MP4/MOV/AVI/MKV video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if video_file is not None:
        st.video(video_file)

        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_file.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)

        frame_number = 0
        results = []

        frame_area = st.empty()
        result_area = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            if frame_number % 20 != 0:
                continue

            img = preprocess_image(frame)

            with st.spinner("🔍 Analyzing digit..."):
                digit, confidence, prediction = predict_digit(img)

            results.append({
                "Frame": frame_number,
                "Digit": digit,
                "Confidence": round(confidence, 2)
            })

            cv2.putText(
                frame,
                f"Digit: {digit} | {confidence:.2f}%",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            frame_area.image(frame, channels="BGR", use_container_width=True)
            result_area.success(
                f"Frame {frame_number}: Digit {digit} | Confidence {confidence:.2f}%"
            )

        cap.release()

        if results:
            digits = [r["Digit"] for r in results]
            confidences = [r["Confidence"] for r in results]

            final_digit = max(set(digits), key=digits.count)
            avg_confidence = np.mean(confidences)

            st.success(f"Final Video Prediction: {final_digit}")
            st.info(f"Average Confidence: {avg_confidence:.2f}%")
            st.dataframe(results)
        else:
            st.warning("No frames were processed from the video.")

elif menu == "Drawing Canvas":

    st.subheader("🖌️ Draw a Digit")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:

        img = canvas_result.image_data[:, :, 0].astype(np.uint8)

        contours, _ = cv2.findContours(
            img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            img = img[y:y+h, x:x+w]

            img = cv2.resize(img, (20, 20))

            padded_img = np.zeros((28, 28), dtype=np.uint8)
            padded_img[4:24, 4:24] = img

            img = padded_img / 255.0
            img = img.reshape(1, 28, 28, 1)

            digit, confidence, prediction = predict_digit(img)

            st.markdown(
    f"""
    <div class="metric-card">
        <h2>Predicted Digit</h2>
        <h1>{digit}</h1>
        <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """,
    unsafe_allow_html=True
)
            show_prediction_graph(prediction)

if menu == "Multi-Digit OCR":

    st.subheader("🔢 Multi-Digit OCR Recognition")

    uploaded_file = st.file_uploader(
        "Upload image containing multiple digits or text",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded OCR Image",
            use_container_width=True
        )

        img_array = np.array(image)

        results = ocr_reader.readtext(img_array)

        if results:
            st.success("OCR Detection Completed")

            for bbox, text, confidence in results:
                st.write(f"**Detected Text:** {text}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.markdown("---")
        else:
            st.warning("No text or digit detected.")

st.markdown("---")
st.markdown("Developed by **Kanhaiya Pathak** | CNN + TensorFlow + Streamlit + OCR")