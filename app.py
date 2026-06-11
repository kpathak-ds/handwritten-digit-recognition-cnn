import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import easyocr
from streamlit_drawable_canvas import st_canvas

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DigitVision AI",
    page_icon="✍️",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("DigitVision AI")
    st.caption("Handwritten Digit Recognition System")

    theme = st.selectbox(
        "🎨 Choose Theme",
        ["Royal Blue", "Dark Purple", "Emerald Green"]
    )

    menu = st.radio(
        "📌 Select Input Method",
        [
            "🏠 Home",
            "📤 Upload Image",
            "📷 Camera Capture",
            "🎥 Upload Video",
            "🖌️ Drawing Canvas",
            "🔢 Multi-Digit OCR",
            "📚 Resources"
        ]
    )

    st.markdown("---")
    auto_predict = st.toggle("⚡ Auto Prediction", value=True)
    show_processed = st.toggle("🔍 Show Processed Image", value=True)

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.success("History cleared")

# ---------------- THEME COLORS ----------------
themes = {
    "Royal Blue": {
        "primary": "#2563eb",
        "secondary": "#1e40af",
        "bg": "#f8fafc",
        "card": "#ffffff",
        "text": "#0f172a"
    },
    "Dark Purple": {
        "primary": "#7c3aed",
        "secondary": "#4c1d95",
        "bg": "#f5f3ff",
        "card": "#ffffff",
        "text": "#1e1b4b"
    },
    "Emerald Green": {
        "primary": "#059669",
        "secondary": "#065f46",
        "bg": "#ecfdf5",
        "card": "#ffffff",
        "text": "#064e3b"
    }
}

c = themes[theme]

# ---------------- CUSTOM CSS ----------------
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(135deg, {c["bg"]}, #ffffff);
}}

.hero {{
    padding: 35px;
    border-radius: 24px;
    background: linear-gradient(135deg, {c["primary"]}, {c["secondary"]});
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 12px 30px rgba(0,0,0,0.18);
}}

.hero h1 {{
    font-size: 46px;
    margin-bottom: 5px;
}}

.hero p {{
    font-size: 18px;
    opacity: 0.95;
}}

.card {{
    background-color: {c["card"]};
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.08);
    margin-bottom: 18px;
    border: 1px solid rgba(0,0,0,0.05);
}}

.feature-card {{
    background-color: white;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid {c["primary"]};
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    height: 170px;
}}

.big-number {{
    font-size: 42px;
    font-weight: 800;
    color: {c["primary"]};
}}

.stButton>button {{
    background-color: {c["primary"]};
    color: white;
    border-radius: 12px;
    height: 45px;
    font-weight: 700;
    border: none;
}}

.stButton>button:hover {{
    background-color: {c["secondary"]};
    color: white;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {c["secondary"]}, {c["primary"]});
}}

[data-testid="stSidebar"] * {{
    color: white;
}}

.footer {{
    text-align: center;
    padding: 25px;
    color: #64748b;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model("models/digit_model.h5")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"])

model = load_digit_model()
ocr_reader = load_ocr_reader()

# ---------------- HELPER FUNCTIONS ----------------
def center_digit(img):
    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]

    h, w = img.shape

    if h == 0 or w == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    if h > w:
        new_h = 20
        new_w = max(1, int(w * (20 / h)))
    else:
        new_w = 20
        new_h = max(1, int(h * (20 / w)))

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded_img = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    padded_img[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = img

    return padded_img

def preprocess_image(image):
    image = image.convert("L")
    img = np.array(image)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    img = center_digit(img)

    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

def preprocess_video_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    gray = center_digit(gray)

    gray = gray.astype("float32") / 255.0
    gray = gray.reshape(1, 28, 28, 1)

    return gray

def preprocess_canvas(canvas_img):
    img = canvas_img[:, :, 0].astype(np.uint8)
    img = center_digit(img)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def predict_digit(img):
    prediction = model.predict(img, verbose=0)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)
    return digit, confidence, prediction

def save_history(source, digit, confidence):
    st.session_state.history.append({
        "Source": source,
        "Prediction": digit,
        "Confidence": round(confidence, 2)
    })

def show_prediction_graph(prediction):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(10), prediction[0])
    ax.set_title("Digit Probability Distribution")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(10))
    st.pyplot(fig)

def prediction_result_ui(digit, confidence):
    if confidence >= 80:
        st.success("High confidence prediction")
    elif confidence >= 50:
        st.warning("Medium confidence prediction")
    else:
        st.error("Low confidence prediction. Try uploading a clearer image.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Digit", digit)

    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>✍️ DigitVision AI</h1>
    <p>Professional Handwritten Digit Recognition System using CNN, TensorFlow, Streamlit and OCR.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- HOME ----------------
if menu == "🏠 Home":

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>📤 Image Prediction</h3>
            <p>Upload handwritten digit images and get instant AI prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>🖌️ Drawing Canvas</h3>
            <p>Draw digits directly inside the app and predict in real time.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔢 OCR Support</h3>
            <p>Detect multiple digits or text using EasyOCR integration.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model", "CNN")
    col2.metric("Input Size", "28×28")
    col3.metric("Classes", "0-9")
    col4.metric("Framework", "TensorFlow")

    st.info(
        "This project is useful for handwritten digit recognition, OCR learning, "
        "computer vision basics, and recruiter-friendly AI portfolio demonstration."
    )

# ---------------- UPLOAD IMAGE ----------------
elif menu == "📤 Upload Image":

    st.subheader("📤 Upload Handwritten Digit Image")

    uploaded_file = st.file_uploader(
        "Upload a clear handwritten digit image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Original Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if auto_predict:
            with st.spinner("AI is preprocessing and predicting..."):
                img = preprocess_image(image)
                digit, confidence, prediction = predict_digit(img)

            save_history("Upload Image", digit, confidence)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                prediction_result_ui(digit, confidence)

                if show_processed:
                    st.image(
                        img.reshape(28, 28),
                        caption="Processed Image Seen by Model",
                        width=160
                    )

                st.markdown('</div>', unsafe_allow_html=True)

            show_prediction_graph(prediction)

# ---------------- CAMERA ----------------
elif menu == "📷 Camera Capture":

    st.subheader("📷 Capture Handwritten Digit")

    camera_image = st.camera_input("Take a clear photo of one handwritten digit")

    if camera_image is not None:
        image = Image.open(camera_image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)

        if auto_predict:
            with st.spinner("Processing camera image..."):
                img = preprocess_image(image)
                digit, confidence, prediction = predict_digit(img)

            save_history("Camera Capture", digit, confidence)

            with col2:
                prediction_result_ui(digit, confidence)

                if show_processed:
                    st.image(
                        img.reshape(28, 28),
                        caption="Processed Camera Image",
                        width=160
                    )

            show_prediction_graph(prediction)

# ---------------- VIDEO ----------------
elif menu == "🎥 Upload Video":

    st.subheader("🎥 Upload Video for Digit Prediction")

    video_file = st.file_uploader(
        "Upload video containing handwritten digit",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if video_file is not None:
        st.video(video_file)

        if auto_predict:
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(video_file.read())
            temp_video.close()

            cap = cv2.VideoCapture(temp_video.name)

            frame_number = 0
            results = []

            frame_area = st.empty()
            result_area = st.empty()

            with st.spinner("Analyzing video frames..."):
                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1

                    if frame_number % 20 != 0:
                        continue

                    img = preprocess_video_frame(frame)
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
                    result_area.info(
                        f"Frame {frame_number}: Digit {digit} | Confidence {confidence:.2f}%"
                    )

            cap.release()

            if results:
                digits = [r["Digit"] for r in results]
                confidences = [r["Confidence"] for r in results]

                final_digit = max(set(digits), key=digits.count)
                avg_confidence = np.mean(confidences)

                save_history("Upload Video", final_digit, avg_confidence)

                st.success(f"Final Video Prediction: {final_digit}")
                st.info(f"Average Confidence: {avg_confidence:.2f}%")
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("No valid frames processed.")

# ---------------- DRAWING CANVAS ----------------
elif menu == "🖌️ Drawing Canvas":

    st.subheader("🖌️ Draw a Digit")

    st.info("Draw one digit using white color on black background.")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None and auto_predict:
        img = preprocess_canvas(canvas_result.image_data)

        digit, confidence, prediction = predict_digit(img)
        save_history("Drawing Canvas", digit, confidence)

        prediction_result_ui(digit, confidence)

        if show_processed:
            st.image(
                img.reshape(28, 28),
                caption="Processed Drawing Seen by Model",
                width=160
            )

        show_prediction_graph(prediction)

# ---------------- OCR ----------------
elif menu == "🔢 Multi-Digit OCR":

    st.subheader("🔢 Multi-Digit OCR Recognition")

    uploaded_file = st.file_uploader(
        "Upload image containing multiple digits or text",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="OCR Input Image", use_container_width=True)

        if auto_predict:
            with st.spinner("EasyOCR is detecting text..."):
                img_array = np.array(image)
                results = ocr_reader.readtext(img_array)

            if results:
                st.success("OCR Detection Completed")

                for bbox, text, confidence in results:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write(f"**Detected Text:** {text}")
                    st.write(f"**Confidence:** {confidence * 100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.session_state.history.append({
                        "Source": "Multi-Digit OCR",
                        "Prediction": text,
                        "Confidence": round(confidence * 100, 2)
                    })
            else:
                st.warning("No text or digit detected.")

# ---------------- RESOURCES ----------------
elif menu == "📚 Resources":

    st.subheader("📚 Project Resources")

    st.markdown("""
    <div class="card">
        <h3>🧠 Model Used</h3>
        <p>This app uses a Convolutional Neural Network trained on handwritten digit data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>⚙️ Technologies</h3>
        <ul>
            <li>Python</li>
            <li>TensorFlow / Keras</li>
            <li>OpenCV</li>
            <li>Streamlit</li>
            <li>EasyOCR</li>
            <li>Matplotlib</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🎯 Accuracy Improvement Tips</h3>
        <ul>
            <li>Upload only one digit at a time for CNN prediction.</li>
            <li>Use black or white background with high contrast.</li>
            <li>Keep the digit centered in the image.</li>
            <li>Avoid blurry, tilted, or very small digits.</li>
            <li>For multiple digits, use the OCR section.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------- HISTORY ----------------
st.markdown("---")
st.subheader("📊 Prediction History")

if st.session_state.history:
    st.dataframe(st.session_state.history, use_container_width=True)
else:
    st.info("No prediction history yet.")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Developed by <b>Kanhaiya Pathak</b> | CNN + TensorFlow + Streamlit + OCR
</div>
""", unsafe_allow_html=True)