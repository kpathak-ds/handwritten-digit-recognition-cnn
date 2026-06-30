st.write("APP STARTED")
st.write("STEP 1")
import streamlit as st

st.title("Application Started")

st.write("Step 1 completed")
import tensorflow as tf

st.write("TensorFlow imported")
import numpy as np
import cv2
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import easyocr
from streamlit_drawable_canvas import st_canvas
model = load_digit_model()
st.write("STEP 2")
# ─── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralScript · Digit AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme toggle via session state ─────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ─── CSS ────────────────────────────────────────────────────────
def inject_css(theme):
    if theme == "dark":
        bg       = "#0a0a0f"
        bg2      = "#12121a"
        bg3      = "#1a1a26"
        card     = "#16161f"
        border   = "#2a2a3d"
        text     = "#e8e8f0"
        muted    = "#7878a0"
        accent   = "#7c6af7"
        accent2  = "#4fc3f7"
        success  = "#4ade80"
        warn     = "#fbbf24"
        grad1    = "linear-gradient(135deg,#7c6af7 0%,#4fc3f7 100%)"
    else:
        bg       = "#f0f0f8"
        bg2      = "#ffffff"
        bg3      = "#e8e8f4"
        card     = "#ffffff"
        border   = "#d0d0e8"
        text     = "#1a1a2e"
        muted    = "#6060a0"
        accent   = "#5a4dd6"
        accent2  = "#0288d1"
        success  = "#16a34a"
        warn     = "#d97706"
        grad1    = "linear-gradient(135deg,#5a4dd6 0%,#0288d1 100%)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap');

    /* ── Reset ── */
    html, body, [class*="css"] {{
        font-family: 'Space Grotesk', sans-serif !important;
        background-color: {bg} !important;
        color: {text} !important;
    }}
    .stApp {{ background-color: {bg} !important; }}
    .block-container {{ padding: 1.5rem 2rem 4rem 2rem !important; max-width: 1300px; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {bg2} !important;
        border-right: 1px solid {border} !important;
    }}
    section[data-testid="stSidebar"] * {{ color: {text} !important; }}
    .sidebar-logo {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        background: {grad1};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 1rem 0 0.5rem 0;
        text-align: center;
    }}
    .sidebar-tagline {{
        text-align: center;
        font-size: 0.72rem;
        color: {muted};
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }}
    .nav-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 4px 0;
        cursor: pointer;
        font-size: 0.9rem;
        font-weight: 500;
        color: {muted};
        transition: all 0.2s;
        border: 1px solid transparent;
    }}
    .nav-item.active {{
        background: {bg3};
        color: {accent};
        border-color: {border};
    }}

    /* ── Hero header ── */
    .hero-wrap {{
        padding: 2.5rem 0 1.5rem 0;
        text-align: center;
    }}
    .hero-eyebrow {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: {accent};
        background: {'rgba(124,106,247,0.12)' if theme=='dark' else 'rgba(90,77,214,0.10)'};
        border: 1px solid {'rgba(124,106,247,0.3)' if theme=='dark' else 'rgba(90,77,214,0.25)'};
        padding: 4px 14px;
        border-radius: 999px;
        margin-bottom: 1rem;
    }}
    .hero-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.2rem, 5vw, 3.4rem);
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin-bottom: 0.6rem;
        color: {text};
    }}
    .hero-title span {{
        background: {grad1};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hero-sub {{
        font-size: 1.05rem;
        color: {muted};
        font-weight: 400;
        max-width: 560px;
        margin: 0 auto 1.4rem auto;
        line-height: 1.6;
    }}

    /* ── Index tab bar (pill nav) ── */
    .index-bar {{
        display: flex;
        justify-content: center;
        gap: 6px;
        flex-wrap: wrap;
        margin-bottom: 2.5rem;
    }}
    .idx-pill {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 6px 16px;
        border-radius: 999px;
        border: 1px solid {border};
        color: {muted};
        background: {bg2};
        cursor: pointer;
        text-decoration: none !important;
        transition: all 0.2s;
        white-space: nowrap;
    }}

    /* ── Stat cards ── */
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 2.5rem;
    }}
    @media (max-width: 900px) {{ .stat-grid {{ grid-template-columns: repeat(2,1fr); }} }}
    .stat-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        position: relative;
        overflow: hidden;
    }}
    .stat-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: {grad1};
    }}
    .stat-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: {text};
        line-height: 1;
        margin-bottom: 4px;
    }}
    .stat-label {{
        font-size: 0.78rem;
        color: {muted};
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    .stat-icon {{
        position: absolute;
        top: 1rem; right: 1rem;
        font-size: 1.4rem;
        opacity: 0.5;
    }}

    /* ── Feature info cards ── */
    .info-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 2rem;
    }}
    @media (max-width: 850px) {{ .info-grid {{ grid-template-columns: 1fr; }} }}
    .info-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 1.4rem;
    }}
    .info-card-icon {{
        font-size: 1.8rem;
        margin-bottom: 0.7rem;
    }}
    .info-card-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {text};
        margin-bottom: 0.4rem;
    }}
    .info-card-body {{
        font-size: 0.82rem;
        color: {muted};
        line-height: 1.6;
    }}

    /* ── Section heading ── */
    .sec-heading {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 2rem 0 1rem 0;
    }}
    .sec-heading-line {{
        flex: 1;
        height: 1px;
        background: {border};
    }}
    .sec-heading-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {muted};
    }}

    /* ── Architecture pipeline ── */
    .pipeline {{
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0;
        margin-bottom: 2rem;
        background: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 1.6rem;
    }}
    .pipe-node {{
        text-align: center;
        padding: 0.8rem 1.1rem;
        border-radius: 10px;
        background: {bg3};
        border: 1px solid {border};
        min-width: 90px;
    }}
    .pipe-node-icon {{ font-size: 1.5rem; }}
    .pipe-node-label {{
        font-size: 0.7rem;
        font-weight: 600;
        color: {muted};
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .pipe-arrow {{
        color: {accent};
        font-size: 1.2rem;
        padding: 0 6px;
        font-weight: 700;
    }}

    /* ── Result box ── */
    .result-box {{
        background: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
    }}
    .result-digit {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 5rem;
        font-weight: 700;
        background: {grad1};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }}
    .result-label {{
        font-size: 0.78rem;
        color: {muted};
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .conf-bar-wrap {{
        margin-top: 1rem;
        background: {bg3};
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
    }}
    .conf-bar {{
        height: 100%;
        border-radius: 999px;
        background: {grad1};
        transition: width 0.6s ease;
    }}

    /* ── Uploader styling ── */
    [data-testid="stFileUploader"] {{
        border: 2px dashed {border} !important;
        border-radius: 14px !important;
        background: {bg2} !important;
        padding: 1rem !important;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {accent} !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: {grad1} !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
        font-size: 0.9rem !important;
        transition: opacity 0.2s !important;
    }}
    .stButton > button:hover {{ opacity: 0.85 !important; }}

    /* ── Radio ── */
    .stRadio > div {{ gap: 6px !important; }}
    .stRadio label {{
        background: {bg3} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
        padding: 6px 12px !important;
        font-size: 0.85rem !important;
        color: {text} !important;
        cursor: pointer !important;
    }}

    /* ── Success / info ── */
    .stSuccess {{ background: {'rgba(74,222,128,0.1)' if theme=='dark' else 'rgba(22,163,74,0.08)'} !important;
                  border: 1px solid {'rgba(74,222,128,0.25)' if theme=='dark' else 'rgba(22,163,74,0.2)'} !important;
                  border-radius: 10px !important; color: {success} !important; }}
    .stInfo    {{ background: {'rgba(79,195,247,0.1)' if theme=='dark' else 'rgba(2,136,209,0.08)'} !important;
                  border: 1px solid {'rgba(79,195,247,0.25)' if theme=='dark' else 'rgba(2,136,209,0.2)'} !important;
                  border-radius: 10px !important; }}

    /* ── Divider ── */
    hr {{ border-color: {border} !important; }}

    /* ── Matplotlib charts ── */
    .stPlotlyChart, .stPyplot {{ border-radius: 12px !important; overflow: hidden; }}

    /* ── Footer ── */
    .footer {{
        text-align: center;
        font-size: 0.78rem;
        color: {muted};
        padding: 2rem 0 0.5rem 0;
        border-top: 1px solid {border};
        margin-top: 3rem;
    }}
    .footer span {{
        background: {grad1};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css(st.session_state.theme)

# ─── Models ────────────────────────────────────────────────────

@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model("models/digit_model.h5")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"])

model     = load_digit_model()
ocr_reader = load_ocr_reader()

st.write("Model loaded")

# ─── Helpers ───────────────────────────────────────────────────
def preprocess_image(image):
    image = image.convert("L")
    img = np.array(image)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    return img.reshape(1, 28, 28, 1)

def preprocess_video_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gray = cv2.resize(gray, (28, 28))
    gray = gray / 255.0
    return gray.reshape(1, 28, 28, 1)

def predict_digit(img):
    prediction = model.predict(img, verbose=0)
    digit      = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)
    return digit, confidence, prediction

def show_confidence_chart(prediction, theme):
    bg_col  = "#12121a" if theme == "dark" else "#ffffff"
    txt_col = "#e8e8f0" if theme == "dark" else "#1a1a2e"
    bar_col = ["#7c6af7" if i == int(np.argmax(prediction[0])) else
               ("#2a2a3d" if theme == "dark" else "#d0d0e8") for i in range(10)]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor(bg_col)
    ax.set_facecolor(bg_col)
    bars = ax.bar(range(10), prediction[0], color=bar_col, width=0.65, zorder=3)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)], color=txt_col, fontsize=11)
    ax.tick_params(colors=txt_col, labelsize=9)
    ax.yaxis.set_visible(False)
    ax.spines[:].set_visible(False)
    ax.set_title("Probability per digit", color=txt_col, fontsize=10, pad=10)
    ax.grid(axis='y', color='#333', linewidth=0.4, zorder=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def result_panel(digit, confidence):
    st.markdown(f"""
    <div class="result-box">
      <div class="result-digit">{digit}</div>
      <div class="result-label">Predicted digit</div>
      <div style="font-size:0.85rem;color:#7878a0;margin-top:0.8rem;">
        Confidence &nbsp;·&nbsp;
        <strong style="color:#e8e8f0;">{confidence:.1f}%</strong>
      </div>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:{confidence}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">NeuralScript</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Digit Recognition AI</div>', unsafe_allow_html=True)

    # Theme toggle
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("☀️ Light", use_container_width=True):
            st.session_state.theme = "light"; st.rerun()
    with col_b:
        if st.button("🌙 Dark", use_container_width=True):
            st.session_state.theme = "dark"; st.rerun()

    st.markdown("---")
    st.markdown("**Navigate**")
    menu = st.radio("", [
        "🏠  Overview",
        "🖼  Upload Image",
        "📷  Camera",
        "🎥  Video",
        "✏️  Draw Canvas",
        "🔢  Multi-Digit OCR"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#7878a0;">
    <b style="color:#e8e8f0;">Stack</b><br>
    TensorFlow · CNN · EasyOCR<br>Streamlit · OpenCV · NumPy
    </div>
    """, unsafe_allow_html=True)

# ─── OVERVIEW PAGE ─────────────────────────────────────────────
if "Overview" in menu:
    # Hero
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-eyebrow">✦ Deep Learning · Computer Vision</div>
      <h1 class="hero-title">Handwritten <span>Digit Recognition</span><br>powered by CNN</h1>
      <p class="hero-sub">Upload an image, draw a number, or point your camera — the model reads your handwriting in real time with 99 %+ accuracy on MNIST.</p>
    </div>
    """, unsafe_allow_html=True)

    # Index pill bar
    st.markdown("""
    <div class="index-bar">
      <span class="idx-pill">📊 Dataset</span>
      <span class="idx-pill">🧠 Architecture</span>
      <span class="idx-pill">🏋️ Training</span>
      <span class="idx-pill">📈 Results</span>
      <span class="idx-pill">🔬 Preprocessing</span>
      <span class="idx-pill">🖼 Inputs</span>
      <span class="idx-pill">🔢 OCR</span>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-icon">📦</div>
        <div class="stat-val">70 K</div>
        <div class="stat-label">MNIST samples</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-val">99.2%</div>
        <div class="stat-label">Test accuracy</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">⚡</div>
        <div class="stat-val">&lt; 50 ms</div>
        <div class="stat-label">Inference time</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🔢</div>
        <div class="stat-val">10</div>
        <div class="stat-label">Classes (0 – 9)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Section: Dataset
    st.markdown('<div class="sec-heading"><div class="sec-heading-line"></div><div class="sec-heading-label">📊 Dataset</div><div class="sec-heading-line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-grid">
      <div class="info-card">
        <div class="info-card-icon">🗄️</div>
        <div class="info-card-title">MNIST Benchmark</div>
        <div class="info-card-body">60,000 training + 10,000 test grayscale images (28×28 px) of handwritten digits 0–9, sourced from census workers and high-school students.</div>
      </div>
      <div class="info-card">
        <div class="info-card-icon">⚖️</div>
        <div class="info-card-title">Balanced Classes</div>
        <div class="info-card-body">Each digit class has ~6,000 training examples ensuring the model does not bias toward any single digit during learning.</div>
      </div>
      <div class="info-card">
        <div class="info-card-icon">🔄</div>
        <div class="info-card-title">Augmentation</div>
        <div class="info-card-body">Random rotations ±10°, width/height shifts, and zoom applied on-the-fly to improve generalization to real-world handwriting styles.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Section: Architecture pipeline
    st.markdown('<div class="sec-heading"><div class="sec-heading-line"></div><div class="sec-heading-label">🧠 CNN Architecture</div><div class="sec-heading-line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
      <div class="pipe-node"><div class="pipe-node-icon">🖼️</div><div class="pipe-node-label">Input 28×28</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="pipe-node-icon">🔳</div><div class="pipe-node-label">Conv2D 32</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="pipe-node-icon">🔲</div><div class="pipe-node-label">MaxPool</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="pipe-node-icon">🔳</div><div class="pipe-node-label">Conv2D 64</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="pipe-node-icon">🔲</div><div class="pipe-node-label">MaxPool</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="pipe-node-icon">📉</div><div class="pipe-node-label">Dropout</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="pipe-node-icon">🔗</div><div class="pipe-node-label">Dense 128</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-node"><div class="pipe-node-icon">🎯</div><div class="pipe-node-label">Softmax 10</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Section: Training & Results
    st.markdown('<div class="sec-heading"><div class="sec-heading-line"></div><div class="sec-heading-label">🏋️ Training · 📈 Results</div><div class="sec-heading-line"></div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-card">
          <div class="info-card-icon">🏋️</div>
          <div class="info-card-title">Training Setup</div>
          <div class="info-card-body">
            <b>Optimizer</b> — Adam (lr 0.001)<br>
            <b>Loss</b> — Categorical Cross-Entropy<br>
            <b>Epochs</b> — 15 with early stopping<br>
            <b>Batch size</b> — 128<br>
            <b>Validation split</b> — 10 %
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
          <div class="info-card-icon">📈</div>
          <div class="info-card-title">Test Results</div>
          <div class="info-card-body">
            <b>Accuracy</b> — 99.2 %<br>
            <b>Loss</b> — 0.027<br>
            <b>Precision</b> — 99.1 %<br>
            <b>Recall</b> — 99.2 %<br>
            <b>F1-Score</b> — 99.15 %
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Section: Preprocessing + Input modes
    st.markdown('<div class="sec-heading"><div class="sec-heading-line"></div><div class="sec-heading-label">🔬 Preprocessing</div><div class="sec-heading-line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-grid">
      <div class="info-card">
        <div class="info-card-icon">🌑</div>
        <div class="info-card-title">Grayscale + Otsu Threshold</div>
        <div class="info-card-body">Converts colour images to grayscale, then applies adaptive Otsu binarisation to separate digit pixels from background automatically.</div>
      </div>
      <div class="info-card">
        <div class="info-card-icon">✂️</div>
        <div class="info-card-title">Contour Crop + Resize</div>
        <div class="info-card-body">Finds the digit's bounding contour, crops tightly around it, then resizes to 28×28 px — matching the exact MNIST format the model expects.</div>
      </div>
      <div class="info-card">
        <div class="info-card-icon">📐</div>
        <div class="info-card-title">Normalise & Reshape</div>
        <div class="info-card-body">Pixel values scaled to [0, 1] and tensor reshaped to (1, 28, 28, 1) before being passed to the CNN for inference.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    

    st.markdown('<div class="sec-heading"><div class="sec-heading-line"></div><div class="sec-heading-label">🖼 Input Modes</div><div class="sec-heading-line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-grid">
      <div class="info-card"><div class="info-card-icon">🖼</div><div class="info-card-title">Image Upload</div><div class="info-card-body">PNG / JPG / JPEG file of a single handwritten digit. Preprocessed and predicted in one step.</div></div>
      <div class="info-card"><div class="info-card-icon">📷</div><div class="info-card-title">Live Camera</div><div class="info-card-body">Snap a photo with your device camera — works on phones and laptops without any extra setup.</div></div>
      <div class="info-card"><div class="info-card-icon">🎥</div><div class="info-card-title">Video Upload</div><div class="info-card-body">Analyse an MP4/MOV/AVI frame-by-frame. Every 20th frame is sampled and the most common prediction is returned.</div></div>
      <div class="info-card"><div class="info-card-icon">✏️</div><div class="info-card-title">Drawing Canvas</div><div class="info-card-body">Draw directly in-browser on a 280×280 canvas. The stroke is auto-cropped and fed to the CNN.</div></div>
      <div class="info-card"><div class="info-card-icon">🔢</div><div class="info-card-title">Multi-Digit OCR</div><div class="info-card-body">EasyOCR detects any number of digits or text in a single image, returning each detection with its confidence score.</div></div>
      <div class="info-card"><div class="info-card-icon">⚙️</div><div class="info-card-title">Canvas + Contour</div><div class="info-card-body">Drawn strokes are extracted via OpenCV contour detection before inference — identical pipeline to scanned images.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="footer">Built by <span>Kanhaiya Pathak</span> · NeuralScript v1.0 · CNN + TensorFlow + EasyOCR + Streamlit</div>', unsafe_allow_html=True)

# ─── UPLOAD IMAGE ──────────────────────────────────────────────
elif "Upload Image" in menu:
    st.markdown('<div class="hero-wrap" style="padding:1.5rem 0 1rem 0;"><div class="hero-eyebrow">Upload Mode</div><h2 class="hero-title" style="font-size:2rem;">Upload an <span>Image</span></h2></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop a PNG / JPG / JPEG here", type=["png","jpg","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.image(image, caption="Your image", use_container_width=True)
        img = preprocess_image(image)
        digit, confidence, prediction = predict_digit(img)
        with c2:
            result_panel(digit, confidence)
            st.markdown("<br>", unsafe_allow_html=True)
            show_confidence_chart(prediction, st.session_state.theme)

# ─── CAMERA ────────────────────────────────────────────────────
elif "Camera" in menu:
    st.markdown('<div class="hero-wrap" style="padding:1.5rem 0 1rem 0;"><div class="hero-eyebrow">Camera Mode</div><h2 class="hero-title" style="font-size:2rem;">Capture & <span>Predict</span></h2></div>', unsafe_allow_html=True)
    camera_image = st.camera_input("Point at a handwritten digit")
    if camera_image:
        image = Image.open(camera_image)
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.image(image, caption="Captured", use_container_width=True)
        img = preprocess_image(image)
        digit, confidence, prediction = predict_digit(img)
        with c2:
            result_panel(digit, confidence)
            st.markdown("<br>", unsafe_allow_html=True)
            show_confidence_chart(prediction, st.session_state.theme)

# ─── VIDEO ─────────────────────────────────────────────────────
elif "Video" in menu:
    st.markdown('<div class="hero-wrap" style="padding:1.5rem 0 1rem 0;"><div class="hero-eyebrow">Video Mode</div><h2 class="hero-title" style="font-size:2rem;">Frame-by-Frame <span>Analysis</span></h2></div>', unsafe_allow_html=True)
    video_file = st.file_uploader("Upload MP4 / MOV / AVI / MKV", type=["mp4","mov","avi","mkv"])
    if video_file:
        st.video(video_file)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_file.read()); tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        frame_number, results = 0, []
        frame_area  = st.empty()
        result_area = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_number += 1
            if frame_number % 20 != 0: continue
            img = preprocess_video_frame(frame)
            digit, confidence, prediction = predict_digit(img)
            results.append({"Frame": frame_number, "Digit": digit, "Confidence": round(confidence,2)})
            cv2.putText(frame, f"Digit: {digit} | {confidence:.1f}%", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (124,106,247), 2)
            frame_area.image(frame, channels="BGR", use_container_width=True)
            result_area.success(f"Frame {frame_number} → Digit **{digit}** ({confidence:.1f}%)")
        cap.release()
        if results:
            digits = [r["Digit"] for r in results]
            final_digit = max(set(digits), key=digits.count)
            avg_conf    = np.mean([r["Confidence"] for r in results])
            c1, c2 = st.columns(2)
            with c1: st.success(f"Final prediction: **{final_digit}**")
            with c2: st.info(f"Average confidence: **{avg_conf:.1f}%**")
            st.dataframe(results, use_container_width=True)

# ─── DRAW CANVAS ───────────────────────────────────────────────
elif "Draw" in menu:
    st.markdown('<div class="hero-wrap" style="padding:1.5rem 0 1rem 0;"><div class="hero-eyebrow">Canvas Mode</div><h2 class="hero-title" style="font-size:2rem;">Draw a <span>Digit</span></h2></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("**Draw below** (white stroke on black)")
        canvas_result = st_canvas(
            fill_color="black", stroke_width=18, stroke_color="white",
            background_color="black", height=280, width=280,
            drawing_mode="freedraw", key="canvas"
        )
    with c2:
        if canvas_result.image_data is not None:
            img = canvas_result.image_data[:, :, 0].astype(np.uint8)
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                img = img[y:y+h, x:x+w]
                img = cv2.resize(img, (20, 20))
                padded = np.zeros((28,28), dtype=np.uint8)
                padded[4:24, 4:24] = img
                inp = padded / 255.0
                inp = inp.reshape(1, 28, 28, 1)
                digit, confidence, prediction = predict_digit(inp)
                result_panel(digit, confidence)
                st.markdown("<br>", unsafe_allow_html=True)
                show_confidence_chart(prediction, st.session_state.theme)
            else:
                st.info("Start drawing to see a prediction →")

# ─── MULTI-DIGIT OCR ───────────────────────────────────────────
elif "OCR" in menu:
    st.markdown('<div class="hero-wrap" style="padding:1.5rem 0 1rem 0;"><div class="hero-eyebrow">OCR Mode</div><h2 class="hero-title" style="font-size:2rem;">Multi-Digit <span>OCR</span></h2></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload image with multiple digits or text", type=["png","jpg","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input image", use_container_width=True)
        img_array = np.array(image)
        with st.spinner("Loading OCR and processing image..."):
            results = ocr_reader.readtext(img_array)
        if results:
            st.success(f"Found **{len(results)}** text region(s)")
            for i, (bbox, text, confidence) in enumerate(results):
                with st.expander(f"Region {i+1} · '{text}'"):
                    st.markdown(f"**Text:** `{text}`")
                    st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                    cc = st.columns(10)
                    for j, ch in enumerate(text):
                        cc[j % 10].markdown(f"<div class='result-box' style='padding:0.5rem;'><div style='font-size:1.6rem;font-family:JetBrains Mono,monospace;font-weight:700;background:linear-gradient(135deg,#7c6af7,#4fc3f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{ch}</div></div>", unsafe_allow_html=True)
        else:
            st.warning("No text detected in the image.")
ocr_reader = load_ocr_reader()
st.write("OCR LOADED")

st.markdown("---")
st.markdown("Developed by **Kanhaiya Pathak** | CNN + TensorFlow + Streamlit + OCR")
