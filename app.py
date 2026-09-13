import streamlit as st
from PIL import Image
import cv2
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Helmet & Seatbelt Detection",
    page_icon="🪖",
    layout="wide"
)

# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>
.reportview-container, .main, .block-container {
    background-color: #0d0d0d;
    color: #ffffff;
    font-family: 'Arial', sans-serif;
}

.title {
    font-size: 3rem;
    font-weight: 800;
    color: #00d4ff;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 1.2rem;
    color: #aaaaaa;
    text-align: center;
    margin-bottom: 50px;
}

.card {
    background-color: #1a1a1a;
    padding: 40px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.7);
    max-width: 900px;
    margin: auto;
}

.features {
    text-align: center;
    font-size: 1.05rem;
    margin-bottom: 50px;
    line-height: 1.8;
    color: #ffffff;
}

div.stButton > button {
    background-color: #333333;
    color: #00d4ff;
    padding: 14px 36px;
    border-radius: 30px;
    font-size: 1.2rem;
    font-weight: bold;
    border: 2px solid #00d4ff;
    cursor: pointer;
    transition: all 0.3s ease;
    display: block;
    margin: auto;
}

div.stButton > button:hover {
    background-color: #555555;
    transform: scale(1.05);
}

p {
    text-align:center;
    color:#888;
    margin-top:40px;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- YOLO MODELS ----------------
# Cached so the models load once per session, not on every rerun/button click
@st.cache_resource
def load_models():
    model_helmet = YOLO("models/helmet_model.pt")
    model_seatbelt = YOLO("models/seatbelt_model.pt")
    return model_helmet, model_seatbelt

model_helmet, modelseatbelt = load_models()

# ---------------- SESSION STATE ----------------
if 'page' not in st.session_state:
    st.session_state.page = 'front'
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'detection_type' not in st.session_state:
    st.session_state.detection_type = None

# ---------------- FRONT PAGE ----------------
if st.session_state.page == 'front':
    st.write("")
    st.write("")
    st.markdown('<div class="title">Helmet & Seatbelt Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">YOLO-Based Deep Learning System for Road Safety</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="features">
            🪖 <b>Helmet Detection</b><br>
            🚗 <b>Seatbelt Detection</b><br>
            🤖 <b>Real-time YOLO Inference</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    col1, col2, col3 = st.columns([3.8, 3, 2])
    with col2:
        if st.button("🚀 Get Started"):
            st.session_state.page = 'input'
            st.rerun()

# ---------------- INPUT PAGE ----------------
elif st.session_state.page == 'input':
    st.markdown('<div class="title">Upload Image for Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Select detection type and upload an image</div>', unsafe_allow_html=True)

    detection_type = st.radio("Select Detection Type:", ('Helmet Detection', 'Seatbelt Detection'), horizontal=True)
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.session_state.image_uploaded = True
        st.session_state.uploaded_file = uploaded_file
        st.success("Image uploaded successfully!")

    st.write("")
    st.write("")

    if st.session_state.image_uploaded:
        col1, col2, col3 = st.columns([3.8, 3, 2])
        with col2:
            if st.button("🛠 Detect"):
                st.session_state.detection_type = detection_type
                st.session_state.page = 'results'
                st.rerun()

# ---------------- RESULTS PAGE ----------------
elif st.session_state.page == 'results':
    st.markdown('<div class="title">Detection Results</div>', unsafe_allow_html=True)
    img = Image.open(st.session_state.uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("")

    col1, col2, col3 = st.columns([3.8, 3, 2])
    with col2:
        if st.button("🔍 Show Results"):
            with st.spinner("Processing... this may take a few seconds"):
                # Select model
                model = model_helmet if st.session_state.detection_type == 'Helmet Detection' else modelseatbelt

                # Run prediction directly on the PIL image (no temp file needed)
                results = model.predict(img, conf=0.25)

                # Plot results and show full-width image
                for r in results:
                    im_array = r.plot(conf=True, labels=True, line_width=2)
                    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                    annotated_img = Image.fromarray(im_rgb)
                    st.image(
                        annotated_img,
                        caption=f"{st.session_state.detection_type} Result",
                        use_container_width=True
                    )

    st.write("")
    col1, col2, col3 = st.columns([3.8, 3, 2])
    with col2:
        if st.button("⬅ Try Another Image"):
            st.session_state.page = 'input'
            st.session_state.image_uploaded = False
            st.rerun()

# ---------------- FOOTER ----------------
st.markdown("<p>Built with Streamlit • YOLO • Computer Vision</p>", unsafe_allow_html=True)
