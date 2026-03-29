import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DR Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:        #08090c;
    --surface:   #0f1117;
    --surface2:  #161924;
    --border:    #1e2433;
    --border2:   #2a3145;
    --accent:    #4f8ef7;
    --green:     #22c55e;
    --yellow:    #f59e0b;
    --orange:    #f97316;
    --red:       #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1200px !important; }

h1,h2,h3 { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.5px; }

[data-testid="stFileUploadDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border2) !important;
    border-radius: 14px !important;
    transition: border-color .2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 13px 0 !important;
    width: 100% !important;
    letter-spacing: .5px;
    transition: opacity .15s, transform .1s;
}
.stButton > button:hover  { opacity: .88 !important; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stImage"] img { border-radius: 12px !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 5)
        )
    def forward(self, x):
        return self.head(self.backbone(x))


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        m = DRModel()
        state = torch.load(
            "best_model.pth",
            map_location=torch.device("cpu"),
            weights_only=False
        )
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state, strict=False)
        m.eval()
        return m
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


# ─────────────────────────────────────────────
#  LABELS + COLOURS
# ─────────────────────────────────────────────
CLASSES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
GRADE_COLOR = {
    0: ("#22c55e", "#052e16"),
    1: ("#f59e0b", "#1c1206"),
    2: ("#f97316", "#1c0f06"),
    3: ("#ef4444", "#1f0505"),
    4: ("#dc2626", "#1f0505"),
}
GRADE_DESC = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Microaneurysms only. Monitor regularly.",
    2: "More than mild NPDR. Closer follow-up advised.",
    3: "Severe NPDR. High risk of progression — refer promptly.",
    4: "Proliferative DR detected. Urgent ophthalmology referral required.",
}


# ─────────────────────────────────────────────
#  PREPROCESS  (original logic, untouched)
# ─────────────────────────────────────────────
def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blur, -4, 128)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img


# ─────────────────────────────────────────────
#  HTML HELPERS
# ─────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style="
        background:#0f1117;
        border:1px solid #1e2433;
        border-radius:16px;
        padding:28px 32px;
        margin-bottom:28px;
        display:flex; align-items:center; gap:20px;
    ">
        <div style="font-size:42px; line-height:1;">🩺</div>
        <div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:20px;
                        font-weight:600; color:#e2e8f0; margin-bottom:4px;">
                Diabetic Retinopathy Screening
            </div>
            <div style="font-size:13px; color:#64748b; margin-bottom:10px;">
                EfficientNet-B0 · 5-class grading · Grad-CAM explainability
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                <span style="background:#0f2837;border:1px solid #1e4060;color:#4f8ef7;
                             font-size:11px;font-family:'IBM Plex Mono',monospace;
                             padding:3px 10px;border-radius:20px;">RESEARCH TOOL</span>
                <span style="background:#052e16;border:1px solid #14532d;color:#22c55e;
                             font-size:11px;font-family:'IBM Plex Mono',monospace;
                             padding:3px 10px;border-radius:20px;">NOT FOR CLINICAL USE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(grade: int, conf: float):
    color, bg = GRADE_COLOR[grade]
    pct = int(conf * 100)
    st.markdown(f"""
    <div style="
        background:{bg};
        border:1.5px solid {color}55;
        border-radius:14px;
        padding:24px 26px;
        margin-bottom:4px;
    ">
        <div style="display:flex; justify-content:space-between;
                    align-items:flex-start; margin-bottom:14px;">
            <div>
                <div style="font-size:10px; font-family:'IBM Plex Mono',monospace;
                            letter-spacing:2px; color:{color}99; margin-bottom:6px;">
                    PRIMARY DIAGNOSIS
                </div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:26px;
                            font-weight:600; color:{color};">
                    {CLASSES[grade]}
                </div>
            </div>
            <div style="
                background:{color}22;
                border:1px solid {color}55;
                border-radius:10px;
                padding:10px 16px;
                text-align:center; min-width:72px;
            ">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:22px;
                            font-weight:600; color:{color};">{pct}%</div>
                <div style="font-size:10px; color:{color}99; letter-spacing:1px;">CONF.</div>
            </div>
        </div>
        <div style="font-size:13px; color:#94a3b8; line-height:1.65; margin-bottom:14px;">
            {GRADE_DESC[grade]}
        </div>
        <div style="background:#0008; border-radius:99px; height:6px; overflow:hidden;">
            <div style="width:{pct}%; height:100%; background:{color}; border-radius:99px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_runner_up(grade: int, conf: float):
    color, _ = GRADE_COLOR[grade]
    pct = int(conf * 100)
    st.markdown(f"""
    <div style="
        background:#0f1117;
        border:1px solid #1e2433;
        border-radius:12px;
        padding:14px 18px;
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom:16px;
    ">
        <div>
            <div style="font-size:10px; color:#64748b; letter-spacing:1.5px;
                        font-family:'IBM Plex Mono',monospace; margin-bottom:3px;">
                RUNNER-UP
            </div>
            <div style="font-size:15px; color:{color}; font-weight:500;">
                {CLASSES[grade]}
            </div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:16px; color:{color};">
            {pct}%
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_all_probs(probs_np):
    st.markdown("""
    <div style="font-size:11px; font-family:'IBM Plex Mono',monospace;
                letter-spacing:2px; color:#64748b; margin-bottom:12px;">
        ALL CLASS PROBABILITIES
    </div>
    """, unsafe_allow_html=True)
    for i, p in enumerate(probs_np):
        color, _ = GRADE_COLOR[i]
        pct = int(p * 100)
        st.markdown(f"""
        <div style="margin-bottom:9px;">
            <div style="display:flex; justify-content:space-between;
                        font-size:12px; margin-bottom:4px;">
                <span style="color:#94a3b8;">{CLASSES[i]}</span>
                <span style="font-family:'IBM Plex Mono',monospace;
                             color:{color}; font-weight:600;">{pct}%</span>
            </div>
            <div style="background:#1e2433; border-radius:99px; height:5px; overflow:hidden;">
                <div style="width:{pct}%; height:100%; background:{color}; border-radius:99px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def section_label(label: str):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin:20px 0 12px;">
        <div style="font-size:11px; font-family:'IBM Plex Mono',monospace;
                    letter-spacing:2px; color:#64748b;">{label}</div>
        <div style="flex:1; height:1px; background:#1e2433;"></div>
    </div>
    """, unsafe_allow_html=True)


def render_disclaimer():
    st.markdown("""
    <div style="
        background:#0f1117;
        border:1px solid #1e2433;
        border-left:3px solid #ef4444;
        border-radius:10px;
        padding:14px 18px;
        margin-top:20px;
        font-size:12px;
        color:#64748b;
        line-height:1.7;
    ">
        <span style="color:#ef4444; font-weight:600;">Disclaimer:</span>
        This tool is intended for research and educational purposes only.
        It does not constitute medical advice and must not be used as a
        substitute for professional clinical evaluation by a qualified ophthalmologist.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
render_header()
model = load_model()

left, right = st.columns([1, 1.4], gap="large")

# ── LEFT: Upload ──────────────────────────────
with left:
    section_label("INPUT")
    uploaded = st.file_uploader(
        "Drop a fundus image", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image  = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)
        st.image(image, use_container_width=True)

        w, h = image.size
        kb   = round(uploaded.size / 1024, 1)
        st.markdown(f"""
        <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
            <div style="background:#161924;border:1px solid #1e2433;border-radius:8px;
                        padding:6px 12px;font-size:12px;color:#64748b;
                        font-family:'IBM Plex Mono',monospace;">{w}×{h}px</div>
            <div style="background:#161924;border:1px solid #1e2433;border-radius:8px;
                        padding:6px 12px;font-size:12px;color:#64748b;
                        font-family:'IBM Plex Mono',monospace;">{kb} KB</div>
            <div style="background:#161924;border:1px solid #1e2433;border-radius:8px;
                        padding:6px 12px;font-size:12px;color:#64748b;
                        font-family:'IBM Plex Mono',monospace;">EfficientNet-B0</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("ANALYSE IMAGE", use_container_width=True)
    else:
        st.markdown("""
        <div style="
            background:#0f1117; border:1.5px dashed #1e2433;
            border-radius:14px; padding:56px 24px;
            text-align:center; margin-top:4px;
        ">
            <div style="font-size:40px; margin-bottom:14px; opacity:.35;">🔬</div>
            <div style="font-size:13px; color:#334155; line-height:1.7;">
                Upload a retinal fundus image<br>to begin screening
            </div>
        </div>
        """, unsafe_allow_html=True)
        run = False

# ── RIGHT: Results ────────────────────────────
with right:
    if uploaded and run:
        with st.spinner("Running inference…"):
            img_proc = preprocess_image(img_np)
            tensor = (
                torch.tensor(img_proc, dtype=torch.float32)
                .permute(2, 0, 1).unsqueeze(0)
            )

            with torch.no_grad():
                output = model(tensor)
                probs  = torch.softmax(output, dim=1)

            probs_np = probs[0].numpy()
            top2     = torch.topk(probs, 2)
            pred1    = top2.indices[0][0].item()
            pred2    = top2.indices[0][1].item()
            conf1    = top2.values[0][0].item()
            conf2    = top2.values[0][1].item()

            # Grad-CAM
            target_layer  = model.backbone.conv_head
            cam_extractor = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam_extractor(input_tensor=tensor)[0]
            heatmap = show_cam_on_image(
                img_proc.astype(np.float32), grayscale_cam, use_rgb=True
            )

        section_label("DIAGNOSIS")
        render_result_card(pred1, conf1)
        render_runner_up(pred2, conf2)
        render_all_probs(probs_np)

        section_label("GRAD-CAM EXPLAINABILITY")
        cam1, cam2 = st.columns(2)
        with cam1:
            st.image(
                (img_proc * 255).astype(np.uint8),
                caption="Preprocessed input",
                use_container_width=True
            )
        with cam2:
            st.image(heatmap, caption="Activation heatmap", use_container_width=True)

        st.markdown("""
        <div style="font-size:12px; color:#475569; margin-top:8px; line-height:1.7;">
            Warmer regions (red/yellow) indicate areas most influential in the model's
            decision. Grad-CAM highlights retinal features such as microaneurysms,
            haemorrhages, or neovascularisation.
        </div>
        """, unsafe_allow_html=True)

        render_disclaimer()

    elif not uploaded:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center;
            justify-content:center; height:480px; text-align:center;
        ">
            <div style="font-size:56px; margin-bottom:18px; opacity:.15;">🩺</div>
            <div style="font-size:14px; color:#334155; font-family:'IBM Plex Mono',monospace;">
                Results appear here
            </div>
            <div style="font-size:12px; color:#1e2433; margin-top:8px;">
                Upload an image and click Analyse
            </div>
        </div>
        """, unsafe_allow_html=True)
