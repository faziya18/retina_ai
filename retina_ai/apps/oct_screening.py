# app_oct_only.py — OCT Screening App (8-class) with MFA OTP Login + AI-based Clinical Reasoning

import os
import secrets
import hashlib
import sqlite3
from datetime import datetime, timedelta
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import torch
import torch.nn.functional as F
from torchvision import models, transforms

# ================== CONFIG ==================
ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "app.db"

# OTP settings
OTP_EXP_MIN = 5
OTP_MAX_ATTEMPTS = 3
OTP_RESEND_COOLDOWN_SEC = 45
OTP_RATE_LIMIT_PER_HOUR = 5
DEV_SHOW_OTP_ON_PAGE = True

# OCT classes (8-class version)
OCT_CLASSES = ["NORMAL", "DME", "CNV", "DRUSEN", "AMD", "CSR", "DR", "MH"]
WEIGHTS_OCT = ROOT_DIR / "runs" / "oct_resnet18_c8_val" / "oct_resnet18_c8.pth"

# AI-based clinical explanations
EXPLANATIONS = {
    "NORMAL": (
        "The OCT scan shows a physiologically normal retina with distinct and intact retinal layers — including the inner limiting membrane, outer nuclear layer, and retinal pigment epithelium (RPE). "
        "A well-defined foveal pit with preserved photoreceptor integrity suggests optimal macular health. "
        "No signs of cystoid spaces, subretinal fluid, or structural distortion are detected. "
        "The AI model classified this as NORMAL likely due to the symmetric layer alignment and uniform reflectivity patterns across the macula, consistent with healthy retinal morphology."
    ),

    "DME": (
        "Diabetic Macular Edema (DME) is characterized by the presence of intraretinal cystoid spaces caused by leakage from compromised microaneurysms and dilated capillaries secondary to chronic hyperglycemia. "
        "OCT reveals hyporeflective (dark) cavities within the inner nuclear and outer plexiform layers, often accompanied by diffuse retinal thickening and loss of foveal contour. "
        "The model identified this condition due to irregular reflectivity and increased retinal thickness indicative of fluid accumulation. "
        "Clinically, DME leads to blurred or distorted central vision; early intervention with anti-VEGF therapy and glycemic control can stabilize visual function."
    ),

    "CNV": (
        "Choroidal Neovascularization (CNV) involves abnormal angiogenesis originating from the choroid penetrating Bruch’s membrane and the RPE. "
        "OCT imaging shows subretinal hyperreflective material associated with intraretinal or subretinal fluid and possible pigment epithelial detachment (PED). "
        "The AI model recognized complex reflectance patterns typical of neovascular membranes and fluid exudation. "
        "CNV is a hallmark feature of wet Age-Related Macular Degeneration (AMD), and rapid treatment using anti-VEGF injections is essential to prevent fibrotic scarring and irreversible central vision loss."
    ),

    "DRUSEN": (
        "Drusen represent focal extracellular lipid-protein deposits between the RPE and Bruch’s membrane. "
        "OCT shows dome-like elevations of the RPE with smooth or irregular contours, often with preserved overlying photoreceptor structure. "
        "The AI likely classified this image as DRUSEN due to multiple sub-RPE elevations with homogeneous reflectivity, a strong indicator of early or intermediate AMD. "
        "Although typically asymptomatic, drusen accumulation increases the risk of progression to advanced atrophic or neovascular AMD, warranting regular follow-up."
    ),

    "AMD": (
        "Age-Related Macular Degeneration (AMD) is a progressive degenerative disease of the macula that disrupts central vision. "
        "OCT findings vary by stage — from RPE thinning and drusen deposition in early AMD to subretinal fibrosis or CNV in the neovascular form. "
        "The AI model likely detected irregularities in RPE elevation, outer retinal layer disruption, and potential subretinal deposits indicative of disease progression. "
        "Clinically, AMD presents with central blurring, metamorphopsia, and scotomas; long-term monitoring, nutritional support, and anti-VEGF therapy are standard management approaches."
    ),

    "CSR": (
        "Central Serous Retinopathy (CSR) occurs when serous fluid accumulates beneath the neurosensory retina due to focal defects in the RPE barrier, often stress-related or corticosteroid-induced. "
        "OCT displays a dome-shaped detachment of the neurosensory retina with clear subretinal fluid but without intraretinal cysts. "
        "The AI system likely identified the hyperreflective RPE and smooth subretinal fluid cavity as the defining CSR features. "
        "Clinically, CSR causes sudden central vision blurring or metamorphopsia and often resolves spontaneously within weeks, though chronic or recurrent cases may require laser or photodynamic therapy."
    ),

    "DR": (
        "Diabetic Retinopathy (DR) is a chronic microvascular complication of diabetes characterized by capillary leakage, ischemia, and neovascular proliferation. "
        "On OCT, DR manifests as diffuse retinal thickening, microaneurysm shadows, and intraretinal cystoid fluid; proliferative stages show tractional membranes or vitreoretinal adhesions. "
        "The model likely detected irregular layer boundaries and heterogeneous reflectivity typical of diabetic vascular damage. "
        "DR progresses from non-proliferative to proliferative forms and is a leading cause of blindness; strict blood sugar control, laser photocoagulation, and anti-VEGF therapy can preserve vision."
    ),

    "MH": (
        "Macular Hole (MH) is a full-thickness discontinuity in the foveal retina caused by anteroposterior vitreomacular traction. "
        "OCT images demonstrate a circular foveal defect with cystoid edges and separation of inner and outer retinal layers. "
        "The AI likely recognized the abrupt break in retinal continuity and loss of foveal depression as diagnostic features. "
        "Clinically, MH results in central vision distortion or loss; surgical repair through pars plana vitrectomy with gas tamponade has a high success rate for visual recovery."
    )
}

# Assets
ASSETS = {
    "hero_oct": "assets/hero_oct.png",
    "login_side": "assets/login_side.png",
}

# ================== DB ==================
def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        phone TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS otps(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        otp_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        attempts_left INTEGER NOT NULL,
        request_id TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        file_name TEXT,
        prediction TEXT,
        probs_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit(); conn.close()

# ================== OTP AUTH ==================
def _hash_code(code, salt):
    return hashlib.sha256((salt + code).encode()).hexdigest()

def _find_or_create_user(identifier):
    conn = get_db()
    row = conn.execute("SELECT user_id FROM users WHERE email=?", (identifier,)).fetchone()
    if row:
        uid = row[0]
    else:
        uid = conn.execute("INSERT INTO users(email) VALUES (?)", (identifier,)).lastrowid
    conn.commit(); conn.close()
    return uid

def send_otp(identifier):
    uid = _find_or_create_user(identifier)
    conn = get_db()

    # Rate limit
    one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
    cnt = conn.execute("SELECT COUNT(*) FROM otps WHERE user_id=? AND created_at>=?", (uid, one_hour_ago)).fetchone()[0]
    if cnt >= OTP_RATE_LIMIT_PER_HOUR:
        conn.close(); return None, "Too many OTP requests in last hour."

    # Cooldown
    last = conn.execute("SELECT created_at FROM otps WHERE user_id=? ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
    if last:
        delta = (datetime.utcnow() - datetime.fromisoformat(last[0])).total_seconds()
        if delta < OTP_RESEND_COOLDOWN_SEC:
            conn.close(); return None, f"Resend after {OTP_RESEND_COOLDOWN_SEC-int(delta)}s."

    code = f"{secrets.randbelow(1_000_000):06d}"
    salt = secrets.token_hex(8)
    expires_at = (datetime.utcnow() + timedelta(minutes=OTP_EXP_MIN)).isoformat()
    request_id = secrets.token_urlsafe(12)

    conn.execute("""INSERT INTO otps(user_id, otp_hash, salt, expires_at, attempts_left, request_id)
                    VALUES (?,?,?,?,?,?)""",
                 (uid, _hash_code(code, salt), salt, expires_at, OTP_MAX_ATTEMPTS, request_id))
    conn.commit(); conn.close()

    print(f"[DEV] OTP for {identifier} → {code}")
    return {"user_id": uid, "request_id": request_id, "code": code if DEV_SHOW_OTP_ON_PAGE else None}, None

def verify_otp(user_id, request_id, code):
    conn = get_db()
    row = conn.execute("""SELECT id, otp_hash, salt, expires_at, attempts_left
                          FROM otps WHERE user_id=? AND request_id=? ORDER BY id DESC LIMIT 1""",
                       (user_id, request_id)).fetchone()
    if not row:
        conn.close(); return False, "Invalid request"
    _id, otp_hash, salt, expires_at, attempts_left = row
    if datetime.utcnow() > datetime.fromisoformat(expires_at):
        conn.close(); return False, "Code expired"
    if attempts_left <= 0:
        conn.close(); return False, "Too many attempts"
    if _hash_code(code, salt) == otp_hash:
        conn.execute("DELETE FROM otps WHERE id=?", (_id,))
        conn.commit(); conn.close()
        return True, None
    attempts_left -= 1
    conn.execute("UPDATE otps SET attempts_left=? WHERE id=?", (attempts_left, _id))
    conn.commit(); conn.close()
    return False, f"Invalid code ({attempts_left} tries left)"

# ================== SESSION ==================
def set_session(uid, identifier):
    st.session_state["auth.user_id"] = uid
    st.session_state["auth.identifier"] = identifier
    st.session_state["auth.is_authenticated"] = True

def clear_session():
    for k in list(st.session_state.keys()):
        if k.startswith("auth.") or k.startswith("ui."):
            del st.session_state[k]

def is_authed(): 
    return st.session_state.get("auth.is_authenticated", False)

# ================== MODEL ==================
_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_oct_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda im: im.convert("L")),
    transforms.Lambda(lambda im: Image.merge("RGB", (im, im, im))),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

@st.cache_resource
def load_oct_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(OCT_CLASSES))
    if os.path.exists(WEIGHTS_OCT):
        state = torch.load(WEIGHTS_OCT, map_location="cpu")
        model.load_state_dict(state)
    model.eval().to(_DEVICE)
    return model

def predict_oct(img: Image.Image):
    x = _oct_tf(img).unsqueeze(0).to(_DEVICE)
    model = load_oct_model()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
    pred = OCT_CLASSES[int(np.argmax(probs))]
    explanation = EXPLANATIONS.get(pred, "No clinical explanation available.")
    return pred, probs, explanation

# ================== PDF ==================
def build_pdf(identifier, file_name, prediction, probs, explanation):
    buf = BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    w,h = A4; y=h-50
    c.setFont("Helvetica-Bold", 18); c.drawString(40,y,"OCT Screening Report"); y-=30
    c.setFont("Helvetica",11)
    c.drawString(40,y,f"User: {identifier}"); y-=16
    c.drawString(40,y,f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"); y-=16
    c.drawString(40,y,f"File: {file_name}"); y-=24
    c.setFont("Helvetica-Bold",12); c.drawString(40,y,f"Prediction: {prediction}"); y-=18
    c.setFont("Helvetica",11)
    for cls,p in zip(OCT_CLASSES, probs):
        c.drawString(60,y,f"{cls:10s} {p*100:6.2f}%"); y-=14
    y -= 18
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "AI Clinical Reasoning:")
    y -= 16
    c.setFont("Helvetica", 10)
    # Wrap explanation text
    from reportlab.lib.utils import simpleSplit
    lines = simpleSplit(explanation, "Helvetica", 10, w-80)
    for line in lines:
        c.drawString(60, y, line)
        y -= 12
        if y < 60:
            c.showPage()
            y = h-50
            c.setFont("Helvetica", 10)
    c.showPage(); c.save(); buf.seek(0)
    return buf.read()

# ================== UI ==================
def page_login():
    # Simple login page: dark theme, centered, minimal styling
    st.markdown("""
<style>
body, .stApp {
    background: #0e1117 !important;
}

/* Kill all Streamlit container backgrounds */
div[data-testid="stAppViewContainer"],
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
div[data-testid="stBlock"],
div[data-testid="block-container"],
section.main,
header,
footer,
div[style*="border-radius"],
div[style*="background-color: rgb"],
div[style*="background: rgb"],
div[style*="rgba"],
.stApp:before,
.stApp:after {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

/* Remove the mysterious top rounded rectangle */
section.main > div:first-child,
.stApp > div:first-child {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}

/* Main login card */
.simple-login-container {
    max-width: 400px;
    margin: 8vh auto 0 auto;
    background: #1c1f26;
    border-radius: 1.2em;
    box-shadow: 0 2px 18px rgba(20,24,31,0.8);
    padding: 2.5em 2em;
    color: #e4e6eb;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.simple-login-title {
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.7em;
}
.simple-login-desc {
    font-size: 1.08rem;
    color: #ccc;
    margin-bottom: 2em;
}
button, .stButton>button {
    background: linear-gradient(90deg,#2563eb 0%,#4a90e2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 0.6em !important;
}
input, .stTextInput input {
    background: #2a2f3a !important;
    color: #e4e6eb !important;
    border: 1.5px solid #22242a !important;
    border-radius: 0.5em !important;
}
</style>
""", unsafe_allow_html=True)
    
    # Remove any extra st.markdown() or HTML container above the login box
    # Only keep the single login container:
    st.markdown('<div class="simple-login-container">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="simple-login-title">Sign in</div>
        <div class="simple-login-desc">
            Use email or phone to receive a one-time code (OTP).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # OTP Request Form
    with st.form("otp_request"):
        email = st.text_input(
            "Email or Phone",
            placeholder="e.g. you@example.com or +1234567890",
            key="login_email",
            label_visibility="collapsed",
        )
        send = st.form_submit_button("Send OTP", type="primary")
    if send and email:
        result, err = send_otp(email.strip())
        if err:
            st.error(err)
        else:
            st.session_state["auth.pending"] = {
                "identifier": email.strip(),
                "user_id": result["user_id"],
                "request_id": result["request_id"],
            }
            st.success("OTP sent to email (check console in dev).")
            if DEV_SHOW_OTP_ON_PAGE and result.get("code"):
                st.info(f"DEV OTP: {result['code']}")

    if "auth.pending" in st.session_state:
        st.markdown(
            '<div style="margin-top:1.5em; margin-bottom:0.4em; font-weight:600; font-size:1.07rem; color:#111;">Enter OTP</div>',
            unsafe_allow_html=True,
        )
        with st.form("verify_otp"):
            otp = st.text_input(
                "One-Time Passcode",
                placeholder="6-digit code",
                max_chars=6,
                key="login_otp",
                label_visibility="collapsed",
            )
            verify = st.form_submit_button("Verify", type="primary")
        if verify:
            p = st.session_state["auth.pending"]
            ok, err = verify_otp(p["user_id"], p["request_id"], otp.strip())
            if ok:
                set_session(p["user_id"], p["identifier"])
                del st.session_state["auth.pending"]
                st.success("Logged in successfully.")
                st.balloons()
                st.rerun()
            else:
                st.error(err)

    st.markdown('<div class="simple-divider">Prefer to try without sign-in?</div>', unsafe_allow_html=True)

    # Guest logic
    guest_btn = st.button("Continue as guest (limited)", key="guest_btn", help="Guest access", type="secondary")
    if guest_btn:
        set_session(-1, "Guest")
        st.session_state["auth.is_guest"] = True
        st.success("Continuing as Guest.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def page_oct_scan():
    st.title("👁️ OCT Screening (8-class)")
    uploaded = st.file_uploader("Upload OCT image", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
        if st.button("Analyze OCT", type="primary"):
            pred, probs, explanation = predict_oct(img)
            st.success(f"Prediction: {pred}")
            st.markdown(f"**Clinical Reason:** {explanation}")
            st.table({"Class": OCT_CLASSES, "Probability": [f"{p*100:.2f}%" for p in probs]})
            # Save prediction to DB
            user_id = st.session_state.get("auth.user_id")
            conn = get_db()
            import sqlite3
            try:
                conn.execute(
                    "INSERT INTO scans(user_id, file_name, prediction, probs_json, created_at, modality) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        uploaded.name,
                        pred,
                        json.dumps({"probs": probs, "explanation": explanation}),
                        datetime.utcnow().isoformat(),
                        "OCT"
                    ),
                )
            except sqlite3.OperationalError:
                conn.execute(
                    "INSERT INTO scans(user_id, file_name, prediction, probs_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        user_id,
                        uploaded.name,
                        pred,
                        json.dumps({"probs": probs, "explanation": explanation}),
                        datetime.utcnow().isoformat()
                    ),
                )
            conn.commit(); conn.close()
            pdf_bytes = build_pdf(st.session_state["auth.identifier"], uploaded.name, pred, probs, explanation)
            st.download_button("📄 Download PDF Report", pdf_bytes, file_name="oct_report.pdf", mime="application/pdf")

# =========== HISTORY PAGE ===========
def page_history():
    st.title("🕓 Scan History")
    user_id = st.session_state.get("auth.user_id")
    conn = get_db()
    rows = conn.execute(
        "SELECT file_name, prediction, probs_json, created_at FROM scans WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    if not rows:
        st.info("No scan history found.")
        return
    for i, (file_name, prediction, probs_json, created_at) in enumerate(rows):
        try:
            data = json.loads(probs_json)
            probs = data.get("probs", [])
            explanation = data.get("explanation", "")
        except Exception:
            probs = []
            explanation = ""
        with st.expander(f"{created_at} — {file_name} — Prediction: {prediction}"):
            st.write(f"**Prediction:** {prediction}")
            st.write(f"**Timestamp:** {created_at}")
            st.write(f"**File:** {file_name}")
            if probs:
                st.table({"Class": OCT_CLASSES, "Probability": [f"{p*100:.2f}%" for p in probs]})
            if explanation:
                st.markdown(f"**Clinical Reason:** {explanation}")

# ================== MAIN ==================
def main():
    st.set_page_config(page_title="OCT Screening (MFA)", layout="wide")
    init_db()
    if not is_authed():
        page_login()
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["OCT Scan", "History", "Logout"])
    if page == "Logout":
        clear_session(); st.rerun()
    elif page == "OCT Scan":
        page_oct_scan()
    elif page == "History":
        page_history()

if __name__ == "__main__":
    main()