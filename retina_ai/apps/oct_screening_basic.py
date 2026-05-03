# app_oct_only.py — OCT Screening App (8-class) with MFA OTP Login

import os
import secrets
import hashlib
import sqlite3
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
from PIL import Image
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import torch
import torch.nn.functional as F
from torchvision import models, transforms

# ================== CONFIG ==================
DB_PATH = "app.db"

# OTP settings
OTP_EXP_MIN = 5
OTP_MAX_ATTEMPTS = 3
OTP_RESEND_COOLDOWN_SEC = 45
OTP_RATE_LIMIT_PER_HOUR = 5
DEV_SHOW_OTP_ON_PAGE = True

# OCT classes (8-class version)
OCT_CLASSES = ["NORMAL","DME","CNV","DRUSEN","AMD","CSR","DR","MH"]
WEIGHTS_OCT = os.getenv("OCT_WEIGHTS", "oct_resnet18_c8.pth")

# Assets
ASSETS = {
    "hero_oct": "assets/hero_oct.png",
    "login_side": "assets/login_side.png",
}

# ================== DB ==================
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    return pred, probs

# ================== PDF ==================
def build_pdf(identifier, file_name, prediction, probs):
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
    c.showPage(); c.save(); buf.seek(0)
    return buf.read()

# ================== UI ==================
def page_login():
    st.title("🔐 OCT Screening — Sign In")
    with st.form("otp_request"):
        email = st.text_input("Enter email for OTP")
        send = st.form_submit_button("Send OTP")
    if send and email:
        result, err = send_otp(email.strip())
        if err: st.error(err)
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
        with st.form("verify_otp"):
            otp = st.text_input("Enter OTP")
            verify = st.form_submit_button("Verify")
        if verify:
            p = st.session_state["auth.pending"]
            ok, err = verify_otp(p["user_id"], p["request_id"], otp.strip())
            if ok:
                set_session(p["user_id"], p["identifier"])
                del st.session_state["auth.pending"]
                st.success("Logged in successfully.")
                st.balloons(); st.rerun()
            else:
                st.error(err)

def page_oct_scan():
    st.title("👁️ OCT Screening (8-class)")
    uploaded = st.file_uploader("Upload OCT image", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
        if st.button("Analyze OCT", type="primary"):
            pred, probs = predict_oct(img)
            st.success(f"Prediction: {pred}")
            st.table({"Class": OCT_CLASSES, "Probability": [f"{p*100:.2f}%" for p in probs]})
            pdf_bytes = build_pdf(st.session_state["auth.identifier"], uploaded.name, pred, probs)
            st.download_button("📄 Download PDF Report", pdf_bytes, file_name="oct_report.pdf", mime="application/pdf")

# ================== MAIN ==================
def main():
    st.set_page_config(page_title="OCT Screening (MFA)", layout="wide")
    init_db()
    if not is_authed():
        page_login()
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["OCT Scan", "Logout"])
    if page == "Logout":
        clear_session(); st.rerun()
    else:
        page_oct_scan()

if __name__ == "__main__":
    main()
