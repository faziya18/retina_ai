
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
import numpy as np
import json, os, time
from pathlib import Path
from passlib.context import CryptContext
import pyotp

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_TITLE = "Diabetic Retinopathy & OCT – Secure App"
USERS_DB = ROOT_DIR / "users.json"
DEFAULT_DR_MODEL = ROOT_DIR / "dr_resnet50.pth"     # fundus model (provided)
DEFAULT_OCT_MODEL = ROOT_DIR / "oct_resnet50.pth"   # optional, if you have one

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -------------------------- Auth helpers --------------------------
def load_users():
    if not os.path.exists(USERS_DB):
        with open(USERS_DB, "w") as f:
            json.dump({}, f)
    with open(USERS_DB, "r") as f:
        return json.load(f)

def save_users(db):
    with open(USERS_DB, "w") as f:
        json.dump(db, f, indent=2)

def register_user(username, password):
    db = load_users()
    if username in db:
        return False, "User already exists."
    secret = pyotp.random_base32()
    db[username] = {
        "password_hash": pwd_context.hash(password),
        "totp_secret": secret
    }
    save_users(db)
    uri = pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name="DR Secure App")
    return True, uri

def validate_login(username, password, otp):
    db = load_users()
    if username not in db:
        return False, "No such user"
    if not pwd_context.verify(password, db[username]["password_hash"]):
        return False, "Wrong password"
    totp = pyotp.TOTP(db[username]["totp_secret"])
    if not totp.verify(otp, valid_window=1):
        return False, "Invalid OTP"
    return True, "OK"

# -------------------------- Models --------------------------
@st.cache_resource
def load_model(model_path, num_classes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(img, is_oct=False):
    # For OCT, convert to grayscale then to 3-channels for ResNet
    if is_oct:
        img = ImageOps.grayscale(img).convert("RGB")
    else:
        img = img.convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ])
    return tfm(img).unsqueeze(0)

@torch.inference_mode()
def predict_image(model, device, img_tensor, class_names):
    img_tensor = img_tensor.to(device)
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    idx = int(np.argmax(probs))
    return class_names[idx], probs

FUNDUS_CLASSES = ['Mild','Moderate','No_DR','Proliferate_DR','Severe']
OCT_CLASSES    = ['Normal','AMD','DME','Drusen','Other']  # placeholder

# -------------------------- UI --------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🔐", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False
if "user" not in st.session_state:
    st.session_state.user = None

st.title(APP_TITLE)

with st.expander("First-time setup: Create an account (MFA enabled)"):
    colr1, colr2 = st.columns(2)
    with colr1:
        new_user = st.text_input("New username", key="reg_user")
        new_pass = st.text_input("New password", type="password", key="reg_pass")
        if st.button("Register"):
            if not new_user or not new_pass:
                st.warning("Username and password required")
            else:
                ok, msg = register_user(new_user, new_pass)
                if ok:
                    st.success("Registered! Add this account to Google Authenticator (or any TOTP app):")
                    st.code(msg, language="text")
                    st.info("Open your Authenticator app → Add account → 'Enter a setup key' → paste the secret shown after 'secret=' in the URL above.")
                else:
                    st.error(msg)

st.subheader("Login")
col1, col2, col3 = st.columns(3)
with col1:
    username = st.text_input("Username", key="login_user")
with col2:
    password = st.text_input("Password", type="password", key="login_pass")
with col3:
    otp = st.text_input("6-digit OTP code", max_chars=6, key="login_otp")
login_clicked = st.button("Login")

if login_clicked:
    ok, msg = validate_login(username, password, otp)
    if ok:
        st.session_state.auth = True
        st.session_state.user = username
        st.success(f"Welcome, {username}!")
    else:
        st.session_state.auth = False
        st.error(msg)

if not st.session_state.auth:
    st.stop()

st.success(f"Logged in as {st.session_state.user} (MFA verified)")

tab1, tab2 = st.tabs(["🏥 Fundus DR Inference", "🧠 OCT Inference"])

with tab1:
    st.write("Upload a retinal **fundus** image to predict DR grade.")
    uploaded = st.file_uploader("Fundus image (JPG/PNG)", type=["jpg","jpeg","png"], key="fundus_up")
    model_path = st.text_input("DR model path", value=DEFAULT_DR_MODEL)
    if uploaded and st.button("Run Fundus Prediction"):
        try:
            img = Image.open(uploaded)
            model, device = load_model(model_path, num_classes=len(FUNDUS_CLASSES))
            x = preprocess_image(img, is_oct=False)
            label, probs = predict_image(model, device, x, FUNDUS_CLASSES)
            st.image(img, caption="Input Fundus", use_column_width=True)
            st.metric("Prediction", label)
            st.json({cls: float(p) for cls, p in zip(FUNDUS_CLASSES, probs)})
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.write("Upload an **OCT** B-scan image to classify. (Needs an OCT model if available.)")
    uploaded_oct = st.file_uploader("OCT image (JPG/PNG)", type=["jpg","jpeg","png"], key="oct_up")
    oct_model_path = st.text_input("OCT model path (optional)", value=DEFAULT_OCT_MODEL)
    if uploaded_oct and st.button("Run OCT Prediction"):
        try:
            img = Image.open(uploaded_oct)
            if os.path.exists(oct_model_path):
                model, device = load_model(oct_model_path, num_classes=len(OCT_CLASSES))
                x = preprocess_image(img, is_oct=True)
                label, probs = predict_image(model, device, x, OCT_CLASSES)
                st.image(img, caption="Input OCT", use_column_width=True)
                st.metric("Prediction", label)
                st.json({cls: float(p) for cls, p in zip(OCT_CLASSES, probs)})
            else:
                st.warning("No OCT model found. Showing preprocessed image only.")
                st.image(ImageOps.grayscale(Image.open(uploaded_oct)), caption="OCT (grayscale)", use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
