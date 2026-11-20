import os

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core import auth
from core.DiseaseModel import DiseaseModel
from core.gemini_advisor import get_ai_advice
from core.helper import prepare_symptoms_array
from core.ui import (PAGES, disease_card, doctor_card, esc, hero, inject_css, login_art, logo_src, page_header,
                     result_image, section, stats_strip, topbar)

load_dotenv()  # reads GEMINI_API_KEY (and optional ADMIN_EMAIL / ADMIN_PASSWORD) from the .env file

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedPredict – Multiple Disease Prediction",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ─── CONFIG / SECRETS ───────────────────────────────────────────────────────────
def get_secret(name: str, default: str = "") -> str:
    """Configuration comes from environment variables / the .env file (never from the UI)."""
    return os.environ.get(name, default).strip()


@st.cache_resource
def init_auth():
    auth.init_db(get_secret("ADMIN_EMAIL"), get_secret("ADMIN_PASSWORD"))
    return True


init_auth()

# ─── LOAD MODELS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return dict(
        diabetes=joblib.load("models/diabetes_model.sav"),
        heart=joblib.load("models/heart_disease_model.sav"),
        parkinson=joblib.load("models/parkinsons_model.sav"),
        lung=joblib.load("models/lung_cancer_model.sav"),
        breast=joblib.load("models/breast_cancer.sav"),
        chronic=joblib.load("models/chronic_model.sav"),
        hepatitis=joblib.load("models/hepititisc_model.sav"),
        liver=joblib.load("models/liver_model.sav"),
    )


_m = load_models()
diabetes_model, heart_model, parkinson_model = _m["diabetes"], _m["heart"], _m["parkinson"]
lung_cancer_model, breast_cancer_model = _m["lung"], _m["breast"]
chronic_disease_model, hepatitis_model, liver_model = _m["chronic"], _m["hepatitis"], _m["liver"]

# ─── SESSION STATE / NAVIGATION ─────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("page", "Home")
ss.setdefault("user", None)
ss.setdefault("after_login", None)
ss.setdefault("login_notice", "")

PREDICTION_PAGES = list(PAGES.keys())
USER_PAGES = ["My History", "Account"]
ADMIN_PAGES = ["Admin Dashboard"]
PROTECTED = set(PREDICTION_PAGES + USER_PAGES)


def go(page: str):
    """Navigate. Protected pages send signed-out visitors to Login and come back after sign-in."""
    user = ss.user
    if page in PROTECTED and not user:
        ss.after_login = page
        label = page.replace(" Prediction", "")
        ss.login_notice = f"Please sign in to open **{label}**."
        ss.page = "Login"
    elif page in ADMIN_PAGES and (not user or user["role"] != "admin"):
        ss.after_login = page
        ss.login_notice = "Administrator sign-in required."
        ss.page = "Login"
    else:
        ss.page = page
        if page not in ("Login", "Register"):
            ss.after_login = None  # user chose somewhere else; forget the pending target
    st.rerun()


def logout():
    ss.user = None
    ss.after_login = None
    ss.page = "Home"
    st.rerun()


# Safety net: a stale/blocked page in state never renders for a signed-out user
if ss.page in PROTECTED and not ss.user:
    ss.after_login, ss.page = ss.page, "Login"
if ss.page in ADMIN_PAGES and not (ss.user and ss.user["role"] == "admin"):
    ss.page = "Login" if not ss.user else "Home"

page = ss.page

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div class='side-brand'><img src='{logo_src()}'/><b>MedPredict</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;opacity:.65;margin-bottom:6px'>AI-powered disease screening</div>",
                unsafe_allow_html=True)

    def nav_button(label, target, key):
        if st.button(label, key=key, use_container_width=True, type="primary" if page == target else "secondary"):
            go(target)

    user = ss.user
    if user:
        role_label = "Administrator" if user["role"] == "admin" else (user.get("specialization") or "Doctor")
        prefix = "" if user["role"] == "admin" else "Dr. "
        st.markdown(f"<div class='side-user'>👤 <b>{prefix}{esc(user['full_name'])}</b><br>"
                    f"<span style='opacity:.7'>{esc(role_label)}</span></div>", unsafe_allow_html=True)
    nav_button("🏠  Home", "Home", "nav_home")
    nav_button("👩‍⚕️  Find a Doctor", "Find a Doctor", "nav_find")

    if user:
        st.markdown("<div class='side-label'>Screenings</div>", unsafe_allow_html=True)
        for p in PREDICTION_PAGES:
            nav_button(f"{PAGES[p]['icon']}  {PAGES[p]['short']}", p, f"nav_{p}")
        st.markdown("<div class='side-label'>Workspace</div>", unsafe_allow_html=True)
        if user["role"] == "admin":
            nav_button("🛡️  Admin Dashboard", "Admin Dashboard", "nav_admin")
        nav_button("🗂️  My History", "My History", "nav_hist")
        nav_button("⚙️  Account", "Account", "nav_acct")
        if st.button("🚪  Sign out", key="nav_logout", use_container_width=True):
            logout()
    else:
        st.markdown("<div class='side-label'>Account</div>", unsafe_allow_html=True)
        nav_button("🔐  Sign in", "Login", "nav_login")
        nav_button("🩺  Doctor registration", "Register", "nav_register")

    st.markdown("<hr style='opacity:.15'/><p style='text-align:center;opacity:.6;font-size:11px'>"
                "⚕️ For educational purposes only.<br>Always consult a doctor.</p>", unsafe_allow_html=True)


# ─── HELPERS ────────────────────────────────────────────────────────────────────
def log_prediction_safe(disease, patient_name, result):
    """Audit-log a prediction; never let a logging problem break the page."""
    try:
        if ss.user:
            auth.log_prediction(ss.user["id"], disease, patient_name, result)
    except Exception:
        pass


def show_ai_recommendations(disease_name, is_positive, patient_summary):
    """Calls Gemini and renders formatted health recommendations."""
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        st.info("💡 **AI Advisor is not configured.** The administrator can enable personalised diet, lifestyle "
                "& precaution recommendations by adding `GEMINI_API_KEY` to the `.env` file and restarting the app.")
        return
    st.markdown(
        f"<div class='ai-head'><b>✨ Gemini AI Health Advisor</b><br>"
        f"<span>Personalised guidance for <b>{disease_name}</b></span></div>", unsafe_allow_html=True)
    with st.spinner("🧠 Gemini is preparing personalised recommendations..."):
        advice = get_ai_advice(disease_name, is_positive, patient_summary, api_key,
                               model=get_secret("GEMINI_MODEL") or None)
    st.markdown(advice)


def show_result(name, positive_msg, negative_msg, prediction, positive_condition=1):
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    positive = prediction == positive_condition
    if positive:
        st.error(f"⚠️ {name + ' — ' if name else ''}{positive_msg}")
    else:
        st.success(f"✅ {name + ' — ' if name else ''}{negative_msg}")
    result_image(positive)
    log_prediction_safe(PAGES.get(ss.page, {}).get("short", ss.page), name, "Positive" if positive else "Negative")
    st.markdown("---")
    st.info("⚠️ This is an AI prediction. Please consult a certified medical professional for proper diagnosis.")


def calc_bmi(weight_kg, height_cm):
    if height_cm > 0:
        h_m = height_cm / 100.0
        return round(weight_kg / (h_m * h_m), 2)
    return 0.0


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#3498db"
    elif bmi < 25:
        return "Normal weight ✅", "#27ae60"
    elif bmi < 30:
        return "Overweight ⚠️", "#f39c12"
    else:
        return "Obese ❗", "#e74c3c"


def sign_in(user):
    ss.user = user
    target = ss.after_login
    ss.after_login = None
    ss.login_notice = ""
    if target and (target not in ADMIN_PAGES or user["role"] == "admin"):
        ss.page = target
    else:
        ss.page = "Admin Dashboard" if user["role"] == "admin" else "Home"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════════
# HOME / LANDING
# ══════════════════════════════════════════════════════════════════════════════════
if page == "Home":
    c_logo, c_d, c_a, c_b = st.columns([4, 1.4, 1.2, 1.4])
    with c_logo:
        topbar()
    with c_d:
        if st.button("Find a doctor", key="top_find", use_container_width=True):
            go("Find a Doctor")
    if ss.user:
        with c_b:
            if st.button("Open workspace", key="top_ws", use_container_width=True, type="primary"):
                go("Admin Dashboard" if ss.user["role"] == "admin" else "My History")
    else:
        with c_a:
            if st.button("Sign in", key="top_login", use_container_width=True):
                go("Login")
        with c_b:
            if st.button("Doctor sign-up", key="top_reg", use_container_width=True, type="primary"):
                go("Register")

    hero()
    stats_strip()

    section("Choose a screening", "Select a condition to open its prediction page"
            + ("" if ss.user else " — you'll be asked to sign in first."))
    pages_list = list(PAGES.keys())
    for row_start in range(0, len(pages_list), 3):
        cols = st.columns(3)
        for col, p in zip(cols, pages_list[row_start:row_start + 3]):
            with col:
                disease_card(p)
                if st.button(f"Open {PAGES[p]['short']} →", key=f"card_{p}", use_container_width=True,
                             type="primary" if p in ("Lung Cancer Prediction", "Breast Cancer Prediction") else "secondary"):
                    go(p)

    section("Need to consult a doctor?", "Browse our registered doctors and contact them by phone or email.")
    if st.button("👩‍⚕️ Browse the doctor directory →", key="home_find", type="primary"):
        go("Find a Doctor")

    section("How it works")
    s1, s2, s3 = st.columns(3)
    steps = [("1", "Sign in", "Doctors register in under a minute; administrators manage access."),
             ("2", "Enter patient details", "Pick a screening and fill in the clinical measurements."),
             ("3", "Review & advise", "Get the model's prediction plus Gemini-written lifestyle and follow-up guidance.")]
    for col, (n, t, d) in zip((s1, s2, s3), steps):
        col.markdown(f"<div class='step'><div class='n'>{n}</div><b>{t}</b><p>{d}</p></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-banner' style='margin-top:26px'>
    ⚠️ <strong>Disclaimer:</strong> This application is built for educational and research purposes only.
    Predictions made by this system should NOT be used as a substitute for professional medical advice.
    Always consult a licensed healthcare provider for medical decisions.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# FIND A DOCTOR (public directory)
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "Find a Doctor":
    topbar()
    st.markdown("## 👩‍⚕️ Find a Doctor")
    st.markdown("<div class='sec-sub'>Registered doctors who can be contacted for a consultation. "
                "Tap a phone number or email to reach them.</div>", unsafe_allow_html=True)
    directory = auth.list_directory()
    if not directory:
        st.info("No doctors are listed yet. Doctors can register from the **Doctor registration** page.")
    else:
        f1, f2 = st.columns([2, 1.4])
        q = f1.text_input("Search by name or hospital", placeholder="e.g. Sharma, City Hospital", key="doc_q")
        specs = sorted({d["specialization"] for d in directory if d["specialization"]})
        spec_f = f2.selectbox("Specialization", ["All"] + specs, key="doc_spec")
        shown = [d for d in directory
                 if (spec_f == "All" or d["specialization"] == spec_f)
                 and (not q.strip() or q.strip().lower() in f"{d['full_name']} {d['hospital'] or ''}".lower())]
        st.caption(f"{len(shown)} doctor(s) found")
        cols = st.columns(2)
        for i, d in enumerate(shown):
            with cols[i % 2]:
                doctor_card(d)
        if not shown:
            st.warning("No doctors match your search.")
    st.markdown("<div class='info-banner'>🚑 For medical emergencies, call your local emergency number "
                "immediately instead of using this directory.</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "Login":
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        login_art("Welcome back", ["Secure, role-based access for doctors and admins",
                                   "Nine screening models in one workspace",
                                   "Gemini AI guidance after every prediction"])
    with right:
        topbar()
        st.markdown("## Sign in")
        if ss.login_notice:
            st.info(ss.login_notice)
        role_choice = st.radio("Sign in as", ["Doctor", "Admin"], horizontal=True, key="login_role")
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@hospital.com")
            pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            user, err = auth.authenticate(email, pw, "admin" if role_choice == "Admin" else "doctor")
            if err:
                st.error(err)
            else:
                sign_in(user)
        if role_choice == "Doctor":
            st.markdown("<div style='margin-top:10px;color:#5b6b86'>New here? Create a doctor account.</div>",
                        unsafe_allow_html=True)
            if st.button("🩺 Register as a doctor", key="login_to_reg"):
                ss.page = "Register"
                st.rerun()
        else:
            st.caption("Administrator accounts are created by the system owner and cannot self-register.")

# ══════════════════════════════════════════════════════════════════════════════════
# DOCTOR REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "Register":
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        login_art("Join MedPredict", ["Free doctor account",
                                      "Run screenings and keep your own history",
                                      "Admin can disable access at any time"])
    with right:
        topbar()
        st.markdown("## Doctor registration")
        with st.form("register_form"):
            full_name = st.text_input("Full name", placeholder="Dr. Jane Doe")
            c1, c2 = st.columns(2)
            email = c1.text_input("Email", placeholder="you@hospital.com")
            phone = c2.text_input("Mobile number", placeholder="+91 98765 43210")
            c3, c4 = st.columns(2)
            license_no = c3.text_input("Medical registration no.")
            spec = c4.selectbox("Specialization", auth.SPECIALIZATIONS)
            hospital = st.text_input("Hospital / clinic (optional)")
            c5, c6 = st.columns(2)
            pw = c5.text_input("Password", type="password", help="8+ characters with a letter and a number")
            pw2 = c6.text_input("Confirm password", type="password")
            consent = st.checkbox("I agree that my name, specialization, mobile number and email are shown in the "
                                  "public **Find a Doctor** directory so people can contact me.")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            ok, msg = auth.register_doctor(full_name, email, pw, pw2, spec, license_no, hospital, phone, consent)
            if ok:
                st.success(msg)
                user, _ = auth.authenticate(email, pw, "doctor")
                if user:
                    sign_in(user)
            else:
                st.error(msg)
        if st.button("Already registered? Sign in", key="reg_to_login"):
            ss.page = "Login"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "Admin Dashboard":
    st.markdown("## 🛡️ Admin Dashboard")
    if ss.user.get("default_password"):
        st.warning("You are still using the default admin password. Open **Account** and change it now.")
    s = auth.stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registered doctors", s["doctors"])
    m2.metric("Active doctors", s["active"])
    m3.metric("Predictions run", s["predictions"])
    m4.metric("Positive results", s["positive"])

    tab_doc, tab_log, tab_dis = st.tabs(["👩‍⚕️ Doctors", "🗂️ All predictions", "📈 By disease"])
    with tab_doc:
        docs = auth.list_doctors()
        if not docs:
            st.info("No doctors have registered yet.")
        for d in docs:
            c1, c2, c3 = st.columns([4, 1.2, 1.2])
            active = d["status"] == "active"
            c1.markdown(f"<b>{esc(d['full_name'])}</b> · {esc(d['specialization'] or '—')}<br>"
                        f"<span style='color:#5b6b86;font-size:13px'>{esc(d['email'])} · 📞 {esc(d['phone'] or '—')} · "
                        f"Reg no. {esc(d['license_no'] or '—')}"
                        f"{' · ' + esc(d['hospital']) if d['hospital'] else ''} · joined {esc(d['created_at'][:10])} · "
                        f"{'listed in directory' if d['listed'] else 'not listed'}</span>",
                        unsafe_allow_html=True)
            c2.markdown("🟢 Active" if active else "🔴 Disabled")
            if c3.button("Disable" if active else "Enable", key=f"tog_{d['id']}", use_container_width=True):
                auth.set_status(d["id"], "disabled" if active else "active")
                st.rerun()
            st.markdown("<hr style='margin:6px 0;opacity:.15'/>", unsafe_allow_html=True)
    with tab_log:
        rows = auth.get_predictions()
        if rows:
            df = pd.DataFrame(rows).rename(columns={"created_at": "Time (UTC)", "disease": "Screening",
                                                    "patient_name": "Patient", "result": "Result", "doctor": "Doctor"})
            st.dataframe(df.drop(columns=["id"]), use_container_width=True, hide_index=True)
        else:
            st.info("No predictions have been run yet.")
    with tab_dis:
        bd = auth.by_disease()
        if bd:
            st.bar_chart(pd.DataFrame(bd).set_index("disease")["n"])
        else:
            st.info("Nothing to chart yet.")

# ══════════════════════════════════════════════════════════════════════════════════
# MY HISTORY
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "My History":
    st.markdown("## 🗂️ My prediction history")
    rows = auth.get_predictions(user_id=ss.user["id"])
    if rows:
        df = pd.DataFrame(rows).rename(columns={"created_at": "Time (UTC)", "disease": "Screening",
                                                "patient_name": "Patient", "result": "Result"})
        st.dataframe(df.drop(columns=["id", "doctor"]), use_container_width=True, hide_index=True)
    else:
        st.info("You haven't run any predictions yet. Pick a screening from the sidebar.")

# ══════════════════════════════════════════════════════════════════════════════════
# ACCOUNT
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "Account":
    st.markdown("## ⚙️ Account")
    u = ss.user
    st.markdown(f"**{u['full_name']}**  \n{u['email']} · {u['role'].title()}")
    if u["role"] == "doctor":
        st.markdown("### Contact details & directory listing")
        cur = auth.get_contact(u["id"])
        with st.form("contact_form"):
            ph = st.text_input("Mobile number", value=cur.get("phone") or "")
            hosp_in = st.text_input("Hospital / clinic", value=cur.get("hospital") or "")
            listed_in = st.checkbox("Show me in the public Find a Doctor directory (name, specialization, "
                                    "mobile, email)", value=bool(cur.get("listed")))
            save_c = st.form_submit_button("Save contact details")
        if save_c:
            ok, msg = auth.update_contact(u["id"], ph, hosp_in, listed_in)
            (st.success if ok else st.error)(msg)
    st.markdown("### Change password")
    with st.form("pw_form"):
        old = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        new2 = st.text_input("Confirm new password", type="password")
        go_btn = st.form_submit_button("Update password")
    if go_btn:
        if new != new2:
            st.error("New passwords do not match.")
        else:
            ok, msg = auth.change_password(u["id"], old, new)
            (st.success if ok else st.error)(msg)
            if ok:
                ss.user["default_password"] = 0


# ══════════════════════════════════════════════════════════════════════════════════
# PREDICTION PAGES (original forms, restyled)
# ══════════════════════════════════════════════════════════════════════════════════
elif page == 'Disease Prediction':
    page_header('Disease Prediction')

    with st.expander("ℹ️ How does this work?"):
        st.write("Select your symptoms from the dropdown below. Our XGBoost model trained on 132 diseases and 377 symptoms will predict the most likely disease along with its probability.")

    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    symptoms = st.multiselect('🩺 What are your symptoms?', options=disease_model.all_symptoms,
                               help="Select all symptoms you are experiencing")

    if symptoms:
        st.info(f"📋 You have selected **{len(symptoms)}** symptom(s): {', '.join(symptoms)}")

    X = prepare_symptoms_array(symptoms)

    if st.button('🔍 Predict Disease'):
        if not symptoms:
            st.warning("⚠️ Please select at least one symptom before predicting.")
        else:
            prediction, prob = disease_model.predict(X)
            st.markdown(f"## 🎯 Predicted Disease: **{prediction}**")
            st.metric("Confidence", f"{prob*100:.2f}%")

            tab1, tab2 = st.tabs(["📖 Description", "💊 Precautions"])
            with tab1:
                st.write(disease_model.describe_predicted_disease())
            with tab2:
                precautions = disease_model.predicted_disease_precautions()
                for i in range(4):
                    st.write(f"**{i+1}.** {precautions[i]}")
            log_prediction_safe('Symptom Checker', '', f'Predicted: {prediction}')
            patient_summary = f"Symptoms reported: {', '.join(symptoms)}"
            show_ai_recommendations(str(prediction), True, patient_summary)

elif page == 'Diabetes Prediction':
    page_header('Diabetes Prediction')

    with st.expander("ℹ️ About Diabetes Prediction"):
        st.write("This model uses the PIMA Indian Diabetes Dataset. It predicts the likelihood of diabetes based on medical measurements like glucose, blood pressure, and BMI.")

    name = st.text_input("👤 Patient Name:")
    st.markdown("### 📋 Enter Medical Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("🤰 Number of Pregnancies", min_value=0, max_value=20, value=0)
    with col2:
        Glucose = st.number_input("🩸 Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    with col3:
        BloodPressure = st.number_input("💓 Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)

    with col1:
        SkinThickness = st.number_input("📏 Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    with col2:
        Insulin = st.number_input("💊 Insulin Level (μU/mL)", min_value=0, max_value=900, value=80)

    # ── BMI AUTO-CALCULATOR ──────────────────────────────────────────────────────
    st.markdown("### ⚖️ BMI Calculator")
    st.markdown("<div class='info-banner'>Enter your <b>Weight</b> and <b>Height</b> to auto-calculate your BMI.</div>", unsafe_allow_html=True)
    bmi_col1, bmi_col2, bmi_col3 = st.columns(3)
    with bmi_col1:
        weight_kg = st.number_input("⚖️ Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5)
    with bmi_col2:
        height_cm = st.number_input("📐 Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
    with bmi_col3:
        BMI = calc_bmi(weight_kg, height_cm)
        category, color = bmi_category(BMI)
        st.markdown(f"<div class='bmi-box' style='background:linear-gradient(135deg,{color},{color}99);'>BMI: {BMI}<br><small>{category}</small></div>", unsafe_allow_html=True)
    # ────────────────────────────────────────────────────────────────────────────

    col1, col2 = st.columns(2)
    with col1:
        DiabetesPedigreefunction = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    with col2:
        Age = st.number_input("📅 Age", min_value=1, max_value=120, value=25)

    st.markdown("---")
    st.markdown("### 📊 Input Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Glucose", f"{Glucose} mg/dL")
    m2.metric("Blood Pressure", f"{BloodPressure} mm Hg")
    m3.metric("BMI", f"{BMI}")
    m4.metric("Age", f"{int(Age)}")

    if st.button("🔍 Predict Diabetes"):
        prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])
        show_result(name,
                    "It seems like you may be Diabetic. Please consult a doctor.",
                    "Great news! You are not Diabetic.",
                    prediction[0])
        patient_summary = (f"Glucose: {Glucose} mg/dL, BMI: {BMI}, Age: {int(Age)}, "
                           f"Blood Pressure: {BloodPressure} mm Hg, Pregnancies: {Pregnancies}, "
                           f"Insulin: {Insulin} μU/mL, Pedigree Function: {DiabetesPedigreefunction:.2f}")
        show_ai_recommendations("Diabetes", prediction[0] == 1, patient_summary)

elif page == 'Heart Disease Prediction':
    page_header('Heart Disease Prediction')

    with st.expander("ℹ️ About Heart Disease Prediction"):
        st.write("This model predicts the presence of heart disease using 13 clinical features including age, cholesterol levels, ECG results, and exercise-induced parameters.")

    name = st.text_input("👤 Patient Name:")
    st.markdown("### 📋 Enter Clinical Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("📅 Age", min_value=1, max_value=120, value=45)
    with col2:
        sex = 0
        sex_val = st.selectbox("⚧ Gender", ["Male", "Female"])
        sex = 1 if sex_val == "Male" else 0
    with col3:
        cp = 0
        cp_val = st.selectbox("💔 Chest Pain Type",
                               ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        cp = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp_val)

    with col1:
        trestbps = st.number_input("💓 Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    with col2:
        chol = st.number_input("🩸 Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    with col3:
        restecg = 0
        ecg_val = st.selectbox("📈 Resting ECG",
                                ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(ecg_val)

    with col1:
        thalach = st.number_input("💗 Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
    with col2:
        oldpeak = st.number_input("📉 ST Depression (Exercise vs Rest)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with col3:
        slope_val = st.selectbox("📐 Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        slope = ["Upsloping", "Flat", "Downsloping"].index(slope_val)

    with col1:
        ca = st.number_input("🔬 Major Vessels Colored (0–3)", min_value=0, max_value=3, value=0)
    with col2:
        thal_val = st.selectbox("🧬 Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal_val)
    with col3:
        exang = 1 if st.checkbox("🏃 Exercise Induced Angina") else 0

    with col1:
        fbs = 1 if st.checkbox("🍬 Fasting Blood Sugar > 120 mg/dL") else 0

    st.markdown("---")
    st.markdown("### 📊 Input Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Age", f"{int(age)}")
    m2.metric("Cholesterol", f"{chol} mg/dL")
    m3.metric("Resting BP", f"{trestbps} mm Hg")
    m4.metric("Max Heart Rate", f"{int(thalach)}")

    if st.button("🔍 Predict Heart Disease"):
        prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        show_result(name,
                    "It seems like you may have Heart Disease. Please consult a cardiologist.",
                    "Great news! No signs of Heart Disease detected.",
                    prediction[0])
        patient_summary = (f"Age: {int(age)}, Gender: {sex_val}, Cholesterol: {chol} mg/dL, "
                           f"Resting BP: {trestbps} mm Hg, Max Heart Rate: {int(thalach)}, "
                           f"Chest Pain Type: {cp_val}, ST Depression: {oldpeak}, Fasting BS>120: {'Yes' if fbs else 'No'}")
        show_ai_recommendations("Heart Disease", prediction[0] == 1, patient_summary)

elif page == 'Parkinson Prediction':
    page_header('Parkinson Prediction')

    with st.expander("ℹ️ About Parkinson's Prediction"):
        st.write("This model uses voice measurement features (MDVP, jitter, shimmer) to detect Parkinson's disease. These measurements capture subtle vocal changes caused by the disease.")

    name = st.text_input("👤 Patient Name:")
    st.markdown("### 🎙️ Voice Measurement Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        MDVP      = st.number_input("MDVP:Fo(Hz)", min_value=0.0, value=119.99)
        MDVPFIZ   = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, value=157.30)
        MDVPFLO   = st.number_input("MDVP:Flo(Hz)", min_value=0.0, value=74.99)
        MDVPJITTER= st.number_input("MDVP:Jitter(%)", min_value=0.0, value=0.00784, format="%.5f")
        MDVPJitterAbs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, value=0.00007, format="%.5f")
        MDVPRAP   = st.number_input("MDVP:RAP", min_value=0.0, value=0.0037, format="%.5f")
        MDVPPPQ   = st.number_input("MDVP:PPQ", min_value=0.0, value=0.0054, format="%.5f")
        JitterDDP = st.number_input("Jitter:DDP", min_value=0.0, value=0.0111, format="%.5f")

    with col2:
        MDVPShimmer    = st.number_input("MDVP:Shimmer", min_value=0.0, value=0.0299, format="%.5f")
        MDVPShimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, value=0.282)
        Shimmer_APQ3   = st.number_input("Shimmer:APQ3", min_value=0.0, value=0.0160, format="%.5f")
        ShimmerAPQ5    = st.number_input("Shimmer:APQ5", min_value=0.0, value=0.0218, format="%.5f")
        MDVP_APQ       = st.number_input("MDVP:APQ", min_value=0.0, value=0.0300, format="%.5f")
        ShimmerDDA     = st.number_input("Shimmer:DDA", min_value=0.0, value=0.0480, format="%.5f")
        NHR            = st.number_input("NHR", min_value=0.0, value=0.0117, format="%.5f")
        HNR            = st.number_input("HNR", min_value=0.0, value=21.03)

    with col3:
        RPDE    = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.498)
        DFA     = st.number_input("DFA", min_value=0.0, max_value=1.0, value=0.718)
        spread1 = st.number_input("spread1", value=-5.684)
        spread2 = st.number_input("spread2", value=0.227)
        D2      = st.number_input("D2", min_value=0.0, value=2.301)
        PPE     = st.number_input("PPE", min_value=0.0, value=0.284)

    if st.button("🔍 Predict Parkinson's Disease"):
        prediction = parkinson_model.predict([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs,
                                               MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,
                                               MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5,
                                               MDVP_APQ, ShimmerDDA, NHR, HNR,
                                               RPDE, DFA, spread1, spread2, D2, PPE]])
        show_result(name,
                    "Signs of Parkinson's Disease detected. Please consult a neurologist.",
                    "No signs of Parkinson's Disease detected.",
                    prediction[0])
        patient_summary = (f"HNR: {HNR}, NHR: {NHR:.5f}, RPDE: {RPDE}, DFA: {DFA}, PPE: {PPE}, "
                           f"MDVP:Fo(Hz): {MDVP}, spread1: {spread1:.3f}, D2: {D2}")
        show_ai_recommendations("Parkinson's Disease", prediction[0] == 1, patient_summary)

elif page == 'Liver Prediction':
    page_header('Liver Prediction')

    with st.expander("ℹ️ About Liver Disease Prediction"):
        st.write("This model analyzes liver enzyme levels and bilirubin values to predict liver disease. Early detection can prevent severe liver damage.")

    name = st.text_input("👤 Patient Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex_val = st.selectbox("⚧ Gender", ["Male", "Female"])
        Sex = 0 if Sex_val == "Male" else 1
    with col2:
        age = st.number_input("📅 Age", min_value=1, max_value=120, value=35)
    with col3:
        Total_Bilirubin = st.number_input("🩸 Total Bilirubin", min_value=0.0, value=0.9, step=0.1)

    with col1:
        Direct_Bilirubin = st.number_input("🩸 Direct Bilirubin", min_value=0.0, value=0.2, step=0.1)
    with col2:
        Alkaline_Phosphotase = st.number_input("🧪 Alkaline Phosphatase", min_value=0, value=200)
    with col3:
        Alamine_Aminotransferase = st.number_input("🧪 Alamine Aminotransferase", min_value=0, value=25)

    with col1:
        Aspartate_Aminotransferase = st.number_input("🧪 Aspartate Aminotransferase", min_value=0, value=30)
    with col2:
        Total_Protiens = st.number_input("🧪 Total Proteins", min_value=0.0, value=7.0, step=0.1)
    with col3:
        Albumin = st.number_input("🧪 Albumin", min_value=0.0, value=3.5, step=0.1)

    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("🧪 Albumin & Globulin Ratio", min_value=0.0, value=1.0, step=0.1)

    if st.button("🔍 Predict Liver Disease"):
        prediction = liver_model.predict([[Sex, age, Total_Bilirubin, Direct_Bilirubin,
                                           Alkaline_Phosphotase, Alamine_Aminotransferase,
                                           Aspartate_Aminotransferase, Total_Protiens,
                                           Albumin, Albumin_and_Globulin_Ratio]])
        show_result(name,
                    "Signs of Liver Disease detected. Please consult a hepatologist.",
                    "No signs of Liver Disease detected.",
                    prediction[0])
        patient_summary = (f"Age: {int(age)}, Gender: {Sex_val}, Total Bilirubin: {Total_Bilirubin}, "
                           f"Direct Bilirubin: {Direct_Bilirubin}, Albumin: {Albumin}, "
                           f"Total Proteins: {Total_Protiens}, Alk. Phosphatase: {Alkaline_Phosphotase}")
        show_ai_recommendations("Liver Disease", prediction[0] == 1, patient_summary)

elif page == 'Hepatitis Prediction':
    page_header('Hepatitis Prediction')

    with st.expander("ℹ️ About Hepatitis Prediction"):
        st.write("This model uses blood test parameters to predict Hepatitis risk. The features include liver enzyme levels and protein markers.")

    name = st.text_input("👤 Patient Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("📅 Age", min_value=1, max_value=120, value=35)
    with col2:
        sex_val = st.selectbox("⚧ Gender", ["Male", "Female"])
        sex = 1 if sex_val == "Male" else 2
    with col3:
        total_bilirubin = st.number_input("🩸 Total Bilirubin (ALB)", min_value=0.0, value=38.0, step=0.1)

    with col1:
        direct_bilirubin = st.number_input("🩸 Direct Bilirubin (ALP)", min_value=0.0, value=52.5, step=0.1)
    with col2:
        alkaline_phosphatase = st.number_input("🧪 Alkaline Phosphatase (ALT)", min_value=0.0, value=7.7, step=0.1)
    with col3:
        alamine_aminotransferase = st.number_input("🧪 Alamine Aminotransferase (AST)", min_value=0.0, value=22.1, step=0.1)

    with col1:
        aspartate_aminotransferase = st.number_input("🧪 Aspartate Aminotransferase (BIL)", min_value=0.0, value=7.5, step=0.1)
    with col2:
        total_proteins = st.number_input("🧪 Total Proteins (CHE)", min_value=0.0, value=6.93, step=0.1)
    with col3:
        albumin = st.number_input("🧪 Albumin (CHOL)", min_value=0.0, value=3.23, step=0.1)

    with col1:
        albumin_and_globulin_ratio = st.number_input("🧪 Albumin & Globulin Ratio (CREA)", min_value=0.0, value=106.0, step=1.0)
    with col2:
        your_ggt_value = st.number_input("🧪 GGT Value", min_value=0.0, value=12.1, step=0.1)
    with col3:
        your_prot_value = st.number_input("🧪 PROT Value", min_value=0.0, value=69.0, step=0.1)

    if st.button("🔍 Predict Hepatitis"):
        user_data = pd.DataFrame({
            'Age': [age], 'Sex': [sex],
            'ALB': [total_bilirubin], 'ALP': [direct_bilirubin],
            'ALT': [alkaline_phosphatase], 'AST': [alamine_aminotransferase],
            'BIL': [aspartate_aminotransferase], 'CHE': [total_proteins],
            'CHOL': [albumin], 'CREA': [albumin_and_globulin_ratio],
            'GGT': [your_ggt_value], 'PROT': [your_prot_value]
        })
        prediction = hepatitis_model.predict(user_data)
        show_result(name,
                    "Signs of Hepatitis detected. Please consult a doctor immediately.",
                    "No signs of Hepatitis detected.",
                    prediction[0])
        patient_summary = (f"Age: {int(age)}, Gender: {sex_val}, ALB: {total_bilirubin}, ALP: {direct_bilirubin}, "
                           f"ALT: {alkaline_phosphatase}, AST: {alamine_aminotransferase}, GGT: {your_ggt_value}, PROT: {your_prot_value}")
        show_ai_recommendations("Hepatitis", prediction[0] == 1, patient_summary)

elif page == 'Lung Cancer Prediction':
    page_header('Lung Cancer Prediction')

    lung_cancer_data = pd.read_csv('data/lung_cancer.csv')
    lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

    with st.expander("ℹ️ About Lung Cancer Prediction"):
        st.write("This model uses lifestyle and symptom-based features to screen for lung cancer risk. It does not replace CT scans or medical biopsy.")

    name = st.text_input("👤 Patient Name:")
    col1, col2, col3 = st.columns(3)

    def yn_select(label, col):
        return col.selectbox(label, ['NO', 'YES'])

    with col1:
        gender = st.selectbox("⚧ Gender:", lung_cancer_data['GENDER'].unique())
    with col2:
        age = st.number_input("📅 Age", min_value=1, max_value=120, value=45)
    with col3:
        smoking = yn_select("🚬 Smoking:", col3)

    with col1:
        yellow_fingers = yn_select("🤞 Yellow Fingers:", col1)
    with col2:
        anxiety = yn_select("😰 Anxiety:", col2)
    with col3:
        peer_pressure = yn_select("👥 Peer Pressure:", col3)

    with col1:
        chronic_disease = yn_select("🏥 Chronic Disease:", col1)
    with col2:
        fatigue = yn_select("😴 Fatigue:", col2)
    with col3:
        allergy = yn_select("🤧 Allergy:", col3)

    with col1:
        wheezing = yn_select("💨 Wheezing:", col1)
    with col2:
        alcohol_consuming = yn_select("🍺 Alcohol Consuming:", col2)
    with col3:
        coughing = yn_select("😷 Coughing:", col3)

    with col1:
        shortness_of_breath = yn_select("😤 Shortness of Breath:", col1)
    with col2:
        swallowing_difficulty = yn_select("🍽️ Swallowing Difficulty:", col2)
    with col3:
        chest_pain = yn_select("💔 Chest Pain:", col3)

    if st.button("🔍 Predict Lung Cancer"):
        user_data = pd.DataFrame({
            'GENDER': [gender], 'AGE': [age],
            'SMOKING': [smoking], 'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety], 'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease], 'FATIGUE': [fatigue],
            'ALLERGY': [allergy], 'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming], 'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty], 'CHESTPAIN': [chest_pain]
        })
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)
        user_data.columns = user_data.columns.str.strip()
        numeric_cols = ['AGE','FATIGUE','ALLERGY','ALCOHOLCONSUMING','COUGHING','SHORTNESSOFBREATH']
        user_data[numeric_cols] = user_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        prediction = lung_cancer_model.predict(user_data)
        show_result(name,
                    "Risk of Lung Cancer detected. Please consult an oncologist immediately.",
                    "No significant risk of Lung Cancer detected.",
                    prediction[0], positive_condition='YES')
        patient_summary = (f"Age: {int(age)}, Gender: {gender}, Smoking: {smoking}, Alcohol: {alcohol_consuming}, "
                           f"Chest Pain: {chest_pain}, Shortness of Breath: {shortness_of_breath}, "
                           f"Chronic Disease: {chronic_disease}, Coughing: {coughing}, Wheezing: {wheezing}")
        show_ai_recommendations("Lung Cancer", prediction[0] == 'YES', patient_summary)

elif page == 'Chronic Kidney Prediction':
    page_header('Chronic Kidney Prediction')

    with st.expander("ℹ️ About Chronic Kidney Disease Prediction"):
        st.write("CKD is a long-term condition where the kidneys don't work as well as they should. This model uses 24 clinical features to predict CKD risk.")

    name = st.text_input("👤 Patient Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("📅 Age", 1, 100, 25)
        al  = st.slider("🧪 Albumin", 0, 5, 0)
        rbc = 1 if st.selectbox("🔴 Red Blood Cells", ["Normal", "Abnormal"]) == "Normal" else 0
        bgr = st.slider("🩸 Blood Glucose Random", 50, 500, 120)
        sod = st.slider("🧂 Sodium", 100, 200, 140)
        pcv = st.slider("🔬 Packed Cell Volume", 20, 60, 40)
        htn = 1 if st.selectbox("💊 Hypertension", ["Yes", "No"]) == "Yes" else 0
        appet = 1 if st.selectbox("🍽️ Appetite", ["Good", "Poor"]) == "Good" else 0

    with col2:
        bp  = st.slider("💓 Blood Pressure", 50, 200, 80)
        su  = st.slider("🍬 Sugar", 0, 5, 0)
        pc  = 1 if st.selectbox("🔬 Pus Cells", ["Normal", "Abnormal"]) == "Normal" else 0
        bu  = st.slider("🧪 Blood Urea", 10, 400, 50)
        pot = st.slider("⚡ Potassium", 2, 10, 4)
        wc  = st.slider("🩺 White Blood Cell Count", 2000, 20000, 8000)
        dm  = 1 if st.selectbox("💉 Diabetes Mellitus", ["Yes", "No"]) == "Yes" else 0
        pe  = 1 if st.selectbox("🦵 Pedal Edema", ["Yes", "No"]) == "Yes" else 0

    with col3:
        sg   = st.slider("💧 Specific Gravity", 1.000, 1.050, 1.020, step=0.001, format="%.3f")
        pcc  = 1 if st.selectbox("🔬 Pus Cell Clumps", ["Present", "Not Present"]) == "Present" else 0
        ba   = 1 if st.selectbox("🦠 Bacteria", ["Present", "Not Present"]) == "Present" else 0
        sc   = st.slider("🧪 Serum Creatinine", 0, 20, 1)
        hemo = st.slider("🩸 Hemoglobin", 3, 20, 13)
        rc   = st.slider("🔴 Red Blood Cell Count", 2, 10, 5)
        cad  = 1 if st.selectbox("❤️ Coronary Artery Disease", ["Yes", "No"]) == "Yes" else 0
        ane  = 1 if st.selectbox("🩺 Anemia", ["Yes", "No"]) == "Yes" else 0

    st.markdown("---")
    st.markdown("### 📊 Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Age", age)
    m2.metric("Blood Pressure", f"{bp} mm Hg")
    m3.metric("Hemoglobin", f"{hemo} g/dL")
    m4.metric("Serum Creatinine", sc)

    if st.button("🔍 Predict Chronic Kidney Disease"):
        user_input = pd.DataFrame({
            'age':[age],'bp':[bp],'sg':[sg],'al':[al],'su':[su],
            'rbc':[rbc],'pc':[pc],'pcc':[pcc],'ba':[ba],'bgr':[bgr],
            'bu':[bu],'sc':[sc],'sod':[sod],'pot':[pot],'hemo':[hemo],
            'pcv':[pcv],'wc':[wc],'rc':[rc],'htn':[htn],'dm':[dm],
            'cad':[cad],'appet':[appet],'pe':[pe],'ane':[ane]
        })
        prediction = chronic_disease_model.predict(user_input)
        show_result(name,
                    "Signs of Chronic Kidney Disease detected. Please consult a nephrologist.",
                    "No signs of Chronic Kidney Disease detected.",
                    prediction[0])
        patient_summary = (f"Age: {age}, Blood Pressure: {bp} mm Hg, Hemoglobin: {hemo} g/dL, "
                           f"Serum Creatinine: {sc}, Potassium: {pot}, Sodium: {sod}, "
                           f"Blood Glucose: {bgr} mg/dL, Hypertension: {'Yes' if htn else 'No'}, Diabetes: {'Yes' if dm else 'No'}")
        show_ai_recommendations("Chronic Kidney Disease", prediction[0] == 1, patient_summary)

elif page == 'Breast Cancer Prediction':
    page_header('Breast Cancer Prediction')

    with st.expander("ℹ️ About Breast Cancer Prediction"):
        st.write("This model uses tumor cell features (radius, texture, perimeter, area, smoothness, etc.) computed from breast mass images to classify tumors as benign or malignant.")

    name = st.text_input("👤 Patient Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.slider("Radius Mean", 6.0, 30.0, 15.0)
        texture_mean = st.slider("Texture Mean", 9.0, 40.0, 20.0)
        perimeter_mean = st.slider("Perimeter Mean", 43.0, 190.0, 90.0)
        fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.01, 0.1, 0.05)
        radius_se = st.slider("Radius SE", 0.1, 3.0, 1.0)
        texture_se = st.slider("Texture SE", 0.2, 2.0, 1.0)
        symmetry_se = st.slider("Symmetry SE", 0.1, 1.0, 0.5)
        fractal_dimension_se = st.slider("Fractal Dimension SE", 0.01, 0.1, 0.05)
        concavity_worst = st.slider("Concavity Worst", 0.0, 0.8, 0.4)
        concave_points_worst = st.slider("Concave Points Worst", 0.0, 0.2, 0.1)
        symmetry_worst = st.slider("Symmetry Worst", 0.1, 1.0, 0.5)

    with col2:
        area_mean = st.slider("Area Mean", 143.0, 2501.0, 750.0)
        smoothness_mean = st.slider("Smoothness Mean", 0.05, 0.25, 0.1)
        compactness_mean = st.slider("Compactness Mean", 0.02, 0.3, 0.15)
        perimeter_se = st.slider("Perimeter SE", 1.0, 30.0, 10.0)
        area_se = st.slider("Area SE", 6.0, 500.0, 150.0)
        smoothness_se = st.slider("Smoothness SE", 0.001, 0.03, 0.01)
        radius_worst = st.slider("Radius Worst", 7.0, 40.0, 20.0)
        texture_worst = st.slider("Texture Worst", 12.0, 50.0, 25.0)
        perimeter_worst = st.slider("Perimeter Worst", 50.0, 250.0, 120.0)
        fractal_dimension_worst = st.slider("Fractal Dimension Worst", 0.01, 0.2, 0.1)

    with col3:
        concavity_mean = st.slider("Concavity Mean", 0.0, 0.5, 0.2)
        concave_points_mean = st.slider("Concave Points Mean", 0.0, 0.2, 0.1)
        symmetry_mean = st.slider("Symmetry Mean", 0.1, 1.0, 0.5)
        compactness_se = st.slider("Compactness SE", 0.002, 0.2, 0.1)
        concavity_se = st.slider("Concavity SE", 0.0, 0.05, 0.02)
        concave_points_se = st.slider("Concave Points SE", 0.0, 0.03, 0.01)
        area_worst = st.slider("Area Worst", 185.0, 4250.0, 1500.0)
        smoothness_worst = st.slider("Smoothness Worst", 0.07, 0.3, 0.15)
        compactness_worst = st.slider("Compactness Worst", 0.03, 0.6, 0.3)

    if st.button("🔍 Predict Breast Cancer"):
        user_input = pd.DataFrame({
            'radius_mean': [radius_mean], 'texture_mean': [texture_mean],
            'perimeter_mean': [perimeter_mean], 'area_mean': [area_mean],
            'smoothness_mean': [smoothness_mean], 'compactness_mean': [compactness_mean],
            'concavity_mean': [concavity_mean], 'concave points_mean': [concave_points_mean],
            'symmetry_mean': [symmetry_mean], 'fractal_dimension_mean': [fractal_dimension_mean],
            'radius_se': [radius_se], 'texture_se': [texture_se],
            'perimeter_se': [perimeter_se], 'area_se': [area_se],
            'smoothness_se': [smoothness_se], 'compactness_se': [compactness_se],
            'concavity_se': [concavity_se], 'concave points_se': [concave_points_se],
            'symmetry_se': [symmetry_se], 'fractal_dimension_se': [fractal_dimension_se],
            'radius_worst': [radius_worst], 'texture_worst': [texture_worst],
            'perimeter_worst': [perimeter_worst], 'area_worst': [area_worst],
            'smoothness_worst': [smoothness_worst], 'compactness_worst': [compactness_worst],
            'concavity_worst': [concavity_worst], 'concave points_worst': [concave_points_worst],
            'symmetry_worst': [symmetry_worst], 'fractal_dimension_worst': [fractal_dimension_worst],
        })
        prediction = breast_cancer_model.predict(user_input)
        show_result(name,
                    "The model predicts a Malignant tumor. Please consult an oncologist immediately.",
                    "The model predicts a Benign tumor. No cancer detected.",
                    prediction[0])
        patient_summary = (f"Radius Mean: {radius_mean}, Texture Mean: {texture_mean}, Area Mean: {area_mean}, "
                           f"Perimeter Mean: {perimeter_mean}, Concavity Mean: {concavity_mean}, "
                           f"Concave Points Mean: {concave_points_mean}, Smoothness Mean: {smoothness_mean}")
        show_ai_recommendations("Breast Cancer", prediction[0] == 1, patient_summary)

