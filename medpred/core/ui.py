"""UI helpers: theme CSS, embedded local images, hero, disease cards, page headers."""
import base64
import functools
import html
import os
import re
from urllib.parse import quote

import streamlit as st

ASSETS = "assets"

# page name -> metadata. 'key' = banner image in assets/, order = order on the landing page
PAGES = {
    "Lung Cancer Prediction":    dict(key="lung",      icon="🫁", short="Lung Cancer",
                                      desc="Lifestyle and symptom-based lung cancer risk screening."),
    "Breast Cancer Prediction":  dict(key="breast",    icon="🎀", short="Breast Cancer",
                                      desc="Tumour cell features classified as benign or malignant."),
    "Disease Prediction":        dict(key="general",   icon="🔬", short="Symptom Checker",
                                      desc="Pick symptoms; XGBoost ranks the most likely disease."),
    "Heart Disease Prediction":  dict(key="heart",     icon="❤️", short="Heart Disease",
                                      desc="Cardiovascular risk from 13 clinical features."),
    "Diabetes Prediction":       dict(key="diabetes",  icon="💉", short="Diabetes",
                                      desc="Diabetes risk from glucose, BMI, insulin and more."),
    "Parkinson Prediction":      dict(key="parkinson", icon="🧠", short="Parkinson's",
                                      desc="Voice-measurement analysis for Parkinson's disease."),
    "Liver Prediction":          dict(key="liver",     icon="🟤", short="Liver Disease",
                                      desc="Liver enzyme and bilirubin panel analysis."),
    "Hepatitis Prediction":      dict(key="hepatitis", icon="🩺", short="Hepatitis",
                                      desc="Hepatitis risk from routine blood-test markers."),
    "Chronic Kidney Prediction": dict(key="kidney",    icon="🫘", short="Chronic Kidney",
                                      desc="24-feature kidney function assessment."),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea, select { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1200px; }
.stApp { background: #f4f7fb; }

/* sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1b3f 0%,#0e2a5c 55%,#0f4c6b 100%); }
section[data-testid="stSidebar"] * { color: #e6eefc !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent; border: 1px solid transparent; text-align: left; justify-content: flex-start;
    border-radius: 10px; padding: 8px 12px; font-weight: 500; box-shadow: none; width: 100%; }
section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,.10); transform: none; box-shadow: none; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: rgba(45,212,191,.22); border-color: rgba(45,212,191,.55); }
section[data-testid="stSidebar"] input { color: #0b1b3f !important; }
.side-brand { display:flex; align-items:center; gap:10px; margin: 4px 0 2px; }
.side-brand img { width:38px; height:38px; border-radius:10px; }
.side-brand b { font-size: 19px; letter-spacing:.2px; }
.side-user { background: rgba(255,255,255,.08); border-radius: 12px; padding: 10px 12px; margin: 10px 0; font-size: 13px; }
.side-label { font-size: 11px; letter-spacing: 1.2px; text-transform: uppercase; opacity:.6; margin: 14px 0 4px 4px; }

/* headings */
h1, h2, h3 { color: #0b1b3f !important; letter-spacing: -.3px; }
h1 { font-weight: 800 !important; }

/* buttons */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 12px; font-weight: 600; padding: .55rem 1.1rem; border: 1px solid #d5dfee;
    transition: transform .15s, box-shadow .15s; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(14,116,144,.18); border-color:#0e7490; color:#0e7490; }
div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg,#0e7490,#2563eb); color: #fff; border: none; }
div.stButton > button[kind="primary"]:hover { color:#fff; box-shadow: 0 8px 22px rgba(37,99,235,.35); }

/* inputs */
div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 10px !important; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e3ebf6; border-left: 4px solid #0e7490; border-radius: 12px; padding: 12px 16px; box-shadow: 0 2px 10px rgba(11,27,63,.05); }
div[data-testid="stMetric"] * { color:#0b1b3f !important; }
div[data-testid="stExpander"] { background:#fff; border-radius: 12px; border: 1px solid #e3ebf6; }

/* hero */
.hero { position: relative; border-radius: 24px; overflow: hidden; min-height: 330px; box-shadow: 0 18px 50px rgba(11,27,63,.25); }
.hero img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }
.hero .txt { position: relative; padding: 52px 48px; max-width: 640px; color:#fff; }
.hero .pill { display:inline-block; background:rgba(45,212,191,.18); border:1px solid rgba(45,212,191,.5); color:#99f6e4; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:600; letter-spacing:.6px; }
.hero h1 { color:#fff !important; font-size: 40px; line-height:1.12; margin: 14px 0 10px; }
.hero p { font-size: 16.5px; opacity:.88; margin:0; line-height:1.55; }
.strip { display:flex; gap:14px; flex-wrap:wrap; margin: 18px 0 6px; }
.strip div { flex:1; min-width:160px; background:#fff; border-radius:14px; padding:14px 18px; border:1px solid #e3ebf6; box-shadow:0 2px 10px rgba(11,27,63,.05); }
.strip b { display:block; font-size:22px; color:#0e7490; }
.strip span { font-size:13px; color:#5b6b86; }

/* disease cards */
.dcard { background:#fff; border-radius:18px; overflow:hidden; border:1px solid #e3ebf6; box-shadow:0 4px 16px rgba(11,27,63,.07); transition: transform .18s, box-shadow .18s; margin-bottom: 8px; }
.dcard:hover { transform: translateY(-3px); box-shadow: 0 14px 32px rgba(11,27,63,.16); }
.dcard img { width:100%; height:130px; object-fit:cover; object-position: 78% center; display:block; }
.dcard .body { padding: 14px 16px 12px; }
.dcard h4 { margin:0 0 4px; color:#0b1b3f; font-size:17px; }
.dcard p { margin:0; color:#5b6b86; font-size:13.5px; line-height:1.45; min-height: 40px; }

.sec-title { font-size: 24px; font-weight: 800; color:#0b1b3f; margin: 30px 0 4px; }
.sec-sub { color:#5b6b86; margin-bottom: 14px; }
.step { background:#fff; border:1px solid #e3ebf6; border-radius:16px; padding:18px 20px; height:100%; }
.step .n { width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,#0e7490,#2563eb); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; margin-bottom:10px; }
.step b { color:#0b1b3f; } .step p { color:#5b6b86; font-size:14px; margin:6px 0 0; }

/* page header banner */
.phead { position:relative; border-radius:20px; overflow:hidden; margin-bottom: 18px; box-shadow:0 10px 30px rgba(11,27,63,.15); }
.phead img { width:100%; height:170px; object-fit:cover; display:block; }
.phead .t { position:absolute; left:32px; top:50%; transform:translateY(-50%); color:#fff; max-width: 60%; }
.phead .t h2 { color:#fff !important; margin:0; font-size:30px; }
.phead .t span { opacity:.9; font-size:14.5px; }

/* doctor directory */
.doc-card { background:#fff; border:1px solid #e3ebf6; border-radius:18px; padding:18px 20px; margin-bottom:14px;
            box-shadow:0 4px 16px rgba(11,27,63,.07); display:flex; gap:16px; align-items:flex-start; }
.doc-card .av { flex:0 0 52px; height:52px; border-radius:14px; background:linear-gradient(135deg,#0e7490,#2563eb);
                color:#fff; font-weight:700; font-size:20px; display:flex; align-items:center; justify-content:center; }
.doc-card h4 { margin:0; color:#0b1b3f; font-size:17px; }
.doc-card .spec { color:#0e7490; font-weight:600; font-size:13.5px; margin:2px 0 2px; }
.doc-card .hosp { color:#5b6b86; font-size:13px; margin-bottom:8px; }
.doc-card .links a { display:inline-block; margin:4px 8px 0 0; padding:6px 12px; border-radius:999px; font-size:13px; font-weight:600;
                     background:#eef6ff; color:#1d4ed8 !important; text-decoration:none; border:1px solid #cfe0ff; }
.doc-card .links a:hover { background:#dbeafe; }

/* login */
.login-art { position:relative; border-radius:22px; overflow:hidden; min-height: 520px; box-shadow:0 18px 50px rgba(11,27,63,.25); }
.login-art img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }
.login-art .t { position:relative; padding: 36px 32px; color:#fff; }
.login-art .t h3 { color:#fff !important; font-size: 26px; margin-bottom: 6px; }
.login-art .t li { margin: 6px 0; opacity:.92; }

.info-banner { background:#eef6ff; border-left:4px solid #2563eb; border-radius:10px; padding:12px 16px; margin:10px 0; font-size:14px; color:#1e3a8a; }
.bmi-box { border-radius:12px; padding:12px; text-align:center; color:#fff; font-size:20px; font-weight:700; margin:10px 0; }
.result-img { border-radius:16px; width:100%; max-width:420px; display:block; box-shadow:0 8px 24px rgba(11,27,63,.18); }
.ai-head { background: linear-gradient(135deg,#0b1b3f,#0e7490); border-radius:14px; padding:14px 20px; margin: 6px 0 10px; color:#fff; }
.ai-head b { font-size:18px; } .ai-head span { opacity:.85; font-size:13px; }
.topbar { display:flex; align-items:center; gap:12px; margin-bottom: 14px; }
.topbar img { width:46px; height:46px; border-radius:12px; }
.topbar b { font-size:22px; color:#0b1b3f; letter-spacing:-.3px; }
.topbar span { color:#5b6b86; font-size:13px; display:block; margin-top:-3px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


@functools.lru_cache(maxsize=None)
def b64(filename: str) -> str:
    path = os.path.join(ASSETS, filename)
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def logo_src() -> str:
    return b64("logo.png")


def topbar():
    st.markdown(
        f"<div class='topbar'><img src='{logo_src()}'/><div><b>MedPredict</b>"
        f"<span>AI-assisted disease screening for clinicians</span></div></div>", unsafe_allow_html=True)


def hero():
    st.markdown(
        f"<div class='hero'><img src='{b64('hero.jpg')}'/><div class='txt'>"
        f"<span class='pill'>MACHINE LEARNING × GEMINI AI</span>"
        f"<h1>Smarter screening.<br>Clearer next steps.</h1>"
        f"<p>Nine machine-learning screening models trained on public datasets, plus a Gemini-powered advisor that turns every "
        f"result into personalised diet, lifestyle and follow-up guidance.</p></div></div>",
        unsafe_allow_html=True)


def stats_strip():
    st.markdown(
        "<div class='strip'>"
        "<div><b>9</b><span>Disease screening models</span></div>"
        "<div><b>Gemini</b><span>AI health advisor on every result</span></div>"
        "<div><b>Open to all</b><span>Screenings need no sign-up</span></div>"
        "<div><b>Instant</b><span>Results in a single click</span></div></div>",
        unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f"<div class='sec-title'>{title}</div><div class='sec-sub'>{sub}</div>", unsafe_allow_html=True)


def disease_card(page_name: str):
    m = PAGES[page_name]
    st.markdown(
        f"<div class='dcard'><img src='{b64(m['key'] + '.jpg')}'/><div class='body'>"
        f"<h4>{m['icon']} {m['short']}</h4><p>{m['desc']}</p></div></div>", unsafe_allow_html=True)


def page_header(page_name: str):
    m = PAGES[page_name]
    title = page_name.replace(" Prediction", "") + " Prediction"
    st.markdown(
        f"<div class='phead'><img src='{b64(m['key'] + '.jpg')}'/><div class='t'>"
        f"<h2>{m['icon']} {title}</h2><span>{m['desc']}</span></div></div>", unsafe_allow_html=True)


def result_image(positive: bool):
    f = "result_positive.jpg" if positive else "result_negative.jpg"
    st.markdown(f"<img class='result-img' src='{b64(f)}'/>", unsafe_allow_html=True)


def login_art(title: str, bullets):
    lis = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(
        f"<div class='login-art'><img src='{b64('login_art.jpg')}'/><div class='t'>"
        f"<h3>{title}</h3><ul style='padding-left:18px;margin-top:14px'>{lis}</ul></div></div>",
        unsafe_allow_html=True)


def esc(value) -> str:
    """HTML-escape user-supplied text before it is placed inside unsafe_allow_html markup."""
    return html.escape("" if value is None else str(value))


def doctor_card(d: dict):
    name = (d.get("full_name") or "Doctor").strip()
    initials = "".join(w[0] for w in re.sub(r"^(dr\.?\s+)", "", name, flags=re.I).split()[:2]).upper() or "D"
    links = ""
    phone = (d.get("phone") or "").strip()
    if phone:
        tel = re.sub(r"[^\d+]", "", phone)
        links += f"<a href='tel:{esc(tel)}'>📞 {esc(phone)}</a>"
    email = (d.get("email") or "").strip()
    if email:
        links += f"<a href='mailto:{esc(quote(email))}'>✉️ {esc(email)}</a>"
    hosp = f"<div class='hosp'>🏥 {esc(d['hospital'])}</div>" if d.get("hospital") else ""
    st.markdown(
        f"<div class='doc-card'><div class='av'>{esc(initials)}</div><div>"
        f"<h4>{esc(name)}</h4><div class='spec'>{esc(d.get('specialization') or 'Doctor')}</div>{hosp}"
        f"<div class='links'>{links}</div></div></div>", unsafe_allow_html=True)
