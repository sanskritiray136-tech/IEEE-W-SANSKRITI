import io
import urllib.request
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Student Success Copilot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URLS = [
    # Primary mirror (the old uciml URL now returns HTTP 404).
    "https://raw.githubusercontent.com/rupakc/UCI-Data-Analysis/master/Student%20Dataset/Student/student-mat.csv",
    # Secondary mirror in case the primary mirror is unavailable.
    "https://raw.githubusercontent.com/oserradilla/students-data-analysis/master/student-mat.csv",
]
FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences", "school", "sex", "address", "famsize",
    "Pstatus", "Mjob", "Fjob", "reason", "guardian"
]
NUMERIC = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(145deg, #f6f8ff 0%, #f8fbff 48%, #f4f7ff 100%); }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
.hero { padding: 2rem 2.2rem; border-radius: 26px; color: white;
        background: linear-gradient(115deg, #171b4d 0%, #3438a6 54%, #7167f8 100%);
        box-shadow: 0 18px 45px rgba(52,56,166,.20); }
.hero h1 { font-family: 'Manrope', sans-serif; font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.08; margin: .4rem 0 .8rem 0; color: white; }
.hero p { color: #e2e5ff; font-size: 1.06rem; max-width: 760px; margin-bottom: .5rem; }
.eyebrow { color: #c8caff; text-transform: uppercase; font-size: .76rem; font-weight: 800; letter-spacing: .16em; }
.pill { display:inline-block; border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
        padding: .35rem .75rem; color:#f2f2ff; background:rgba(255,255,255,.1); font-size:.8rem; margin-right:.35rem; }
.metric-card { background: white; padding: 1rem 1.2rem; border: 1px solid #e6e9f5; border-radius: 18px; box-shadow: 0 6px 20px rgba(35,42,100,.04); }
.section-title { font-family: 'Manrope', sans-serif; color:#171b4d; font-weight:800; font-size:1.3rem; margin:.35rem 0 .25rem; }
.muted { color:#6b7192; font-size:.92rem; }
div[data-testid="stForm"] { background: white; border: 1px solid #e6e9f5; padding: 1.2rem; border-radius: 20px; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button { border-radius: 12px; min-height: 3rem; font-weight:700; border:0; background:linear-gradient(100deg,#3438a6,#7167f8); color:white; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { color:white; border:0; filter:brightness(1.06); }
[data-testid="stSidebar"] { background: #eff1ff; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    errors = []
    for url in DATA_URLS:
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "StudentSuccessCopilot/1.0"},
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
            frame = pd.read_csv(io.BytesIO(raw), sep=";")
            required = set(FEATURES + ["G1", "G2", "G3"])
            if not required.issubset(frame.columns):
                raise ValueError("Downloaded file does not have the expected dataset columns.")
            frame = frame.drop_duplicates().copy()
            frame["passed"] = (frame["G3"] >= 10).astype(int)
            return frame
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Could not download the student dataset from the available sources. "
        "Please try again later. Details: " + " | ".join(errors)
    )

@st.cache_resource(show_spinner="Training the student success models…")
def train_models():
    frame = load_data()
    X = frame[FEATURES].copy()
    y = frame["passed"].copy()
    num_pipe_tree = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep_tree = ColumnTransformer([
        ("num", num_pipe_tree, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL)
    ])
    num_pipe_knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    prep_knn = ColumnTransformer([
        ("num", num_pipe_knn, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    models = {
        "Random Forest": Pipeline([
            ("prep", prep_tree),
            ("model", RandomForestClassifier(
                n_estimators=250, max_depth=7, min_samples_leaf=3,
                class_weight="balanced", random_state=42
            ))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("prep", prep_knn),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ])
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = accuracy_score(y_test, model.predict(X_test))
    return frame, models, scores

def select(label, options, default_index=0, key=None):
    return st.selectbox(label, options, index=default_index, key=key)

def build_student_input():
    with st.form("student_profile"):
        st.markdown('<div class="section-title">Build a student profile</div>', unsafe_allow_html=True)
        st.markdown('<p class="muted">Use information you know. This demo estimates a pattern, not a student’s potential.</p>', unsafe_allow_html=True)
        tabs = st.tabs(["📚 Learning habits", "🏡 Support & context", "🌿 Wellbeing"])
        values = {}
        with tabs[0]:
            a, b, c = st.columns(3)
            with a:
                values["studytime"] = st.select_slider("Weekly study time", options=[1,2,3,4], value=2, format_func=lambda x: {1:"< 2 hours",2:"2–5 hours",3:"5–10 hours",4:"> 10 hours"}[x])
                values["failures"] = st.select_slider("Past class failures", options=[0,1,2,3], value=0)
                values["absences"] = st.number_input("School absences", min_value=0, max_value=93, value=3, step=1)
                values["traveltime"] = st.select_slider("Travel time to school", options=[1,2,3,4], value=2, format_func=lambda x:{1:"< 15 min",2:"15–30 min",3:"30–60 min",4:"> 60 min"}[x])
            with b:
                values["higher"] = select("Wants higher education", ["yes","no"], key="higher")
                values["paid"] = select("Extra paid classes", ["no","yes"], key="paid")
                values["activities"] = select("Extracurricular activities", ["yes","no"], key="activities")
                values["internet"] = select("Internet access at home", ["yes","no"], key="internet")
            with c:
                values["age"] = st.slider("Age", 15, 22, 17)
                values["freetime"] = st.slider("Free time after school (1–5)", 1, 5, 3)
                values["goout"] = st.slider("Going out with friends (1–5)", 1, 5, 3)
                values["school"] = select("School", ["GP","MS"], key="school")
        with tabs[1]:
            a,b,c = st.columns(3)
            with a:
                values["schoolsup"] = select("Extra school support", ["no","yes"], key="schoolsup")
                values["famsup"] = select("Family learning support", ["yes","no"], key="famsup")
                values["guardian"] = select("Main guardian", ["mother","father","other"], key="guardian")
                values["Pstatus"] = select("Parents living together", ["T","A"], key="Pstatus")
            with b:
                values["Medu"] = st.slider("Mother's education (0–4)",0,4,2)
                values["Fedu"] = st.slider("Father's education (0–4)",0,4,2)
                values["Mjob"] = select("Mother's job", ["other","services","teacher","health","at_home"], key="Mjob")
                values["Fjob"] = select("Father's job", ["other","services","teacher","health","at_home"], key="Fjob")
            with c:
                values["address"] = select("Home area", ["U","R"], key="address", format_func=lambda x: "Urban" if x=="U" else "Rural")
                values["famsize"] = select("Family size", ["GT3","LE3"], key="famsize", format_func=lambda x: "More than 3" if x=="GT3" else "3 or fewer")
                values["reason"] = select("School choice reason", ["course","reputation","home","other"], key="reason")
                values["sex"] = select("Sex (dataset category)", ["F","M"], key="sex", format_func=lambda x: {"F":"Female","M":"Male"}[x])
        with tabs[2]:
            a,b,c = st.columns(3)
            with a:
                values["health"] = st.slider("Self-rated health (1–5)",1,5,3)
                values["famrel"] = st.slider("Family relationship quality (1–5)",1,5,4)
            with b:
                values["Dalc"] = st.slider("Weekday alcohol consumption (1–5)",1,5,1)
                values["Walc"] = st.slider("Weekend alcohol consumption (1–5)",1,5,1)
            with c:
                values["romantic"] = select("In a romantic relationship", ["no","yes"], key="romantic")
        submitted = st.form_submit_button("✨ Generate student success insights", use_container_width=True)
    return values, submitted

with st.sidebar:
    st.markdown("## 🎓 Student Success Copilot")
    st.markdown("A friendly ML demo that turns student context into a simple risk signal and practical next steps.")
    st.markdown("---")
    st.markdown("### What it does")
    st.markdown("1. Learns from historical student data\n2. Estimates pass likelihood\n3. Suggests supportive, low-risk actions")
    st.markdown("---")
    st.caption("Built for learning. Not a diagnostic or decision-making tool.")
    st.markdown("[Dataset source: UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)")

st.markdown("""
<div class="hero">
  <div class="eyebrow">YOUR STUDENT SUPPORT DASHBOARD</div>
  <h1>Make every student’s next step<br>feel a little clearer.</h1>
  <p>Student Success Copilot uses a machine-learning model to estimate pass likelihood and turn the result into practical, encouraging support ideas.</p>
  <span class="pill">✦ ML-powered</span><span class="pill">✦ Action-focused</span><span class="pill">✦ Beginner-friendly</span>
</div>
""", unsafe_allow_html=True)
st.write("")

try:
    frame, models, scores = train_models()
except Exception as e:
    st.error("The app could not load the dataset. Check your internet connection and refresh the page.")
    st.caption(f"Technical detail: {e}")
    st.stop()

m1,m2,m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric-card"><div class="muted">Training dataset</div><h2 style="margin:.2rem 0;color:#22245f">395 students</h2><div class="muted">Math course records after cleaning</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="muted">Best held-out accuracy</div><h2 style="margin:.2rem 0;color:#22245f">{max(scores.values()):.0%}</h2><div class="muted">One 80/20 test split; results may vary</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="muted">What matters most</div><h2 style="margin:.2rem 0;color:#22245f">Support, not labels</h2><div class="muted">Use the result to open a conversation</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.12, .88], gap="large")
with left:
    st.markdown('<div class="section-title">🧩 Student profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Fill out the profile and get a tailored, supportive action plan.</div>', unsafe_allow_html=True)
    student, submitted = build_student_input()

with right:
    st.markdown('<div class="section-title">📈 Model performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">A quick comparison on held-out test data.</div>', unsafe_allow_html=True)
    perf = pd.DataFrame({"Model": list(scores.keys()), "Accuracy": list(scores.values())})
    fig = px.bar(perf, x="Accuracy", y="Model", orientation="h", text=perf["Accuracy"].map(lambda x:f"{x:.1%}"),
                 color="Model", color_discrete_sequence=["#6d63f6","#25b8a6"], range_x=[0,1])
    fig.update_layout(height=250, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=5,r=15,t=10,b=5), xaxis_title="Test accuracy", yaxis_title="")
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Accuracy alone is not enough. The original notebook also includes confusion matrices and class-level metrics.")

if submitted:
    student_df = pd.DataFrame([{col: student[col] for col in FEATURES}])[FEATURES]
    chosen_model_name = max(scores, key=scores.get)
    chosen_model = models[chosen_model_name]
    probabilities = chosen_model.predict_proba(student_df)[0]
    pass_idx = list(chosen_model.named_steps["model"].classes_).index(1)
    pass_probability = float(probabilities[pass_idx])
    st.markdown("---")
    st.markdown('<div class="section-title">✨ Your student support snapshot</div>', unsafe_allow_html=True)
    c1,c2 = st.columns([.85,1.15], gap="large")
    with c1:
        st.metric("Estimated pass likelihood", f"{pass_probability:.0%}")
        st.progress(int(round(pass_probability*100)))
        if pass_probability >= .70:
            st.success("Model signal: relatively higher estimated likelihood")
        elif pass_probability >= .45:
            st.warning("Model signal: mixed — check in and identify barriers")
        else:
            st.error("Model signal: lower estimated likelihood — offer supportive help early")
        st.caption(f"Model used: {chosen_model_name}. This is a statistical estimate, not a certainty.")
    actions = []
    if student["studytime"] <= 2:
        actions.append(("Create a small study rhythm", "Try four 25-minute focused study blocks across the week, with one clear topic per block."))
    if student["failures"] > 0:
        actions.append(("Arrange a no-judgment check-in", "Ask which topics feel hardest and agree on one manageable goal plus a follow-up date."))
    if student["absences"] >= 5:
        actions.append(("Understand attendance barriers", "Privately ask whether transport, health, workload, or another barrier is making attendance difficult."))
    if student["schoolsup"] == "no":
        actions.append(("Explore available learning support", "Offer tutoring, teacher office hours, a peer study group, or free learning resources."))
    if student["famsup"] == "no":
        actions.append(("Find a supportive study partner", "If helpful to the student, connect them with a trusted peer, mentor, or teacher."))
    if student["internet"] == "no":
        actions.append(("Make resources accessible offline", "Provide printed notes, library access, or downloadable materials that work without internet."))
    if student["higher"] == "yes":
        actions.append(("Connect learning to future goals", "Help map one current subject skill to a course, career, or further-study goal the student cares about."))
    if not actions:
        actions.append(("Keep momentum with a weekly check-in", "Celebrate what is working, choose one learning goal for next week, and revisit it together."))
    with c2:
        st.markdown("#### A few practical next steps")
        for i,(title,desc) in enumerate(actions[:4],1):
            st.markdown(f"**{i}. {title}**")
            st.write(desc)
        st.download_button("⬇️ Download support snapshot (CSV)",
            data=pd.DataFrame([{
                "estimated_pass_likelihood": round(pass_probability,4),
                "model": chosen_model_name,
                "suggested_actions": " | ".join(t for t,d in actions[:4]),
                "note": "Educational estimate only; do not use for high-stakes decisions."
            }]).to_csv(index=False).encode("utf-8"),
            file_name="student_support_snapshot.csv", mime="text/csv")

st.markdown("---")
with st.expander("ℹ️ How this model works and its limitations"):
    st.markdown("""
    - The app uses the UCI Student Performance mathematics dataset.
    - A pass is defined as final grade `G3 >= 10` on a 0–20 scale.
    - `G1`, `G2`, and `G3` are excluded from model inputs to prevent grade leakage.
    - The app compares a Random Forest and K-Nearest Neighbors model on a held-out test split.
    - Dataset patterns may not transfer to other schools, countries, or student populations.
    - Demographic and family-context variables can encode unfair historical patterns. Treat the output as a prompt for support, never as a label or a basis for high-stakes decisions.
    - The action suggestions are simple rules designed to encourage supportive follow-up; they are not generated by the classifier.
    """)
st.caption("Student Success Copilot · A student-support learning prototype · Not for high-stakes use")
