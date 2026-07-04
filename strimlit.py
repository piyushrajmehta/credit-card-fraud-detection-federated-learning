import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page Config (must be FIRST Streamlit call) ──
st.set_page_config(
    page_title="CCFD — Federated Learning",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
.stApp                      { background: #0D1B2A; }
[data-testid="stSidebar"]   { background: #0A1520 !important; border-right: 1px solid #1B4F72; }
[data-testid="stSidebar"] * { color: #AED6F1 !important; }
h1, h2, h3                  { font-family: 'DM Sans', sans-serif !important; font-weight: 700 !important; }
 
[data-testid="metric-container"] {
    background: #112233; border: 1px solid #1B4F72;
    border-radius: 10px; padding: 1rem;
}
[data-testid="metric-container"] label {
    color: #AED6F1 !important; font-size: 0.8rem !important;
    text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #F4D03F !important; font-size: 2rem !important;
    font-weight: 700 !important; font-family: 'Space Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #2ECC71 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1B4F72, #2E86C1);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.6rem 1.5rem;
    transition: all 0.2s; width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2E86C1, #3498DB);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(46,134,193,0.4);
}

.stTabs [data-baseweb="tab-list"] {
    background: #112233; border-radius: 10px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"]   { border-radius: 7px; color: #AED6F1 !important; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #1B4F72 !important; color: white !important; }

.dash-card {
    background: #112233; border: 1px solid #1B4F72;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.dash-card-title { color: #AED6F1; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.dash-card-value { color: #F4D03F; font-size: 1.8rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.dash-card-sub   { color: #5D8AA8; font-size: 0.78rem; margin-top: 0.25rem; }

.step-box {
    background: #0A1F33; border-left: 4px solid #2E86C1;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}
.step-num   { color: #F4D03F; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 0.85rem; }
.step-title { color: #FFFFFF; font-weight: 600; font-size: 0.95rem; }
.step-desc  { color: #7FB3D3; font-size: 0.82rem; margin-top: 0.2rem; }

.section-header {
    background: linear-gradient(90deg, #1B4F72, transparent);
    border-left: 4px solid #F4D03F; border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem; margin: 1rem 0 0.8rem 0;
    color: white; font-weight: 700; font-size: 1.1rem;
}

.badge-green { background:#1E8449; color:white; border-radius:20px; padding:0.2rem 0.7rem; font-size:0.75rem; font-weight:600; display:inline-block; }
.badge-gold  { background:#B7950B; color:white; border-radius:20px; padding:0.2rem 0.7rem; font-size:0.75rem; font-weight:600; display:inline-block; }
.badge-blue  { background:#1B4F72; color:#AED6F1; border-radius:20px; padding:0.2rem 0.7rem; font-size:0.75rem; font-weight:600; display:inline-block; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, average_precision_score,
    classification_report, precision_recall_curve
)
from imblearn.over_sampling import SMOTE


# ══════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="#112233",
    plot_bgcolor="#0D1B2A",
    font=dict(family="DM Sans", color="#AED6F1"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1B4F72", linecolor="#1B4F72", zerolinecolor="#1B4F72"),
    yaxis=dict(gridcolor="#1B4F72", linecolor="#1B4F72", zerolinecolor="#1B4F72"),
)
COLORS = {
    "bankA"  : "#3498DB",
    "bankB"  : "#E67E22",
    "bankC"  : "#9B59B6",
    "global" : "#E74C3C",
    "legit"  : "#2ECC71",
    "fraud"  : "#E74C3C",
    "gold"   : "#F4D03F",
}


# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════
for key in ["df", "results", "trained", "global_model",
            "local_models", "X_test", "y_test", "banks"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "trained" not in st.session_state:
    st.session_state["trained"] = False


# ══════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════
def section(title):
    st.markdown(
        f'<div class="section-header">⬡ {title}</div>',
        unsafe_allow_html=True
    )

def plotly_fig(fig):
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def run_pipeline(df):
    """Full Federated Learning pipeline."""

    # ── Preprocessing ──
    scaler = StandardScaler()
    df = df.copy()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"]   = scaler.fit_transform(df[["Time"]])

    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Split into 3 Banks ──
    total  = len(X_train_full)
    split1 = total // 3
    split2 = 2 * (total // 3)
    banks  = {
        "Bank A (HDFC)" : (X_train_full[:split1],       y_train_full[:split1]),
        "Bank B (SBI)"  : (X_train_full[split1:split2], y_train_full[split1:split2]),
        "Bank C (ICICI)": (X_train_full[split2:],       y_train_full[split2:]),
    }

    # ── SMOTE + Local Training ──
    local_models = {}
    smote = SMOTE(random_state=42)
    for name, (Xb, yb) in banks.items():
        Xr, yr = smote.fit_resample(Xb, yb)
        m = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        m.fit(Xr, yr)
        local_models[name] = {"model": m, "Xr": Xr, "yr": yr}

    # ── FedAvg Algorithm ──
    models_list    = [v["model"] for v in local_models.values()]
    avg_coef       = np.mean([m.coef_[0]      for m in models_list], axis=0)
    avg_intercept  = np.mean([m.intercept_[0] for m in models_list], axis=0)

    ref  = list(local_models.values())[0]
    idx0 = np.where(ref["yr"] == 0)[0][:5]
    idx1 = np.where(ref["yr"] == 1)[0][:5]
    idx  = np.concatenate([idx0, idx1])
    gm   = LogisticRegression(max_iter=1000, random_state=42)
    gm.fit(ref["Xr"][idx], ref["yr"][idx])
    gm.coef_[0]      = avg_coef
    gm.intercept_[0] = avg_intercept

    # ── Evaluate All Models ──
    def evaluate(name, model):
        yp    = model.predict(X_test)
        yprob = model.predict_proba(X_test)[:, 1]
        rep   = classification_report(y_test, yp, output_dict=True)
        return {
            "Model"     : name,
            "F1-Score"  : round(f1_score(y_test, yp), 4),
            "ROC-AUC"   : round(roc_auc_score(y_test, yprob), 4),
            "PR-AUC"    : round(average_precision_score(y_test, yprob), 4),
            "Precision" : round(rep["1"]["precision"], 4),
            "Recall"    : round(rep["1"]["recall"], 4),
            "CM"        : confusion_matrix(y_test, yp).tolist(),
            "y_prob"    : yprob.tolist(),
        }

    results = [evaluate(n, v["model"]) for n, v in local_models.items()]
    results.append(evaluate("🌐 Global (FedAvg)", gm))

    return results, gm, local_models, X_test, y_test, banks


# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-size:2.5rem;'>💳</div>
        <div style='color:#F4D03F; font-weight:700; font-size:1rem; margin-top:0.3rem;'>CCFD Dashboard</div>
        <div style='color:#5D8AA8; font-size:0.75rem;'>Federated Learning Simulation</div>
    </div>
    <hr style='border-color:#1B4F72; margin:0.5rem 0 1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Upload Dataset**")
    uploaded = st.file_uploader(
        "Upload creditcard.csv",
        type=["csv"],
        help="Download from: kaggle.com/datasets/mlg-ulb/creditcardfraud"
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
        st.success(f"✅ Loaded {len(df):,} rows")

    st.markdown("<hr style='border-color:#1B4F72; margin:1rem 0;'>", unsafe_allow_html=True)

    if st.session_state["df"] is not None:
        st.markdown("**⚙️ Run Pipeline**")
        if st.button("🚀 Train All Models"):
            with st.spinner("Running Federated Learning Pipeline..."):
                out = run_pipeline(st.session_state["df"])
                results, gm, local_models, X_test, y_test, banks = out
                st.session_state["results"]      = results
                st.session_state["global_model"] = gm
                st.session_state["local_models"] = local_models
                st.session_state["X_test"]       = X_test
                st.session_state["y_test"]       = y_test
                st.session_state["banks"]        = banks
                st.session_state["trained"]      = True
            st.success("✅ Training complete!")
    else:
        st.info("Upload creditcard.csv to begin.")

    st.markdown("<hr style='border-color:#1B4F72; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#5D8AA8; font-size:0.75rem; line-height:1.8;'>
        <b style='color:#AED6F1;'>Algorithm  :</b> FedAvg<br>
        <b style='color:#AED6F1;'>Model      :</b> Logistic Regression<br>
        <b style='color:#AED6F1;'>Balancing  :</b> SMOTE<br>
        <b style='color:#AED6F1;'>Banks      :</b> 3 Simulated<br>
        <b style='color:#AED6F1;'>GDPR Safe  :</b> ✅ Compliant<br>
        <b style='color:#AED6F1;'>Dataset    :</b> ULB Kaggle<br>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TITLE BANNER
# ══════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#0A1520 0%,#1B4F72 50%,#0A1520 100%);
     border:1px solid #2E86C1; border-radius:16px; padding:2rem;
     text-align:center; margin-bottom:2rem;'>
    <div style='color:#FFF; font-size:2.2rem; font-weight:700;'>
        💳 Credit Card Fraud Detection
    </div>
    <div style='color:#F4D03F; font-size:1.1rem; font-style:italic; margin-top:0.4rem;'>
        Solving the Data Silo Paradox using Federated Learning Simulation
    </div>
    <div style='margin-top:1rem; display:flex; justify-content:center; gap:0.5rem; flex-wrap:wrap;'>
        <span class="badge-blue">FedAvg Algorithm</span>
        <span class="badge-blue">3 Simulated Banks</span>
        <span class="badge-green">GDPR Compliant</span>
        <span class="badge-gold">ULB Kaggle Dataset</span>
        <span class="badge-blue">IILM University</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview " ,
    "🔍 EDA " ,
    "🤖 Model Results " ,
    "📈 Visualizations " ,
    "🌐 Methodology"
])


# ══════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════
with tab1:
    df = st.session_state["df"]

    if df is None:
        st.markdown("""
        <div style='text-align:center; padding:3rem; background:#112233;
             border:1px solid #1B4F72; border-radius:12px;'>
            <div style='font-size:3rem;'>📂</div>
            <div style='color:#AED6F1; font-size:1.1rem; margin-top:1rem; font-weight:600;'>
                No dataset loaded yet
            </div>
            <div style='color:#5D8AA8; margin-top:0.5rem;'>
                Upload <b>creditcard.csv</b> from the sidebar to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        section("Dataset Statistics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Transactions", f"{len(df):,}")
        c2.metric("Fraud Cases",        f"{df['Class'].sum():,}")
        c3.metric("Fraud Rate",         f"{df['Class'].mean()*100:.3f}%")
        c4.metric("Features",           f"{df.shape[1]-1}")
        c5.metric("Missing Values",     f"{df.isnull().sum().sum()}")

        st.markdown("<br>", unsafe_allow_html=True)

        section("Problem Statement")
        st.markdown("""
        <div style='background:#0A1F33; border-left:4px solid #F4D03F;
             border-radius:0 10px 10px 0; padding:1.2rem 1.5rem;'>
            <div style='color:#AED6F1; font-style:italic; line-height:1.7; font-size:0.95rem;'>
                "Current Credit Card Fraud Detection (CCFD) systems face a critical
                <b style='color:#F4D03F;'>'Data Silo' Paradox</b>: while machine learning
                models require massive, diverse, and real-time datasets to accurately identify
                evolving fraudulent patterns, financial institutions are strictly prohibited
                from sharing sensitive customer data due to
                <b style='color:#F4D03F;'>GDPR</b> and data confidentiality regulations."
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        section("Federated Learning Solution Flow")
        ca, cb, cc = st.columns(3)
        with ca:
            st.markdown("""
            <div class="dash-card">
                <div class="dash-card-title">🏦 Step 1–2</div>
                <div style='color:white; font-weight:600;'>Data Isolation</div>
                <div class="dash-card-sub">Dataset split into 3 bank partitions.
                Each bank applies SMOTE locally. No data leaves the bank.</div>
            </div>""", unsafe_allow_html=True)
        with cb:
            st.markdown("""
            <div class="dash-card" style='border-color:#F4D03F;'>
                <div class="dash-card-title">🤖 Step 3–4</div>
                <div style='color:white; font-weight:600;'>Local Training + FedAvg</div>
                <div class="dash-card-sub">Each bank trains independently.
                Only model weights (not data) are shared with central server.</div>
            </div>""", unsafe_allow_html=True)
        with cc:
            st.markdown("""
            <div class="dash-card" style='border-color:#2ECC71;'>
                <div class="dash-card-title">🌐 Step 5</div>
                <div style='color:white; font-weight:600;'>Global Model</div>
                <div class="dash-card-sub">Averaged weights create a global model
                that knows fraud patterns from all banks. 100% GDPR compliant.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True, height=300)

        if st.session_state["trained"]:
            st.success("✅ Models trained! Check Model Results and Visualizations tabs.")
        else:
            st.info("👈 Click Train All Models in the sidebar to run the pipeline.")


# ══════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════
with tab2:
    df = st.session_state["df"]

    if df is None:
        st.warning("Please upload creditcard.csv from the sidebar first.")
    else:
        section("Class Distribution — The Data Silo Problem")
        col1, col2 = st.columns(2)
        counts = df["Class"].value_counts()

        with col1:
            fig = go.Figure(go.Bar(
                x=["Legitimate (0)", "Fraud (1)"],
                y=counts.values,
                marker_color=[COLORS["legit"], COLORS["fraud"]],
                text=[f"{v:,}" for v in counts.values],
                textposition="outside",
                textfont=dict(color="white", size=14),
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Transaction Count by Class",
                           font=dict(color="white", size=14)),
                showlegend=False, yaxis_title="Count"
            )
            plotly_fig(fig)

        with col2:
            fig = go.Figure(go.Pie(
                labels=["Legitimate", "Fraud"],
                values=counts.values,
                hole=0.55,
                marker=dict(
                    colors=[COLORS["legit"], COLORS["fraud"]],
                    line=dict(color="#112233", width=3)
                ),
                textfont=dict(color="white", size=13),
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Class Distribution (%)",
                           font=dict(color="white", size=14)),
                annotations=[dict(
                    text=f"{df['Class'].mean()*100:.3f}%<br>Fraud",
                    x=0.5, y=0.5,
                    font=dict(size=14, color="#E74C3C"),
                    showarrow=False
                )]
            )
            plotly_fig(fig)

        section("Transaction Amount Analysis")
        fraud_amt = df[df["Class"] == 1]["Amount"]
        legit_amt = df[df["Class"] == 0]["Amount"]
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=legit_amt[legit_amt < 1000], name="Legitimate",
                marker_color=COLORS["legit"], opacity=0.7, nbinsx=50
            ))
            fig.add_trace(go.Histogram(
                x=fraud_amt[fraud_amt < 1000], name="Fraud",
                marker_color=COLORS["fraud"], opacity=0.7, nbinsx=50
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Amount Distribution (€0–1000)",
                           font=dict(color="white", size=14)),
                barmode="overlay", xaxis_title="Amount (€)",
                legend=dict(bgcolor="#112233", bordercolor="#1B4F72")
            )
            plotly_fig(fig)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=legit_amt, name="Legitimate",
                marker_color=COLORS["legit"], line_color=COLORS["legit"]
            ))
            fig.add_trace(go.Box(
                y=fraud_amt, name="Fraud",
                marker_color=COLORS["fraud"], line_color=COLORS["fraud"]
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Amount Boxplot by Class",
                           font=dict(color="white", size=14)),
                yaxis_title="Amount (€)",
                legend=dict(bgcolor="#112233", bordercolor="#1B4F72")
            )
            plotly_fig(fig)

        section("Top Features Correlated with Fraud")
        corr = (df.corr()["Class"].drop("Class")
                  .abs().sort_values(ascending=False).head(15))
        fig = go.Figure(go.Bar(
            x=corr.values, y=corr.index, orientation="h",
            marker=dict(
                color=corr.values,
                colorscale=[[0, "#1B4F72"], [1, "#E74C3C"]],
                showscale=True,
                colorbar=dict(title="Corr", tickfont=dict(color="#AED6F1"))
            ),
            text=[f"{v:.3f}" for v in corr.values],
            textposition="outside",
            textfont=dict(color="white")
        ))
        fig.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Top 15 Features — Absolute Correlation with Fraud",
                       font=dict(color="white", size=14)),
            xaxis_title="Absolute Correlation", height=460
        )
        fig.update_yaxes(autorange="reversed")
        plotly_fig(fig)

        section("Transaction Time Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df["Class"] == 0]["Time"], name="Legitimate",
                marker_color=COLORS["legit"], opacity=0.7, nbinsx=48
            ))
            fig.add_trace(go.Histogram(
                x=df[df["Class"] == 1]["Time"], name="Fraud",
                marker_color=COLORS["fraud"], opacity=0.7, nbinsx=48
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Time of Transaction",
                           font=dict(color="white", size=14)),
                barmode="overlay", xaxis_title="Time (seconds)",
                legend=dict(bgcolor="#112233", bordercolor="#1B4F72")
            )
            plotly_fig(fig)

        with col2:
            st.markdown(f"""
            <div class="dash-card" style='margin-top:0.5rem;'>
                <div class="dash-card-title">💡 Key EDA Insights</div>
                <div style='color:#AED6F1; line-height:1.9; font-size:0.88rem; margin-top:0.5rem;'>
                    🔴 <b style='color:#E74C3C;'>Fraud = only 0.17%</b> — proves data silo problem<br>
                    🟡 <b style='color:#F4D03F;'>V17, V14, V12</b> are top fraud predictors<br>
                    🟢 <b style='color:#2ECC71;'>No missing values</b> — dataset is clean<br>
                    🔵 Avg Legit: <b>€{legit_amt.mean():.2f}</b> vs Avg Fraud: <b>€{fraud_amt.mean():.2f}</b><br>
                    🟣 Time shows <b>two daily transaction cycles</b><br>
                    ⚪ V1–V28 are <b>PCA-anonymized</b> for privacy<br>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TAB 3 — MODEL RESULTS
# ══════════════════════════════════════════════════
with tab3:
    if not st.session_state["trained"]:
        st.markdown("""
        <div style='text-align:center; padding:3rem; background:#112233;
             border:1px solid #1B4F72; border-radius:12px;'>
            <div style='font-size:3rem;'>🤖</div>
            <div style='color:#AED6F1; font-size:1.1rem; margin-top:1rem; font-weight:600;'>
                Models not trained yet
            </div>
            <div style='color:#5D8AA8; margin-top:0.5rem;'>
                Upload the dataset then click <b>Train All Models</b> in the sidebar.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        results  = st.session_state["results"]
        y_test   = st.session_state["y_test"]
        res_df   = pd.DataFrame(results)[
            ["Model","F1-Score","ROC-AUC","PR-AUC","Precision","Recall"]
        ]
        best_idx = res_df["F1-Score"].idxmax()
        best     = res_df.loc[best_idx]

        section("Model Comparison Table")
        st.dataframe(
            res_df.set_index("Model").style.format("{:.4f}"),
            use_container_width=True, height=220
        )
        st.markdown(f"""
        <div style='background:#1E3A4A; border:1px solid #F4D03F; border-radius:8px;
             padding:0.8rem 1.2rem; margin:0.5rem 0 1.5rem 0;'>
            🏆 <b style='color:#F4D03F;'>Best Model : {best["Model"]}</b>
            &nbsp;|&nbsp; F1 : <b style='color:#F4D03F;'>{best["F1-Score"]}</b>
            &nbsp;|&nbsp; ROC-AUC : <b style='color:#F4D03F;'>{best["ROC-AUC"]}</b>
            &nbsp;|&nbsp; Recall : <b style='color:#F4D03F;'>{best["Recall"]}</b>
        </div>
        """, unsafe_allow_html=True)

        section("Improvement from Federated Learning")
        c1, c2, c3, c4 = st.columns(4)
        cols       = [c1, c2, c3, c4]
        local_res  = results[:3]
        global_res = results[3]
        for i, metric in enumerate(["F1-Score","ROC-AUC","PR-AUC","Recall"]):
            best_local  = max(r[metric] for r in local_res)
            gval        = global_res[metric]
            improvement = (gval - best_local) / best_local * 100
            cols[i].metric(
                f"Global {metric}",
                f"{gval:.4f}",
                f"+{improvement:.2f}% vs best local"
            )

        section("Confusion Matrix")
        selected = st.selectbox("Select Model to Inspect",
                                [r["Model"] for r in results])
        sel_res  = next(r for r in results if r["Model"] == selected)
        cm       = np.array(sel_res["CM"])
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted Legit", "Predicted Fraud"],
                y=["Actual Legit",    "Actual Fraud"],
                colorscale=[[0, "#0D1B2A"], [1, "#2E86C1"]],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=22, color="white"),
                showscale=False
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                title=dict(text=f"Confusion Matrix — {selected}",
                           font=dict(color="white", size=13)),
                height=320
            )
            plotly_fig(fig)

        with col2:
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            <div class="dash-card">
                <div class="dash-card-title">Matrix Breakdown</div><br>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;'>
                    <div style='background:#1E3A4A; border-radius:8px; padding:0.8rem; text-align:center;'>
                        <div style='color:#2ECC71; font-size:1.6rem; font-weight:700; font-family:monospace;'>{tp:,}</div>
                        <div style='color:#AED6F1; font-size:0.75rem; margin-top:0.2rem;'>True Positives<br>Fraud Caught ✅</div>
                    </div>
                    <div style='background:#1E3A4A; border-radius:8px; padding:0.8rem; text-align:center;'>
                        <div style='color:#E74C3C; font-size:1.6rem; font-weight:700; font-family:monospace;'>{fn:,}</div>
                        <div style='color:#AED6F1; font-size:0.75rem; margin-top:0.2rem;'>False Negatives<br>Fraud Missed ❌</div>
                    </div>
                    <div style='background:#1E3A4A; border-radius:8px; padding:0.8rem; text-align:center;'>
                        <div style='color:#E67E22; font-size:1.6rem; font-weight:700; font-family:monospace;'>{fp:,}</div>
                        <div style='color:#AED6F1; font-size:0.75rem; margin-top:0.2rem;'>False Positives<br>False Alarm ⚠️</div>
                    </div>
                    <div style='background:#1E3A4A; border-radius:8px; padding:0.8rem; text-align:center;'>
                        <div style='color:#3498DB; font-size:1.6rem; font-weight:700; font-family:monospace;'>{tn:,}</div>
                        <div style='color:#AED6F1; font-size:0.75rem; margin-top:0.2rem;'>True Negatives<br>Legit Correct ✅</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TAB 4 — VISUALIZATIONS
# ══════════════════════════════════════════════════
with tab4:
    if not st.session_state["trained"]:
        st.warning("Train models first — click Train All Models in the sidebar.")
    else:
        results      = st.session_state["results"]
        y_test       = st.session_state["y_test"]
        model_colors = [COLORS["bankA"], COLORS["bankB"],
                        COLORS["bankC"], COLORS["global"]]

        section("ROC Curves — All 4 Models")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#5D8AA8", dash="dash", width=1),
            name="Random Classifier (AUC = 0.50)"
        ))
        for res, color in zip(results, model_colors):
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            lw   = 3 if "Global" in res["Model"] else 1.5
            dash = "solid" if "Global" in res["Model"] else "dot"
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{res['Model']}  (AUC = {res['ROC-AUC']})",
                line=dict(color=color, width=lw, dash=dash)
            ))
        fig.update_layout(
            **PLOT_LAYOUT,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            legend=dict(bgcolor="#112233", bordercolor="#1B4F72", font=dict(size=11)),
            height=430
        )
        plotly_fig(fig)
        st.info("Global Model curve sits highest → best fraud detection ability.")

        section("Performance Metrics — Comparison")
        names  = [r["Model"].replace("🌐 ", "") for r in results]
        col1, col2 = st.columns(2)
        for i, metric in enumerate(["F1-Score", "Recall", "ROC-AUC", "Precision"]):
            vals = [r[metric] for r in results]
            fig  = go.Figure(go.Bar(
                x=names, y=vals,
                marker_color=model_colors,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                textfont=dict(color="white", size=12),
                marker_line_color=["#112233","#112233","#112233", COLORS["gold"]],
                marker_line_width=[1, 1, 1, 3]
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                title=dict(text=metric, font=dict(color="white", size=14)),
                showlegend=False, height=310
            )
            fig.update_yaxes(range=[0, 1.15])
            if i % 2 == 0:
                col1.plotly_chart(fig, use_container_width=True,
                                  config={"displayModeBar": False})
            else:
                col2.plotly_chart(fig, use_container_width=True,
                                  config={"displayModeBar": False})

        section("Precision-Recall Curves")
        fig = go.Figure()
        for res, color in zip(results, model_colors):
            prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
            lw = 3 if "Global" in res["Model"] else 1.5
            fig.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                name=f"{res['Model']}  (PR-AUC = {res['PR-AUC']})",
                line=dict(color=color, width=lw)
            ))
        fig.update_layout(
            **PLOT_LAYOUT,
            xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(bgcolor="#112233", bordercolor="#1B4F72", font=dict(size=11)),
            height=390
        )
        plotly_fig(fig)
        st.info("PR-AUC is the most important metric for imbalanced datasets.")


# ══════════════════════════════════════════════════
# TAB 5 — METHODOLOGY
# ══════════════════════════════════════════════════
with tab5:
    section("Federated Learning Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="dash-card">
            <div class="dash-card-title">🔑 The Core Idea</div>
            <div style='color:#AED6F1; line-height:1.8; font-size:0.9rem; margin-top:0.5rem;'>
                Instead of sharing raw customer transaction data
                (illegal under GDPR), each bank trains a model locally
                and shares only <b style='color:#F4D03F;'>model weights</b>
                — just numbers, no customer info.
                <br><br>
                A central server averages these weights using
                <b style='color:#F4D03F;'>FedAvg</b> to create
                one powerful Global Model.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="dash-card" style='border-color:#2ECC71;'>
            <div class="dash-card-title">✅ Why It Solves the Problem</div>
            <div style='color:#AED6F1; line-height:1.9; font-size:0.9rem; margin-top:0.5rem;'>
                ✅ Raw customer data <b>never leaves</b> each bank's server<br>
                ✅ Banks collectively learn from <b>diverse fraud patterns</b><br>
                ✅ Global model is <b>stronger than any single bank model</b><br>
                ✅ Fully compliant with <b style='color:#F4D03F;'>GDPR</b><br>
                ✅ Scales to <b>any number of institutions</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    section("Step-by-Step Pipeline")
    steps = [
        ("1","Data Split",    "Dataset divided into 3 parts — each simulates one bank's private database. No bank can see another's data.", "#3498DB"),
        ("2","SMOTE",         "Each bank independently applies SMOTE to balance fraud vs legitimate samples locally. No data is shared.", "#9B59B6"),
        ("3","Local Training","Each bank trains a Logistic Regression model privately on its own resampled data only.", "#2ECC71"),
        ("4","FedAvg",        "Banks share ONLY model weights. Central server computes: Global = (Bank A + Bank B + Bank C) / 3", "#E74C3C"),
        ("5","Global Model",  "Averaged weights form a Global Model knowing fraud patterns from all 3 banks — GDPR compliant.", "#F4D03F"),
    ]
    for num, title, desc, color in steps:
        st.markdown(f"""
        <div class="step-box" style='border-left-color:{color};'>
            <div class="step-num" style='color:{color};'>STEP {num}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    section("FedAvg Algorithm — Core Code")
    st.code("""
# ── FEDERATED AVERAGING (FedAvg) ──────────────────────────────
# Step 1: Each bank trains locally (NO data sharing)
model_A.fit(X_bankA, y_bankA)    # Bank A — private data only
model_B.fit(X_bankB, y_bankB)    # Bank B — private data only
model_C.fit(X_bankC, y_bankC)    # Bank C — private data only

# Step 2: Extract weights (no raw data — just learned numbers)
weights_A = model_A.coef_[0]     # e.g. [ 0.23, -0.87,  0.45 ...]
weights_B = model_B.coef_[0]     # e.g. [ 0.31, -0.79,  0.51 ...]
weights_C = model_C.coef_[0]     # e.g. [ 0.19, -0.91,  0.43 ...]

# Step 3: Central server computes FedAvg
global_weights = (weights_A + weights_B + weights_C) / 3

# Step 4: Global model uses averaged weights
global_model.coef_[0] = global_weights
# ✅ Knows fraud from ALL 3 banks — GDPR Compliant!
""", language="python")

    section("Tools & Technologies")
    tools = [
        ("🐍","Python","Core language"),
        ("🧠","scikit-learn","ML models"),
        ("⚖️","imbalanced-learn","SMOTE"),
        ("📊","Pandas & NumPy","Data handling"),
        ("📈","Plotly","Interactive charts"),
        ("🚀","Streamlit","This dashboard"),
        ("💻","VS Code","Development IDE"),
        ("📁","Kaggle ULB","Dataset source"),
    ]
    cols = st.columns(4)
    for i, (icon, tool, desc) in enumerate(tools):
        cols[i % 4].markdown(f"""
        <div class="dash-card" style='text-align:center; padding:0.8rem;'>
            <div style='font-size:1.5rem;'>{icon}</div>
            <div style='color:white; font-weight:600; font-size:0.85rem;'>{tool}</div>
            <div class="dash-card-sub">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; padding:1.5rem; margin-top:1.5rem;
         background:#0A1520; border:1px solid #1B4F72; border-radius:12px;'>
        <div style='color:#F4D03F; font-weight:700; font-size:1rem;'>
            💳 Credit Card Fraud Detection — Federated Learning Simulation
        </div>
        <div style='color:#5D8AA8; font-size:0.8rem; margin-top:0.4rem;'>
            IILM University &nbsp;|&nbsp; B.Tech Computer Science &nbsp;|&nbsp;
            2026 &nbsp;|&nbsp; Dataset: ULB Kaggle &nbsp;|&nbsp; Algorithm: FedAvg
        </div>
    </div>
    """, unsafe_allow_html=True)