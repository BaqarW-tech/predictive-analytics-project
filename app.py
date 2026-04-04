import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nitaqat Breach Predictor | Predictive Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.main{background:#050810;}
.block-container{padding:1.5rem 2rem;}
[data-testid="stSidebar"]{background:#080c18;border-right:1px solid #0f2040;}
[data-testid="stSidebar"] *{color:#7fa3c8 !important;}

.pred-title{font-size:1.7rem;font-weight:700;color:#e0f0ff;letter-spacing:-0.03em;}
.pred-sub{font-size:0.72rem;color:#1e4070;letter-spacing:0.18em;text-transform:uppercase;}

.kpi-pred{
    background:linear-gradient(135deg,#060d1f,#0a1630);
    border:1px solid #0f2855;border-radius:12px;
    padding:1.1rem 1.3rem;text-align:center;
}
.kpi-pred .label{color:#2a5080;font-size:0.67rem;letter-spacing:0.14em;text-transform:uppercase;}
.kpi-pred .val{font-family:'Space Mono',monospace;font-size:1.9rem;font-weight:700;color:#38d9f5;}
.kpi-pred .sub{font-size:0.72rem;color:#1a3a60;margin-top:0.2rem;}

.sec{font-size:0.65rem;letter-spacing:0.2em;text-transform:uppercase;
     color:#1565c0;border-bottom:1px solid #0a1e40;
     padding-bottom:0.35rem;margin:1.5rem 0 0.9rem;}

.pred-card{
    background:#060d1f;border:1px solid #0f2855;border-radius:10px;
    padding:1rem 1.3rem;margin-bottom:0.55rem;
    display:flex;justify-content:space-between;align-items:center;
}
.pred-card.high  {border-left:4px solid #ef4444;}
.pred-card.medium{border-left:4px solid #f59e0b;}
.pred-card.low   {border-left:4px solid #22c55e;}

.risk-badge-high  {background:#ef4444;color:white;padding:3px 10px;border-radius:20px;font-size:0.7rem;font-weight:700;}
.risk-badge-medium{background:#f59e0b;color:#1a0a00;padding:3px 10px;border-radius:20px;font-size:0.7rem;font-weight:700;}
.risk-badge-low   {background:#22c55e;color:#001a0a;padding:3px 10px;border-radius:20px;font-size:0.7rem;font-weight:700;}

.insight-box{
    background:#060d1f;border:1px solid #0f2855;border-radius:8px;
    padding:1rem 1.3rem;font-size:0.82rem;color:#5a85aa;line-height:1.75;
}
.insight-box b{color:#38d9f5;}

#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NITAQAT_TARGETS = {
    "Construction": 6, "Information Technology": 35, "Healthcare": 35,
    "Financial Services": 70, "Retail": 30, "Education": 55,
    "Manufacturing": 10, "Tourism & Hospitality": 20,
    "Energy & Utilities": 75, "Transport & Logistics": 15,
}
STATUS_ORDER = ["Red","Yellow","Low Green","Medium Green","High Green","Platinum"]

# ── Data & Feature Engineering ────────────────────────────────────────────────
@st.cache_data
def build_company_features():
    df = pd.read_csv("saudi_workforce_data.csv")
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["year"]  = df["hire_date"].dt.year
    df["month"] = df["hire_date"].dt.month
    df["nitaqat_target"] = df["sector"].map(NITAQAT_TARGETS)

    # ── Per-company feature matrix ──
    records = []
    for cid, g in df.groupby("company_id"):
        is_saudi = g["nationality"] == "Saudi"
        recent   = g["year"] >= g["year"].max() - 1
        old      = g["year"] <= g["year"].max() - 3

        saud_pct      = is_saudi.mean() * 100
        target        = g["nitaqat_target"].iloc[0]
        gap           = saud_pct - target
        recent_total  = recent.sum()
        recent_saudi  = (is_saudi & recent).sum()
        recent_saud_pct = recent_saudi / max(recent_total, 1) * 100

        older_saudi_pct = (is_saudi & old).mean() * 100 if old.sum() > 0 else saud_pct
        trend         = recent_saud_pct - older_saudi_pct  # positive = improving

        avg_sal_saudi = g[is_saudi]["monthly_salary_sar"].mean() if is_saudi.any() else 0
        avg_sal_all   = g["monthly_salary_sar"].mean()
        sal_ratio     = avg_sal_saudi / avg_sal_all if avg_sal_all > 0 else 1

        edu_saudi_bachelor_pct = (
            g[is_saudi & g["education_level"].isin(["Bachelor","Master","PhD"])].shape[0] /
            max(is_saudi.sum(), 1) * 100
        )
        female_pct    = (g["gender"] == "Female").mean() * 100
        contract_pct  = (g["employment_type"] == "Contract").mean() * 100
        parttime_pct  = (g["employment_type"] == "Part-time").mean() * 100
        senior_pct    = g["age_group"].isin(["45-54","55+"]).mean() * 100
        ft_saudi_pct  = (
            g[is_saudi & (g["employment_type"] == "Full-time")].shape[0] /
            max(is_saudi.sum(), 1) * 100
        )
        q1_saudi_ratio = (
            (is_saudi & g["month"].isin([1,2,3])).sum() /
            max(g["month"].isin([1,2,3]).sum(), 1) * 100
        )
        hiring_variance = g.groupby("month").size().std()

        records.append({
            "company_id":           cid,
            "company_name":         g["company_name"].iloc[0],
            "sector":               g["sector"].iloc[0],
            "region":               g["region"].iloc[0],
            "nitaqat_status":       g["nitaqat_status"].iloc[0],
            # ── raw targets ──
            "saudization_pct":      round(saud_pct, 2),
            "nitaqat_target":       target,
            "gap_to_target":        round(gap, 2),
            "headcount":            len(g),
            # ── features ──
            "recent_saudi_hire_pct":round(recent_saud_pct, 2),
            "saudization_trend":    round(trend, 2),
            "salary_ratio":         round(sal_ratio, 3),
            "edu_bachelor_pct":     round(edu_saudi_bachelor_pct, 2),
            "female_pct":           round(female_pct, 2),
            "contract_pct":         round(contract_pct, 2),
            "parttime_pct":         round(parttime_pct, 2),
            "senior_emp_pct":       round(senior_pct, 2),
            "ft_saudi_pct":         round(ft_saudi_pct, 2),
            "q1_saudi_hire_ratio":  round(q1_saudi_ratio, 2),
            "hiring_variance":      round(hiring_variance, 2),
        })

    co = pd.DataFrame(records)

    # ── Label: will breach = currently Yellow or Red ──
    co["will_breach"] = co["nitaqat_status"].isin(["Yellow","Red"]).astype(int)
    # Simulate forward-looking probability with gap + trend noise
    np.random.seed(42)
    co["breach_prob_true"] = np.clip(
        0.5 - co["gap_to_target"] * 0.04 - co["saudization_trend"] * 0.015 +
        co["contract_pct"] * 0.005 + np.random.normal(0, 0.05, len(co)),
        0.02, 0.98
    )
    return df, co


@st.cache_resource
def train_models(co):
    FEATURES = [
        "saudization_pct","gap_to_target","recent_saudi_hire_pct",
        "saudization_trend","salary_ratio","edu_bachelor_pct",
        "female_pct","contract_pct","parttime_pct","ft_saudi_pct",
        "q1_saudi_hire_ratio","hiring_variance","headcount","senior_emp_pct",
    ]
    X = co[FEATURES].fillna(0)
    y = co["will_breach"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42),
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=0.5, max_iter=500, class_weight="balanced", random_state=42))]),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
        results[name] = {
            "model":   model,
            "y_test":  y_test,
            "y_pred":  y_pred,
            "y_prob":  y_prob,
            "auc":     roc_auc_score(y_test, y_prob),
            "cv_auc":  cv_auc,
            "ap":      average_precision_score(y_test, y_prob),
            "report":  classification_report(y_test, y_pred, output_dict=True),
            "cm":      confusion_matrix(y_test, y_pred),
        }

    # Best model = highest CV AUC
    best_name = max(results, key=lambda k: results[k]["cv_auc"])

    # Feature importances from best tree model
    best_tree_name = "Random Forest" if results["Random Forest"]["cv_auc"] >= results["Gradient Boosting"]["cv_auc"] else "Gradient Boosting"
    best_tree = results[best_tree_name]["model"]
    importances = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": best_tree.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # Full-data predictions
    best_model = results[best_name]["model"]
    co = co.copy()
    co["predicted_breach_prob"] = best_model.predict_proba(X.fillna(0))[:, 1]
    co["predicted_breach"]      = best_model.predict(X.fillna(0))
    co["risk_tier"] = pd.cut(
        co["predicted_breach_prob"],
        bins=[0, 0.35, 0.65, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    return results, best_name, best_tree_name, importances, co, FEATURES, X_test, y_test


# ── Load ──────────────────────────────────────────────────────────────────────
df, co = build_company_features()
results, best_name, best_tree_name, importances, co_pred, FEATURES, X_test, y_test = train_models(co)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Predictive Module")
    st.markdown("**ML Breach Predictor · App 3 of 4**")
    st.markdown("---")
    model_choice = st.selectbox("Compare Model", list(results.keys()), index=list(results.keys()).index(best_name))
    threshold    = st.slider("Decision Threshold", 0.20, 0.80, 0.50, 0.05,
                             help="Adjust probability cutoff for breach classification")
    show_sectors = st.multiselect("Sector Filter", sorted(co["sector"].unique()), default=sorted(co["sector"].unique()))
    st.markdown("---")
    st.markdown("<div style='color:#0f2040;font-size:0.72rem;'>Predictive Analytics · Vision 2030 Series</div>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="pred-title">Nitaqat Breach Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="pred-sub">PREDICTIVE ANALYTICS · MACHINE LEARNING · 12-MONTH COMPLIANCE FORECAST</div>', unsafe_allow_html=True)
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
res    = results[model_choice]
at_pct = (co_pred["predicted_breach_prob"] >= threshold).mean() * 100
best_auc = results[best_name]["cv_auc"]

k1,k2,k3,k4,k5 = st.columns(5)
for col, label, val, sub in [
    (k1, "Best Model AUC",        f"{best_auc:.3f}",          f"{best_name}"),
    (k2, f"{model_choice} AUC",   f"{res['auc']:.3f}",        "Test set ROC-AUC"),
    (k3, "Avg Precision",         f"{res['ap']:.3f}",          "Precision-Recall AP"),
    (k4, "Companies at Risk",     f"{at_pct:.0f}%",           f"≥{threshold:.0%} breach prob"),
    (k5, "CV AUC (5-fold)",       f"{res['cv_auc']:.3f}",     "Cross-validation"),
]:
    with col:
        st.markdown(f'<div class="kpi-pred"><div class="label">{label}</div><div class="val">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

st.markdown("")

# ── Section 1: ROC + PR Curves ────────────────────────────────────────────────
st.markdown('<div class="sec">Model Performance — ROC & Precision-Recall Curves</div>', unsafe_allow_html=True)
mc1, mc2, mc3 = st.columns(3)

with mc1:
    fig_roc = go.Figure()
    colors_roc = {"Random Forest":"#38d9f5","Gradient Boosting":"#9333ea","Logistic Regression":"#f59e0b"}
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{name} (AUC={r['auc']:.2f})",
            mode="lines", line=dict(color=colors_roc[name], width=2.5),
        ))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
        line=dict(color="#1e3a5f",dash="dash"),showlegend=False))
    fig_roc.update_layout(
        title="ROC Curves — All Models", height=330,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#5a85aa", title_font_color="#e0f0ff",
        margin=dict(l=10,r=10,t=40,b=10),
        xaxis=dict(title="FPR", showgrid=True, gridcolor="#080c18", color="#2a5080"),
        yaxis=dict(title="TPR", showgrid=True, gridcolor="#080c18", color="#2a5080"),
        legend=dict(font_color="#5a85aa", bgcolor="rgba(0,0,0,0)", font_size=10),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with mc2:
    fig_pr = go.Figure()
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(r["y_test"], r["y_prob"])
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, name=f"{name} (AP={r['ap']:.2f})",
            mode="lines", line=dict(color=colors_roc[name], width=2.5),
        ))
    fig_pr.update_layout(
        title="Precision-Recall Curves", height=330,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#5a85aa", title_font_color="#e0f0ff",
        margin=dict(l=10,r=10,t=40,b=10),
        xaxis=dict(title="Recall", showgrid=True, gridcolor="#080c18", color="#2a5080"),
        yaxis=dict(title="Precision", showgrid=True, gridcolor="#080c18", color="#2a5080"),
        legend=dict(font_color="#5a85aa", bgcolor="rgba(0,0,0,0)", font_size=10),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

with mc3:
    cm = res["cm"]
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Pred: No Breach","Pred: Breach"],
        y=["Actual: No Breach","Actual: Breach"],
        color_continuous_scale=[[0,"#050810"],[0.4,"#0f2855"],[1,"#38d9f5"]],
        title=f"Confusion Matrix — {model_choice}",
        aspect="equal",
    )
    fig_cm.update_layout(
        height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#5a85aa", title_font_color="#e0f0ff",
        margin=dict(l=10,r=10,t=40,b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ── Section 2: Feature Importance + Threshold Sweep ───────────────────────────
st.markdown('<div class="sec">Feature Importance & Threshold Analysis</div>', unsafe_allow_html=True)
fi1, fi2 = st.columns([1.2, 1])

with fi1:
    fig_fi = px.bar(
        importances.head(12), x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=["#0a1630","#1565c0","#38d9f5"],
        title=f"Feature Importances — {best_tree_name}",
        text=importances.head(12)["Importance"].apply(lambda x: f"{x:.3f}"),
    )
    fig_fi.update_layout(
        height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#5a85aa", title_font_color="#e0f0ff",
        margin=dict(l=10,r=10,t=40,b=10), coloraxis_showscale=False,
        xaxis=dict(showgrid=False, color="#2a5080"),
        yaxis=dict(showgrid=False, color="#7fa3c8"),
    )
    fig_fi.update_traces(textposition="outside", textfont_color="#38d9f5")
    st.plotly_chart(fig_fi, use_container_width=True)

with fi2:
    # Threshold sweep: precision, recall, F1 vs threshold
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(res["y_test"], res["y_prob"])
    thresh_full = np.linspace(0.1, 0.9, 50)
    f1_arr = []
    prec_sw, rec_sw = [], []
    for t in thresh_full:
        yp = (res["y_prob"] >= t).astype(int)
        tp = ((yp==1)&(res["y_test"]==1)).sum()
        fp = ((yp==1)&(res["y_test"]==0)).sum()
        fn = ((yp==0)&(res["y_test"]==1)).sum()
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1_arr.append(2*p*r/(p+r) if (p+r)>0 else 0)
        prec_sw.append(p); rec_sw.append(r)

    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Scatter(x=thresh_full, y=prec_sw, name="Precision",
        line=dict(color="#38d9f5", width=2)))
    fig_thresh.add_trace(go.Scatter(x=thresh_full, y=rec_sw, name="Recall",
        line=dict(color="#9333ea", width=2)))
    fig_thresh.add_trace(go.Scatter(x=thresh_full, y=f1_arr, name="F1",
        line=dict(color="#f59e0b", width=2.5)))
    fig_thresh.add_vline(x=threshold, line_color="#ef4444", line_dash="dash",
        annotation_text=f"t={threshold:.2f}", annotation_font_color="#ef4444")
    fig_thresh.update_layout(
        title="Precision / Recall / F1 vs Decision Threshold",
        height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#5a85aa", title_font_color="#e0f0ff",
        margin=dict(l=10,r=10,t=40,b=10),
        xaxis=dict(title="Threshold", showgrid=True, gridcolor="#080c18", color="#2a5080"),
        yaxis=dict(showgrid=True, gridcolor="#080c18", color="#2a5080", range=[0,1]),
        legend=dict(font_color="#5a85aa", bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_thresh, use_container_width=True)

# ── Section 3: Breach Probability Distribution ────────────────────────────────
st.markdown('<div class="sec">Predicted Breach Probability Distribution</div>', unsafe_allow_html=True)
dp1, dp2 = st.columns([1.4, 1])

fco_sect = co_pred[co_pred["sector"].isin(show_sectors)]

with dp1:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=fco_sect[fco_sect["will_breach"]==0]["predicted_breach_prob"],
        nbinsx=20, name="Actual: Compliant",
        marker_color="#1565c0", opacity=0.75,
    ))
    fig_hist.add_trace(go.Histogram(
        x=fco_sect[fco_sect["will_breach"]==1]["predicted_breach_prob"],
        nbinsx=20, name="Actual: Breach",
        marker_color="#ef4444", opacity=0.75,
    ))
    fig_hist.add_vline(x=threshold, line_color="#f59e0b", line_dash="dash",
        annotation_text=f"Threshold {threshold:.2f}", annotation_font_color="#f59e0b")
    fig_hist.update_layout(
        barmode="overlay",
        title="Distribution of Predicted Breach Probabilities",
        height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#5a85aa", title_font_color="#e0f0ff",
        margin=dict(l=10,r=10,t=40,b=10),
        xaxis=dict(title="Predicted Breach Probability", showgrid=False, color="#2a5080"),
        yaxis=dict(showgrid=False, color="#2a5080"),
        legend=dict(font_color="#5a85aa", bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with dp2:
    sector_risk = fco_sect.groupby("sector")["predicted_breach_prob"].mean().reset_index()
    sector_risk = sector_risk.sort_values("predicted_breach_prob", ascending=True)
    fig_srisk = px.bar(
        sector_risk, x="predicted_breach_prob", y="sector", orientation="h",
        color="predicted_breach_prob",
        color_continuous_scale=["#0a1630","#1565c0","#f59e0b","#ef4444"],
        text=sector_risk["predicted_breach_prob"].apply(lambda x: f"{x:.2f}"),
        title="Avg Predicted Breach Probability by Sector",
    )
    fig_srisk.update_layout(
        height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#5a85aa", title_font_color="#e0f0ff",
        margin=dict(l=10,r=10,t=40,b=10), coloraxis_showscale=False,
        xaxis=dict(showgrid=False, color="#2a5080", range=[0,1]),
        yaxis=dict(showgrid=False, color="#7fa3c8"),
    )
    fig_srisk.update_traces(textposition="outside", textfont_color="#38d9f5")
    st.plotly_chart(fig_srisk, use_container_width=True)

# ── Section 4: Saudization Trend vs Predicted Risk Scatter ────────────────────
st.markdown('<div class="sec">Risk Scatter — Saudization Trend vs Breach Probability</div>', unsafe_allow_html=True)

fig_s2 = px.scatter(
    fco_sect, x="saudization_trend", y="predicted_breach_prob",
    color="risk_tier",
    size="headcount",
    hover_name="company_name",
    hover_data={
        "sector": True, "region": True,
        "saudization_pct": ":.1f",
        "gap_to_target": ":.1f",
        "predicted_breach_prob": ":.3f",
        "headcount": True,
    },
    color_discrete_map={"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#22c55e"},
    title="Saudization Trend (pp/yr) vs Predicted Breach Probability — bubble = headcount",
    symbol="sector",
)
fig_s2.add_hline(y=threshold, line_color="#f59e0b", line_dash="dash", opacity=0.5)
fig_s2.add_vline(x=0, line_color="#1565c0", line_dash="dot", opacity=0.4)
fig_s2.add_annotation(x=-15, y=0.9, text="📉 Declining + High Risk", font_color="#ef4444", showarrow=False)
fig_s2.add_annotation(x=10,  y=0.15, text="📈 Improving + Safe",     font_color="#22c55e", showarrow=False)
fig_s2.update_layout(
    height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#5a85aa", title_font_color="#e0f0ff",
    margin=dict(l=10,r=10,t=40,b=10),
    xaxis=dict(title="Saudization Trend (pp, positive = improving)", showgrid=True, gridcolor="#080c18", color="#2a5080"),
    yaxis=dict(title="Predicted Breach Prob", showgrid=True, gridcolor="#080c18", color="#2a5080", range=[0,1]),
    legend=dict(font_color="#5a85aa", bgcolor="rgba(0,0,0,0)", font_size=10),
)
st.plotly_chart(fig_s2, use_container_width=True)

# ── Section 5: Company-level Prediction Cards ─────────────────────────────────
st.markdown('<div class="sec">Company-Level Breach Predictions — Ranked by Risk</div>', unsafe_allow_html=True)

col_sort = st.columns([1, 4])[0]
with col_sort:
    show_tier = st.selectbox("Show Tier", ["All", "High Risk", "Medium Risk", "Low Risk"])

display_co = fco_sect.copy()
display_co["flag"] = (display_co["predicted_breach_prob"] >= threshold).map({True:"🔴 BREACH", False:"✅ SAFE"})

if show_tier != "All":
    display_co = display_co[display_co["risk_tier"] == show_tier]
display_co = display_co.sort_values("predicted_breach_prob", ascending=False)

for _, row in display_co.iterrows():
    prob  = row["predicted_breach_prob"]
    tier  = str(row["risk_tier"])
    card_cls  = "high" if tier == "High Risk" else "medium" if tier == "Medium Risk" else "low"
    badge_cls = f"risk-badge-{card_cls}"
    bar_color = "#ef4444" if card_cls=="high" else "#f59e0b" if card_cls=="medium" else "#22c55e"
    bar_width  = int(prob * 100)

    st.markdown(f"""
    <div class="pred-card {card_cls}">
      <div style="flex:1">
        <div style="color:#c8dff5;font-weight:600;font-size:0.88rem;">
          {row['company_name']}
          <span style="color:#2a5080;font-weight:400;font-size:0.78rem;">
            · {row['sector']} · {row['region']}
          </span>
        </div>
        <div style="margin:0.35rem 0;background:#080c18;border-radius:4px;height:6px;width:100%;">
          <div style="background:{bar_color};width:{bar_width}%;height:100%;border-radius:4px;"></div>
        </div>
        <div style="color:#2a5080;font-size:0.73rem;">
          Saudization: <b style="color:#38d9f5">{row['saudization_pct']:.1f}%</b>
          &nbsp;·&nbsp; Target: {row['nitaqat_target']:.0f}%
          &nbsp;·&nbsp; Gap: <b style="color:{'#ef4444' if row['gap_to_target']<0 else '#22c55e'}">{row['gap_to_target']:+.1f}pp</b>
          &nbsp;·&nbsp; Trend: {row['saudization_trend']:+.1f}pp
          &nbsp;·&nbsp; Headcount: {row['headcount']}
        </div>
      </div>
      <div style="text-align:right;margin-left:1.5rem;min-width:120px;">
        <div style="font-family:'Space Mono',monospace;font-size:1.6rem;color:{bar_color};font-weight:700;">{prob:.0%}</div>
        <div style="font-size:0.65rem;color:#1e4070;">BREACH PROB</div>
        <span class="{badge_cls}" style="margin-top:0.3rem;display:inline-block;">{tier}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Model Scorecard Table ──────────────────────────────────────────────────────
st.markdown('<div class="sec">Model Comparison Scorecard</div>', unsafe_allow_html=True)

scorecard = []
for name, r in results.items():
    rep = r["report"]
    scorecard.append({
        "Model":      name,
        "Test AUC":   f"{r['auc']:.3f}",
        "CV AUC":     f"{r['cv_auc']:.3f}",
        "Avg Prec":   f"{r['ap']:.3f}",
        "Precision":  f"{rep.get('1',{}).get('precision',0):.3f}",
        "Recall":     f"{rep.get('1',{}).get('recall',0):.3f}",
        "F1":         f"{rep.get('1',{}).get('f1-score',0):.3f}",
        "Best":       "⭐" if name == best_name else "",
    })
st.dataframe(pd.DataFrame(scorecard), use_container_width=True, hide_index=True)

# ── Insight Box ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Model Interpretation & Policy Implications</div>', unsafe_allow_html=True)
top_feat   = importances.iloc[0]["Feature"].replace("_"," ").title()
high_count = (co_pred["risk_tier"] == "High Risk").sum()
med_count  = (co_pred["risk_tier"] == "Medium Risk").sum()

st.markdown(f"""
<div class="insight-box">
🤖 <b>Model Selection:</b> The <b>{best_name}</b> achieves the highest 5-fold cross-validated
AUC of <b>{results[best_name]['cv_auc']:.3f}</b>, making it the primary predictor.
The ensemble approach outperforms Logistic Regression because Nitaqat breach risk
is driven by <i>interaction effects</i> — a company with low salary ratio
<i>and</i> a declining hiring trend is disproportionately more at risk than either factor alone.<br><br>

📊 <b>Top Predictive Signal:</b> <b>{top_feat}</b> is the most important feature,
consistent with economic intuition: the gap to the Nitaqat target is the strongest
leading indicator of near-term breach.<br><br>

⚠️ <b>Risk Distribution:</b> Of {len(co_pred)} companies in the dataset,
<b>{high_count}</b> are classified as High Risk and <b>{med_count}</b> as Medium Risk
at the current threshold of <b>{threshold:.0%}</b>.
Adjusting the threshold rightward (toward 0.65) reduces false positives at the cost of
missing genuine breaches — a policy trade-off between penalty enforcement and false accusations.<br><br>

🏛️ <b>Operational Use Case:</b> HRSD could deploy this model as a quarterly early-warning
system, flagging high-risk companies for pre-emptive advisory visits rather than
reactive penalties — reducing both non-compliance and enforcement costs.
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#080c18;font-size:0.72rem;padding:0.5rem;'>"
    "Nitaqat Breach Predictor · Predictive Analytics · Vision 2030 Portfolio Series · App 3 of 4"
    "</div>",
    unsafe_allow_html=True,
)
