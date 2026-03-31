"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         KPI Discovery & Rule Engine — Production Streamlit App              ║
║         Single-file, cloud-deployable, multi-tenant ready                   ║
║         Run: streamlit run kpi_dashboard.py                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import re
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KPI Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.stApp { background: #0C0F0E; color: #E4E0D6; }
.stApp > header { background: transparent; }
section[data-testid="stSidebar"] { background: #111511 !important; border-right: 1px solid #1e2a1e; }
section[data-testid="stSidebar"] * { color: #C5C1B6 !important; }

/* ── Sidebar nav items ── */
section[data-testid="stSidebar"] .stRadio label {
    cursor: pointer;
    padding: 6px 12px;
    border-radius: 4px;
    transition: background 0.15s;
    font-size: 13px;
    letter-spacing: 0.03em;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: #1a2a1a; }

/* ── Headers ── */
h1 { font-family: 'Fraunces', serif !important; font-weight: 300 !important; font-size: 2.1rem !important; color: #AECFA8 !important; letter-spacing: -0.02em; }
h2 { font-family: 'Fraunces', serif !important; font-weight: 300 !important; color: #D4D0C6 !important; font-size: 1.4rem !important; }
h3 { font-family: 'DM Mono', monospace !important; font-weight: 500 !important; color: #8EBF87 !important; font-size: 0.9rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── KPI Cards ── */
.kpi-card {
    background: #131A13;
    border: 1px solid #1e2a1e;
    border-radius: 6px;
    padding: 18px 22px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #3a5c35; }
.kpi-card .label { font-size: 11px; color: #6b7d66; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-card .value { font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 300; color: #AECFA8; line-height: 1; }
.kpi-card .delta { font-size: 12px; margin-top: 4px; }
.kpi-card .delta.pos { color: #6EC36A; }
.kpi-card .delta.neg { color: #C36A6A; }
.kpi-card .delta.neu { color: #7a7a6a; }

/* ── Rule pill ── */
.rule-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0e1a10; border: 1px solid #2a4a2a;
    border-radius: 20px; padding: 5px 14px;
    font-size: 12px; color: #A8D4A3; margin: 3px;
}
.rule-pill.violation { border-color: #6a2a2a; color: #D4A8A8; background: #1a0e0e; }

/* ── Section divider ── */
.section-divider { border: none; border-top: 1px solid #1a2a1a; margin: 20px 0; }

/* ── Badge ── */
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; font-weight: 500;
}
.badge-green { background: #1a3a1a; color: #6EC36A; }
.badge-amber { background: #2a2010; color: #C3A06A; }
.badge-red   { background: #2a1010; color: #C36A6A; }
.badge-blue  { background: #101a2a; color: #6A9EC3; }

/* ── Data table ── */
.stDataFrame { border-radius: 6px; overflow: hidden; }
.stDataFrame thead tr th { background: #131A13 !important; color: #8EBF87 !important; font-size: 11px !important; letter-spacing: 0.06em; }

/* ── Button ── */
.stButton > button {
    background: #1a3a1a; color: #AECFA8; border: 1px solid #2a5a2a;
    border-radius: 4px; font-family: 'DM Mono', monospace; font-size: 12px;
    letter-spacing: 0.05em; padding: 6px 16px; transition: all 0.15s;
}
.stButton > button:hover { background: #2a4a2a; border-color: #4a8a4a; color: #C4EFC4; }

/* ── Selectbox / inputs ── */
.stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div,
.stNumberInput > div > div { background: #111511 !important; color: #C5C1B6 !important; border-color: #1e2a1e !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background: #131A13 !important; color: #8EBF87 !important; font-size: 13px !important; }
.streamlit-expanderContent { background: #0e1510 !important; border-color: #1e2a1e !important; }

/* ── Progress / spinner ── */
.stProgress > div > div { background: #4a8a4a !important; }

/* ── Tooltip marker ── */
abbr[title] { text-decoration: underline dotted #3a5c35; cursor: help; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0C0F0E; }
::-webkit-scrollbar-thumb { background: #2a3a2a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG SYSTEM (multi-tenant)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "client_name": "Default Client",
    "theme_accent": "#AECFA8",
    "kpi_overrides": {},          # client-specific KPI definitions
    "rule_presets": [],            # pre-loaded rules for this client
    "anomaly_zscore_threshold": 2.5,
    "top_n_providers": 10,
    "date_format": "auto",
}

# Sample client configs for demo
CLIENT_CONFIGS = {
    "Default": DEFAULT_CONFIG,
    "HealthPlan Co.": {
        **DEFAULT_CONFIG,
        "client_name": "HealthPlan Co.",
        "kpi_overrides": {
            "total_paid": {"label": "Total Claims Paid", "format": "currency"},
            "denial_rate": {"label": "Denial Rate", "format": "percent"},
        },
        "rule_presets": [
            {"name": "High Denial Alert", "column": "denial_rate", "operator": ">", "threshold": 0.15, "priority": "High"},
        ],
    },
    "RetailMax": {
        **DEFAULT_CONFIG,
        "client_name": "RetailMax",
        "kpi_overrides": {
            "revenue": {"label": "Net Revenue", "format": "currency"},
            "units_sold": {"label": "Units Sold", "format": "number"},
        },
    },
}

def load_config(client_name: str) -> dict:
    return CLIENT_CONFIGS.get(client_name, DEFAULT_CONFIG)

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_data() -> pd.DataFrame:
    """Generate a realistic claims/sales-like dataset for demo purposes."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D").to_series().sample(n, replace=True).values
    statuses = np.random.choice(["paid","denied","pending","void"], n, p=[0.68, 0.18, 0.10, 0.04])
    providers = [f"Provider_{np.random.randint(1,51):03d}" for _ in range(n)]
    diag_codes = np.random.choice([f"Z{np.random.randint(10,99)}" for _ in range(20)], n)
    charge_amt = np.random.lognormal(7.5, 1.2, n).round(2)
    paid_amt   = np.where(statuses=="paid", charge_amt * np.random.uniform(0.55, 0.95, n), 0).round(2)
    allowed    = (charge_amt * np.random.uniform(0.60, 0.98, n)).round(2)
    members    = [f"MBR{np.random.randint(1,201):04d}" for _ in range(n)]
    network    = np.random.choice(["in-network","out-of-network"], n, p=[0.75, 0.25])
    service_types = np.random.choice(["Medical","Pharmacy","Dental","Vision","Mental Health"], n, p=[0.5,0.2,0.15,0.1,0.05])
    plan_names = np.random.choice(["Gold PPO","Silver HMO","Bronze EPO","Platinum PPO"], n)

    return pd.DataFrame({
        "claim_id":     [f"CLM{i:06d}" for i in range(1, n+1)],
        "claim_date":   pd.to_datetime(dates),
        "paid_date":    pd.to_datetime(dates) + pd.to_timedelta(np.random.randint(1,45,n), unit="D"),
        "member_id":    members,
        "provider_id":  providers,
        "diag_code":    diag_codes,
        "service_type": service_types,
        "plan_name":    plan_names,
        "network":      network,
        "status":       statuses,
        "charge_amt":   charge_amt,
        "allowed_amt":  allowed,
        "paid_amt":     paid_amt,
    })

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Keywords suggesting a column is KPI-relevant
KPI_KEYWORDS = [
    "revenue","sales","profit","loss","cost","expense","paid","charge","amount","amt",
    "count","total","sum","rate","ratio","score","value","growth","volume","spend",
    "denied","approved","pending","rate","pct","percent","avg","mean","median",
    "utilization","frequency","days","duration","lag","gap","conversion","retention",
]

def detect_column_types(df: pd.DataFrame) -> dict:
    """Classify each column as numeric, categorical, date, or id."""
    schema = {}
    for col in df.columns:
        col_lower = col.lower()
        sample = df[col].dropna()
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            kind = "date"
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Distinguish id-like integers from real metrics
            nunique_ratio = df[col].nunique() / max(len(df), 1)
            if nunique_ratio > 0.95 and df[col].dtype in [np.int64, np.int32]:
                kind = "id"
            else:
                kind = "numeric"
        elif sample.dtype == object:
            # Try to coerce to date
            try:
                pd.to_datetime(sample.head(50), infer_datetime_format=True)
                kind = "date_candidate"
            except Exception:
                nunique = df[col].nunique()
                kind = "id" if nunique / max(len(df), 1) > 0.9 else "categorical"
        else:
            kind = "other"
        # Mark KPI-likely fields
        is_kpi = any(kw in col_lower for kw in KPI_KEYWORDS) and kind in ("numeric", "categorical")
        schema[col] = {"kind": kind, "is_kpi": is_kpi, "nunique": df[col].nunique(), "nulls": df[col].isna().sum()}
    return schema

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame, schema: dict) -> tuple:
    """Clean, coerce types, and return a log of actions taken."""
    log = []
    df = df.copy()

    # Coerce date candidates
    for col, info in schema.items():
        if info["kind"] == "date_candidate":
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                schema[col]["kind"] = "date"
                log.append(f"✦ '{col}' coerced to datetime")
            except Exception:
                pass

    # Numeric: strip currency chars, coerce
    for col, info in schema.items():
        if info["kind"] == "numeric" and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r"[\$,\s%]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            log.append(f"✦ '{col}' cleaned and coerced to numeric")

    # Fill missing numerics with median; categoricals with 'Unknown'
    null_cols = {col: info["nulls"] for col, info in schema.items() if info["nulls"] > 0}
    for col, cnt in null_cols.items():
        if col not in df.columns:
            continue
        if schema[col]["kind"] == "numeric":
            med = df[col].median()
            df[col].fillna(med, inplace=True)
            log.append(f"✦ '{col}' — {cnt} nulls filled with median ({med:.2f})")
        elif schema[col]["kind"] == "categorical":
            df[col].fillna("Unknown", inplace=True)
            log.append(f"✦ '{col}' — {cnt} nulls filled with 'Unknown'")

    # Deduplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < before:
        log.append(f"✦ {before - len(df)} duplicate rows removed")

    if not log:
        log.append("✦ Data already clean — no transformations needed")

    return df, log

# ─────────────────────────────────────────────────────────────────────────────
# KPI DETECTION & SCORING
# ─────────────────────────────────────────────────────────────────────────────

def detect_kpis(df: pd.DataFrame, schema: dict, config: dict) -> list:
    """
    Dynamically generate KPI definitions from numeric columns.
    Returns list of KPI dicts with name, label, formula, value, importance_score.
    """
    kpis = []
    numeric_cols = [c for c, info in schema.items() if info["kind"] == "numeric"]
    date_cols    = [c for c, info in schema.items() if info["kind"] == "date"]
    cat_cols     = [c for c, info in schema.items() if info["kind"] == "categorical"]

    def importance(col):
        """Score 0–1 based on keyword match + variance coefficient."""
        col_lower = col.lower()
        kw_score  = sum(1 for kw in KPI_KEYWORDS if kw in col_lower) / 3
        vals = df[col].dropna()
        if len(vals) == 0 or vals.std() == 0:
            var_score = 0
        else:
            var_score = min(vals.std() / (abs(vals.mean()) + 1e-9), 1.0)
        return min(round(0.6 * kw_score + 0.4 * var_score, 3), 1.0)

    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        imp = importance(col)
        override = config.get("kpi_overrides", {}).get(col, {})

        # Detect format hint from column name
        col_lower = col.lower()
        if any(x in col_lower for x in ["rate","pct","percent","ratio"]):
            fmt_hint = "percent"
        elif any(x in col_lower for x in ["amt","amount","cost","revenue","paid","charge","spend","value"]):
            fmt_hint = "currency"
        else:
            fmt_hint = "number"

        fmt_hint = override.get("format", fmt_hint)
        label    = override.get("label", col.replace("_"," ").title())

        base_kpis = [
            {
                "id":        f"sum_{col}",
                "column":    col,
                "label":     f"Total {label}",
                "formula":   "SUM",
                "value":     float(vals.sum()),
                "fmt":       fmt_hint,
                "importance": round(imp + 0.1, 3),
                "category":  "Volume",
                "selected":  imp > 0.2,
            },
            {
                "id":        f"avg_{col}",
                "column":    col,
                "label":     f"Avg {label}",
                "formula":   "MEAN",
                "value":     float(vals.mean()),
                "fmt":       fmt_hint,
                "importance": round(imp, 3),
                "category":  "Cost",
                "selected":  imp > 0.3,
            },
            {
                "id":        f"median_{col}",
                "column":    col,
                "label":     f"Median {label}",
                "formula":   "MEDIAN",
                "value":     float(vals.median()),
                "fmt":       fmt_hint,
                "importance": round(imp * 0.8, 3),
                "category":  "Cost",
                "selected":  False,
            },
            {
                "id":        f"p90_{col}",
                "column":    col,
                "label":     f"P90 {label}",
                "formula":   "PERCENTILE(90)",
                "value":     float(np.percentile(vals, 90)),
                "fmt":       fmt_hint,
                "importance": round(imp * 0.9, 3),
                "category":  "Outliers",
                "selected":  imp > 0.4,
            },
        ]
        kpis.extend(base_kpis)

    # Categorical KPIs (rates, distributions)
    for col in cat_cols:
        if df[col].nunique() > 15:
            continue
        vc = df[col].value_counts(normalize=True)
        for val, rate in vc.items():
            safe_val = str(val).lower().replace(" ", "_")
            kpis.append({
                "id":        f"rate_{col}_{safe_val}",
                "column":    col,
                "label":     f"{str(val).title()} Rate ({col.replace('_',' ').title()})",
                "formula":   f"COUNT({val}) / N",
                "value":     float(rate),
                "fmt":       "percent",
                "importance": 0.6 if any(kw in col.lower() for kw in ["status","network","type"]) else 0.3,
                "category":  "Utilization",
                "selected":  any(kw in col.lower() for kw in ["status","network","denial","denied"]) and rate > 0.05,
            })

    # Sort by importance descending
    kpis.sort(key=lambda k: k["importance"], reverse=True)
    return kpis

# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, col: str, z_thresh: float = 2.5) -> pd.Series:
    """Return boolean mask of anomalous rows using Z-score method."""
    vals = pd.to_numeric(df[col], errors="coerce")
    z = (vals - vals.mean()) / (vals.std() + 1e-9)
    return z.abs() > z_thresh

# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

OPERATORS = {
    ">":  lambda s, t: s > t,
    ">=": lambda s, t: s >= t,
    "<":  lambda s, t: s < t,
    "<=": lambda s, t: s <= t,
    "==": lambda s, t: s == t,
    "!=": lambda s, t: s != t,
}

def evaluate_rule(df: pd.DataFrame, rule: dict) -> dict:
    """Evaluate a single rule against the dataframe. Returns violation info."""
    col  = rule["column"]
    op   = rule["operator"]
    thr  = float(rule["threshold"])

    if col not in df.columns:
        return {"violations": 0, "total": len(df), "rows": pd.DataFrame(), "error": f"Column '{col}' not found"}

    if rule.get("formula") == "SUM":
        series_val = df[col].sum()
        violated = OPERATORS[op](series_val, thr)
        return {
            "violations": int(violated),
            "total": 1,
            "aggregate_value": series_val,
            "rows": df if violated else pd.DataFrame(),
            "error": None,
        }
    elif rule.get("formula") == "MEAN":
        series_val = df[col].mean()
        violated = OPERATORS[op](series_val, thr)
        return {
            "violations": int(violated),
            "total": 1,
            "aggregate_value": series_val,
            "rows": df if violated else pd.DataFrame(),
            "error": None,
        }
    else:
        # Row-level rule
        mask = OPERATORS[op](pd.to_numeric(df[col], errors="coerce"), thr)
        return {
            "violations": int(mask.sum()),
            "total": len(df),
            "rows": df[mask],
            "error": None,
        }

# ─────────────────────────────────────────────────────────────────────────────
# VALUE FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def format_value(val: float, fmt: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if fmt == "currency":
        if abs(val) >= 1_000_000:
            return f"${val/1_000_000:.2f}M"
        elif abs(val) >= 1_000:
            return f"${val/1_000:.1f}K"
        return f"${val:,.2f}"
    elif fmt == "percent":
        return f"{val*100:.1f}%"
    else:
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:.2f}M"
        elif abs(val) >= 1_000:
            return f"{val/1_000:.1f}K"
        return f"{val:,.2f}"

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

CHART_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "font":          {"color": "#C5C1B6", "family": "DM Mono, monospace", "size": 11},
    "colorway":      ["#AECFA8","#6EC36A","#C3A06A","#6A9EC3","#C36A6A","#A8A8D4","#C3C3A0","#6AD4C3"],
    "xaxis":         {"gridcolor": "#1a2a1a", "linecolor": "#1a2a1a", "tickcolor": "#2a3a2a"},
    "yaxis":         {"gridcolor": "#1a2a1a", "linecolor": "#1a2a1a", "tickcolor": "#2a3a2a"},
}

def apply_theme(fig):
    fig.update_layout(**{k: v for k, v in CHART_THEME.items() if k not in ("xaxis","yaxis")})
    fig.update_xaxes(gridcolor=CHART_THEME["xaxis"]["gridcolor"], linecolor=CHART_THEME["xaxis"]["linecolor"])
    fig.update_yaxes(gridcolor=CHART_THEME["yaxis"]["gridcolor"], linecolor=CHART_THEME["yaxis"]["linecolor"])
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 20px; border-bottom: 1px solid #1e2a1e; margin-bottom: 20px;">
            <div style="font-family: 'Fraunces', serif; font-size: 1.3rem; color: #AECFA8; font-weight: 300;">◈ KPI Intelligence</div>
            <div style="font-size: 10px; color: #4a5a44; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 3px;">Rule Engine Platform</div>
        </div>
        """, unsafe_allow_html=True)

        # Client selector
        client = st.selectbox("Client Profile", list(CLIENT_CONFIGS.keys()), key="client_select")

        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

        # Navigation
        page = st.radio(
            "Navigation",
            ["① Upload & Preview", "② Schema & Cleaning", "③ KPI Selection", "④ Rule Engine", "⑤ Dashboard"],
            label_visibility="collapsed",
        )

        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

        # Quick stats if data loaded
        if st.session_state.get("df") is not None:
            df = st.session_state["df"]
            _rows = format(len(df), ",")
            _cols = len(df.columns)
            st.markdown(f"""
            <div style="font-size: 10px; color: #4a5a44; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">Dataset</div>
            <div style="font-size: 12px; color: #8EBF87; margin-bottom: 3px;">{_rows} rows</div>
            <div style="font-size: 12px; color: #8EBF87; margin-bottom: 3px;">{_cols} columns</div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="position: fixed; bottom: 20px; left: 0; width: 230px; padding: 0 20px; font-size: 10px; color: #2a3a2a; letter-spacing: 0.06em;">
            KPI Intelligence Platform v1.0
        </div>
        """, unsafe_allow_html=True)

    return page, client

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def page_upload(config: dict):
    st.title("Data Ingestion")
    st.markdown("<p style='color:#6b7d66; font-size:13px; margin-top:-8px;'>Upload a CSV or Excel file to begin KPI discovery</p>", unsafe_allow_html=True)

    col_up, col_demo = st.columns([3, 1], gap="large")

    with col_up:
        uploaded = st.file_uploader(
            "Drop file here",
            type=["csv","xlsx","xls"],
            help="Max ~50 MB recommended. First sheet used for Excel files.",
        )

    with col_demo:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Load Sample Data", use_container_width=True):
            df = generate_sample_data()
            st.session_state["df"]       = df
            st.session_state["schema"]   = {}
            st.session_state["kpis"]     = []
            st.session_state["rules"]    = config.get("rule_presets", [])
            st.session_state["file_name"] = "sample_claims_data.csv"
            st.success(f"✦ Sample dataset loaded — {len(df):,} rows")
            st.rerun()

    if uploaded:
        with st.spinner("Parsing file…"):
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded, sheet_name=0)

                if len(df) > 500_000:
                    st.warning(f"⚠ File has {len(df):,} rows — sampling first 200k for performance.")
                    df = df.sample(200_000, random_state=42).reset_index(drop=True)

                st.session_state["df"]        = df
                st.session_state["schema"]    = {}
                st.session_state["kpis"]      = []
                st.session_state["rules"]     = config.get("rule_presets", [])
                st.session_state["file_name"] = uploaded.name
                st.success(f"✦ '{uploaded.name}' loaded — {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Failed to parse file: {e}")
                return

    if st.session_state.get("df") is not None:
        df = st.session_state["df"]
        st.markdown("### Preview")
        st.dataframe(df.head(20), use_container_width=True, height=340)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Numeric cols", sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns))
        col4.metric("Missing values", f"{df.isna().sum().sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SCHEMA & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def page_schema(config: dict):
    st.title("Schema Detection & Cleaning")

    if st.session_state.get("df") is None:
        st.info("↩ Please upload data first.")
        return

    df = st.session_state["df"]

    with st.spinner("Detecting schema…"):
        schema = detect_column_types(df)
        st.session_state["schema"] = schema

    # Schema table
    st.markdown("### Detected Column Types")
    rows_out = []
    for col, info in schema.items():
        badge = {
            "numeric": '<span class="badge badge-green">numeric</span>',
            "date":    '<span class="badge badge-blue">date</span>',
            "date_candidate": '<span class="badge badge-blue">date?</span>',
            "categorical": '<span class="badge badge-amber">categorical</span>',
            "id":      '<span class="badge badge-red">id/key</span>',
        }.get(info["kind"], '<span class="badge">other</span>')
        kpi_flag = "✦" if info["is_kpi"] else ""
        rows_out.append({
            "Column": col,
            "Type": info["kind"],
            "KPI-like": kpi_flag,
            "Unique Values": info["nunique"],
            "Null Count": info["nulls"],
        })

    schema_df = pd.DataFrame(rows_out)
    st.dataframe(schema_df, use_container_width=True, hide_index=True, height=300)

    # Clean button
    if st.button("⟳ Run Data Cleaning", use_container_width=False):
        with st.spinner("Cleaning data…"):
            df_clean, log = clean_data(df, schema)
            st.session_state["df"]     = df_clean
            st.session_state["schema"] = detect_column_types(df_clean)
            st.session_state["clean_log"] = log

    if st.session_state.get("clean_log"):
        st.markdown("### Cleaning Actions")
        for entry in st.session_state["clean_log"]:
            st.markdown(f"<div style='font-size:12px; color:#8EBF87; padding:2px 0'>{entry}</div>", unsafe_allow_html=True)

    # Missingness chart
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=True)
    if len(nulls):
        st.markdown("### Missingness")
        fig = px.bar(
            x=nulls.values, y=nulls.index,
            orientation="h",
            labels={"x": "Null Count", "y": ""},
            color_discrete_sequence=["#C3A06A"],
        )
        fig.update_layout(height=max(120, len(nulls)*28), margin=dict(l=0,r=0,t=10,b=10))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: KPI SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def page_kpi_selection(config: dict):
    st.title("KPI Discovery & Selection")

    if st.session_state.get("df") is None:
        st.info("↩ Please upload data first.")
        return

    df     = st.session_state["df"]
    schema = st.session_state.get("schema") or detect_column_types(df)

    if not st.session_state.get("kpis"):
        with st.spinner("Generating KPI catalogue…"):
            kpis = detect_kpis(df, schema, config)
            st.session_state["kpis"] = kpis

    kpis = st.session_state["kpis"]

    # Controls
    col_a, col_b, col_c = st.columns([2,2,2])
    with col_a:
        cat_filter = st.multiselect("Filter by category", ["Volume","Cost","Utilization","Outliers"], default=["Volume","Cost","Utilization","Outliers"])
    with col_b:
        imp_min = st.slider("Min importance score", 0.0, 1.0, 0.0, 0.05)
    with col_c:
        if st.button("Select All Visible", use_container_width=True):
            for k in kpis:
                if k["category"] in cat_filter and k["importance"] >= imp_min:
                    k["selected"] = True
        if st.button("Deselect All", use_container_width=True):
            for k in kpis:
                k["selected"] = False

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # KPI cards in a grid
    visible = [k for k in kpis if k["category"] in cat_filter and k["importance"] >= imp_min]

    if not visible:
        st.info("No KPIs match current filters.")
        return

    # Render as columns of 3
    for i in range(0, len(visible), 3):
        cols = st.columns(3, gap="medium")
        for j, col in enumerate(cols):
            if i + j >= len(visible):
                break
            k = visible[i + j]
            imp_pct  = int(k["importance"] * 100)
            imp_color = "#6EC36A" if imp_pct >= 60 else "#C3A06A" if imp_pct >= 30 else "#6b7d66"
            formatted = format_value(k["value"], k["fmt"])

            with col:
                selected = st.checkbox(
                    k["label"],
                    value=k["selected"],
                    key=f"kpi_sel_{k['id']}",
                    help=f"Formula: {k['formula']} · Category: {k['category']}",
                )
                k["selected"] = selected
                st.markdown(f"""
                <div class="kpi-card" style="margin-top:-8px">
                    <div class="label">{k['category']} · {k['formula']}</div>
                    <div class="value">{formatted}</div>
                    <div style="font-size:10px; color:{imp_color}; margin-top:6px;">
                        importance {imp_pct}%
                        <span style="display:inline-block;width:{imp_pct}px;height:3px;background:{imp_color};border-radius:2px;margin-left:6px;vertical-align:middle;max-width:80px;"></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
    selected_count = sum(1 for k in kpis if k["selected"])
    st.markdown(f"<div style='font-size:12px; color:#8EBF87;'>✦ {selected_count} KPIs selected</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def page_rule_engine(config: dict):
    st.title("Rule Engine")
    st.markdown("<p style='color:#6b7d66; font-size:13px; margin-top:-8px;'>Define threshold and trend-based alert rules</p>", unsafe_allow_html=True)

    if st.session_state.get("df") is None:
        st.info("↩ Please upload data first.")
        return

    df     = st.session_state["df"]
    schema = st.session_state.get("schema") or detect_column_types(df)

    if "rules" not in st.session_state:
        st.session_state["rules"] = list(config.get("rule_presets", []))

    rules = st.session_state["rules"]

    # Rule builder
    st.markdown("### Add Rule")
    numeric_cols = [c for c, info in schema.items() if info["kind"] == "numeric"]

    with st.container():
        r1, r2, r3, r4, r5, r6 = st.columns([3, 2, 1.5, 2, 2, 1.5])
        with r1: rule_col   = st.selectbox("Column", numeric_cols, key="rc")
        with r2: rule_formula = st.selectbox("Aggregation", ["Row-level","SUM","MEAN"], key="rf")
        with r3: rule_op    = st.selectbox("Operator", list(OPERATORS.keys()), key="ro")
        with r4: rule_thr   = st.number_input("Threshold", value=0.0, key="rt", format="%.2f")
        with r5: rule_name  = st.text_input("Rule name (optional)", key="rn")
        with r6:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            rule_priority = st.selectbox("Priority", ["High","Med","Low"], key="rp")

        add_col, _ = st.columns([1, 4])
        with add_col:
            if st.button("+ Add Rule", use_container_width=True):
                formula = None if rule_formula == "Row-level" else rule_formula
                new_rule = {
                    "id":        f"rule_{len(rules)+1}_{rule_col}",
                    "name":      rule_name or f"{rule_col} {rule_op} {rule_thr}",
                    "column":    rule_col,
                    "operator":  rule_op,
                    "threshold": rule_thr,
                    "formula":   formula,
                    "priority":  rule_priority,
                }
                rules.append(new_rule)
                st.session_state["rules"] = rules
                st.rerun()

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    if not rules:
        st.info("No rules defined yet. Add one above or load a client profile with presets.")
        return

    st.markdown("### Active Rules")
    for idx, rule in enumerate(rules):
        result = evaluate_rule(df, rule)
        viol   = result.get("violations", 0)
        total  = result.get("total", len(df))
        pct_v  = viol / max(total, 1)
        is_agg = rule.get("formula") in ("SUM", "MEAN")

        priority_colors = {"High": "#C36A6A", "Med": "#C3A06A", "Low": "#6A9EC3"}
        p_color = priority_colors.get(rule.get("priority","Med"), "#8EBF87")

        with st.expander(f"{'🔴' if viol else '🟢'}  {rule['name']}", expanded=viol > 0):
            exp_c1, exp_c2, exp_c3 = st.columns(3)
            with exp_c1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="label">Rule</div>
                    <div style="font-size:13px; color:#C5C1B6; font-family:'DM Mono',monospace;">
                        {'SUM' if rule.get('formula')=='SUM' else 'AVG' if rule.get('formula')=='MEAN' else 'ROW'}({rule['column']}) {rule['operator']} {format(rule['threshold'], ',.2f')}
                    </div>
                    <div style="margin-top:6px;">
                        <span class="badge" style="background:{p_color}22; color:{p_color}">{rule.get('priority','Med')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with exp_c2:
                if is_agg:
                    agg_val = result.get("aggregate_value", 0)
                    _agg_fmt = format(agg_val, ",.2f")
                    _delta_cls = 'neg' if viol else 'pos'
                    _delta_txt = '⚠ VIOLATED' if viol else '✓ OK'
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="label">Aggregate Value</div>
                        <div class="value" style="font-size:1.4rem;">{_agg_fmt}</div>
                        <div class="delta {_delta_cls}">{_delta_txt}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    _viol_color = "#C36A6A" if viol else "#6EC36A"
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="label">Violations</div>
                        <div class="value" style="font-size:1.4rem; color:{_viol_color}">{format(viol, ",")}</div>
                        <div class="delta neu">of {format(total, ",")} rows ({round(pct_v*100,1)}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
            with exp_c3:
                if not is_agg and viol > 0 and len(result.get("rows", [])):
                    viol_rows = result["rows"]
                    cols_to_show = [rule["column"]] + [c for c in df.columns if c != rule["column"]][:3]
                    st.dataframe(viol_rows[cols_to_show].head(5), use_container_width=True, hide_index=True)

            del_col, _ = st.columns([1, 5])
            with del_col:
                if st.button("Delete rule", key=f"del_{idx}"):
                    rules.pop(idx)
                    st.session_state["rules"] = rules
                    st.rerun()

    # Export rules
    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
    rules_json = json.dumps(rules, indent=2)
    st.download_button(
        "↓ Export Rules as JSON",
        data=rules_json,
        file_name=f"rules_{config['client_name'].replace(' ','_').lower()}_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def page_dashboard(config: dict):
    st.title("Executive Dashboard")

    if st.session_state.get("df") is None:
        st.info("↩ Please upload data first.")
        return

    df     = st.session_state["df"]
    schema = st.session_state.get("schema") or detect_column_types(df)
    kpis   = [k for k in st.session_state.get("kpis", []) if k.get("selected")]
    rules  = st.session_state.get("rules", [])

    if not kpis:
        st.info("↩ No KPIs selected. Go to KPI Selection first.")
        return

    numeric_cols  = [c for c, info in schema.items() if info["kind"] == "numeric"]
    date_cols     = [c for c, info in schema.items() if info["kind"] == "date"]
    cat_cols      = [c for c, info in schema.items() if info["kind"] == "categorical"]

    # ── KPI Cards ────────────────────────────────────────────────────────────
    st.markdown("### Key Metrics")
    kpi_cols = st.columns(min(len(kpis), 4), gap="medium")
    for i, k in enumerate(kpis[:8]):
        with kpi_cols[i % 4]:
            formatted = format_value(k["value"], k["fmt"])
            imp_color = "#6EC36A" if k["importance"] > 0.6 else "#C3A06A" if k["importance"] > 0.3 else "#6b7d66"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="label">{k['label']}</div>
                <div class="value">{formatted}</div>
                <div style="font-size:10px; color:{imp_color}; margin-top:4px;">
                    <span class="badge badge-{'green' if k['importance']>0.6 else 'amber' if k['importance']>0.3 else 'blue'}">
                        {k['category']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Rule Violations Summary ───────────────────────────────────────────────
    if rules:
        st.markdown("### Rule Status")
        rule_cols = st.columns(min(len(rules), 4), gap="medium")
        for i, rule in enumerate(rules[:8]):
            res = evaluate_rule(df, rule)
            viol = res.get("violations", 0)
            is_agg = rule.get("formula") in ("SUM","MEAN")
            status_icon = "⚠" if viol else "✓"
            status_color = "#C36A6A" if viol else "#6EC36A"
            disp_val = format(res.get('aggregate_value', 0), ",.0f") if is_agg else f"{format(viol, ',')} rows"
            with rule_cols[i % 4]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="label">{status_icon} {rule.get('priority','Med')} · Rule</div>
                    <div style="font-size:14px; color:{status_color}; font-weight:500; margin-bottom:4px;">{rule['name']}</div>
                    <div class="delta" style="color:{status_color};">{disp_val}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2, gap="large")

    # Time series if date available
    with chart_col1:
        if date_cols and numeric_cols:
            dc   = date_cols[0]
            nc   = kpis[0]["column"] if kpis else numeric_cols[0]
            st.markdown(f"### {nc.replace('_',' ').title()} Over Time")
            ts = df[[dc, nc]].dropna().copy()
            ts[dc] = pd.to_datetime(ts[dc])
            ts = ts.groupby(ts[dc].dt.to_period("M"))[nc].sum().reset_index()
            ts[dc] = ts[dc].dt.to_timestamp()
            fig = px.line(ts, x=dc, y=nc, markers=True, color_discrete_sequence=["#AECFA8"])
            fig.update_traces(line=dict(width=2), marker=dict(size=5))
            fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="", yaxis_title="")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Categorical breakdown
    with chart_col2:
        if cat_cols and numeric_cols:
            best_cat = next((c for c in cat_cols if df[c].nunique() <= 10), cat_cols[0] if cat_cols else None)
            nc2 = kpis[0]["column"] if kpis else numeric_cols[0]
            if best_cat:
                st.markdown(f"### {nc2.replace('_',' ').title()} by {best_cat.replace('_',' ').title()}")
                grp = df.groupby(best_cat)[nc2].sum().sort_values(ascending=True).tail(10)
                fig2 = px.bar(
                    x=grp.values, y=grp.index,
                    orientation="h",
                    color_discrete_sequence=["#6A9EC3"],
                )
                fig2.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="", yaxis_title="")
                apply_theme(fig2)
                st.plotly_chart(fig2, use_container_width=True)

    # Second row
    chart_col3, chart_col4 = st.columns(2, gap="large")

    with chart_col3:
        if len(cat_cols) >= 1:
            vc_col = next((c for c in cat_cols if 1 < df[c].nunique() <= 8), None)
            if vc_col:
                st.markdown(f"### {vc_col.replace('_',' ').title()} Distribution")
                vc = df[vc_col].value_counts()
                fig3 = px.pie(
                    values=vc.values, names=vc.index,
                    hole=0.55,
                    color_discrete_sequence=CHART_THEME["colorway"],
                )
                fig3.update_traces(textinfo="percent+label", textfont_size=10)
                fig3.update_layout(height=260, margin=dict(l=0,r=20,t=10,b=0), showlegend=False)
                apply_theme(fig3)
                st.plotly_chart(fig3, use_container_width=True)

    # Anomaly scatter for first numeric KPI
    with chart_col4:
        if numeric_cols:
            anc = kpis[0]["column"] if kpis else numeric_cols[0]
            st.markdown(f"### {anc.replace('_',' ').title()} — Anomaly Detection")
            anom_mask = detect_anomalies(df, anc, config.get("anomaly_zscore_threshold", 2.5))
            plot_df = df[[anc]].copy()
            plot_df["anomaly"] = anom_mask
            plot_df["index"]   = range(len(plot_df))
            fig4 = px.scatter(
                plot_df, x="index", y=anc,
                color="anomaly",
                color_discrete_map={True: "#C36A6A", False: "#AECFA8"},
                opacity=0.7,
                labels={"index": "Row #", anc: anc.replace("_"," ").title()},
            )
            fig4.update_traces(marker=dict(size=4))
            fig4.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0), showlegend=True,
                               legend=dict(title="Anomaly", font=dict(size=10), yanchor="top", y=0.99, xanchor="right", x=0.99))
            apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)
            anom_count = int(anom_mask.sum())
            st.markdown(f"<div style='font-size:11px; color:#C36A6A; margin-top:-8px;'>⚠ {anom_count} anomalies detected (Z > {config.get('anomaly_zscore_threshold',2.5)})</div>", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.markdown("### Correlation Matrix")
        corr_cols = numeric_cols[:12]
        corr_df = df[corr_cols].dropna().corr()
        fig5 = px.imshow(
            corr_df,
            color_continuous_scale=[[0,"#1a0e0e"],[0.5,"#1a1a2a"],[1,"#1a3a1a"]],
            aspect="auto",
            zmin=-1, zmax=1,
            text_auto=".2f",
        )
        fig5.update_traces(textfont_size=9)
        fig5.update_layout(
            height=max(200, len(corr_cols)*30),
            margin=dict(l=0,r=0,t=10,b=0),
            coloraxis_showscale=False,
        )
        apply_theme(fig5)
        st.plotly_chart(fig5, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        csv_buf = df.to_csv(index=False).encode()
        st.download_button("↓ Export Dataset (CSV)", data=csv_buf, file_name="kpi_data.csv", mime="text/csv")

    with dl_col2:
        kpi_export = [{"id": k["id"], "label": k["label"], "formula": k["formula"],
                        "value": k["value"], "format": k["fmt"], "importance": k["importance"]} for k in kpis]
        st.download_button("↓ Export KPIs (JSON)", data=json.dumps(kpi_export, indent=2),
                           file_name="kpis.json", mime="application/json")

    with dl_col3:
        rules = st.session_state.get("rules", [])
        full_config = {**config, "kpis": kpi_export, "rules": rules, "exported_at": datetime.now().isoformat()}
        st.download_button("↓ Export Full Config (JSON)", data=json.dumps(full_config, indent=2),
                           file_name=f"config_{config['client_name'].replace(' ','_').lower()}.json",
                           mime="application/json")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Initialize session state defaults
    for key, default in [("df", None), ("schema", {}), ("kpis", []), ("rules", []), ("clean_log", [])]:
        if key not in st.session_state:
            st.session_state[key] = default

    page, client = render_sidebar()
    config = load_config(client)

    if   page == "① Upload & Preview":  page_upload(config)
    elif page == "② Schema & Cleaning": page_schema(config)
    elif page == "③ KPI Selection":     page_kpi_selection(config)
    elif page == "④ Rule Engine":       page_rule_engine(config)
    elif page == "⑤ Dashboard":         page_dashboard(config)

if __name__ == "__main__":
    main()
