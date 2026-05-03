import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, classification_report,
    ConfusionMatrixDisplay
)
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Credit Risk Predictor | MSc Project',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🏦'
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.9rem;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}

/* Main background */
.main { background: #f8fafc; }
.block-container { padding: 1.5rem 2rem 3rem; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: white;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
div[data-testid="stMetric"] label {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 600;
    color: #0f172a !important;
}

/* Section titles */
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 0.25rem;
}
.section-sub {
    font-size: 0.875rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}
.badge-green { background: #dcfce7; color: #166534; }
.badge-orange { background: #fff7ed; color: #9a3412; }
.badge-red { background: #fee2e2; color: #991b1b; }
.badge-blue { background: #dbeafe; color: #1e40af; }

/* Divider */
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }

/* Chart card */
.chart-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Risk gauge */
.risk-bar-wrap {
    background: #f1f5f9;
    border-radius: 99px;
    height: 14px;
    overflow: hidden;
    margin: 0.5rem 0;
}

/* Buttons */
.stButton>button {
    background: #1e40af;
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-weight: 500;
    font-size: 0.9rem;
    transition: background 0.2s;
}
.stButton>button:hover { background: #1d4ed8; }

/* Info boxes */
.info-box {
    background: #eff6ff;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.875rem;
    color: #1e40af;
}
.warn-box {
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.875rem;
    color: #92400e;
}

/* Selectbox and radio */
div[data-testid="stRadio"] > label { font-weight: 500; color: #374151; }

/* Hide hamburger */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = '#2563eb'
GREEN  = '#16a34a'
AMBER  = '#d97706'
RED    = '#dc2626'
SLATE  = '#475569'
LIGHT  = '#f8fafc'
BORDER = '#e2e8f0'

# ── Helper: styled matplotlib figure ─────────────────────────────────────────
def styled_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(LIGHT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER)
    ax.spines['bottom'].set_color(BORDER)
    ax.tick_params(colors=SLATE, labelsize=9)
    ax.xaxis.label.set_color(SLATE)
    ax.yaxis.label.set_color(SLATE)
    return fig, ax

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for f in ['model.pkl', 'random_forest_model.pkl', 'credit_model.pkl']:
        p = Path(f)
        if p.exists():
            return joblib.load(p)
    return None

model = load_model()

# ── Session state ─────────────────────────────────────────────────────────────
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk AI")
    st.markdown("<p style='font-size:0.78rem;color:#64748b;margin-top:-0.5rem;margin-bottom:1.2rem;'>MSc Thesis • Sulav Katuwal</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader('Upload CSV Dataset', type=['csv'], label_visibility='visible')

    st.markdown("---")
    menu = st.radio('Navigation', [
        '🏠  Home',
        '📄  Dataset',
        '📊  EDA & Visualization',
        '🤖  Train Models',
        '🔮  Predict Risk',
        '📈  Portfolio Analytics',
        '⚙️  Model Info',
        'ℹ️  About'
    ])
    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem;color:#475569;'>Dataset: UCI Credit Card Default<br>30,000 records • Taiwan proxy</p>", unsafe_allow_html=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.upper()
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    return df

df = None
if uploaded_file:
    df = load_data(uploaded_file)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if '🏠' in menu:
    st.markdown('<p class="section-title">Credit Default Risk Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Predictive Modeling Using Machine Learning Methods & Financial Behavioural Data</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Best Model', 'Random Forest')
    c2.metric('Accuracy', '81.4%', '+13.5% vs baseline')
    c3.metric('ROC-AUC', '0.761', '+0.053 vs baseline')
    c4.metric('Dataset Size', '30,000', 'UCI Credit Card')

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### 📋 Project Overview")
        st.markdown("""
Credit risk assessment in developing economies like **Nepal** remains largely manual,
subjective, and collateral-dependent. This system proposes a machine learning-based
credit risk prediction framework that:

- Implements **Logistic Regression**, **Random Forest**, and **SVM** models
- Engineers three behavioural features: **AVG_BILL**, **AVG_PAY**, **RISK_RATIO**
- Produces a **0–100 probabilistic risk score** mapped to Low / Medium / High categories
- Achieves **81.4% accuracy** and **0.761 ROC-AUC** with Random Forest
        """)

        st.markdown('<div class="info-box">💡 Upload the UCI Credit Card Default CSV from the sidebar to enable training and prediction.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🔬 Model Comparison")
        results_data = {
            'Model': ['Logistic Regression', 'Random Forest ★', 'SVM (RBF)'],
            'Accuracy': [0.679, 0.814, 0.777],
            'ROC-AUC': [0.708, 0.761, 0.753],
            'Recall (Default)': [0.62, 0.34, 0.56]
        }
        results_df = pd.DataFrame(results_data).set_index('Model')
        st.dataframe(results_df.style.highlight_max(axis=0, color='#dbeafe'), use_container_width=True)

        st.markdown("### 🧠 Engineered Features")
        for feat, desc in [
            ('AVG_BILL', 'Avg outstanding credit over 6 months'),
            ('AVG_PAY', 'Avg repayment consistency over 6 months'),
            ('RISK_RATIO', 'AVG_BILL ÷ AVG_PAY — financial stress index'),
        ]:
            st.markdown(f'<span class="badge badge-blue">{feat}</span> &nbsp; <span style="font-size:0.85rem;color:#374151;">{desc}</span>', unsafe_allow_html=True)
            st.write("")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("### 🏗️ System Pipeline")
    cols = st.columns(5)
    steps = [
        ('1', 'Data Loading', 'UCI Dataset\n30K records'),
        ('2', 'Preprocessing', 'Scale, clean\nremove ID'),
        ('3', 'Feature Eng.', 'AVG_BILL\nAVG_PAY\nRISK_RATIO'),
        ('4', 'Model Training', 'LR / RF / SVM\n80:20 split'),
        ('5', 'Risk Scoring', '0–100 scale\nLow/Med/High'),
    ]
    for col, (num, title, body) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;text-align:center;height:130px;">
                <div style="width:28px;height:28px;background:#1e40af;color:white;border-radius:50%;
                    display:flex;align-items:center;justify-content:center;font-size:0.8rem;
                    font-weight:600;margin:0 auto 0.5rem;">{num}</div>
                <p style="font-weight:600;font-size:0.875rem;margin:0 0 0.25rem;color:#0f172a;">{title}</p>
                <p style="font-size:0.75rem;color:#64748b;margin:0;white-space:pre-line;">{body}</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif '📄' in menu:
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">UCI Credit Card Default Dataset — 30,000 records, 25 features</p>', unsafe_allow_html=True)

    if df is None:
        st.warning('⚠️ Please upload the CSV dataset from the sidebar to continue.')
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Total Records', f'{df.shape[0]:,}')
        c2.metric('Features', df.shape[1])
        c3.metric('Missing Values', int(df.isnull().sum().sum()))
        target_col = [c for c in df.columns if 'DEFAULT' in c.upper()]
        if target_col:
            def_rate = df[target_col[0]].mean()
            c4.metric('Default Rate', f'{def_rate:.1%}')

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(['📋 Data Preview', '📊 Statistics', '🔍 Column Details'])

        with tab1:
            st.dataframe(df.head(50), use_container_width=True)

        with tab2:
            st.dataframe(df.describe().T.style.format('{:.2f}'), use_container_width=True)

        with tab3:
            info_data = []
            for col in df.columns:
                info_data.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-Null': df[col].notna().sum(),
                    'Unique Values': df[col].nunique(),
                    'Min': df[col].min() if df[col].dtype != object else '-',
                    'Max': df[col].max() if df[col].dtype != object else '-',
                })
            st.dataframe(pd.DataFrame(info_data).set_index('Column'), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif '📊' in menu:
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Visualizing distributions, correlations, and behavioural patterns</p>', unsafe_allow_html=True)

    if df is None:
        st.warning('⚠️ Please upload the CSV dataset from the sidebar to continue.')
    else:
        target_col = next((c for c in df.columns if 'DEFAULT' in c.upper()), None)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col and target_col in num_cols:
            num_cols_no_target = [c for c in num_cols if c != target_col]
        else:
            num_cols_no_target = num_cols

        # ── Feature engineering for EDA ──
        bill_cols = [c for c in df.columns if 'BILL_AMT' in c or 'BILL AMT' in c]
        pay_cols  = [c for c in df.columns if 'PAY_AMT'  in c or 'PAY AMT'  in c]
        if bill_cols:
            df['AVG_BILL'] = df[bill_cols].mean(axis=1)
        if pay_cols:
            df['AVG_PAY'] = df[pay_cols].mean(axis=1)
        if 'AVG_BILL' in df.columns and 'AVG_PAY' in df.columns:
            df['RISK_RATIO'] = (df['AVG_BILL'] + 1) / (df['AVG_PAY'] + 1)

        # ── Row 1: Class distribution + Credit limit distribution ──
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Default Status Distribution**")
            if target_col:
                vc = df[target_col].value_counts().sort_index()
                fig, ax = styled_fig(5, 3.5)
                bars = ax.bar(['Non-Default (0)', 'Default (1)'], vc.values,
                              color=[GREEN, RED], edgecolor='white', linewidth=1.5,
                              width=0.5)
                for b in bars:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 100,
                            f'{int(b.get_height()):,}', ha='center', va='bottom',
                            fontsize=9, color=SLATE, fontweight='500')
                ax.set_ylabel('Count', fontsize=9)
                ax.set_title(f'Class Imbalance: {vc.iloc[0]/len(df)*100:.1f}% vs {vc.iloc[1]/len(df)*100:.1f}%',
                             fontsize=9, color=SLATE)
                ax.set_ylim(0, vc.max() * 1.15)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with col2:
            st.markdown("**Credit Limit Distribution (LIMIT_BAL)**")
            if 'LIMIT_BAL' in df.columns:
                fig, ax = styled_fig(5, 3.5)
                ax.hist(df['LIMIT_BAL'] / 1000, bins=40, color=BLUE,
                        edgecolor='white', linewidth=0.6, alpha=0.85)
                ax.set_xlabel('Credit Limit (000s NTD)', fontsize=9)
                ax.set_ylabel('Count', fontsize=9)
                ax.set_title('Distribution of Customer Credit Limits', fontsize=9, color=SLATE)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                st.pyplot(fig, use_container_width=True)
                plt.close()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Row 2: Correlation heatmap ──
        st.markdown("**Feature Correlation Matrix**")
        corr_cols = [c for c in num_cols if c in df.columns][:20]
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        cax = ax.imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(cax, ax=ax, fraction=0.03, pad=0.02)
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=7.5, color=SLATE)
        ax.set_yticklabels(corr_cols, fontsize=7.5, color=SLATE)
        ax.set_title('Inter-Feature Correlation Heatmap', fontsize=10, color='#0f172a', pad=10)
        # Annotate cells
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.4:
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=5.5, color='white' if abs(val) > 0.7 else '#0f172a')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Row 3: Engineered features vs default ──
        if target_col and 'AVG_BILL' in df.columns:
            st.markdown("**Engineered Features vs Default Status**")
            col1, col2, col3 = st.columns(3)

            features_to_plot = [('AVG_BILL', 'Average Bill Amount', col1),
                                 ('AVG_PAY',  'Average Payment Amount', col2),
                                 ('RISK_RATIO','Financial Stress Ratio', col3)]

            for feat, title, col in features_to_plot:
                if feat not in df.columns:
                    continue
                with col:
                    df_plot = df[[feat, target_col]].copy()
                    df_plot = df_plot[df_plot[feat] < df_plot[feat].quantile(0.99)]

                    non_def = df_plot[df_plot[target_col] == 0][feat]
                    defaulter = df_plot[df_plot[target_col] == 1][feat]

                    fig, ax = styled_fig(4, 3.2)
                    bp = ax.boxplot([non_def, defaulter],
                                    labels=['Non-Default', 'Default'],
                                    patch_artist=True, notch=False,
                                    whiskerprops=dict(color=SLATE, linewidth=1),
                                    capprops=dict(color=SLATE, linewidth=1),
                                    medianprops=dict(color='white', linewidth=2),
                                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
                    bp['boxes'][0].set_facecolor(GREEN)
                    bp['boxes'][0].set_alpha(0.8)
                    bp['boxes'][1].set_facecolor(RED)
                    bp['boxes'][1].set_alpha(0.8)
                    ax.set_title(title, fontsize=9, color='#0f172a')
                    ax.set_ylabel(feat, fontsize=8)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}' if abs(x) < 1e6 else f'{x/1e3:.0f}K'))
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Row 4: Payment delay distribution + Age distribution ──
        col1, col2 = st.columns(2)

        with col1:
            pay_delay_col = next((c for c in df.columns if c in ['PAY_0', 'PAY_1', 'PAY0']), None)
            if pay_delay_col and target_col:
                st.markdown("**Payment Delay Status by Default Class**")
                grouped = df.groupby([pay_delay_col, target_col]).size().unstack(fill_value=0)
                fig, ax = styled_fig(5, 3.5)
                x = np.arange(len(grouped.index))
                w = 0.35
                ax.bar(x - w/2, grouped.iloc[:, 0], w, label='Non-Default', color=GREEN, alpha=0.85)
                if grouped.shape[1] > 1:
                    ax.bar(x + w/2, grouped.iloc[:, 1], w, label='Default', color=RED, alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(grouped.index, fontsize=9)
                ax.set_xlabel('Payment Delay Status (PAY_0)', fontsize=9)
                ax.set_ylabel('Count', fontsize=9)
                ax.set_title('Repayment Delay vs Default', fontsize=9, color=SLATE)
                ax.legend(fontsize=8)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with col2:
            if 'AGE' in df.columns and target_col:
                st.markdown("**Age Distribution by Default Status**")
                non_def_age = df[df[target_col] == 0]['AGE']
                def_age     = df[df[target_col] == 1]['AGE']
                fig, ax = styled_fig(5, 3.5)
                ax.hist(non_def_age, bins=30, color=GREEN, alpha=0.6, label='Non-Default', density=True)
                ax.hist(def_age,     bins=30, color=RED,   alpha=0.6, label='Default',     density=True)
                ax.set_xlabel('Age', fontsize=9)
                ax.set_ylabel('Density', fontsize=9)
                ax.set_title('Age Distribution by Default Class', fontsize=9, color=SLATE)
                ax.legend(fontsize=8)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        # ── Row 5: Education & Marriage ──
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if 'EDUCATION' in df.columns and target_col:
                st.markdown("**Default Rate by Education Level**")
                edu_map = {1: 'Graduate', 2: 'University', 3: 'High School', 4: 'Others'}
                df_edu = df.copy()
                df_edu['EDU_LABEL'] = df_edu['EDUCATION'].map(edu_map).fillna('Other')
                edu_def = df_edu.groupby('EDU_LABEL')[target_col].mean().sort_values(ascending=False)
                fig, ax = styled_fig(5, 3)
                colors = [BLUE if v < 0.25 else AMBER if v < 0.30 else RED for v in edu_def.values]
                bars = ax.barh(edu_def.index, edu_def.values * 100, color=colors, edgecolor='white', height=0.5)
                for b in bars:
                    ax.text(b.get_width() + 0.3, b.get_y() + b.get_height()/2,
                            f'{b.get_width():.1f}%', va='center', fontsize=8.5, color=SLATE)
                ax.set_xlabel('Default Rate (%)', fontsize=9)
                ax.set_title('Default Rate by Education', fontsize=9, color=SLATE)
                ax.set_xlim(0, edu_def.max() * 100 * 1.25)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with col2:
            if 'MARRIAGE' in df.columns and target_col:
                st.markdown("**Default Rate by Marital Status**")
                mar_map = {1: 'Married', 2: 'Single', 3: 'Others'}
                df_mar = df.copy()
                df_mar['MAR_LABEL'] = df_mar['MARRIAGE'].map(mar_map).fillna('Other')
                mar_def = df_mar.groupby('MAR_LABEL')[target_col].mean().sort_values(ascending=False)
                fig, ax = styled_fig(5, 3)
                colors = [BLUE if v < 0.22 else AMBER if v < 0.25 else RED for v in mar_def.values]
                bars = ax.bar(mar_def.index, mar_def.values * 100, color=colors, edgecolor='white', width=0.4)
                for b in bars:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                            f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=9, color=SLATE)
                ax.set_ylabel('Default Rate (%)', fontsize=9)
                ax.set_title('Default Rate by Marital Status', fontsize=9, color=SLATE)
                st.pyplot(fig, use_container_width=True)
                plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif '🤖' in menu:
    st.markdown('<p class="section-title">Train Machine Learning Models</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Build, evaluate, and compare Logistic Regression, Random Forest, and SVM</p>', unsafe_allow_html=True)

    if df is None:
        st.warning('⚠️ Please upload the CSV dataset from the sidebar to continue.')
    else:
        target_col = next((c for c in df.columns if 'DEFAULT' in c.upper()), None)

        if not target_col:
            st.error('Could not find a target column. Make sure the CSV has a column with "DEFAULT" in its name.')
        else:
            # Feature engineering
            df_fe = df.copy()
            bill_cols = [c for c in df_fe.columns if 'BILL_AMT' in c]
            pay_cols  = [c for c in df_fe.columns if 'PAY_AMT'  in c]
            if bill_cols: df_fe['AVG_BILL']   = df_fe[bill_cols].mean(axis=1)
            if pay_cols:  df_fe['AVG_PAY']    = df_fe[pay_cols].mean(axis=1)
            if 'AVG_BILL' in df_fe.columns and 'AVG_PAY' in df_fe.columns:
                df_fe['RISK_RATIO'] = (df_fe['AVG_BILL'] + 1) / (df_fe['AVG_PAY'] + 1)

            feature_cols = [c for c in df_fe.select_dtypes(include=np.number).columns
                            if c != target_col]

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### ⚙️ Training Configuration")
                test_size = st.slider('Test Split Size', 0.1, 0.4, 0.2, 0.05)
                n_estimators = st.slider('RF: n_estimators', 50, 500, 300, 50)
                cv_folds = st.slider('Cross-Validation Folds', 3, 10, 5)
                use_class_weight = st.checkbox('Use class_weight=balanced', value=True)

                models_to_train = st.multiselect(
                    'Select Models to Train',
                    ['Logistic Regression', 'Random Forest', 'SVM (RBF)'],
                    default=['Logistic Regression', 'Random Forest', 'SVM (RBF)']
                )

                train_btn = st.button('🚀 Train Selected Models', use_container_width=True)

            with col2:
                if train_btn and models_to_train:
                    X = df_fe[feature_cols].fillna(0)
                    y = df_fe[target_col]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )

                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_test_sc  = scaler.transform(X_test)

                    cw = 'balanced' if use_class_weight else None

                    trained = {}
                    progress = st.progress(0, text='Training models...')
                    total = len(models_to_train)

                    for i, mname in enumerate(models_to_train):
                        progress.progress((i + 0.5) / total, text=f'Training {mname}...')

                        if mname == 'Logistic Regression':
                            clf = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)
                        elif mname == 'Random Forest':
                            clf = RandomForestClassifier(n_estimators=n_estimators, class_weight=cw,
                                                          random_state=42, n_jobs=-1)
                        else:
                            clf = SVC(kernel='rbf', probability=True, class_weight=cw,
                                      cache_size=1000, random_state=42)

                        clf.fit(X_train_sc, y_train)
                        pred = clf.predict(X_test_sc)
                        prob = clf.predict_proba(X_test_sc)[:, 1]

                        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(clf, X_train_sc, y_train, cv=cv, scoring='roc_auc')

                        trained[mname] = {
                            'model': clf,
                            'pred': pred,
                            'prob': prob,
                            'acc': accuracy_score(y_test, pred),
                            'auc': roc_auc_score(y_test, prob),
                            'cm': confusion_matrix(y_test, pred),
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std(),
                            'report': classification_report(y_test, pred, output_dict=True)
                        }
                        progress.progress((i + 1) / total, text=f'Done: {mname}')

                    st.session_state.trained_models = trained
                    st.session_state.scaler = scaler
                    st.session_state.X_test = X_test_sc
                    st.session_state.y_test = y_test
                    st.session_state.feature_cols = feature_cols
                    st.session_state.rf_model = trained.get('Random Forest', {}).get('model')

                    st.success(f'✅ Training complete! {len(trained)} model(s) trained on {len(X_train):,} samples.')

            # ── Display results if available ──────────────────────────────────
            if st.session_state.trained_models:
                trained = st.session_state.trained_models
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown("### 📊 Training Results")

                # Summary table
                summary = []
                for mname, mdata in trained.items():
                    r = mdata['report']
                    summary.append({
                        'Model': mname,
                        'Accuracy': f"{mdata['acc']:.3f}",
                        'ROC-AUC': f"{mdata['auc']:.3f}",
                        'CV AUC': f"{mdata['cv_mean']:.3f} ± {mdata['cv_std']:.3f}",
                        'Precision (Def)': f"{r.get('1', {}).get('precision', 0):.3f}",
                        'Recall (Def)': f"{r.get('1', {}).get('recall', 0):.3f}",
                        'F1 (Def)': f"{r.get('1', {}).get('f1-score', 0):.3f}",
                    })
                st.dataframe(pd.DataFrame(summary).set_index('Model'), use_container_width=True)

                st.markdown('<div class="warn-box">⚠️ <strong>Important:</strong> Random Forest has high accuracy but low recall on the default class (misses ~66% of defaulters). In banking, false negatives are costly — consider threshold tuning for deployment.</div>', unsafe_allow_html=True)

                # ── Accuracy & AUC bars ──
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Accuracy Comparison**")
                    fig, ax = styled_fig(5, 3)
                    names = list(trained.keys())
                    accs  = [trained[m]['acc'] for m in names]
                    colors_bar = [GREEN if a == max(accs) else BLUE for a in accs]
                    bars = ax.bar(names, accs, color=colors_bar, edgecolor='white', width=0.4)
                    for b in bars:
                        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9.5,
                                fontweight='500', color='#0f172a')
                    ax.set_ylim(0.5, 1.0)
                    ax.set_ylabel('Accuracy', fontsize=9)
                    ax.set_title('Model Accuracy Comparison', fontsize=9, color=SLATE)
                    plt.xticks(rotation=15, fontsize=8.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                with col2:
                    st.markdown("**ROC-AUC Comparison**")
                    fig, ax = styled_fig(5, 3)
                    aucs = [trained[m]['auc'] for m in names]
                    colors_bar = [GREEN if a == max(aucs) else AMBER for a in aucs]
                    bars = ax.bar(names, aucs, color=colors_bar, edgecolor='white', width=0.4)
                    for b in bars:
                        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9.5,
                                fontweight='500', color='#0f172a')
                    ax.set_ylim(0.5, 1.0)
                    ax.set_ylabel('ROC-AUC', fontsize=9)
                    ax.set_title('ROC-AUC Comparison', fontsize=9, color=SLATE)
                    plt.xticks(rotation=15, fontsize=8.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # ── Confusion Matrices ──
                st.markdown("### 🔲 Confusion Matrices")
                cm_cols = st.columns(len(trained))
                for col, (mname, mdata) in zip(cm_cols, trained.items()):
                    with col:
                        st.markdown(f"**{mname}**")
                        cm = mdata['cm']
                        fig, ax = plt.subplots(figsize=(3.5, 3.2))
                        fig.patch.set_facecolor('white')
                        disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Default', 'Default'])
                        disp.plot(ax=ax, cmap='Blues', colorbar=False)
                        ax.set_title(f'AUC: {mdata["auc"]:.3f}', fontsize=9, color=SLATE)
                        ax.set_xticklabels(['Non-Def', 'Default'], fontsize=7.5)
                        ax.set_yticklabels(['Non-Def', 'Default'], fontsize=7.5, rotation=90, va='center')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # ── ROC Curves ──
                st.markdown("### 📈 ROC Curves")
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = styled_fig(6, 4.5)
                    colors_roc = [BLUE, GREEN, AMBER, RED]
                    for (mname, mdata), c in zip(trained.items(), colors_roc):
                        fpr, tpr, _ = roc_curve(st.session_state.y_test, mdata['prob'])
                        ax.plot(fpr, tpr, lw=2, color=c, label=f"{mname} (AUC={mdata['auc']:.3f})")
                    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
                    ax.set_xlabel('False Positive Rate', fontsize=9)
                    ax.set_ylabel('True Positive Rate', fontsize=9)
                    ax.set_title('ROC Curve Comparison', fontsize=10, color='#0f172a')
                    ax.legend(fontsize=8, loc='lower right')
                    ax.set_xlim([-0.01, 1.01])
                    ax.set_ylim([-0.01, 1.05])
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                with col2:
                    fig, ax = styled_fig(6, 4.5)
                    for (mname, mdata), c in zip(trained.items(), colors_roc):
                        prec, rec, _ = precision_recall_curve(st.session_state.y_test, mdata['prob'])
                        ax.plot(rec, prec, lw=2, color=c, label=mname)
                    ax.set_xlabel('Recall', fontsize=9)
                    ax.set_ylabel('Precision', fontsize=9)
                    ax.set_title('Precision-Recall Curve Comparison', fontsize=10, color='#0f172a')
                    ax.legend(fontsize=8)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # ── Feature Importance (RF only) ──
                if 'Random Forest' in trained:
                    st.markdown("### 🌲 Random Forest Feature Importance")
                    rf = trained['Random Forest']['model']
                    fi = pd.Series(rf.feature_importances_,
                                   index=st.session_state.feature_cols).sort_values(ascending=True).tail(15)

                    fig, ax = styled_fig(8, 5)
                    bar_colors = [RED if i in ['RISK_RATIO', 'AVG_BILL', 'AVG_PAY'] else BLUE
                                  for i in fi.index]
                    bars = ax.barh(fi.index, fi.values, color=bar_colors, edgecolor='white', height=0.65)
                    for b in bars:
                        ax.text(b.get_width() + 0.0003, b.get_y() + b.get_height()/2,
                                f'{b.get_width():.4f}', va='center', fontsize=8, color=SLATE)
                    ax.set_xlabel('Feature Importance Score', fontsize=9)
                    ax.set_title('Top 15 Features — Random Forest Importance', fontsize=10, color='#0f172a')
                    red_patch = mpatches.Patch(color=RED, label='Engineered features')
                    blue_patch = mpatches.Patch(color=BLUE, label='Original features')
                    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc='lower right')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                    st.markdown('<div class="info-box">💡 <strong>Key Insight:</strong> PAY_0 (repayment delay history) is the strongest predictor, followed by LIMIT_BAL and AGE. The engineered RISK_RATIO also appears in the top features, validating the feature engineering approach.</div>', unsafe_allow_html=True)

                # ── Cross-validation scores ──
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown("### 🔁 Cross-Validation Results")
                cv_data = {mname: [f"{mdata['cv_mean']:.3f}", f"{mdata['cv_std']:.4f}",
                                   f"{mdata['cv_mean'] - 2*mdata['cv_std']:.3f}",
                                   f"{mdata['cv_mean'] + 2*mdata['cv_std']:.3f}"]
                           for mname, mdata in trained.items()}
                cv_df = pd.DataFrame(cv_data, index=['Mean AUC', 'Std Dev', 'Lower CI (95%)', 'Upper CI (95%)'])
                st.dataframe(cv_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT RISK
# ══════════════════════════════════════════════════════════════════════════════
elif '🔮' in menu:
    st.markdown('<p class="section-title">Individual Credit Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter applicant financial details to compute a probabilistic risk score</p>', unsafe_allow_html=True)

    with st.form('predict_form'):
        st.markdown("#### 👤 Applicant Profile")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            age   = st.slider('Age', 18, 80, 30)
            sex   = st.selectbox('Gender', [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
            edu   = st.selectbox('Education', [1, 2, 3, 4],
                                  format_func=lambda x: {1:'Graduate', 2:'University', 3:'High School', 4:'Other'}[x])
            mar   = st.selectbox('Marital Status', [1, 2, 3],
                                  format_func=lambda x: {1:'Married', 2:'Single', 3:'Other'}[x])

        with c2:
            st.markdown("**Credit Information**")
            limit_bal = st.number_input('Credit Limit (NTD)', 10_000, 1_000_000, 200_000, 10_000,
                                         format='%d')
            pay_0     = st.slider('Current Payment Delay Status', -2, 8, 0,
                                   help='-2=No consumption, -1=Paid in full, 0=Revolving credit, 1-8=Months delayed')
            avg_bill  = st.number_input('Avg Monthly Bill Amount (NTD)', 0, 1_000_000, 50_000, 1_000)
            avg_pay   = st.number_input('Avg Monthly Payment Amount (NTD)', 0, 1_000_000, 30_000, 1_000)

        with c3:
            st.markdown("**Derived Indicators**")
            risk_ratio = (avg_bill + 1) / (avg_pay + 1)
            st.metric('Financial Stress Ratio', f'{risk_ratio:.2f}',
                       delta='⚠️ High stress' if risk_ratio > 3 else '✅ Normal',
                       delta_color='inverse')
            st.metric('Payment Delay', f'{pay_0} months' if pay_0 > 0 else ('Revolving' if pay_0 == 0 else 'Paid Full'),
                       delta='⚠️ Delayed' if pay_0 > 0 else None, delta_color='inverse')

            st.markdown("")
            st.markdown("**Risk Guidelines**")
            st.markdown("""
            <div style="font-size:0.8rem;line-height:1.8;color:#475569;">
            🟢 <b>Low Risk</b>: Score &lt; 40<br>
            🟡 <b>Medium Risk</b>: Score 40–70<br>
            🔴 <b>High Risk</b>: Score &gt; 70
            </div>
            """, unsafe_allow_html=True)

        submitted = st.form_submit_button('🔮 Calculate Risk Score', use_container_width=True)

    if submitted:
        rf_model = st.session_state.trained_models.get('Random Forest', {}).get('model') if st.session_state.trained_models else model
        scaler = st.session_state.scaler

        input_features = {
            'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': edu,
            'MARRIAGE': mar, 'AGE': age, 'PAY_0': pay_0,
            'AVG_BILL': avg_bill, 'AVG_PAY': avg_pay, 'RISK_RATIO': risk_ratio
        }

        prob = None
        if rf_model is not None:
            try:
                X_input = pd.DataFrame([input_features])
                if scaler is not None and hasattr(scaler, 'feature_names_in_'):
                    # Align features
                    feat_cols = st.session_state.feature_cols
                    X_aligned = pd.DataFrame([{c: input_features.get(c, 0) for c in feat_cols}])
                    X_scaled = scaler.transform(X_aligned)
                    prob = float(rf_model.predict_proba(X_scaled)[0][1])
                else:
                    prob = float(rf_model.predict_proba(X_input)[0][1])
            except Exception:
                prob = None

        if prob is None:
            # Fallback formula
            delay_penalty = max(pay_0 * 0.06, 0)
            ratio_score = min(risk_ratio / 4, 0.6)
            limit_factor = max(0, 1 - (limit_bal / 800_000)) * 0.1
            prob = min(max(ratio_score + delay_penalty + limit_factor, 0.02), 0.97)
            used_fallback = True
        else:
            used_fallback = False

        score = int(prob * 100)

        if prob < 0.4:
            cat, cat_color, advice, badge_class, icon = (
                'Low Risk', '#16a34a', 'Approve Loan', 'badge-green', '✅')
        elif prob < 0.7:
            cat, cat_color, advice, badge_class, icon = (
                'Medium Risk', '#d97706', 'Manual Review Required', 'badge-orange', '⚠️')
        else:
            cat, cat_color, advice, badge_class, icon = (
                'High Risk', '#dc2626', 'Reject / Investigate', 'badge-red', '🚫')

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 🎯 Risk Assessment Result")

        rc1, rc2, rc3 = st.columns([1, 1.5, 1])

        with rc1:
            st.metric('Default Probability', f'{score}%')
            st.metric('Risk Category', cat)
            st.metric('Recommendation', advice)

        with rc2:
            # Gauge-style bar
            bar_color = '#16a34a' if score < 40 else '#d97706' if score < 70 else '#dc2626'
            st.markdown(f"""
            <div style="margin-top:0.5rem;">
                <p style="font-size:0.85rem;color:#64748b;margin-bottom:4px;">Risk Score</p>
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                    <span style="font-size:2.5rem;font-weight:700;color:{cat_color};">{score}</span>
                    <span style="font-size:1rem;color:#94a3b8;">/100</span>
                </div>
                <div style="background:#f1f5f9;border-radius:99px;height:18px;overflow:hidden;border:1px solid #e2e8f0;">
                    <div style="width:{score}%;height:100%;background:{bar_color};
                        border-radius:99px;transition:width 0.5s ease;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:4px;">
                    <span style="font-size:0.7rem;color:#94a3b8;">0</span>
                    <span style="font-size:0.7rem;color:#16a34a;font-weight:500;">40 (Low)</span>
                    <span style="font-size:0.7rem;color:#d97706;font-weight:500;">70 (Med)</span>
                    <span style="font-size:0.7rem;color:#94a3b8;">100</span>
                </div>
            </div>

            <div style="margin-top:1.5rem;background:#f8fafc;border:1px solid #e2e8f0;
                border-radius:10px;padding:1rem;">
                <p style="font-size:0.8rem;color:#64748b;margin:0 0 0.5rem;">Key Risk Factors</p>
                <div style="font-size:0.82rem;line-height:2;color:#374151;">
                    {'⚠️' if pay_0 > 1 else '✅'} Payment delay: <b>{pay_0}</b> months<br>
                    {'⚠️' if risk_ratio > 3 else '✅'} Stress ratio: <b>{risk_ratio:.2f}</b><br>
                    {'⚠️' if limit_bal < 100000 else '✅'} Credit limit: <b>{limit_bal:,}</b><br>
                    {'ℹ️'} Age: <b>{age}</b> years
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rc3:
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                padding:1.25rem;text-align:center;height:100%;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">{icon}</div>
                <div style="font-size:1.1rem;font-weight:600;color:{cat_color};
                    margin-bottom:0.75rem;">{cat}</div>
                <div style="font-size:0.85rem;color:#475569;margin-bottom:1rem;">{advice}</div>
                <div style="background:{'#dcfce7' if score < 40 else '#fff7ed' if score < 70 else '#fee2e2'};
                    border-radius:8px;padding:0.75rem;font-size:0.8rem;
                    color:{cat_color};">
                    <b>Model:</b> {'Random Forest (Trained)' if not used_fallback else 'Heuristic Formula*'}<br>
                    <b>Probability:</b> {prob:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if used_fallback:
            st.markdown('<div class="warn-box">* No trained model found. Prediction uses a heuristic formula based on RISK_RATIO and PAY_0. Train a model in the "Train Models" page for more accurate predictions.</div>', unsafe_allow_html=True)

        # Input summary
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Applicant Input Summary**")
        input_df = pd.DataFrame([{
            'Age': age, 'Gender': 'Male' if sex == 1 else 'Female',
            'Education': {1:'Graduate',2:'University',3:'High School',4:'Other'}[edu],
            'Marital Status': {1:'Married',2:'Single',3:'Other'}[mar],
            'Credit Limit': f'NTD {limit_bal:,}',
            'Payment Delay': pay_0,
            'Avg Bill': f'NTD {avg_bill:,}',
            'Avg Payment': f'NTD {avg_pay:,}',
            'Stress Ratio': f'{risk_ratio:.2f}',
            'Risk Score': f'{score}/100',
            'Category': cat
        }])
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif '📈' in menu:
    st.markdown('<p class="section-title">Portfolio Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Aggregate risk metrics and portfolio distribution</p>', unsafe_allow_html=True)

    if df is not None and st.session_state.trained_models:
        target_col = next((c for c in df.columns if 'DEFAULT' in c.upper()), None)
        rf = st.session_state.trained_models.get('Random Forest', {}).get('model')
        y_test = st.session_state.y_test
        y_pred = st.session_state.trained_models['Random Forest']['pred'] if 'Random Forest' in st.session_state.trained_models else None

        # Metrics
        if y_pred is not None and y_test is not None:
            y_prob = st.session_state.trained_models['Random Forest']['prob']
            low_risk    = (y_prob < 0.4).sum()
            med_risk    = ((y_prob >= 0.4) & (y_prob < 0.7)).sum()
            high_risk   = (y_prob >= 0.7).sum()
            total_test  = len(y_prob)
            default_rate = y_test.mean()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('Test Applicants', f'{total_test:,}')
            c2.metric('Low Risk', f'{low_risk:,}', f'{low_risk/total_test:.1%}')
            c3.metric('Medium Risk', f'{med_risk:,}', f'{med_risk/total_test:.1%}')
            c4.metric('High Risk', f'{high_risk:,}', f'{high_risk/total_test:.1%}')
            c5.metric('Actual Default Rate', f'{default_rate:.1%}')

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Risk Category Distribution**")
                fig, ax = styled_fig(5, 3.5)
                categories = ['Low Risk', 'Medium Risk', 'High Risk']
                counts = [low_risk, med_risk, high_risk]
                colors_pie = [GREEN, AMBER, RED]
                bars = ax.bar(categories, counts, color=colors_pie, edgecolor='white', linewidth=1.5, width=0.5)
                for b in bars:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                            f'{int(b.get_height()):,}\n({int(b.get_height())/total_test:.1%})',
                            ha='center', va='bottom', fontsize=8.5, color=SLATE)
                ax.set_ylabel('Count', fontsize=9)
                ax.set_ylim(0, max(counts) * 1.25)
                ax.set_title('Applicant Risk Categories (Test Set)', fontsize=9, color=SLATE)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col2:
                st.markdown("**Risk Score Distribution (0–100)**")
                scores = (y_prob * 100).astype(int)
                fig, ax = styled_fig(5, 3.5)
                ax.hist(scores, bins=50, color=BLUE, edgecolor='white', linewidth=0.5, alpha=0.85)
                ax.axvline(40, color=GREEN, lw=2, linestyle='--', label='Low/Med boundary (40)')
                ax.axvline(70, color=RED,   lw=2, linestyle='--', label='Med/High boundary (70)')
                ax.set_xlabel('Risk Score', fontsize=9)
                ax.set_ylabel('Count', fontsize=9)
                ax.set_title('Distribution of Predicted Risk Scores', fontsize=9, color=SLATE)
                ax.legend(fontsize=7.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # Default detection analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Defaulters Detected by Risk Category**")
                pred_cats = pd.cut(y_prob, bins=[-0.01, 0.4, 0.7, 1.01],
                                    labels=['Low', 'Medium', 'High'])
                detection = pd.DataFrame({'Category': pred_cats, 'Actual': y_test.values})
                det_summary = detection.groupby('Category')['Actual'].agg(['sum', 'count', 'mean'])
                det_summary.columns = ['Defaults Caught', 'Total', 'Default Rate']

                fig, ax = styled_fig(5, 3.5)
                colors_d = [GREEN, AMBER, RED]
                bars = ax.bar(det_summary.index, det_summary['Default Rate'] * 100,
                               color=colors_d, edgecolor='white', width=0.4)
                for b in bars:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                            f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=9.5,
                            fontweight='500', color=SLATE)
                ax.set_ylabel('Actual Default Rate (%)', fontsize=9)
                ax.set_title('Default Rate Within Each Risk Category', fontsize=9, color=SLATE)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col2:
                st.markdown("**Score Calibration: Predicted vs Actual**")
                bins_c = np.linspace(0, 1, 11)
                bin_labels = np.arange(5, 100, 10)
                pred_bin = np.digitize(y_prob, bins_c) - 1
                pred_bin = np.clip(pred_bin, 0, 9)
                cal_df = pd.DataFrame({'bin': pred_bin, 'actual': y_test.values, 'prob': y_prob})
                cal_summary = cal_df.groupby('bin').agg(
                    mean_pred=('prob', 'mean'),
                    mean_actual=('actual', 'mean'),
                    count=('actual', 'count')
                ).reset_index()

                fig, ax = styled_fig(5, 3.5)
                ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Perfect calibration')
                ax.scatter(cal_summary['mean_pred'], cal_summary['mean_actual'],
                           s=cal_summary['count'] / 8, color=BLUE, alpha=0.8, zorder=5,
                           label='Model calibration')
                ax.set_xlabel('Mean Predicted Probability', fontsize=9)
                ax.set_ylabel('Actual Default Rate', fontsize=9)
                ax.set_title('Calibration Plot', fontsize=9, color=SLATE)
                ax.legend(fontsize=8)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.02, 1.02)
                st.pyplot(fig, use_container_width=True)
                plt.close()

    else:
        st.info('Train models first using the "Train Models" page to see portfolio analytics.')

        # Show simulated data as placeholder
        st.markdown("### 📋 Sample Portfolio Metrics (Illustrative)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Applicants Processed', '6,000')
        c2.metric('Low Risk', '5,023 (83.7%)')
        c3.metric('Medium Risk', '685 (11.4%)')
        c4.metric('High Risk', '292 (4.9%)')

        fig, ax = styled_fig(7, 3.5)
        labels = ['Low Risk\n5,023', 'Medium Risk\n685', 'High Risk\n292']
        values = [5023, 685, 292]
        bars = ax.bar(labels, values, color=[GREEN, AMBER, RED], edgecolor='white', width=0.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                    f'{int(b.get_height()):,}', ha='center', va='bottom', fontsize=10,
                    fontweight='500', color=SLATE)
        ax.set_ylabel('Applicant Count', fontsize=9)
        ax.set_title('Illustrative Risk Category Distribution (Test Set — from paper)', fontsize=9, color=SLATE)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif '⚙️' in menu:
    st.markdown('<p class="section-title">Model Performance Summary</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Complete results from the research paper</p>', unsafe_allow_html=True)

    # Paper results
    tab1, tab2, tab3, tab4 = st.tabs(['📊 Performance Table', '🔲 Confusion Matrices', '📈 ROC Analysis', '🌲 Feature Importance'])

    with tab1:
        st.markdown("#### Classification Report (from Paper)")
        paper_data = {
            'Model': ['Logistic Regression', 'Random Forest ★', 'SVM (RBF)'],
            'Accuracy': [0.679, 0.814, 0.777],
            'ROC-AUC': [0.708, 0.761, 0.753],
            'CV AUC (k=5)': ['—', '0.767 ± 0.006', '—'],
            'Precision (Non-Def)': [0.87, 0.84, 0.82],
            'Recall (Non-Def)': [0.70, 0.95, 0.91],
            'F1 (Non-Def)': [0.77, 0.89, 0.86],
            'Precision (Default)': [0.37, 0.65, 0.57],
            'Recall (Default)': [0.62, 0.34, 0.36],
            'F1 (Default)': [0.46, 0.45, 0.44],
        }
        paper_df = pd.DataFrame(paper_data).set_index('Model')
        st.dataframe(paper_df.style.highlight_max(axis=0, color='#dbeafe', subset=['Accuracy','ROC-AUC']), use_container_width=True)

        st.markdown('<div class="warn-box">⚠️ <strong>Critical Note:</strong> Random Forest has the best accuracy (81.4%) and ROC-AUC (0.761) but the worst recall on defaults (0.34). This means ~66% of actual defaulters are predicted as safe — a significant risk in real banking deployment.</div>', unsafe_allow_html=True)

        st.markdown("#### Hyperparameters Used")
        hp_data = {
            'Parameter': ['n_estimators', 'kernel', 'C', 'gamma', 'max_iter', 'class_weight', 'random_state', 'cache_size'],
            'Logistic Regression': ['—', '—', '—', '—', '1000', 'balanced', '42', '—'],
            'Random Forest': ['300', '—', '—', '—', '—', 'balanced', '42', '—'],
            'SVM (RBF)': ['—', 'RBF', 'default', 'auto-scale', '—', 'balanced', '42', '1000 MB'],
        }
        st.dataframe(pd.DataFrame(hp_data).set_index('Parameter'), use_container_width=True)

    with tab2:
        st.markdown("#### Confusion Matrices (Paper Results)")
        col1, col2, col3 = st.columns(3)

        cms = [
            ('Logistic Regression', np.array([[3249, 1424], [504, 823]]),  BLUE),
            ('Random Forest ★',     np.array([[4425, 248],  [870, 457]]),  GREEN),
            ('SVM (RBF)',            np.array([[3921, 752],  [586, 741]]),  AMBER),
        ]

        for col, (mname, cm, color) in zip([col1, col2, col3], cms):
            with col:
                tn, fp, fn, tp = cm.ravel()
                acc = (tn + tp) / (tn + fp + fn + tp)
                rec_def = tp / (tp + fn)
                fig, ax = plt.subplots(figsize=(3.8, 3.5))
                fig.patch.set_facecolor('white')
                disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Def', 'Default'])
                disp.plot(ax=ax, cmap='Blues', colorbar=False)
                ax.set_title(f'{mname}\nAcc={acc:.3f} | Recall(Def)={rec_def:.2f}', fontsize=8.5)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.markdown(f"""
                <div style="font-size:0.78rem;color:#475569;line-height:1.8;margin-top:0.25rem;">
                    TN={tn:,} | FP={fp:,}<br>
                    FN={fn:,} | TP={tp:,}
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("#### ROC Curve (Illustrative — Paper Values)")

        fig, ax = styled_fig(7, 5)
        # Approximate ROC curves from paper AUC values
        np.random.seed(42)
        for (mname, auc_val, color, style) in [
            ('Logistic Regression', 0.708, BLUE, '--'),
            ('Random Forest ★', 0.761, GREEN, '-'),
            ('SVM (RBF)', 0.753, AMBER, '-.'),
        ]:
            # Generate smooth ROC curve approximation
            fpr_approx = np.linspace(0, 1, 200)
            tpr_approx = np.where(fpr_approx < 0.5,
                                   2 * auc_val * fpr_approx,
                                   2 * auc_val * fpr_approx - (2 * auc_val - 1) * fpr_approx)
            tpr_approx = np.clip(fpr_approx ** (1 / (2 * auc_val - 1 + 0.001)), 0, 1)
            ax.plot(fpr_approx, tpr_approx, lw=2.5, color=color, linestyle=style,
                    label=f'{mname} (AUC = {auc_val:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random classifier (AUC=0.5)')
        ax.fill_between([0, 1], [0, 1], alpha=0.04, color='gray')
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title('ROC Curve Comparison (Paper Results)', fontsize=11, color='#0f172a')
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="info-box">An AUC of 0.761 indicates moderate-to-strong discriminative ability — suitable for decision-support but not fully autonomous lending.</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown("#### Feature Importance (Random Forest — Paper Results)")
        features = {
            'PAY_0': 0.0997,
            'LIMIT_BAL': 0.0564,
            'AGE': 0.0548,
            'AVG_PAY': 0.0512,
            'RISK_RATIO': 0.0498,
            'BILL_AMT1': 0.0441,
            'AVG_BILL': 0.0428,
            'PAY_AMT1': 0.0389,
            'BILL_AMT2': 0.0362,
            'PAY_AMT2': 0.0341,
            'PAY_2': 0.0318,
            'PAY_AMT3': 0.0291,
            'BILL_AMT3': 0.0278,
            'BILL_AMT4': 0.0261,
            'BILL_AMT5': 0.0244,
        }
        fi_series = pd.Series(features).sort_values()
        engineered = ['AVG_PAY', 'RISK_RATIO', 'AVG_BILL']

        fig, ax = styled_fig(8, 5.5)
        colors_fi = [RED if f in engineered else BLUE for f in fi_series.index]
        bars = ax.barh(fi_series.index, fi_series.values, color=colors_fi, edgecolor='white', height=0.65)
        for b in bars:
            ax.text(b.get_width() + 0.0003, b.get_y() + b.get_height()/2,
                    f'{b.get_width():.4f}', va='center', fontsize=8.5, color=SLATE)
        ax.set_xlabel('Feature Importance Score', fontsize=9)
        ax.set_title('Top 15 Features — Random Forest (Paper Results)', fontsize=10, color='#0f172a')
        red_patch  = mpatches.Patch(color=RED,  label='Engineered features (new)')
        blue_patch = mpatches.Patch(color=BLUE, label='Original dataset features')
        ax.legend(handles=[red_patch, blue_patch], fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif 'ℹ️' in menu:
    st.markdown('<p class="section-title">About This Project</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("""
### Research Summary

**Title:** Predictive Modeling of Credit Default Risk Using Machine Learning Methods and Financial Behavioral Data

**Author:** Sulav Katuwal | MSc | April 2026

**Abstract:**
Credit risk assessment in developing economies like Nepal remains largely manual, subjective, and collateral-dependent.
This study proposes a machine learning-based credit risk prediction system using the UCI Credit Card Default Dataset
(30,000 records) as a proxy for Nepal-specific data. The framework implements Logistic Regression, Random Forest, and
SVM, enhanced by a structured feature engineering pipeline deriving behavioural indicators: AVG_BILL, AVG_PAY, and RISK_RATIO.
The Random Forest ensemble achieved the best performance with **81.4% accuracy** and **0.761 ROC-AUC**.
        """)

        st.markdown("---")
        st.markdown("#### 🛠️ Tech Stack")
        for lib, purpose in [
            ('Python 3.10+', 'Core language'),
            ('Streamlit', 'Web application framework'),
            ('Scikit-learn', 'ML models, preprocessing, evaluation'),
            ('Pandas / NumPy', 'Data manipulation'),
            ('Matplotlib', 'All visualizations'),
            ('Joblib', 'Model persistence'),
        ]:
            st.markdown(f'`{lib}` — {purpose}')

    with col2:
        st.markdown("#### 📊 Key Achievements")
        for item in [
            'Implemented 3 ML models (LR, RF, SVM)',
            'Engineered 3 behavioural features',
            '81.4% accuracy (Random Forest)',
            '0.761 ROC-AUC (Random Forest)',
            '0.767 CV AUC (5-fold stratified)',
            'Probabilistic risk scoring (0–100)',
            'Streamlit deployment-ready app',
        ]:
            st.markdown(f'✅ {item}')

        st.markdown("---")
        st.markdown("#### 🔗 Resources")
        st.markdown("""
- 📂 [GitHub Repository](https://github.com/sulav97/Predictive-Modeling-of-Credit-Default-Risk-Using-Machine-Learning-Methods.git)
- 📊 [UCI Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
        """)

        st.markdown("---")
        st.markdown("#### ⚖️ Ethical Considerations")
        st.markdown("""
<div style="font-size:0.85rem;line-height:1.8;color:#374151;">
• Demographic features (SEX, EDUCATION, MARRIAGE) risk introducing discriminatory bias<br>
• Black-box RF models require SHAP/LIME for regulatory compliance<br>
• Class imbalance may under-serve minority borrowers<br>
• Dataset is not Nepal-specific (proxy limitation)<br>
• Human oversight required for all lending decisions
</div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🔮 Future Work")
        for item in ['Deep learning (ANN, TabNet)', 'Real-time scoring API',
                      'Nepal Rastra Bank macroeconomic data', 'SHAP/LIME explainability', 'Nepal-localized dataset']:
            st.markdown(f'→ {item}')