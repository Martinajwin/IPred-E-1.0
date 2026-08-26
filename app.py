# ==========================================================
# - IPRED-E 1.0 (Sequential Prediction Pipeline + Diagnostics)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from mordred import Calculator, descriptors
import joblib
from graphviz import Digraph
import plotly.express as px
import plotly.graph_objects as go
import base64
import os

# ==========================================================
#  Page Configuration & Advanced Glassmorphism CSS
# ==========================================================
st.set_page_config(
    page_title="EGFR Inhibitor Predictor | IPRED-E 1.0", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize persistent session states
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'diagnostic_df' not in st.session_state:
    st.session_state.diagnostic_df = None
if 'profile_idx' not in st.session_state:
    st.session_state.profile_idx = 0
if 'mol_idx' not in st.session_state:
    st.session_state.mol_idx = 0
if 'smiles_input_value' not in st.session_state:
    st.session_state.smiles_input_value = ""

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0A0E17 !important;
        color: #E2E8F0 !important;
        font-size: 1.05rem;
    }
    
    .stCodeBlock, code, pre, .SMILES-font {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.1rem !important;
        color: #00E5FF !important;
    }
    
    [data-testid="collapsedControl"] {display: none;}
    #MainMenu, header, footer {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 96%; }

    div[data-testid="stColumn"]:first-child {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 2rem !important;
        height: fit-content !important;
        align-self: flex-start !important; 
        z-index: 999;
    }

    .floating-nav-panel {
        background: rgba(16, 24, 39, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }

    div.row-widget.stRadio > div[role="radiogroup"] { gap: 18px; display: flex; flex-direction: column; }
    div.row-widget.stRadio > div[role="radiogroup"] label {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 18px 24px !important;
        border-radius: 12px;
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 1.15rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    div.row-widget.stRadio > div[role="radiogroup"] label:hover {
        border-color: #00E5FF;
        color: #FFFFFF !important;
        transform: translateX(5px);
        box-shadow: -4px 0px 15px rgba(0, 229, 255, 0.15);
    }
    div.row-widget.stRadio > div[role="radiogroup"] [data-checked="true"] label {
        background: rgba(0, 229, 255, 0.05) !important;
        border: 1px solid #00E5FF !important;
        color: #00E5FF !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }

    .metric-panel {
        background: rgba(16, 24, 39, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 2.5rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-panel:hover {
        transform: translateY(-5px);
        border-color: rgba(255,255,255,0.2);
    }
    .metric-active { border-bottom: 3px solid #00E5FF; box-shadow: 0 10px 30px rgba(0, 229, 255, 0.05); }
    .metric-inactive { border-bottom: 3px solid #FF003C; }
    
    .mp-value { font-size: 4.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }
    .val-cyan { color: #00E5FF; text-shadow: 0 0 15px rgba(0, 229, 255, 0.4); }
    .val-crimson { color: #FF003C; text-shadow: 0 0 15px rgba(255, 0, 60, 0.4); }
    .val-neutral { color: #F8FAFC; }
    .mp-label { font-size: 1.3rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-top: 15px; }

    .main-title-bar {
        background: linear-gradient(135deg, rgba(16, 24, 39, 0.8) 0%, rgba(10, 14, 23, 0.9) 100%);
        backdrop-filter: blur(20px);
        padding: 3rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 2.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-title-bar h1 { font-size: 3.2rem !important; font-weight: 800; color: #FFFFFF; letter-spacing: -1px; margin:0; }
    
    .stButton>button {
        background: transparent;
        color: #00E5FF;
        border: 1px solid #00E5FF;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: inset 0 0 10px rgba(0, 229, 255, 0.05);
    }
    .stButton>button:hover {
        background: rgba(0, 229, 255, 0.1);
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.3), inset 0 0 15px rgba(0, 229, 255, 0.2);
        transform: translateY(-2px);
        color: #FFFFFF;
    }
    
    [data-testid="stDataFrame"] { background-color: transparent !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# ⚙️ Algorithmic Infrastructure
# -----------------------------

clf_features = [
    'ATSC5dv', 'NaaN', 'VSA_EState7', 'PEOE_VSA10', 'SlogP_VSA8', 'PEOE_VSA9', 
    'ATSC4dv', 'ATS0Z', 'PEOE_VSA3', 'IC1', 'ATSC8i', 'nAromAtom', 'nAtom', 
    'EState_VSA8', 'SMR_VSA9', 'AATS0i', 'nN', 'ECIndex', 'SlogP_VSA1', 
    'VSA_EState4', 'ATSC2se', 'NdCH2', 'ATSC6m', 'VSA_EState3', 'PEOE_VSA8', 
    'ATSC8dv', 'EState_VSA9', 'n6HRing', 'ATSC6v', 'SlogP_VSA10', 'AATS0Z', 
    'JGI5', 'ATSC5Z', 'ATSC2m', 'ATSC6i', 'ATSC5i', 'ATSC4Z', 'JGI4', 'ATSC6pe', 
    'EState_VSA3', 'ATSC3dv', 'ATSC7se', 'GeomDiameter', 'ATSC2dv', 'PEOE_VSA6', 
    'PEOE_VSA2', 'PEOE_VSA1', 'JGI6', 'ATSC5v', 'NdsCH'
]

reg_features = [
    'NaaN', 'SlogP_VSA8', 'nAtom', 'EState_VSA8', 'nAromAtom', 'SMR_VSA4', 
    'PEOE_VSA4', 'ATSC5dv', 'FilterItLogS', 'ATSC8i', 'VSA_EState7', 'ATS0Z', 
    'PEOE_VSA3', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA9', 'AATSC0d', 'n6HRing', 
    'ATSC6m', 'VSA_EState4', 'PEOE_VSA8', 'JGI5', 'VSA_EState3', 'ATSC8v', 
    'JGI4', 'AATS0i', 'GeomDiameter', 'ATSC8dv', 'IC1', 'ATSC4v', 'ATSC6i', 
    'SMR_VSA9', 'ATSC2se', 'ATSC5i', 'PEOE_VSA2', 'JGI8', 'JGI3', 'JGI6', 
    'ATSC6dv', 'JGI9'
]

@st.cache_resource
def load_models_and_params():
    """
    Load sequential models, scalers, and AD parameters.
    """
    svm_clf = joblib.load("models/final_model_clf_svm.joblib")
    xgb_reg = joblib.load("models/final_model_reg_xgboost.joblib")
    
    scaler_clf = joblib.load("models/scaler_clf.joblib")
    knn_clf = joblib.load("models/knn_clf.joblib")
    ad_thresh_clf = 45.0 

    scaler_reg = joblib.load("models/scaler_reg.joblib")
    knn_reg = joblib.load("models/knn_reg.joblib")
    ad_thresh_reg = 45.0 

    return (
        svm_clf, xgb_reg,
        scaler_clf, knn_clf, ad_thresh_clf,
        scaler_reg, knn_reg, ad_thresh_reg
    )

def prepare_features(df_desc, feature_list):
    """
    Strictly aligns DF to the explicit manual feature lists.
    """
    X = df_desc.copy()
    X = X.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
    for col in feature_list:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_list]
    X = X.fillna(X.mean(numeric_only=True)).fillna(0)
    return X.astype(float)

def calculate_knn_ad(X_scaled, knn_model, threshold):
    """
    k-NN Applicability Domain check.
    """
    distances, _ = knn_model.kneighbors(X_scaled)
    mean_dist = distances.mean(axis=1)
    status = ["Within" if d <= threshold else "Outside" for d in mean_dist]
    return mean_dist, status

def sequential_prediction(clf_pred, clf_prob, clf_ad, reg_pred, reg_ad):
    """
    Sequential Routing Logic: SVM Classifier -> XGBoost Regressor
    Always trusts model prediction, appends AD warnings when extrapolated.
    """
    final_pred = []
    final_score = []
    ad_status = []

    for i in range(len(clf_pred)):
        c_pred = clf_pred[i]
        c_ad = clf_ad[i]
        
        if c_pred == 1:
            r_pred = reg_pred[i]
            r_ad = reg_ad[i]
            
            if c_ad == "Within" and r_ad == "Within":
                final_pred.append("Active")
                final_score.append(r_pred)
                ad_status.append("Highly Reliable (Within AD)")
            else:
                final_pred.append("Active (AD Flagged)")
                final_score.append(r_pred)
                ad_status.append(f"Unreliable Extrapolation (CLF: {c_ad}, REG: {r_ad})")
        else:
            if c_ad == "Within":
                final_pred.append("Inactive")
                final_score.append("NIL") 
                ad_status.append("Reliable (Within CLF AD)")
            else:
                final_pred.append("Inactive (AD Flagged)")
                final_score.append("NIL")
                ad_status.append("Unreliable (Outside CLF AD)")

    return final_pred, final_score, ad_status

def render_mol_to_base64(mol, neon_glow=False, glow_color="0,229,255"):
    if mol is None:
        return ""
    from io import BytesIO
    img = Draw.MolToImage(mol, size=(420, 420))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    glow = f"box-shadow:0 0 30px rgba({glow_color},0.45);" if neon_glow else ""
    return f"""
    <div style="display:flex;justify-content:center;">
        <img src="data:image/png;base64,{img_b64}" style="width:320px; background:white; border-radius:16px; padding:12px; {glow}">
    </div>
    """

# ==========================================================
# 📱 Application Layout
# ==========================================================
col_nav, col_main = st.columns([1.2, 4.5], gap="large")

with col_nav:
    st.markdown("""
        <div class="floating-nav-panel">
            <h2 style='color:#FFFFFF; font-weight:800; margin-top:0; font-size:2rem; letter-spacing:-1px;'>IPRED-E 1.0</h2>
            <p style='color:#00E5FF; font-family:"JetBrains Mono", monospace; font-size:1rem; margin-bottom:2.5rem;'>INHIBITOR PREDICTOR FOR EPIDERMAL GROWTH FACTOR RECEPTOR</p>
    """, unsafe_allow_html=True)
    
    navigation_selection = st.radio(
        "Navigation",
        ["🔬 Input and Screening", "⚙️ Architecture Methodology", "📊 Validation and Performance", "📚 References & Citation"],
        label_visibility="collapsed",
        key="main_nav_radio"
    )
    
    st.markdown("""
            <hr style="border:none; border-top:1px solid rgba(255,255,255,0.1); margin: 2.5rem 0;">
            <p style='font-size:0.9rem; color:#94A3B8; font-weight:700; letter-spacing:1px;'>PREDICTION TARGET</p>
            <p style='font-size:1.1rem; color:#FFFFFF; font-weight:600;'>EGFR <span style="color:#FF003C;">(Human)</span></p>
            <br>
            <p style='font-size:0.9rem; color:#94A3B8; font-weight:700; letter-spacing:1px;'>DEVELOPED BY:</p>
            <p style='font-size:1.1rem; color:#FFFFFF; font-weight:600;'>D. Kumar, A. J. Martin.</p>
            <p style='font-size:1.1rem; color:#00E5FF; font-weight:600;'>© 2026 Manipal Academy of Higher Education (MAHE).</p>
            <p style='font-size:1.1rem; color:#FFFFFF; font-weight:600;'>All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

with col_main:
    st.markdown("""
        <div class="main-title-bar">
            <div>
                <h1 style='color: white; margin: 0;'>EGFR Target Inhibitor Predictor</h1>
                <p style='color: #94A3B8; margin: 0.8rem 0 0 0; font-size: 1.3rem; font-weight:400;'>Machine Learning based Classification and Regression Models for Screening Potential EGFR Inhibitors</p>
            </div>
            <div style='background: rgba(0, 229, 255, 0.05); padding: 12px 24px; border-radius: 12px; border: 1px solid rgba(0, 229, 255, 0.3); box-shadow: 0 0 20px rgba(0, 229, 255, 0.1);'>
                <span style='color:#00E5FF; font-weight:800; font-size:1.4rem; font-family:"JetBrains Mono", monospace;'>V1.0</span>
                <span style='color:#FFFFFF; margin-left:12px; font-weight:600; font-size:1.1rem;'>Active</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if navigation_selection == "🔬 Input and Screening":
        st.markdown("<h2 style='font-weight:700; margin-bottom:1.5rem;'>Data Input and Verification</h2>", unsafe_allow_html=True)
        col_in1, col_in2 = st.columns([1.2, 1], gap="large")
        
        with col_in1:
            st.markdown("<p style='color:#94A3B8; font-weight:600; margin-bottom:1rem;'>INPUT MOLECULES</p>", unsafe_allow_html=True)
            input_option = st.radio("Configure Input Stream:", ["Raw SMILES Entry", "Batch Dataset (.CSV)"], horizontal=True, label_visibility="collapsed", key="input_stream_radio")
            
            smiles_list, valid_mols = [], []
            
            if input_option == "Batch Dataset (.CSV)":
                uploaded_file = st.file_uploader("Upload CSV file with column heading 'SMILES' in the first column", type=["csv"], key="file_uploader")
                if uploaded_file is not None:
                    df_input = pd.read_csv(uploaded_file)
                    if "SMILES" not in df_input.columns:
                        st.error("Matrix failure: Missing mandatory 'SMILES' header.")
                        st.stop()
                    smiles_list = [s for s in df_input["SMILES"] if isinstance(s, str)]
            else:
                user_smiles = st.text_area("Enter SMILES Sequence (One in Each Line):", height=220, value=st.session_state.smiles_input_value, placeholder="Paste Input SMILES here", key="smiles_textarea")
                st.session_state.smiles_input_value = user_smiles
                smiles_list = [s.strip() for s in user_smiles.split("\n") if s.strip()]

            if smiles_list:
                for s in smiles_list:
                    m = Chem.MolFromSmiles(s)
                    if m is not None: valid_mols.append(m)

            st.write("") 
            execute = st.button("INITIALIZE SEQUENTIAL SCREENING", use_container_width=True, key="exec_btn")

        with col_in2:
            st.markdown("<p style='color:#94A3B8; font-weight:600; margin-bottom:1rem;'>INPUT STRUCTURE</p>", unsafe_allow_html=True)
            
            if valid_mols:
                if st.session_state.mol_idx >= len(valid_mols) or st.session_state.mol_idx < 0:
                    st.session_state.mol_idx = 0
                current_mol = valid_mols[st.session_state.mol_idx]
                st.markdown(render_mol_to_base64(current_mol, neon_glow=True, glow_color="0,229,255"), unsafe_allow_html=True)
                
                st.write("")
                p1, p2, p3 = st.columns([1, 1.5, 1])
                with p1:
                    if st.button("◀ PREV", use_container_width=True, key="prev_input_mol"):
                        st.session_state.mol_idx -= 1
                        st.rerun()
                with p2:
                    st.markdown(f"<div style='text-align:center; padding:0.5rem; color:#00E5FF; font-family:\"JetBrains Mono\"; font-weight:700;'>[ {st.session_state.mol_idx + 1} / {len(valid_mols)} ]</div>", unsafe_allow_html=True)
                with p3:
                    if st.button("NEXT ▶", use_container_width=True, key="next_input_mol"):
                        st.session_state.mol_idx += 1
                        st.rerun()
            else:
                st.markdown("""
                <div style="background: rgba(16,24,39,0.5); border: 1px dashed rgba(255,255,255,0.1); border-radius: 16px; height: 350px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                    <span style="font-size: 3rem; opacity: 0.2;">⬡</span>
                    <p style="color: #64748B; font-family: 'JetBrains Mono'; margin-top: 1rem;">AWAITING_INPUT_SEQUENCE</p>
                </div>
                """, unsafe_allow_html=True)

        if execute and smiles_list:
            st.session_state.profile_idx = 0
            st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.05); margin:3rem 0;'>", unsafe_allow_html=True)

            with st.expander("ANALYZING (View Operations Log)", expanded=False):
                st.write(">> Canonicalizing SMILES...")
                canonical_smiles, mols = [], []
                for smi in smiles_list:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                        mols.append(mol)

                if not mols:
                    st.error("No valid molecules detected.")
                    st.stop()

                st.write(">> Calculating Mordred descriptors...")
                calc = Calculator(descriptors, ignore_3D=True)
                df_desc = calc.pandas(mols)

                st.write(">> Loading Models & AD Matrices...")
                (svm_clf, xgb_reg, scaler_clf, knn_clf, ad_thresh_clf, scaler_reg, knn_reg, ad_thresh_reg) = load_models_and_params()

                st.write(">> Step 1: SVM Classification (Active/Inactive)...")
                X_clf = prepare_features(df_desc, clf_features)
                X_clf_scaled = scaler_clf.transform(X_clf)
                
                md_clf, ad_clf = calculate_knn_ad(X_clf_scaled, knn_clf, ad_thresh_clf)
                pred_clf = svm_clf.predict(X_clf_scaled)
                prob_clf = svm_clf.predict_proba(X_clf_scaled)[:, 1]

                st.write(">> Step 2: XGBoost Regression (Potency) for Actives...")
                X_reg = prepare_features(df_desc, reg_features)
                X_reg_scaled = scaler_reg.transform(X_reg)
                
                md_reg, ad_reg = calculate_knn_ad(X_reg_scaled, knn_reg, ad_thresh_reg)
                pred_reg = xgb_reg.predict(X_reg_scaled)

                st.write(">> Compiling Sequential Protocol...")
                final_prediction, final_score, final_ad_status = sequential_prediction(
                    pred_clf, prob_clf, ad_clf, pred_reg, ad_reg
                )
                
                formatted_score = [np.round(s, 3) if isinstance(s, float) else s for s in final_score]

                st.write(">> Analysis complete.")

            # Compile Results DataFrame
            st.session_state.results_df = pd.DataFrame({
                "SMILES": canonical_smiles,
                "Prediction": final_prediction,
                "Predicted_pIC50": formatted_score,
                "SVM_Prob": np.round(prob_clf, 3),
                "AD_Status": final_ad_status,
                "CLF_Distance": np.round(md_clf, 3),
                "REG_Distance": np.round(md_reg, 3)
            })

            # ==========================================
            # DIAGNOSTIC EXPORT GENERATOR
            # ==========================================
            clf_raw_df = pd.DataFrame(X_clf.values, columns=[f"RAW_CLF_{c}" for c in clf_features])
            clf_scaled_df = pd.DataFrame(X_clf_scaled, columns=[f"SCALED_CLF_{c}" for c in clf_features])
            reg_raw_df = pd.DataFrame(X_reg.values, columns=[f"RAW_REG_{c}" for c in reg_features])
            reg_scaled_df = pd.DataFrame(X_reg_scaled, columns=[f"SCALED_REG_{c}" for c in reg_features])
            
            st.session_state.diagnostic_df = pd.concat([
                st.session_state.results_df.reset_index(drop=True),
                clf_raw_df.reset_index(drop=True),
                clf_scaled_df.reset_index(drop=True),
                reg_raw_df.reset_index(drop=True),
                reg_scaled_df.reset_index(drop=True)
            ], axis=1)

        if st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            if st.session_state.profile_idx >= len(results_df) or st.session_state.profile_idx < 0:
                st.session_state.profile_idx = 0
            
            st.markdown("<h2 style='font-weight:700; margin: 2rem 0 1.5rem 0;'>PREDICTION RESULTS</h2>", unsafe_allow_html=True)
            
            act_count = len(results_df[results_df["Prediction"].isin(["Active", "Active (AD Flagged)"])])
            hit_rate = np.round((act_count / len(results_df)) * 100, 1) if len(results_df) > 0 else 0
            
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.markdown(f'<div class="metric-panel"><div class="mp-value val-neutral">{len(results_df)}</div><div class="mp-label">Total Screened</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-panel metric-active"><div class="mp-value val-cyan">{act_count}</div><div class="mp-label">Predicted Actives</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="metric-panel metric-active"><div class="mp-value val-cyan">{hit_rate}%</div><div class="mp-label">Hit Rate</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="metric-panel"><div class="mp-value val-neutral">2D & 3D</div><div class="mp-label">Descriptor Space</div></div>', unsafe_allow_html=True)

            st.write("")
            v_col1, v_col2 = st.columns([1.5, 1], gap="large")
            
            with v_col1:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Scatter Plot Matrix (Actives)</p>", unsafe_allow_html=True)
                results_df["_Internal_Index"] = results_df.index
                
                active_df = results_df[results_df["Prediction"].isin(["Active", "Active (AD Flagged)"])]
                if not active_df.empty:
                    fig = px.scatter(
                        active_df, x="SVM_Prob", y="Predicted_pIC50", color="Prediction",
                        color_discrete_map={"Active": "#00E5FF", "Active (AD Flagged)": "#FF8C00"},
                        hover_data={"SMILES": True, "Predicted_pIC50": True, "_Internal_Index": False},
                        template="plotly_dark", custom_data=["_Internal_Index"]
                    )
                    
                    if st.session_state.profile_idx in active_df.index:
                        sel_row = results_df.iloc[st.session_state.profile_idx]
                        fig.add_trace(go.Scatter(
                            x=[sel_row["SVM_Prob"]], y=[sel_row["Predicted_pIC50"]], mode="markers",
                            marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color="#FFFFFF", width=3)),
                            showlegend=False, hoverinfo="skip"
                        ))

                    fig.update_layout(
                        margin=dict(l=0, r=0, t=10, b=0), height=420,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="SVM Classification Probability", gridcolor="rgba(255,255,255,0.05)", range=[0.5, 1.05]),
                        yaxis=dict(title="XGBoost Predicted pIC50", gridcolor="rgba(255,255,255,0.05)"),
                        clickmode="event+select"
                    )
                    scatter_selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="scatter_actives")
                    if scatter_selection and "selection" in scatter_selection:
                        points = scatter_selection["selection"].get("points", [])
                        if points and "customdata" in points[0]:
                            st.session_state.profile_idx = points[0]["customdata"][0]
                else:
                    st.info("No active molecules detected to plot.")

            with v_col2:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-align:center;'>pIC50 Value</p>", unsafe_allow_html=True)
                target_row = results_df.iloc[st.session_state.profile_idx]
                target_smi = target_row["SMILES"]
                target_score = target_row["Predicted_pIC50"]
                target_pred = target_row["Prediction"]
                target_ad = target_row["AD_Status"]
                
                if target_pred == "Active": gauge_color, glow_color = "#00E5FF", "0,229,255"
                elif "Flagged" in target_pred: gauge_color, glow_color = "#FF8C00", "255,140,0"
                else: gauge_color, glow_color = "#FF003C", "255,0,60"
                
                if isinstance(target_score, str):
                    disp_score = 0
                else:
                    disp_score = float(target_score)

                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = disp_score,
                    number = {'font': {'color': gauge_color, 'family': 'JetBrains Mono', 'size': 38}},
                    title = {'text': "Predicted pIC50", 'font': {'color': '#94A3B8', 'size': 12}},
                    gauge = {
                        'axis': {'range': [0, 12], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.1)"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'borderwidth': 1, 'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [{'range': [0, 5], 'color': "rgba(255, 0, 60, 0.15)"}, {'range': [5, 12], 'color': "rgba(0, 229, 255, 0.15)"}],
                    }
                ))
                fig_gauge.update_layout(height=160, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")
                
                st.markdown(f"<div style='text-align:center; margin-top:-15px; margin-bottom:15px;'><span style='color:{gauge_color}; font-size: 0.85rem; font-weight:700;'>{target_ad}</span></div>", unsafe_allow_html=True)
                st.markdown(render_mol_to_base64(Chem.MolFromSmiles(target_smi), neon_glow=True, glow_color=glow_color), unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='background:rgba(16,24,39,0.3); border: 1px solid rgba(255,255,255,0.05); padding:12px; border-radius:10px; margin-top:10px;'>
                        <p style='margin:0; font-size:0.85rem; color:#94A3B8;'>SMILES:</p>
                        <p style='margin:0; font-family:"JetBrains Mono"; font-size:0.85rem; color:#00E5FF; word-break:break-all;'>{target_smi}</p>
                        <p style='margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;'>Status: <span style='color:{gauge_color}; font-weight:800;'>{target_pred.upper()}</span></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                p1, p2, p3 = st.columns([1, 1.5, 1])
                with p1:
                    if st.button("◀ COMP", use_container_width=True, key="prev_prof_mol"):
                        st.session_state.profile_idx -= 1
                        st.rerun()
                with p2:
                    st.markdown(f"<div style='text-align:center; padding:0.5rem; color:#E2E8F0; font-family:\"JetBrains Mono\"; font-weight:700;'>Entry {st.session_state.profile_idx + 1} / {len(results_df)}</div>", unsafe_allow_html=True)
                with p3:
                    if st.button("COMP ▶", use_container_width=True, key="next_prof_mol"):
                        st.session_state.profile_idx += 1
                        st.rerun()

            st.divider()
            st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>COMPLETE PREDICTION RESULTS</p>", unsafe_allow_html=True)
            
            display_df = results_df.drop(columns=["_Internal_Index"], errors='ignore')
            
            def style_predictions(x):
                if x == "Active": return "background-color: rgba(0, 229, 255, 0.1); color: #00E5FF; font-weight: bold;"
                elif "Flagged" in str(x): return "background-color: rgba(255, 140, 0, 0.1); color: #FF8C00; font-weight: bold;"
                else: return "color: #FF003C;"

            styled_df = display_df.style.map(style_predictions, subset=["Prediction"])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                csv = display_df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 EXPORT STANDARD RESULTS (CSV)", data=csv, file_name="IPRED-E_Results.csv", mime="text/csv", key="dl_std")
            
            with btn_c2:
                if st.session_state.diagnostic_df is not None:
                    diag_csv = st.session_state.diagnostic_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⚠️ EXPORT FULL AD DIAGNOSTIC (CSV)", data=diag_csv, file_name="AD_Diagnostic_Export.csv", mime="text/csv", key="dl_diag")

    elif navigation_selection == "⚙️ Architecture Methodology":
        st.markdown("<h2 style='font-weight:700; margin-bottom:1.5rem;'>Methodology of IPRED-E 1.0</h2>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns([1, 1.2], gap="large")
        with col_a:
            st.markdown("""
            <strong style="color:#00E5FF; font-size:1.15rem; letter-spacing:0.5px;">Sequential ML Workflow for EGFR Prediction</strong>
            <div style="text-align: justify;">
            IPRED-E 1.0 pipeline uses a sequential machine learning approach to screen and predict potential Epidermal Growth Factor Receptor (EGFR) inhibitors. The workflow combines SMILES canonicalisation, descriptor calculation, feature normalisation, applicability domain assessment, and machine learning model based prdictions for both activity classification and pIC50 prediction.
            </div>
            
            <p style='color:#00E5FF; font-family:"JetBrains Mono"; margin-top:2rem;'>Step 1: MOLECULAR DESCRIPTOR CALCULATION</p>
            <div style="text-align: justify;">
            The workflow starts with canonicalization of raw SMILES structures, followed by calculation of molecular descriptors using the Mordred. To reduce noise and avoid unnecessary redundancy, only pre-selected descriptors required for the models are generated and used. The classification model uses a set of 50 descriptors, while the regression model uses a related but distinct set of 40 descriptors. These descriptor subsets were selected by Global Feature Ranking method, during model development to retain the most informative features while minimizing noise and collinearity.
            </div>
            
            <p style='color:#00E5FF; font-family:"JetBrains Mono"; margin-top:2rem;'>Step 2: CLASSIFICATION MODEL SCREENING</p>
            <div style="text-align: justify;">
            The first major filtering step is done using a calibrated Support Vector Machine (SVM) using a Radial Basis Function (RBF) kernel. The model evaluates the 50 selected descriptors and classifies compounds as either Active or Inactive class. Only compounds predicted to be active, proceed to next stage of the workflow. 
            <br><br>
            <b>Classification Applicability Domain (AD):</b> To ensure that predictions are made only for molecules that are sufficiently similar to training data, a k-Nearest Neighbors (k-NN) based applicability domain check is applied. The Euclidean distance of each incoming molecule is calculated relative to scaled classification training space. Molecules with distance greater than the predefined threshold are flagged with an AD Warning, indicating that their chemical structure lies outside the reliable region of the classification model and may represent an extrapolation beyond the training data.
            </div>

            <p style='color:#00E5FF; font-family:"JetBrains Mono"; margin-top:2rem;'>Step 3: REGRESSION MODEL PREDICTION</p>
            <div style="text-align: justify;">
            Only compounds which pass the SVM active filter are evaluated using the XGBoost Regressor model. A set of selected 40 descriptors are used by the models to estimate the binding potency of each compound, expressed as pIC50.
            <br><br>
            <b>Regression Applicability Domain (AD):</b> A second, independent k-NN-based applicability domain assessment is done using the regression training space. This additional check determines if the compound falls within chemical space where the regression model has sufficient training support. By applying separate AD checks at both the classification and regression stages, the workflow helps distinguish compounds that are merely predicted to be active from those for which the predicted pIC50 values are supported by relevant chemical examples in the training data.
            <br><br>
            <p style='color:#FFB300; margin-top:2rem;'> Together, these sequential filtering steps provide a more conservative framework for screening and prioritizing potential EGFR inhibitors and helps to reduce the risk of advancing compounds based on unreliable model extrapolation.
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Flowchart Pipeline</p>", unsafe_allow_html=True)
            dot = Digraph("SequentialFlow", engine="dot")
            dot.attr(rankdir="TB", splines="ortho", nodesep="0.6", ranksep="0.7", bgcolor="transparent")
            dot.attr("node", shape="rect", style="rounded,filled", fontsize="13", fontname="Inter", color="white", penwidth="1")
            dot.attr("edge", color="#00E5FF", penwidth="2")
            
            dot.node("In", "SMILES Input", fillcolor="#0A0E17", fontcolor="#00E5FF", color="#00E5FF")
            dot.node("Desc", "Mordred Descriptors", fillcolor="#0A0E17", fontcolor="#FFFFFF", color="rgba(255,255,255,0.2)")
            dot.node("CLFAD", "SVM AD Check (k-NN)", fillcolor="#0A0E17", fontcolor="#FFB300", color="#FFB300")
            dot.node("SVM", "SVM Classification", fillcolor="#0A0E17", fontcolor="#FFFFFF", color="rgba(255,255,255,0.2)")
            dot.node("Inact", "Inactive\n(Stop)", fillcolor="#0A0E17", fontcolor="#FF003C", color="#FF003C")
            dot.node("REGAD", "XGBoost AD Check (k-NN)", fillcolor="#0A0E17", fontcolor="#FFB300", color="#FFB300")
            dot.node("XGB", "XGBoost Regression", fillcolor="#0A0E17", fontcolor="#FFFFFF", color="rgba(255,255,255,0.2)")
            dot.node("Out", "Active Hit (with pIC50)", fillcolor="#0A0E17", fontcolor="#00E5FF", color="#00E5FF")
            
            dot.edge("In", "Desc"); dot.edge("Desc", "CLFAD"); dot.edge("CLFAD", "SVM")
            
            with dot.subgraph() as s:
                s.attr(rank="same")
                s.node("Inact")
                s.node("REGAD")
            
            dot.edge("SVM", "Inact", label="  Class 0", fontcolor="#FF003C")
            dot.edge("SVM", "REGAD", label="Class 1", fontcolor="#00E5FF")
            
            dot.edge("REGAD", "XGB"); dot.edge("XGB", "Out")
            st.graphviz_chart(dot, use_container_width=True)

    elif navigation_selection == "📊 Validation and Performance":
        st.markdown("<h2 style='font-weight:700; margin-bottom:1.5rem;'>Model Validation and Performance Metrics</h2>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs([
            "Classification Model Metrics", 
            "Regression Model Metrics"
        ])
        
        with tab1:
            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:1.1rem; margin-top:1rem;'>SUPPORT VECTOR MACHINE (SVM) CLASSIFIER</p>", unsafe_allow_html=True)
            
            clf_metrics = pd.DataFrame({
                "Validation Phase": [
                    "5-fold CV", "Test set (3337 Molecules)", "AD filtered Test set (2954 Molecules)", 
                    "10-fold CV", "LOO", "Scaffold Disjoint", "Y-randomisation"
                ],
                "Accuracy": [0.9083, 0.9119, 0.9225, 0.9088, 0.8420, 0.8457, 0.5208],
                "Precision": [0.9290, 0.9301, 0.9413, 0.9306, 0.8620, 0.8401, 0.6306],
                "Recall": [0.9252, 0.9301, 0.9388, 0.9242, 0.8921, 0.9330, 0.5789],
                "Specificity": [0.8794, 0.8808, 0.8924, 0.8826, 0.7568, 0.6962, 0.4219],
                "F1 Score": [0.9271, 0.9301, 0.9401, 0.9274, 0.8768, 0.8841, 0.6036],
                "Balanced Accuracy": [0.9023, 0.9055, 0.9156, 0.9034, 0.8244, 0.8146, 0.5004]
            })
            
            st.dataframe(clf_metrics.style.format(precision=4), use_container_width=True, hide_index=True)
            
            st.write("")
            c_col1, c_col2 = st.columns([1.5, 1], gap="large")
            
            with c_col1:
                selected_clf_phases = st.multiselect(
                    "Select Validation Phases to visualize:", 
                    options=clf_metrics["Validation Phase"].tolist(),
                    default=["5-fold CV", "Test set (3337 Molecules)", "AD filtered Test set (2954 Molecules)"],
                    key="clf_multi"
                )
                
                if selected_clf_phases:
                    filtered_clf = clf_metrics[clf_metrics["Validation Phase"].isin(selected_clf_phases)]
                    fig_clf = px.bar(
                        filtered_clf.melt(id_vars="Validation Phase", var_name="Metric", value_name="Score"),
                        x="Validation Phase", y="Score", color="Metric", barmode="group",
                        template="plotly_dark", color_discrete_sequence=["#00E5FF", "#00B3CC", "#FFB300", "#CC8F00", "#FF8C00", "#FF003C"]
                    )
                    fig_clf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Score", margin=dict(t=20, b=0, l=0, r=0))
                    st.plotly_chart(fig_clf, use_container_width=True)
            
            with c_col2:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Hyperparameters</p>", unsafe_allow_html=True)
                st.code('{"kernel": "rbf",\n "gamma": "auto",\n "C": 10}', language='json')
                
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase; margin-top:1rem;'>Selected Features (50)</p>", unsafe_allow_html=True)
                with st.expander("View Classification Feature List"):
                    st.write(", ".join(clf_features))

        with tab2:
            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:1.1rem; margin-top:1rem;'>XGBOOST (XGB) REGRESSOR</p>", unsafe_allow_html=True)
            
            reg_metrics = pd.DataFrame({
                "Validation Phase": [
                    "5-fold CV", "Test set (3873 Molecules)", "AD filtered Test set (3235 Molecules)", 
                    "10-fold CV", "LOO", "Scaffold Disjoint", "Y-randomisation"
                ],
                "R2": [0.6870, 0.6870, 0.7026, 0.6856, 0.5288, 0.4911, -0.1232],
                "RMSE": [0.8238, 0.8238, 0.7766, 0.8218, 1.0558, 0.9844, 1.5533],
                "MAE": [0.6170, 0.6170, 0.5808, 0.6161, 0.8248, 0.7529, 1.2813],
                "Pearson Correlation": [0.8303, 0.8303, 0.8395, 0.8292, 0.7276, 0.7014, 0.0015]
            })
            
            st.dataframe(reg_metrics.style.format(precision=4), use_container_width=True, hide_index=True)
            
            st.write("")
            r_col1, r_col2 = st.columns([1.5, 1], gap="large")
            
            with r_col1:
                selected_reg_phases = st.multiselect(
                    "Select Validation Phases to visualize:", 
                    options=reg_metrics["Validation Phase"].tolist(),
                    default=["5-fold CV", "Test set (3873 Molecules)", "AD filtered Test set (3235 Molecules)"],
                    key="reg_multi"
                )
                
                if selected_reg_phases:
                    filtered_reg = reg_metrics[reg_metrics["Validation Phase"].isin(selected_reg_phases)]
                    fig_reg1 = px.bar(
                        filtered_reg.melt(id_vars="Validation Phase", value_vars=["R2", "RMSE", "MAE", "Pearson Correlation"], var_name="Metric", value_name="Score"),
                        x="Validation Phase", y="Score", color="Metric", barmode="group",
                        template="plotly_dark", color_discrete_sequence=["#00E5FF", "#FFB300", "#CC8F00", "#FF8C00"]
                    )
                    fig_reg1.update_layout(title="Regression Metric Comparison", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Score", margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_reg1, use_container_width=True)

            with r_col2:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Hyperparameters</p>", unsafe_allow_html=True)
                st.code('{\n "subsample": 0.8,\n "n_estimators": 500,\n "max_depth": 7,\n "learning_rate": 0.05,\n "colsample_bytree": 0.6\n}', language='json')
                
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase; margin-top:1rem;'>Selected Features (40)</p>", unsafe_allow_html=True)
                with st.expander("View Regression Feature List"):
                    st.write(", ".join(reg_features))
            
    elif navigation_selection == "📚 References & Citation":
        st.markdown("<h2 style='font-weight:700; margin-bottom:1.5rem;'>References, Citation and Scientific Literature</h2>", unsafe_allow_html=True)
        
        st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\";'>DEVELOPERS, INSTITUTIONAL AFFILIATION and COPYRIGHT</p>", unsafe_allow_html=True)
        st.markdown("- **© 2026 Manipal Academy of Higher Education (MAHE).** <span style='color:#94A3B8;'>All rights reserved.</span>", unsafe_allow_html=True)
        st.markdown("- **D.Kumar, A. J. Martin** <span style='color:#94A3B8;'>[Manipal Academy of Higher Education - MAHE, Manipal, Karnataka]</span>", unsafe_allow_html=True)
        
        st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.05); margin: 2rem 0;'>", unsafe_allow_html=True)
        
        st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\";'>How to Cite IPRED-E 1.0</p>", unsafe_allow_html=True)
        st.markdown("If you use the IPRED-E 1.0 webtool in research or publications, please cite:")
        st.markdown("""
        <div style="background: rgba(0, 229, 255, 0.05); padding: 1.5rem; border-left: 4px solid #00E5FF; border-radius: 8px; margin-bottom: 1.5rem;">
            <p style="margin:0; color:#FFFFFF;"><b>IPRED-E 1.0 Webtool</b> | D. Kumar, A. J. Martin | Version 1.0 (2026).</p>
            <p style="margin:0; color:#94A3B8; margin-top: 0.5rem;"><b>Webtool URL:</b> <i>https://ipred-e-1-clf-reg-screening.streamlit.app/</i></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\";'>SCIENTIFIC LITERATURE AND COMPUTATIONAL PACKAGES</p>", unsafe_allow_html=True)
        st.markdown("Below is the complete list of scientific literature, software tools, and computational packages used in the development, validation, and deployment of IPRED-E 1.0.")
        st.markdown("""
        <div style="color:#94A3B8; font-size: 0.95rem;">
        
        <strong style="color:white;">1. Machine Learning & Data Processing</strong>
        *   **Chen, T., & Guestrin, C.** XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD* (2016).
        *   **Cortes, C., & Vapnik, V.** Support-vector networks. *Machine Learning*, 20, 273-297 (1995).
        *   **Pedregosa et al.** Scikit-Learn: Machine Learning in Python. *JMLR* 12, 2825–2830 (2011).

        <strong style="color:white; display:block; margin-top:1rem;">2. Descriptor Generation & Cheminformatics</strong>
        *   **Moriwaki et al.** Mordred: A Comprehensive Descriptor Library for Molecular Descriptors. *J. Cheminf.* 10, 4 (2018).
        *   **RDKit:** Open-source cheminformatics. http://www.rdkit.org.

        <strong style="color:white; display:block; margin-top:1rem;">3. Model Interpretation & Performance Evaluation</strong>
        *   **Powers, D.** Evaluation: Precision, Recall, F-measure, ROC, Informedness, Markedness. *JMLT* 2, 37–63 (2011).
        *   **Brodersen, K. H. et al.** The Balanced Accuracy and Its Posterior Distribution.*Proc. Int. Conf. Pattern Recogn.*, 20 (2010).

        <strong style="color:white; display:block; margin-top:1rem;">4. Applicability Domain (AD)</strong>
        *   **Sahigara, F. et al.** Comparison of Different Approaches to Define the Applicability Domain. *J. Chemometrics* 26, 269–276 (2012).
        *   **Tropsha, A.** Best Practices for QSAR Model Development, Validation, and Exploitation. *Mol. Inf.* 29, 476-488 (2010).

        <strong style="color:white; display:block; margin-top:1rem;">5. Datasets & Source</strong>
        *   **EGFR Bioassay Data:** Retrieved from CHEMBL, PubChem, and BindingDB.

        <strong style="color:white; display:block; margin-top:1rem;">6. Software, Platforms & Versions (Used in IPRED-E 1.0)</strong>
        
        | Software / Package | Version | Purpose |
        | :--- | :--- | :--- |
        | **Python** | 3.10 | Core Development |
        | **Streamlit** | 1.50 | Web Interface Deployment |
        | **RDKit** | 2025.03.6 | SMILES parsing |
        | **Mordred** | 1.2.0 | Topological Descriptor generation |
        | **XGBoost** | 2.1.3 | Regression Modeling |
        | **scikit-learn** | 1.4.2 | Model inference and Normalization |
        | **NumPy** | 1.25.2 | Numerical computing |
        | **Pandas** | 2.3.2 | DataFrame processing |
        | **Graphviz** | latest | Flowchart rendering |
        
        </div>
        """, unsafe_allow_html=True)
