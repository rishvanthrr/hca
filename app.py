import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from federated_triage.aggregator import aggregate_predictions
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Federated Triage Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1c1f26 100%);
    }
    .status-card {
        padding: 20px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00d4ff;
    }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("🏥 Federated Triage Analysis System")
st.markdown("---")

# --- SIDEBAR: PATIENT INPUT ---
with st.sidebar:
    st.header("Patient Assessment")
    st.info("Input patient parameters for multi-clinic federated triage.")
    
    with st.form("patient_form"):
        age = st.slider("Age", 0, 100, 45)
        fever = st.selectbox("Fever Level", ["None", "Mild", "Moderate", "High", "Critical"])
        cough = st.selectbox("Cough Severity", ["None", "Mild", "Persistent", "Severe"])
        fatigue = st.selectbox("Fatigue", ["None", "Mild", "Moderate", "Severe"])
        travel_history = st.checkbox("Recent International Travel")
        comorbidities = st.slider("Number of Comorbidities", 0, 10, 1)
        spo2 = st.slider("SpO2 Levels (%)", 70, 100, 95)
        
        submitted = st.form_submit_button("Perform Triage Analysis")

# --- DATA MAPPING ---
fever_map = {"None": 0, "Mild": 1, "Moderate": 2, "High": 3, "Critical": 4}
cough_map = {"None": 0, "Mild": 1, "Persistent": 2, "Severe": 3}
fatigue_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

patient_data = {
    "age": age,
    "fever": fever_map[fever],
    "cough": cough_map[cough],
    "fatigue": fatigue_map[fatigue],
    "travel_history": 1 if travel_history else 0,
    "comorbidities": comorbidities,
    "spo2": spo2
}

# --- MAIN ANALYSIS ---
if submitted:
    with st.spinner("Aggregating clinic-specific insights..."):
        try:
            results = aggregate_predictions(patient_data)
            
            # Layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Global Triage Recommendation")
                level_colors = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
                color = level_colors.get(results["final_prediction"], "#ffffff")
                
                st.markdown(f"""
                    <div style="padding: 30px; border-radius: 15px; background: {color}22; border: 2px solid {color}; text-align: center;">
                        <h2 style="color: {color}; margin:0;">{results['final_prediction']}</h2>
                        <p style="color: #ccc; margin-top: 10px;">Consensus Triage Level</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.write("**Federated Confidence:**")
                probs = results["global_probabilities"]
                fig = px.bar(
                    x=list(probs.keys()), 
                    y=list(probs.values()),
                    labels={'x': 'Triage Level', 'y': 'Probability'},
                    color=list(probs.keys()),
                    color_discrete_map=level_colors,
                    height=300
                )
                fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Clinic Ensemble Details")
                
                # Clinic Weights Card
                w = results["weights"]
                fig_pie = px.pie(
                    names=list(w.keys()), 
                    values=list(w.values()),
                    title="Model Reliability Weights",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- INDIVIDUAL CLINIC RESULTS ---
            st.markdown("### 🏥 Local Clinic Insights")
            c_a, c_b, c_c = st.columns(3)
            
            with c_a:
                st.markdown(f"""
                <div class="status-card">
                    <b>Clinic A (RF)</b><br/>
                    Prediction: <span style="color:{level_colors[results['clinic_a']['prediction']]}">{results['clinic_a']['prediction']}</span><br/>
                    Conf: {results['clinic_a']['confidence']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
            with c_b:
                st.markdown(f"""
                <div class="status-card">
                    <b>Clinic B (MLP)</b><br/>
                    Prediction: <span style="color:{level_colors[results['clinic_b']['prediction']]}">{results['clinic_b']['prediction']}</span><br/>
                    Conf: {results['clinic_b']['confidence']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
            with c_c:
                st.markdown(f"""
                <div class="status-card">
                    <b>Clinic C (XGB)</b><br/>
                    Prediction: <span style="color:{level_colors[results['clinic_c']['prediction']]}">{results['clinic_c']['prediction']}</span><br/>
                    Conf: {results['clinic_c']['confidence']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error in federated aggregation: {str(e)}")
            st.info("Make sure you have run the model training script first.")

else:
    # --- LANDING STATE ---
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.info("👋 Welcome! Use the sidebar to input patient data and generate a federated triage prediction.")
        st.image("https://img.freepik.com/free-vector/digital-healthcare-concept_1017-30230.jpg?w=800", use_container_width=True)
    
    with col2:
        st.subheader("Model Performance Summary")
        if os.path.exists("models/training_results.json"):
            with open("models/training_results.json", "r") as f:
                res = json.load(f)
                df_perf = pd.DataFrame([
                    {"Clinic": k, "Model": v["model"], "Accuracy": v["accuracy"]} for k,v in res.items()
                ])
                st.table(df_perf)
                
                fig_perf = px.bar(df_perf, x="Clinic", y="Accuracy", color="Clinic", title="Current Local Model Accuracies")
                st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.warning("No training results found. Please run training to see performance metrics.")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 Federated Care Alliance — Advanced Triage Modeling")
