"""
Brain Tumor Prediction System - Professional Medical UI
A comprehensive web application for medical professionals to predict brain tumor outcomes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our inference module
from inference import BrainTumorPredictor

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #4a90e2;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4a90e2;
        padding-left: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .survival-rate {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
    }
    
    .treatment-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-card {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4a90e2, #357abd);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #357abd, #2c5aa0);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;">
        Professional AI-powered tool for medical professionals to predict brain tumor outcomes and treatment recommendations
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        with st.spinner('Loading AI model...'):
            try:
                st.session_state.predictor = BrainTumorPredictor()
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.stop()
    
    # Sidebar for input parameters
    with st.sidebar:
        st.markdown("## 📋 Patient Information")
        st.markdown("---")
        
        # Patient demographics
        st.markdown("### 👤 Demographics")
        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=50,
            help="Patient's age in years"
        )
        
        # Tumor characteristics
        st.markdown("### 🧬 Tumor Characteristics")
        tumor_size = st.number_input(
            "Tumor Size (cm)",
            min_value=0.1,
            max_value=20.0,
            value=5.0,
            step=0.1,
            help="Maximum tumor diameter in centimeters"
        )
        
        tumor_growth_rate = st.selectbox(
            "Tumor Growth Rate",
            options=["Slow", "Moderate", "Rapid"],
            index=1,
            help="Rate of tumor growth based on imaging studies"
        )
        
        tumor_location = st.selectbox(
            "Tumor Location",
            options=["Frontal", "Temporal", "Parietal", "Occipital", "Cerebellum"],
            index=0,
            help="Primary location of the tumor in the brain"
        )
        
        # Clinical findings
        st.markdown("### 🔬 Clinical Findings")
        symptom_severity = st.selectbox(
            "Symptom Severity",
            options=["Mild", "Moderate", "Severe"],
            index=1,
            help="Severity of neurological symptoms"
        )
        
        mri_findings = st.selectbox(
            "MRI Findings",
            options=["Normal", "Abnormal", "Severe"],
            index=1,
            help="Results from MRI imaging"
        )
        
        radiation_exposure = st.selectbox(
            "Radiation Exposure",
            options=["Low", "Medium", "High"],
            index=1,
            help="Patient's history of radiation exposure"
        )
        
        # Prediction button
        st.markdown("---")
        predict_button = st.button("🔮 Generate Prediction", type="primary", use_container_width=True)
        
        # Information section
        st.markdown("---")
        st.markdown("### ℹ️ About This Tool")
        st.info("""
        This AI system analyzes patient data to predict:
        - **Survival Rate**: Probability of survival
        - **Treatment Recommendation**: Optimal treatment approach
        
        **Note**: This is a decision support tool. Always consult with medical specialists for final treatment decisions.
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input summary
        if predict_button:
            st.markdown('<div class="section-header">📊 Patient Data Summary</div>', unsafe_allow_html=True)
            
            # Create input data dictionary
            input_data = {
                'Age': age,
                'Tumor_Size': tumor_size,
                'Tumor_Growth_Rate': tumor_growth_rate,
                'Symptom_Severity': symptom_severity,
                'Tumor_Location': tumor_location,
                'MRI_Findings': mri_findings,
                'Radiation_Exposure': radiation_exposure
            }
            
            # Display input summary
            input_df = pd.DataFrame([input_data]).T
            input_df.columns = ['Value']
            st.dataframe(input_df, use_container_width=True)
            
            # Validate input
            errors = st.session_state.predictor.validate_input(input_data)
            if errors:
                st.error("❌ Input validation errors:")
                for error in errors:
                    st.error(f"• {error}")
            else:
                # Make prediction
                with st.spinner('Analyzing patient data...'):
                    try:
                        result = st.session_state.predictor.predict(input_data)
                        
                        # Display results
                        st.markdown('<div class="section-header">🎯 AI Prediction Results</div>', unsafe_allow_html=True)
                        
                        # Survival rate prediction
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3 style="text-align: center; margin-bottom: 1rem;">Predicted Survival Rate</h3>
                            <div class="survival-rate">{result['survival_rate']}%</div>
                            <p style="text-align: center; margin: 0;">Based on current patient data</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Treatment recommendation
                        st.markdown(f"""
                        <div class="treatment-card">
                            <h4>🎯 Recommended Treatment</h4>
                            <h3 style="color: #28a745; margin: 0.5rem 0;">{result['treatment']}</h3>
                            <p><strong>Confidence:</strong> {result['treatment_confidence']}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Treatment probability breakdown
                        st.markdown("### 📈 Treatment Probability Analysis")
                        treatment_probs = result['all_treatment_probs']
                        
                        # Create bar chart
                        fig = px.bar(
                            x=list(treatment_probs.keys()),
                            y=list(treatment_probs.values()),
                            title="Treatment Recommendation Probabilities",
                            labels={'x': 'Treatment Option', 'y': 'Probability (%)'},
                            color=list(treatment_probs.values()),
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(
                            showlegend=False,
                            height=400,
                            title_font_size=16
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk assessment
                        st.markdown("### ⚠️ Risk Assessment")
                        if result['survival_rate'] < 30:
                            st.markdown("""
                            <div class="warning-card">
                                <h4>⚠️ High Risk Case</h4>
                                <p>This patient shows indicators of high risk. Immediate specialist consultation is recommended.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif result['survival_rate'] < 60:
                            st.markdown("""
                            <div class="info-card">
                                <h4>⚠️ Moderate Risk Case</h4>
                                <p>This patient shows moderate risk indicators. Regular monitoring and specialist follow-up recommended.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="info-card">
                                <h4>✅ Lower Risk Case</h4>
                                <p>This patient shows more favorable indicators. Continue with recommended treatment plan.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {e}")
    
    with col2:
        # Quick stats and information
        st.markdown('<div class="section-header">📊 System Information</div>', unsafe_allow_html=True)
        
        # Model info
        st.markdown("""
        <div class="metric-container">
            <h4>🤖 AI Model Status</h4>
            <p>✅ Model Loaded</p>
            <p>✅ Ready for Predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance (placeholder)
        st.markdown("""
        <div class="metric-container">
            <h4>📈 Key Factors</h4>
            <p>• Tumor Size</p>
            <p>• Growth Rate</p>
            <p>• Symptom Severity</p>
            <p>• MRI Findings</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class="info-card">
            <h4>⚠️ Important Disclaimer</h4>
            <p><small>This AI system is designed to assist medical professionals and should not replace clinical judgment. Always consult with specialists for treatment decisions.</small></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
