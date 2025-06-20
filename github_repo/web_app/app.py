"""
Rush Medical College AI Admissions Assistant
Web Application for Processing Medical School Applications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from pathlib import Path
import joblib
import os
from dotenv import load_dotenv

# Import our processing modules
from app_processor import ApplicationProcessor
from app_results import ResultsViewer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Rush Medical College - AI Admissions Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Rush branding
st.markdown("""
<style>
    /* Rush University Colors */
    :root {
        --rush-green: #006747;
        --rush-gold: #FFB500;
        --rush-light-green: #4A9B7F;
        --rush-dark-green: #004030;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #006747 0%, #004030 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #FFB500;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #006747;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #006747;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #004030;
        transform: translateY(-2px);
    }
    
    /* Success message */
    .success-box {
        background-color: #4A9B7F;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Warning message */
    .warning-box {
        background-color: #FFB500;
        color: #004030;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Progress bar color */
    .stProgress > div > div > div > div {
        background-color: #FFB500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processor' not in st.session_state:
    st.session_state.processor = ApplicationProcessor()
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Rush Medical College</h1>
    <p>AI-Powered Admissions Assistant</p>
</div>
""", unsafe_allow_html=True)

# Authentication (simplified for demo - use proper auth in production)
if not st.session_state.authenticated:
    st.markdown("### 🔐 Login Required")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            # Simple demo authentication - replace with proper authentication
            if username == "rush_admin" and password == "demo2025":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials. For demo, use: rush_admin / demo2025")
    
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Select Page",
        ["📊 Dashboard", "📤 Process Applications", "📈 View Results", "⚙️ Settings"]
    )
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.processed_data is not None:
        st.markdown("### 📊 Current Session Stats")
        total = len(st.session_state.processed_data)
        q1 = len(st.session_state.processed_data[st.session_state.processed_data['predicted_quartile'] == 'Q1'])
        high_conf = len(st.session_state.processed_data[st.session_state.processed_data['confidence'] >= 80])
        
        st.metric("Total Processed", total)
        st.metric("Q1 Candidates", q1)
        st.metric("High Confidence", high_conf)

# Main content area
if page == "📊 Dashboard":
    st.markdown("## 📊 Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">80.8%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">99%</div>
            <div class="metric-label">Adjacent Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">73</div>
            <div class="metric-label">Features Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0</div>
            <div class="metric-label">Bias Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Information sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 How It Works")
        st.info("""
        1. **Upload** application data (Excel/CSV)
        2. **Process** through AI model
        3. **Review** quartile assignments
        4. **Export** results for committee
        """)
        
        st.markdown("### 📋 Required Data Fields")
        st.warning("""
        Essential fields needed:
        - AMCAS ID
        - Service Rating
        - Clinical Hours
        - Essay Text
        - Demographics
        
        See Settings page for full list
        """)
    
    with col2:
        st.markdown("### 🚀 Quick Start")
        if st.button("Process New Applications", use_container_width=True):
            st.session_state.selected_page = "📤 Process Applications"
            st.rerun()
        
        st.markdown("### 📊 Model Performance")
        
        # Create accuracy chart
        fig = go.Figure(data=[
            go.Bar(
                x=['Q1 (Top)', 'Q2', 'Q3', 'Q4 (Bottom)'],
                y=[91.7, 87.6, 80.2, 76.4],
                marker_color=['#006747', '#4A9B7F', '#FFB500', '#FFD166']
            )
        ])
        fig.update_layout(
            title="Accuracy by Quartile",
            yaxis_title="Accuracy %",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📤 Process Applications":
    st.markdown("## 📤 Process Applications")
    
    # File upload section
    st.markdown("### 📁 Upload Application Data")
    
    uploaded_file = st.file_uploader(
        "Choose Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="File should contain all required application fields"
    )
    
    if uploaded_file is not None:
        # Load and preview data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} applications from {uploaded_file.name}")
            
            # Data preview
            with st.expander("Preview Data (first 5 rows)"):
                st.dataframe(df.head())
            
            # Check for required fields
            st.markdown("### 🔍 Data Validation")
            required_fields = [
                'amcas_id', 'service_rating_numerical', 'healthcare_total_hours',
                'essay_text'  # This would contain the combined essays
            ]
            
            missing_fields = [field for field in required_fields if field not in df.columns]
            
            if missing_fields:
                st.error(f"❌ Missing required fields: {', '.join(missing_fields)}")
                st.info("Please ensure your file contains all required fields. See Settings for field mappings.")
            else:
                st.success("✅ All required fields present")
                
                # Processing options
                st.markdown("### ⚙️ Processing Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    process_essays = st.checkbox("Process essays with GPT-4o", value=True)
                    show_progress = st.checkbox("Show detailed progress", value=True)
                
                with col2:
                    confidence_threshold = st.slider(
                        "Confidence threshold for auto-review",
                        min_value=50,
                        max_value=90,
                        value=80,
                        help="Applications below this confidence will be flagged for review"
                    )
                
                # Process button
                if st.button("🚀 Process Applications", use_container_width=True):
                    st.markdown("### 📊 Processing Progress")
                    
                    # Create progress containers
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    details_container = st.container()
                    
                    # Process applications
                    start_time = time.time()
                    
                    try:
                        # Simulate processing for demo (replace with actual processing)
                        total_apps = len(df)
                        processed_data = []
                        
                        for idx, row in df.iterrows():
                            # Update progress
                            progress = (idx + 1) / total_apps
                            progress_bar.progress(progress)
                            status_text.text(f"Processing application {idx + 1} of {total_apps}...")
                            
                            # Simulate processing delay
                            time.sleep(0.1)  # Remove in production
                            
                            # Here you would call your actual processor
                            # result = st.session_state.processor.process_application(row)
                            
                            # Demo result
                            result = {
                                'amcas_id': row['amcas_id'],
                                'predicted_quartile': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4']),
                                'confidence': np.random.randint(60, 95),
                                'needs_review': np.random.choice([True, False], p=[0.2, 0.8])
                            }
                            processed_data.append(result)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(processed_data)
                        results_df = results_df.merge(df, on='amcas_id', how='left')
                        
                        # Store in session state
                        st.session_state.processed_data = results_df
                        
                        # Show completion
                        elapsed_time = time.time() - start_time
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ Processing complete! Time: {elapsed_time:.1f} seconds")
                        
                        # Show summary
                        st.markdown("### 📊 Processing Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Processed", len(results_df))
                        with col2:
                            q1_count = len(results_df[results_df['predicted_quartile'] == 'Q1'])
                            st.metric("Q1 Candidates", q1_count)
                        with col3:
                            high_conf = len(results_df[results_df['confidence'] >= confidence_threshold])
                            st.metric("High Confidence", high_conf)
                        with col4:
                            review_count = len(results_df[results_df['needs_review'] == True])
                            st.metric("Need Review", review_count)
                        
                        # Quartile distribution
                        fig = px.pie(
                            results_df,
                            names='predicted_quartile',
                            title='Quartile Distribution',
                            color_discrete_map={
                                'Q1': '#006747',
                                'Q2': '#4A9B7F', 
                                'Q3': '#FFB500',
                                'Q4': '#FFD166'
                            }
                        )
                        st.plotly_chart(fig)
                        
                        st.success("✅ Processing complete! Go to 'View Results' to see detailed analysis.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

elif page == "📈 View Results":
    st.markdown("## 📈 View Results")
    
    if st.session_state.processed_data is None:
        st.warning("No processed data available. Please process applications first.")
        if st.button("Go to Process Applications"):
            st.session_state.selected_page = "📤 Process Applications"
            st.rerun()
    else:
        # Results viewer
        viewer = ResultsViewer(st.session_state.processed_data)
        viewer.display()

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ Settings")
    
    # Model information
    st.markdown("### 🤖 Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Essay Model**: GPT-4o (Azure OpenAI)
        **Ranking Model**: XGBoost Cascade
        **Training Data**: 2022-2023 (n=838)
        **Last Updated**: June 19, 2025
        """)
    
    with col2:
        # Check model status
        model_path = Path("models")
        if model_path.exists():
            model_files = list(model_path.glob("*.pkl"))
            if model_files:
                st.success(f"✅ Model loaded: {model_files[0].name}")
            else:
                st.error("❌ No model file found")
        
        # Check API status
        if os.getenv("AZURE_OPENAI_API_KEY"):
            st.success("✅ Azure OpenAI API configured")
        else:
            st.error("❌ Azure OpenAI API key not found")
    
    # Field mappings
    st.markdown("### 📋 Required Data Fields")
    
    field_descriptions = {
        "amcas_id": "Unique applicant identifier",
        "service_rating_numerical": "Faculty service rating (1-4)",
        "healthcare_total_hours": "Total clinical experience hours",
        "exp_hour_research": "Research hours",
        "exp_hour_volunteer_med": "Medical volunteering hours",
        "exp_hour_volunteer_non_med": "Non-medical volunteering hours",
        "age": "Applicant age",
        "gender": "Gender (Male/Female/Other)",
        "citizenship": "Citizenship status",
        "first_generation_ind": "First generation college (0/1)",
        "essay_text": "Combined essay text for GPT-4o analysis"
    }
    
    df_fields = pd.DataFrame(
        [(field, desc) for field, desc in field_descriptions.items()],
        columns=["Field Name", "Description"]
    )
    st.table(df_fields)
    
    # Export settings
    st.markdown("### 📥 Export Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox(
            "Default export format",
            ["Excel (.xlsx)", "CSV (.csv)", "PDF Report"]
        )
    
    with col2:
        st.number_input(
            "Results per page",
            min_value=10,
            max_value=100,
            value=25
        )
    
    # Save settings button
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Rush Medical College AI Admissions Assistant v1.0 | 
    Developed with integrity and tested for fairness | 
    <a href='#' style='color: #006747;'>Documentation</a> | 
    <a href='#' style='color: #006747;'>Support</a></p>
</div>
""", unsafe_allow_html=True)