# Imports
import joblib
import requests
import tempfile
import os
import pandas as pd
import plotly_express as px
import streamlit as st

# Initialize session state for dataset storage
if 'selected_dataset' not in st.session_state:
    st.session_state['selected_dataset'] = None
if 'dataset_name' not in st.session_state:
    st.session_state['dataset_name'] = "No dataset selected"

# Model loading
@st.cache_resource
def load_assets():
    try: 
        model_url = 'https://raw.githubusercontent.com/RosellaAM/Production-ML-Showcase/main/megaline_plans/joblib_files/megaline_model.joblib'
        scaler_url = 'https://raw.githubusercontent.com/RosellaAM/Production-ML-Showcase/main/megaline_plans/joblib_files/megaline_scaler.joblib'
        
        model_response = requests.get(model_url)
        scaler_response = requests.get(scaler_url)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as model_file:
            model_file.write(model_response.content)
            model_path = model_file.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as scaler_file:
            scaler_file.write(scaler_response.content)
            scaler_path = scaler_file.name

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Clean up temporary files
        os.unlink(model_path)
        os.unlink(scaler_path)

        return model, scaler, True
    except Exception as e:
        st.error(f"Error details: {e}")
        return None, None, False
    
model, scaler, success = load_assets()

if success:
    st.success('Model deployed successfully!')
else:
    st.error('Error loading model assets')

# Sample datasets
january_df = pd.read_csv('https://raw.githubusercontent.com/RosellaAM/Production-ML-Showcase/main/megaline_plans/datasets/megaline_january.csv')
february_df = pd.read_csv('https://raw.githubusercontent.com/RosellaAM/Production-ML-Showcase/main/megaline_plans/datasets/megaline_february.csv')
march_df = pd.read_csv('https://raw.githubusercontent.com/RosellaAM/Production-ML-Showcase/main/megaline_plans/datasets/megaline_march.csv')

# Functions to handle the datasets
def use_january_data():
    st.session_state['selected_dataset'] = january_df
    st.session_state['dataset_name'] = 'January Data'

def use_february_data():
    st.session_state['selected_dataset'] = february_df
    st.session_state['dataset_name'] = 'February Data'

def use_march_data():
    st.session_state['selected_dataset'] = march_df
    st.session_state['dataset_name'] = 'March Data'

# Title
st.title('Megaline Plan Recommendation Engine')
st.divider()

# Sidebar navegation
with st.sidebar:
    st.title('Navigation')
    page = st.radio('Go to:', [
        '🏠 Introduction',
        '📁 Upload Data', 
        '🤖 Batch Predictions', 
        '👤 Single Client',
        '📊 Results'
    ])

# Page content based on selection
if 'Introduction' in page:
    st.header('Introduction')
    st.write('**Drive Business Growth with AI-Powered Insights**')
    st.write('Welcome to our intelligent plan recommendation system! This tool leverages machine learning to analyze customer usage patterns and recommend optimal mobile plans—helping you reduce churn and increase customer satisfaction through data-driven decisions.')
    st.write('**How it works in 3 simple steps:**')
    st.write('1. 📊 **Upload** your customer usage data or select one of the ones provided (CSV format).')
    st.write('2. ⚡ **Analyze** with our AI model that has learned from thousands of patterns.')
    st.write('3. 💡 **Optimize** with instant Smart vs Ultra plan recommendations.')
    st.write('**Why it works:**')
    st.write('Our optimized Random Forest model achieves **88% accuracy** by analyzing calls, minutes, messages, and data usage to identify when customers need plan upgrades—ensuring perfect fit without overpayment.')
    st.write('*Outperforms baseline models by over 30% accuracy*')
    st.write('**Start by uploading your data or testing with individual customers below!**')
    st.divider()

elif 'Upload Data' in page:
    st.header('Upload Data')
    st.write('Choose from our pre-loaded sample datasets or upload your own customer data to get started with plan recommendations.')
    st.caption("⚠️ Your dataset must include: calls, minutes, messages, mb_used. Missing columns will prevent predictions.")
    tab1, tab2, tab3, tab4 = st.tabs(['January Megaline Dataset', 'February Megaline Dataset', 'March Megaline Dataset', 'Upload Your Own'])

    with tab1:
        st.subheader('January customer usage data')
        st.dataframe(january_df)
        if st.button('Use January Data'):
            use_january_data()
            st.success('January dataset selected!')

# Interpret the result: 0 = Smart plan, 1 = Ultra plan
