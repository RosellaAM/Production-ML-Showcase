# Imports
import joblib
import requests
import tempfile
import os
import pandas as pd
import plotly_express as px
import streamlit as st
from streamlit_navigation_bar import st_navbar

st.set_page_config(initial_sidebar_state="collapsed")


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


# Navegation bar
pages = ['Home', 'Upload Data', 'Batch Predictions', 'Results', 'GitHub']
parent_dir = os.path.dirname(os.path.abspath(__file__))
urls = {'GitHub': 'https://github.com/RosellaAM'}

styles = {
    "nav": {
        "background-color": "maroon",
        "justify-content": "left",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "white",
        "padding": "14px",
    },
    "active": {
        "background-color": "white",
        "color": "var(--text-color)",
        "font-weight": "normal",
        "padding": "14px",
    }
}
options = {
    "show_menu": False,
    "show_sidebar": False,
}

page = st_navbar(
    pages,
    urls=urls,
    styles=styles,
    options=options,
)

# Setting up pages functions
# Home
def show_home():
    st.title('Megaline Plan Recommendation Engine')
    st.divider()
    st.header('Introduction')
    st.write(
        """**Drive Business Growth with AI-Powered Insights**
        Welcome to our intelligent plan recommendation system! This tool leverages machine learning 
        to analyze customer usage patterns and recommend optimal mobile plans, helping you reduce churn 
        and increase customer satisfaction through data-driven decisions.
        **How it works in 3 simple steps:**
        1. 📊 **Upload** your customer usage data or select one of the ones provided (CSV format).
        2. ⚡ **Analyze** with our AI model that has learned from thousands of patterns.
        3. 💡 **Optimize** with instant Smart vs Ultra plan recommendations.
        **Why it works:**
        Our optimized Random Forest model achieves **88% accuracy** by analyzing calls, minutes, messages, 
        and data usage to identify when customers need plan upgrades—ensuring perfect fit without overpayment.
        *Outperforms baseline models by over 30% accuracy*
        **Start by uploading your data or testing with individual customers below!**"""
        )


def show_upload_data():
    st.header('Upload Data')
    st.write('Choose from our pre-loaded sample datasets or upload your own customer data to get started with plan recommendations.')
    st.caption('⚠️ Your dataset must include: calls, minutes, messages, mb_used. Missing columns will prevent predictions.')
    tab1, tab2, tab3, tab4 = st.tabs(['January Megaline Dataset', 'February Megaline Dataset', 'March Megaline Dataset', 'Upload Your Own'])

    with tab1:
        st.subheader('January customer usage data')
        st.dataframe(january_df)
        if st.button('Use January Data'):
            use_january_data()
            st.success('January dataset selected!')
    
    with tab2:
        st.subheader('February customer usage data')
        st.dataframe(february_df)
        if st.button('Use February Data'):
            use_february_data()
            st.success('February dataset selected!')
    
    with tab3:
        st.subheader('March customer usage data')
        st.dataframe(march_df)
        if st.button('Use March Data'):
            use_march_data()
            st.success('March dataset selected!')
    
    with tab4:
        st.subheader('Upload your cutumer usage data')
        uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
        if uploaded_file is not None:
            try:
                user_data = pd.read_csv(uploaded_file)
                st.write('Data preview')
                st.dataframe(user_data)

                # Validates columns
                required_columns = ['calls', 'minutes', 'messages', 'mb_used']
                missing_columns = [col for col in required_columns if col not in user_data.columns]

                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.info("Your dataset must include: calls, minutes, messages, mb_used")
                else:
                    st.success('All required columns present!')
    
                if st.button('Use Uploaded Data'):
                    st.session_state['selected_dataset'] = user_data
                    st.session_state['dataset_name'] = 'Your Uploaded Data'
                    st.success("Uploaded dataset selected!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

functions = {
    "Home": show_home,
    "Upload Data": show_upload_data,
    "Batch Predictions": show_batch_predictions,
    "Results": show_results,
}

go_to = functions.get(page)
if go_to:
    go_to()

# Interpret the result: 0 = Smart plan, 1 = Ultra plan
