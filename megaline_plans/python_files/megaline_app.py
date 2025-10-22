# Imports
import joblib
import os
import pandas as pd
import plotly_express as px
import requests
import streamlit as st
import tempfile
import time

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


# Setting up sidebar
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

st.sidebar.title("Navigation")

if st.sidebar.button("Home", use_container_width=True):
    st.session_state['page'] = 'Home'
if st.sidebar.button("Upload Data", use_container_width=True):
    st.session_state['page'] = 'Upload Data'
if st.sidebar.button("Batch Prediction", use_container_width=True):
    st.session_state['page'] = 'Batch Predictions'
if st.sidebar.button("Results", use_container_width=True):
    st.session_state['page'] = 'Results'

# Setting up pages functions
# Home
if st.session_state['page'] == 'Home':
    st.title('Megaline Plan Recommendation Engine')
    st.divider()
    st.header('Introduction')
    st.write('**Drive Business Growth with AI-Powered Insights**')
    st.write(
        """Welcome to our intelligent plan recommendation system! This tool leverages machine learning 
        to analyze customer usage patterns and recommend optimal mobile plans, helping you reduce churn 
        and increase customer satisfaction through data-driven decisions."""
        )
    st.write('**How it works in 3 simple steps:**')
    st.write('1. 📊 **Upload** your customer usage data or select one of the ones provided (CSV format).')
    st.write('2. ⚡ **Analyze** with our AI model that has learned from thousands of patterns.')
    st.write('3. 💡 **Optimize** with instant Smart vs Ultra plan recommendations.')
    st.write('**Why it works:**')
    st.write(
        """Our optimized Random Forest model achieves **88% accuracy** by analyzing calls, minutes, messages, 
        and data usage to identify when customers need plan upgrades—ensuring perfect fit without overpayment."""
        )
    st.write('*Outperforms baseline models by over 30% accuracy*')
    st.write('**Start by uploading your data or testing with individual customers!**')

# Upload Data
elif st.session_state['page'] == 'Upload Data':
    st.header('Upload Data')
    st.write('Choose from our pre-loaded sample datasets or upload your own customer data to get started with plan recommendations.')
    st.caption('*Note: Once you select or upload your data, you will be redirected to the predictions page*')
    st.caption('⚠️ Your dataset must include: calls, minutes, messages, mb_used. Missing columns will prevent predictions.')
    tab1, tab2, tab3, tab4 = st.tabs(['January Megaline Dataset', 'February Megaline Dataset', 'March Megaline Dataset', 'Upload Your Own'])

    with tab1:
        st.subheader('January customer usage data')
        st.dataframe(january_df.head())
        if st.button('Use January Data'):
            use_january_data()
            st.success('January dataset selected!')
            st.info("Redirecting to Batch Predictions page...")
            time.sleep(3)
            st.session_state['page'] = 'Batch Predictions'
            st.rerun()
    
    with tab2:
        st.subheader('February customer usage data')
        st.dataframe(february_df.head())
        if st.button('Use February Data'):
            use_february_data()
            st.success('February dataset selected!')
            st.info("Redirecting to Batch Predictions page...")
            time.sleep(3)
            st.session_state['page'] = 'Batch Predictions'
            st.rerun()
    
    with tab3:
        st.subheader('March customer usage data')
        st.dataframe(march_df.head())
        if st.button('Use March Data'):
            use_march_data()
            st.success('March dataset selected!')
            st.info("Redirecting to Batch Predictions page...")
            time.sleep(3)
            st.session_state['page'] = 'Batch Predictions'
            st.rerun()
    
    with tab4:
        st.subheader('Upload your cutumer usage data')
        uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
        if uploaded_file is not None:
            try:
                user_data = pd.read_csv(uploaded_file)
                st.write('Data preview')
                st.dataframe(user_data.head())

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
                    st.info("Redirecting to Batch Predictions page...")
                    time.sleep(3)
                    st.session_state['page'] = 'Batch Predictions'
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")

# Batch Predictions
elif st.session_state['page'] == 'Batch Predictions':
    st.header('Batch Predictions')
    st.write('**Choose your analysis approach:**')
    st.write('🔍 **Full Analysis** - Process all customers at once for comprehensive insights.')  
    st.write('🎯 **Targeted Range** - Focus on specific customer segments.')
    st.write('👤 **Individual Assessment** - Get detailed recommendations for a single customer.')
    st.write('**How it works:**')
    st.write('1. Select your preferred analysis method below.')
    st.write('2. Review the data preview.')
    st.write('3. Generate predictions.')
    st.write('4. View interpreted results on the next page.')
    
    st.divider()
    
    # Verify a dataset is selected
    if st.session_state['selected_dataset'] is None:
        st.warning('Please go to **Upload Data** and select a dataset first')
    else:
        st.success(f"Using: {st.session_state['dataset_name']}")
        st.dataframe(st.session_state['selected_dataset'])

    st.divider()
    
    # Initializing session state for analysis methods
    if 'analysis_mode' not in st.session_state:
        st.session_state['analysis_mode'] = None
    if 'selected_range' not in st.session_state:
        st.session_state['selected_range'] = [0, 0]
    if 'selected_clients' not in st.session_state:
        st.session_state['selected_clients'] = []
    if 'data_subset' not in st.session_state:
        st.session_state['data_subset'] = None

    # Setting the diffrent methods
    st.subheader('Analysis Methods')
    methods = st.selectbox( 
        'Choose your prediction method',
        ('Full dataset', 'Targeted customer range', 'Individual client'),
                accept_new_options=False,
        placeholder='select...'
        )

    # Full dataset
    if methods == 'Full dataset':
        st.session_state['analysis_mode'] = 'full'
        st.session_state['data_subset'] = st.session_state['selected_dataset']
        st.success('Full dataset selected!')

    # Targeted range
    if methods == 'Targeted customer range':
        dataset_length = len(st.session_state['selected_dataset'])
        range_selection = st.slider(
            "Select customer range", 
            min_value=0, 
            max_value=dataset_length-1,
            value=[0, min(49, dataset_length-1)]

            ) 
        st.session_state['analysis_mode'] = 'range'
        st.session_state['selected_range'] = range_selection
        st.session_state['data_subset'] = st.session_state['selected_dataset'].iloc[range_selection[0]:range_selection[1]]
        st.success(f'Selected customers {range_selection[0]} to {range_selection[1]}!')
    
    if methods == 'Individual client':
        client_options = st.session_state['selected_dataset'].index.tolist()
        select_clients = st.multiselect(
            'Select specific client/s',
            options=client_options,
            default=[client_options[0]] if client_options else []
        )
        st.session_state['analysis_mode'] = 'clients'
        st.session_state['selected_clients'] = select_clients
        st.session_state['data_subset'] = st.session_state['selected_dataset'].iloc[select_clients]
        st.success(f'{len(select_clients)} clients selected!')
    
    st.divider()
    if st.session_state['analysis_mode']:
        st.subheader("📊 Data Preview")
        st.write(f"Analysis Mode: {st.session_state['analysis_mode']}")
        st.write(f"Rows to analyze: {len(st.session_state['data_subset'])}")
        st.dataframe(st.session_state['data_subset'])

    # Making prediction with selected method
    def make_predictions(data_subset):
        # Extracting columns
        features = data_subset[['calls', 'minutes', 'messages', 'mb_used']]
        # Scaling data
        features_scaled = scaler.transform(features)
        # Makes pedictions
        predictions = model.predict(features_scaled)
        return predictions

    if st.session_state['analysis_mode'] and st.session_state['data_subset'] is not None:
        if st.button('🚀 Creat Prediction'):
            with st.spinner('Generating predictions...'):
                predictions = make_predictions(st.session_state['data_subset'])
                st.session_state['prediction_results'] = predictions
                st.session_state['show_results'] = True

                st.success("Predictions generated successfully!")
                st.info("Redirecting to Results page...")

                time.sleep(4)

                st.session_state['page'] = 'Results'
                st.rerun()

# Results
elif st.session_state['page'] == 'Results':
    st.header('Prediction Results')
    # Verifying pred results
    if 'prediction_results' not in st.session_state or st.session_state['prediction_results'] is None:
        st.warning("No prediction results found. Please generate predictions first.")
        if st.button("Go to Batch Predictions"):
            st.session_state['page'] = 'Batch Predictions'
            st.rerun()
    else:
        predictions = st.session_state['prediction_results']
        data_subset = st.session_state['data_subset']
    
    st.subheader('📈 Summary')
    ultra_count = sum(predictions == 1)
    smart_count = sum(predictions == 0)
    st.metric('Ultra Plan Recommendations', ultra_count)
    st.write('Ideal for moderate usage patterns with balanced calls, messages, and data.')
    st.metric('Smart Plan Recommendations', smart_count)
    st.write('Recommended for high-usage customers needing more data, minutes, or messaging capacity')
    st.divider()

    # Interpretation functions
    def interpret_prediction(prediction, client_data=None):
        """
        Convert 0/1 predictions to business-friendly explanations
        """
        if prediction == 0:
            return "Smart Plan", "Ideal for moderate usage patterns with balanced calls, messages, and data."
        else:
            return "Ultra Plan", "Recommended for high-usage customers needing more data, minutes, or messaging capacity."
    
    def generate_client_insights(client_data, prediction):
        """
        Generate specific insights based on client usage patterns
        """
        plan_type, explanation = interpret_prediction(prediction)
        
        insights = []
        if client_data['minutes'] > 500:
            insights.append(f"High minutes usage ({client_data['minutes']} min)")
        if client_data['mb_used'] > 15000:
            insights.append(f"High data consumption ({client_data['mb_used']} MB)")
        if client_data['calls'] > 100:
            insights.append(f"Frequent caller ({client_data['calls']} calls)")
        
        return plan_type, explanation, insights

    # Showing results
    st.subheader('Detailed Insights')
    for i, (client_idx, prediction) in enumerate(zip(data_subset.index, predictions)) :
        client_data = data_subset.iloc(client_idx)
        plan_type, explenation, insights = generate_client_insights(client_data, predictions)

        # Data for each client
        with st.expander(f"Client {client_idx} - {plan_type}"):
            st.write(f'**Recommendation**: {plan_type}')
            st.write(f'{explenation}')
            if insights:
                    st.write("**Key Usage Patterns:**")
                    for insight in insights:
                        st.write(f"- {insight}")
            
            # Clients actual data
            st.write('Usage Summary')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Calls", client_data['calls'])
            with col2:
                st.metric("Minutes", client_data['minutes'])
            with col3:
                st.metric("Messages", client_data['messages'])
            with col4:
                st.metric("Data Used", f'{client_data['mb_used']} MB')                     
    st.divider()

    # Download results option
    results_df = data_subset.copy(True)
    results_df['Recommended_Plan'] = ['Smart' if p == 0 else 'Ultra' for p in predictions]
    results_csv = results_df.to_csv(index=True)
    st.download_button(
        label='Download Results as CSV',
        data=results_csv,
        file_name='megaline_plan_recommendations.csv',
        mime='text/csv'
        )


