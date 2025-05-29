import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Fraud Detection", layout="wide")

# Title and description
st.title("Credit Card Fraud Detection")
st.markdown("""
This application helps detect fraudulent credit card transactions using multiple machine learning models.
Upload your dataset and get predictions from four different models:
- Random Forest
- Decision Tree
- AdaBoost
- Gradient Boosting
""")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Dataset Overview Section
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            st.metric("Number of Features", len(df.columns))
        with col3:
            if 'Class' in df.columns:
                fraud_count = df['Class'].sum()
                st.metric("Fraudulent Transactions", fraud_count)
            else:
                st.metric("Fraudulent Transactions", "Unknown")
        with col4:
            if 'Class' in df.columns:
                legitimate_count = len(df) - df['Class'].sum()
                st.metric("Legitimate Transactions", legitimate_count)
            else:
                st.metric("Legitimate Transactions", "Unknown")
        
        # Data visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Display first few rows
            st.subheader("Sample Data")
            st.dataframe(df.head(10))
        
        with col2:
            # Class distribution (if Class column exists)
            if 'Class' in df.columns:
                st.subheader("Class Distribution")
                class_counts = df['Class'].value_counts()
                fig_class = px.pie(
                    values=class_counts.values,
                    names=['Legitimate', 'Fraudulent'],
                    title='Transaction Distribution in Dataset'
                )
                st.plotly_chart(fig_class, use_container_width=True)
            else:
                st.subheader("Amount Distribution")
                if 'Amount' in df.columns:
                    fig_amount = px.histogram(
                        df, 
                        x='Amount', 
                        title='Transaction Amount Distribution',
                        nbins=50
                    )
                    st.plotly_chart(fig_amount, use_container_width=True)
        

        
        st.divider()
        
        # Define the required columns in the correct order
        required_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                          'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                          'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        
        # Check if all required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            # Load models and scaler
            try:
                rf_model = joblib.load('random_forest_model.joblib')
                dt_model = joblib.load('decision_tree_model.joblib')
                ab_model = joblib.load('ada_boost_model.joblib')
                gb_model = joblib.load('gradient_boosting_model.joblib')
                scaler = joblib.load('scaler.joblib')
                
                st.header("🤖 Model Predictions")
                
                # Prepare data for prediction - ensure columns are in correct order
                X = df[required_columns].copy()  # This ensures columns are in the correct order
                X['Amount'] = scaler.transform(X['Amount'].values.reshape(-1,1))
                
                if st.button("Make Predictions", type="primary"):
                    with st.spinner("Making predictions with all models..."):
                        # Make predictions with all models
                        predictions = {
                            'RF_Prediction': rf_model.predict(X),
                            'DT_Prediction': dt_model.predict(X),
                            'AB_Prediction': ab_model.predict(X),
                            'GB_Prediction': gb_model.predict(X)
                        }
                        
                        # Create results DataFrame
                        results_df = df.copy()
                        
                        # Add predictions to the dataframe
                        for model_name, preds in predictions.items():
                            results_df[model_name] = preds
                        
                        # Add Transaction ID
                        results_df.insert(0, 'Transaction_ID', range(1, len(df) + 1))
                        
                        # Display results for each model
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Prediction Summary")
                            summary_data = []
                            for model_name, preds in predictions.items():
                                frauds = np.sum(preds == 1)
                                legitimate = np.sum(preds == 0)
                                fraud_rate = (frauds / len(preds)) * 100
                                summary_data.append({
                                    'Model': model_name.replace('_Prediction', ''),
                                    'Fraudulent': frauds,
                                    'Legitimate': legitimate,
                                    'Fraud Rate (%)': f"{fraud_rate:.2f}%"
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                        
                        with col2:
                            # Create pie chart for Random Forest
                            rf_frauds = np.sum(predictions['RF_Prediction'] == 1)
                            rf_legitimate = np.sum(predictions['RF_Prediction'] == 0)
                            fig = px.pie(
                                values=[rf_legitimate, rf_frauds],
                                names=['Legitimate', 'Fraudulent'],
                                title='Transaction Distribution (Random Forest)',
                                color_discrete_map={'Legitimate': '#2E8B57', 'Fraudulent': '#DC143C'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Model agreement analysis
                        st.subheader("Model Agreement Analysis")
                        
                        # Calculate agreement between models
                        pred_df = pd.DataFrame(predictions)
                        agreement_sum = pred_df.sum(axis=1)
                        
                        agreement_counts = {
                            'All Models Agree (Legitimate)': np.sum(agreement_sum == 0),
                            'All Models Agree (Fraudulent)': np.sum(agreement_sum == 4),
                            '3 Models Agree': np.sum((agreement_sum == 1) | (agreement_sum == 3)),
                            '2 Models Split': np.sum(agreement_sum == 2)
                        }
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            for desc, count in agreement_counts.items():
                                st.metric(desc, count)
                        
                        with col2:
                            fig_agreement = px.bar(
                                x=list(agreement_counts.keys()),
                                y=list(agreement_counts.values()),
                                title='Model Agreement Distribution',
                                labels={'x': 'Agreement Type', 'y': 'Number of Transactions'}
                            )
                            st.plotly_chart(fig_agreement, use_container_width=True)
                        
                        # Display sample predictions
                        st.subheader("Sample Predictions")
                        st.dataframe(results_df.head(10), use_container_width=True)
                        
                        # Download buttons
                        st.header("📥 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Save results to CSV
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="📄 Download Predictions CSV File",
                                data=csv_data,
                                file_name='fraud_predictions.csv',
                                mime='text/csv',
                                type="primary"
                            )
                        
                        with col2:
                            # Save results to Excel
                            excel_file = 'prediction_results.xlsx'
                            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                                results_df.to_excel(writer, sheet_name='Predictions', index=False)
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            # Provide download button for Excel
                            with open(excel_file, 'rb') as f:
                                st.download_button(
                                    label="📊 Download Predictions Excel File",
                                    data=f,
                                    file_name=excel_file,
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                )
                    
            except FileNotFoundError as e:
                st.error("Model files not found. Please run save_models.py first to train and save the models.")
                st.info("Required files: random_forest_model.joblib, decision_tree_model.joblib, ada_boost_model.joblib, gradient_boosting_model.joblib, scaler.joblib")
                
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
        st.info("Please make sure your CSV file is properly formatted and contains the required columns.")

else:
    st.info("👆 Please upload a CSV file to begin fraud detection analysis.")
    st.markdown("""
    ### Expected CSV Format:
    Your CSV file should contain the following columns:
    - V1 through V28: Principal components from PCA transformation
    - Amount: Transaction amount
    - Class (optional): Actual labels for visualization (0 for legitimate, 1 for fraudulent)
    """)