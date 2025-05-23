import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction & Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load models and scaler functions
@st.cache_resource
def load_models():
    try:
        model_paths = {
            "Random Forest": "models/supervised/random_forest_model.pkl",
            "Logistic Regression": "models/supervised/Logistic_Regression_model.pkl",
            "SVM": "models/supervised/SVM_model.pkl"
        }
        models = {}
        for name, path in model_paths.items():
            if os.path.exists(path):
                models[name] = joblib.load(path)
            else:
                st.warning(f"Model {name} not found at {path}")
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

@st.cache_resource
def load_scaler():
    try:
        if os.path.exists("data/standard_scaler.pkl"):
            return joblib.load("data/standard_scaler.pkl")
        else:
            st.warning("Scaler not found. Using default StandardScaler.")
            return StandardScaler()
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return StandardScaler()

@st.cache_resource
def load_unsupervised_models():
    try:
        unsupervised_models = {}
        unsupervised_paths = {
            "PCA": "models/unsupervised/pca_model.pkl",
            "KMeans": "models/unsupervised/kmeans_model.pkl",
            "DBSCAN": "models/unsupervised/dbscan_model.pkl",
            "Unsupervised Scaler": "models/unsupervised/unsupervised_scaler.pkl"
        }
        
        for name, path in unsupervised_paths.items():
            if os.path.exists(path):
                unsupervised_models[name] = joblib.load(path)
            else:
                st.warning(f"Unsupervised model {name} not found at {path}")
        return unsupervised_models
    except Exception as e:
        st.error(f"Error loading unsupervised models: {str(e)}")
        return {}

# Data preprocessing functions
def preprocess_data(df):
    """Preprocess the data for model prediction"""
    # Handle missing values
    df_processed = df.copy()
    
    # Replace 0 values with NaN for certain columns where 0 is not physiologically possible
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_replace:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace(0, np.nan)
    
    # Fill missing values with median
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

# Visualization functions
def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    fig = plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_distribution_plots(df):
    """Create distribution plots for all features"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=numeric_columns,
        specs=[[{"secondary_y": False}]*3]*3
    )
    
    for i, col in enumerate(numeric_columns):
        row = i // 3 + 1
        col_num = i % 3 + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, nbinsx=30, opacity=0.7),
            row=row, col=col_num
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Feature Distributions")
    return fig

def create_pairplot_plotly(df, target_col='Outcome'):
    """Create pairplot using plotly"""
    if target_col in df.columns:
        fig = px.scatter_matrix(df, dimensions=df.columns[:-1], color=target_col,
                               title="Feature Pairplot", height=800)
    else:
        fig = px.scatter_matrix(df, dimensions=df.columns, 
                               title="Feature Pairplot", height=800)
    return fig

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Prediction & Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Exploration", "🤖 Supervised Learning", 
         "🔍 Unsupervised Learning", "📋 Manual Prediction", "📈 Model Comparison"]
    )
    
    # Load models and scaler
    models = load_models()
    scaler = load_scaler()
    unsupervised_models = load_unsupervised_models()
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Diabetes Prediction Dashboard
        
        This interactive application provides comprehensive analysis and prediction capabilities for diabetes using machine learning techniques.
        
        ### Features:
        - **Data Exploration**: Visualize and understand your dataset
        - **Supervised Learning**: Make predictions using trained models (Random Forest, Logistic Regression, SVM)
        - **Unsupervised Learning**: Discover patterns using clustering and dimensionality reduction
        - **Manual Prediction**: Input patient data for individual predictions
        - **Model Comparison**: Compare performance across different algorithms
        
        ### Dataset Information:
        The diabetes dataset contains medical information about patients with the following features:
        - **Pregnancies**: Number of pregnancies
        - **Glucose**: Glucose concentration
        - **BloodPressure**: Blood pressure measurement
        - **SkinThickness**: Skin thickness
        - **Insulin**: Insulin level
        - **BMI**: Body Mass Index
        - **DiabetesPedigreeFunction**: Genetic risk factor
        - **Age**: Patient age
        - **Outcome**: Target variable (0: No diabetes, 1: Diabetes)
        
        **Get started by uploading your data or using the sample dataset!**
        """)
        
        # Sample data info
        if st.button("Load Sample Dataset"):
            try:
                sample_df = pd.read_csv("data/diabetes.csv")
                st.session_state.df = sample_df
                st.success("Sample dataset loaded successfully!")
                st.write("Dataset shape:", sample_df.shape)
                st.write(sample_df.head())
            except FileNotFoundError:
                st.error("Sample dataset not found. Please upload your own data.")
    
    elif page == "📊 Data Exploration":
        st.markdown('<h2 class="sub-header">Data Exploration</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
        elif 'df' not in st.session_state:
            try:
                df = pd.read_csv("data/diabetes.csv")
                st.session_state.df = df
                st.info("Using default dataset. Upload your own file to analyze different data.")
            except FileNotFoundError:
                st.error("No data available. Please upload a CSV file.")
                return
        
        df = st.session_state.df
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            if 'Outcome' in df.columns:
                diabetes_rate = df['Outcome'].mean() * 100
                st.metric("Diabetes Rate (%)", f"{diabetes_rate:.1f}")
        
        # Data preview
        st.subheader("Data Preview")
        st.write(df.head(10))
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.write(df.describe())
        
        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Correlation Heatmap", "Distributions", "Target Analysis", "Pairplot"])
        
        with tab1:
            st.subheader("Feature Correlation Heatmap")
            fig = create_correlation_heatmap(df)
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Feature Distributions")
            fig = create_distribution_plots(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if 'Outcome' in df.columns:
                st.subheader("Target Variable Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(df, names='Outcome', title='Diabetes Distribution')
                    st.plotly_chart(fig)
                
                with col2:
                    fig = px.box(df, x='Outcome', y='Glucose', title='Glucose by Diabetes Status')
                    st.plotly_chart(fig)
        
        with tab4:
            st.subheader("Feature Relationships")
            fig = create_pairplot_plotly(df)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Supervised Learning":
        st.markdown('<h2 class="sub-header">Supervised Learning Predictions</h2>', unsafe_allow_html=True)
        
        if 'df' not in st.session_state:
            st.warning("Please upload data in the Data Exploration section first.")
            return
        
        df = st.session_state.df
        
        if len(models) == 0:
            st.error("No models found. Please ensure model files are in the correct directory.")
            return
        
        # Preprocess data
        df_processed = preprocess_data(df)
        
        # Prepare features
        if 'Outcome' in df_processed.columns:
            try:
                X = df_processed.drop(columns=['Outcome'])
                y_true = df_processed['Outcome']
                has_target = True
            except KeyError:
                st.warning("'Outcome' column is missing. Proceeding without target variable.")
                X = df_processed.copy()
                has_target = False
        else:
            X = df_processed.copy()
            has_target = False
        
        # Scale features
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.error(f"Error scaling data: {str(e)}")
            return
        
        # Model selection
        selected_models = st.multiselect(
            "Select models for prediction:",
            list(models.keys()),
            default=list(models.keys())
        )
        
        if selected_models:
            # Make predictions
            predictions = {}
            probabilities = {}
            
            for model_name in selected_models:
                model = models[model_name]
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)[:, 1]
                    probabilities[model_name] = prob
            
            # Display predictions
            st.subheader("Prediction Results")
            
            # Create results dataframe
            results_df = df.copy()
            for model_name, pred in predictions.items():
                results_df[f'{model_name}_Prediction'] = pred
                if model_name in probabilities:
                    results_df[f'{model_name}_Probability'] = probabilities[model_name]
            
            st.write(results_df)
            
            # Download predictions
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="diabetes_predictions.csv",
                mime="text/csv"
            )
            
            # Model performance (if target is available)
            if has_target:
                st.subheader("Model Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Accuracy scores
                    st.write("**Accuracy Scores:**")
                    for model_name, pred in predictions.items():
                        accuracy = accuracy_score(y_true, pred)
                        st.metric(f"{model_name} Accuracy", f"{accuracy:.3f}")
                
                with col2:
                    # Confusion matrices
                    st.write("**Confusion Matrices:**")
                    for model_name, pred in predictions.items():
                        cm = confusion_matrix(y_true, pred)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title(f'{model_name} Confusion Matrix')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                
                # ROC Curves
                if probabilities:
                    st.subheader("ROC Curves")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    for model_name, prob in probabilities.items():
                        fpr, tpr, _ = roc_curve(y_true, prob)
                        auc_score = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
                    
                    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curves Comparison')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
    
    elif page == "🔍 Unsupervised Learning":
        st.markdown('<h2 class="sub-header">Unsupervised Learning Analysis</h2>', unsafe_allow_html=True)
        
        if 'df' not in st.session_state:
            st.warning("Please upload data in the Data Exploration section first.")
            return
        
        df = st.session_state.df
        
        # Prepare data for clustering
        if 'Outcome' in df.columns:
            X = df.drop('Outcome', axis=1)
        else:
            X = df.copy()
        
        # PCA Analysis
        st.subheader("Principal Component Analysis (PCA)")
        
        if 'PCA' in unsupervised_models:
            pca_model = unsupervised_models['PCA']
            if 'Unsupervised Scaler' in unsupervised_models:
                scaler_unsup = unsupervised_models['Unsupervised Scaler']
                X_scaled = scaler_unsup.transform(X)
            else:
                scaler_unsup = StandardScaler()
                X_scaled = scaler_unsup.fit_transform(X)
            
            X_pca = pca_model.transform(X_scaled)
            
            # PCA visualization
            fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], 
                           title='PCA - First Two Components',
                           labels={'x': 'First Principal Component', 'y': 'Second Principal Component'})
            
            if 'Outcome' in df.columns:
                fig.update_traces(marker=dict(color=df['Outcome'], colorscale='Viridis', showscale=True))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explained variance
            if hasattr(pca_model, 'explained_variance_ratio_'):
                st.write(f"Explained variance by first component: {pca_model.explained_variance_ratio_[0]:.3f}")
                st.write(f"Explained variance by second component: {pca_model.explained_variance_ratio_[1]:.3f}")
                st.write(f"Total explained variance: {sum(pca_model.explained_variance_ratio_[:2]):.3f}")
        
        # Clustering Analysis
        st.subheader("Clustering Analysis")
        
        clustering_models = ['KMeans', 'DBSCAN']
        available_clustering = [model for model in clustering_models if model in unsupervised_models]
        
        if available_clustering:
            selected_clustering = st.selectbox("Select clustering algorithm:", available_clustering)
            
            if selected_clustering in unsupervised_models:
                clustering_model = unsupervised_models[selected_clustering]
                
                # Apply clustering
                if 'Unsupervised Scaler' in unsupervised_models:
                    scaler_unsup = unsupervised_models['Unsupervised Scaler']
                    X_scaled = scaler_unsup.transform(X)
                else:
                    scaler_unsup = StandardScaler()
                    X_scaled = scaler_unsup.fit_transform(X)
                
                clusters = clustering_model.fit_predict(X_scaled)
                
                # Visualize clusters in PCA space
                if 'PCA' in unsupervised_models:
                    pca_model = unsupervised_models['PCA']
                    X_pca = pca_model.transform(X_scaled)
                    
                    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=clusters,
                                   title=f'{selected_clustering} Clustering Results in PCA Space',
                                   labels={'x': 'First Principal Component', 'y': 'Second Principal Component'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster statistics
                unique_clusters = np.unique(clusters)
                st.write(f"Number of clusters found: {len(unique_clusters)}")
                
                cluster_df = df.copy()
                cluster_df['Cluster'] = clusters
                
                # Cluster summary
                st.subheader("Cluster Summary")
                for cluster in unique_clusters:
                    if cluster != -1:  # Ignore noise points in DBSCAN
                        cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
                        st.write(f"**Cluster {cluster}**: {len(cluster_data)} samples")
                        if 'Outcome' in df.columns:
                            diabetes_rate = cluster_data['Outcome'].mean() * 100
                            st.write(f"Diabetes rate in cluster {cluster}: {diabetes_rate:.1f}%")
                
                # Download clustered data
                csv = cluster_df.to_csv(index=False)
                st.download_button(
                    label="Download Clustered Data as CSV",
                    data=csv,
                    file_name="clustered_data.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No clustering models found. Please ensure model files are in the correct directory.")
    
    elif page == "📋 Manual Prediction":
        st.markdown('<h2 class="sub-header">Manual Patient Data Prediction</h2>', unsafe_allow_html=True)
        
        if len(models) == 0:
            st.error("No models found. Please ensure model files are in the correct directory.")
            return
        
        st.write("Enter patient information to get a diabetes prediction:")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
        
        if st.button("Predict Diabetes Risk", type="primary"):
            # Create input array
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                  insulin, bmi, diabetes_pedigree, age]])
            
            # Scale input
            try:
                input_scaled = scaler.transform(input_data)
            except Exception as e:
                st.error(f"Error scaling input data: {str(e)}")
                return
            
            # Make predictions with all models
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            for i, (model_name, model) in enumerate(models.items()):
                prediction = model.predict(input_scaled)[0]
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_scaled)[0]
                    risk_probability = probability[1] * 100
                else:
                    risk_probability = "N/A"
                
                result = "High Risk 🔴" if prediction == 1 else "Low Risk 🟢"
                
                if i == 0:
                    with col1:
                        st.metric(f"{model_name}", result)
                        if risk_probability != "N/A":
                            st.write(f"Risk Probability: {risk_probability:.1f}%")
                elif i == 1:
                    with col2:
                        st.metric(f"{model_name}", result)
                        if risk_probability != "N/A":
                            st.write(f"Risk Probability: {risk_probability:.1f}%")
                else:
                    with col3:
                        st.metric(f"{model_name}", result)
                        if risk_probability != "N/A":
                            st.write(f"Risk Probability: {risk_probability:.1f}%")
            
            # Additional insights
            st.subheader("Health Insights")
            
            insights = []
            if glucose > 140:
                insights.append("🔸 Glucose level is elevated (>140 mg/dL)")
            if bmi > 30:
                insights.append("🔸 BMI indicates obesity (>30)")
            if blood_pressure > 90:
                insights.append("🔸 High blood pressure detected (>90 mmHg)")
            if age > 45:
                insights.append("🔸 Age is a risk factor (>45 years)")
            
            if insights:
                for insight in insights:
                    st.write(insight)
            else:
                st.write("🟢 No major risk factors detected in the input values.")
    
    elif page == "📈 Model Comparison":
        st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        if 'df' not in st.session_state:
            st.warning("Please upload data with target labels in the Data Exploration section first.")
            return
        
        df = st.session_state.df
        
        if 'Outcome' not in df.columns:
            st.warning("Target column 'Outcome' not found. Model comparison requires labeled data.")
            return
        
        if len(models) == 0:
            st.error("No models found. Please ensure model files are in the correct directory.")
            return
        
        # Prepare data
        df_processed = preprocess_data(df)
        X = df_processed.drop('Outcome', axis=1)
        y = df_processed['Outcome']
        
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.error(f"Error scaling data: {str(e)}")
            return
        
        # Calculate metrics for all models
        metrics_data = []
        
        for model_name, model in models.items():
            pred = model.predict(X_scaled)
            accuracy = accuracy_score(y, pred)
            
            # Classification report
            report = classification_report(y, pred, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']
            
            metrics_data.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score
            })
        
        # Display metrics table
        metrics_df = pd.DataFrame(metrics_data)
        st.subheader("Model Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualize metrics comparison
        fig = px.bar(metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                    x='Model', y='Score', color='Metric', barmode='group',
                    title='Model Performance Comparison')
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model recommendation
        best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
        best_accuracy = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Accuracy']
        
        st.success(f"🏆 Best performing model: **{best_model}** with accuracy of {best_accuracy:.3f}")

if __name__ == "__main__":
    main()
