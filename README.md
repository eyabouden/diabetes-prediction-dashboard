# Diabetes Prediction & Analysis Dashboard

An interactive web application for diabetes prediction and analysis using machine learning techniques.

## Features

- **Data Exploration**: Visualize and understand your dataset
- **Supervised Learning**: Make predictions using trained models (Random Forest, Logistic Regression, SVM)
- **Unsupervised Learning**: Discover patterns using clustering and dimensionality reduction
- **Manual Prediction**: Input patient data for individual predictions
- **Model Comparison**: Compare performance across different algorithms

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction-dashboard.git
cd diabetes-prediction-dashboard
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Project Structure

```
diabetes-prediction-dashboard/
├── data/
│   ├── diabetes.csv
│   └── standard_scaler.pkl
├── models/
│   ├── supervised/
│   │   ├── random_forest_model.pkl
│   │   ├── Logistic_Regression_model.pkl
│   │   └── SVM_model.pkl
│   └── unsupervised/
│       ├── pca_model.pkl
│       ├── kmeans_model.pkl
│       ├── dbscan_model.pkl
│       └── unsupervised_scaler.pkl
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Dataset Information

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

## Contributing

Feel free to submit issues and enhancement requests! 