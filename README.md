# ❤️ Heart Disease Prediction Project

This project uses machine learning to predict heart disease risk with high accuracy. 🎯 It includes a complete data science pipeline, from data preprocessing and model training to an interactive web application for real-time predictions. The project evaluates several models, including **Logistic Regression, SVM, Decision Trees, Random Forest, and XGBoost**, with the final, optimized XGBoost model deployed in a **Streamlit** web app.

## ✨ Features

- **🌐 Interactive Web Application**: A user-friendly web app built with Streamlit to make predictions easily accessible
- **🔄 End-to-End Machine Learning Pipeline**: Covers the entire workflow from data cleaning and preprocessing to model training and evaluation
- **📊 Comprehensive Model Comparison**: Evaluates various machine learning models, including:
  - **🔍 Supervised Learning**: Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forest, and XGBoost
  - **🔍 Unsupervised Learning**: K-Means clustering to discover hidden patterns in the data
- **⚡ Optimized Model**: The final prediction model (XGBoost) is fine-tuned using hyperparameter optimization for the best possible performance
- **📈 Real-time Predictions**: Get instant heart disease risk assessments with confidence scores
- **🎨 Beautiful UI**: Modern, responsive interface with helpful tooltips and visual feedback

## 🚀 Getting Started

Run the web app online:

https://heart-disease-prediction-app-ml.streamlit.app

Or set up locally:

### 📋 Prerequisites

- Python 3.7 or higher 🐍
- pip package manager 📦

### 🛠️ Installation

1. **📥 Clone the repository:**
   ```bash
   git clone https://github.com/n0urhatem/Heart-Disease-Prediction.git
   cd heart-disease-prediction
   ```

2. **📦 Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **✅ Verify installation:**
   ```bash
   python --version
   streamlit --version
   ```

## 🎮 Usage

You can execute the following command in your terminal to run it locally:

```bash
streamlit run app.py
```

This will open the application in your web browser 🌐, where you can:
- 📝 Input patient data through an intuitive interface
- 🔮 Get instant heart disease risk predictions
- 📊 View confidence scores and recommendations
- 💡 Access helpful information about each medical parameter

## 🧠 About the Model

The predictive model is an **XGBoost Classifier** 🏆 trained on 13 clinical features. The data undergoes a sophisticated preprocessing pipeline that includes:

- **📏 Standardization**: Scaling numerical features to have a mean of 0 and a standard deviation of 1
- **🔬 Principal Component Analysis (PCA)**: Reducing the dimensionality of the data to improve model efficiency and performance

### 📊 Model Performance

| Metric | Score |
|--------|-------|
| 🎯 Accuracy | 86.6% |
| 🔍 F1-Score | 89.6% |
| 📈 Recall | 94.8% |
| ⚖️ Precision | 85.0% |

### 🩺 Input Features

| Feature | Description | Type |
|---------|-------------|------|
| 🎂 **Age** | Patient's age in years | Numerical |
| 👤 **Sex** | Biological sex (male/female) | Categorical |
| 💔 **Chest Pain Type** | Classification of chest pain (0-3) | Categorical |
| 🩸 **Resting Blood Pressure** | Blood pressure at rest (mmHg) | Numerical |
| 🧪 **Cholesterol** | Serum cholesterol level (mg/dl) | Numerical |
| 🍯 **Fasting Blood Sugar** | Whether fasting blood sugar > 120 mg/dl | Binary |
| 📋 **Resting ECG** | Electrocardiogram results at rest | Categorical |
| 💓 **Max Heart Rate** | Maximum heart rate achieved during stress test | Numerical |
| ⚡ **Exercise Angina** | Whether exercise induces angina | Binary |
| 📉 **ST Depression** | ST depression induced by exercise | Numerical |
| 📈 **ST Slope** | Slope of peak exercise ST segment | Categorical |
| 🔬 **Major Vessels** | Number of major vessels colored by fluoroscopy (0-3) | Numerical |
| 🧬 **Thalassemia** | Thalassemia test results | Categorical |
