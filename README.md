# Heart Disease Prediction Project

This project uses machine learning to predict heart disease risk. It includes a complete data science pipeline, from data preprocessing and model training to an interactive web application for real-time predictions. The project evaluates several models, including **Logistic Regression, SVM, Decision Trees, Random Forest, and XGBoost**, with the final, optimized XGBoost model deployed in a **Streamlit** web app.

  \#\# Features

  * **Interactive Web Application**: A user-friendly web app built with Streamlit to make predictions easily accessible.
  * **End-to-End Machine Learning Pipeline**: Covers the entire workflow from data cleaning and preprocessing to model training and evaluation.
  * **Comprehensive Model Comparison**: Evaluates various machine learning models, including:
      * **Supervised Learning**: Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forest, and XGBoost.
      * **Unsupervised Learning**: K-Means clustering to discover hidden patterns in the data.
  * **Optimized Model**: The final prediction model (XGBoost) is fine-tuned using hyperparameter optimization for the best possible performance.

## Getting Started

Follow these instructions to set up the project on your local machine for development and testing.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/n0urhatem/Heart-Disease-Prediction.git
    cd heart-disease-prediction
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the web application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your web browser, where you can input patient data to get a heart disease risk prediction.

## About the Model

The predictive model is an **XGBoost Classifier** trained on 13 clinical features. The data undergoes a preprocessing pipeline that includes:

  * **Standardization**: Scaling numerical features to have a mean of 0 and a standard deviation of 1.
  * **Principal Component Analysis (PCA)**: Reducing the dimensionality of the data to improve model efficiency and performance.

### Input Features

  * **Age**: Patient's age in years
  * **Sex**: Biological sex (male/female)
  * **Chest Pain Type**: Classification of chest pain (0-3)
  * **Resting Blood Pressure**: Blood pressure at rest (mmHg)
  * **Cholesterol**: Serum cholesterol level (mg/dl)
  * **Fasting Blood Sugar**: Whether fasting blood sugar \> 120 mg/dl
  * **Resting ECG**: Electrocardiogram results at rest
  * **Max Heart Rate**: Maximum heart rate achieved during stress test
  * **Exercise Angina**: Whether exercise induces angina
  * **ST Depression**: ST depression induced by exercise
  * **ST Slope**: Slope of peak exercise ST segment
  * **Major Vessels**: Number of major vessels colored by fluoroscopy (0-3)
  * **Thalassemia**: Thalassemia test results
