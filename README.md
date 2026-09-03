# 🏦 Customer Loan Decision AI
## 🌐 Live Demo

[🏦 Customer Loan Decision App](https://customer-loan-ann.streamlit.app/)
## 🖥️ Application Screenshot
A deep-learning project that uses **PyTorch Artificial Neural Networks (ANNs)** to predict loan approval and estimate the loan amount.

## 🚀 Project Overview

The project contains two ANN models:

```text
Customer Data
      │
      ▼
┌─────────────────────┐
│ Classification ANN  │
│ Loan Approval       │
└──────────┬──────────┘
           │
       Approved?
       /       \
     YES        NO
      │          │
      ▼          ▼
Regression     No Loan
ANN
      │
      ▼
Predicted Loan Amount
```

### Classification

Predicts whether a customer's loan should be approved.

```text
Input → 32 → 16 → 1
```

### Regression

If the loan is approved, the regression ANN predicts the estimated loan amount.

```text
Input → 64 → 32 → 16 → 1
```

## 🧠 Technologies

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## 📊 Model Evaluation

### Classification

The classification model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Regression

The regression model is evaluated using:

* MAE
* MSE
* RMSE
* R² Score

## 🔄 Machine Learning Pipeline

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
PyTorch Tensors
   ↓
DataLoader
   ↓
ANN
   ↓
Training
   ↓
Evaluation
   ↓
Model Saving
   ↓
Streamlit Application
```

## 💻 Run the Project

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project:

```bash
cd Customer-Loan-ANN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
Customer-Loan-ANN/
│
├── app.py
├── customer_loan.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebook/
│   └── customer_loan_ann.ipynb
│
├── models/
│   ├── loan_classifier.pth
│   └── loan_regressor.pth
│
└── scalers/
    ├── Classification_scaler.pkl
    ├── Regression_scaler.pkl
    ├── Regression_target_scaler.pkl
    ├── classification_columns.pkl
    └── regression_columns.pkl
```

## 🎯 Key Learning Outcomes

* PyTorch tensors
* GPU/CPU training
* `nn.Module`
* ANN architecture
* Forward propagation
* Loss functions
* Backpropagation
* Adam optimizer
* DataLoader
* Feature scaling
* Binary classification
* Regression
* Model evaluation
* Model serialization
* Streamlit deployment

## ⚠️ Disclaimer

This project is an educational machine-learning project. The predictions are model outputs and should not be used as real-world financial decisions.
