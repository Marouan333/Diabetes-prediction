# 🩺 Diabetes Prediction Model using SVM

A machine learning model that predicts whether a person has diabetes based on medical features. Built using Support Vector Machine (SVM), and deployed with **Streamlit** and **Pickle** for interactive use.

---

## 🚀 Overview

This project uses the Pima Indians Diabetes Dataset to train a classifier that predicts the likelihood of diabetes based on input features like glucose level, BMI, age, and more.

---

## 📊 Dataset

- **Features Used**:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- **Target**:
  - `0` = Non-diabetic
  - `1` = Diabetic

---

## 🧠 Model

- **Algorithm**: Support Vector Machine (SVM)
- **Libraries**: `scikit-learn`, `pickle`,`streamlit`
- **Preprocessing**:
  - Standardization with `StandardScaler`
  - Data split using `train_test_split`
- **Evaluation**:
  - Accuracy Score

---

## 📦 Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
