# Diabetes 130 US Hospitals

Link to have a more detailed look at the dataset:
https://onlinelibrary.wiley.com/doi/10.1155/2014/781670

## 1. Overview of Dataset

#### This dataset represents 10 years of clinical care at 130 US hospitals ranging from 1999 to 2008. The dataset includes over 50 features that represent patient and hospital outcomes. The information was extracted from the database for encounters that satisfy the following criteria:
- It is an inpatient encounter (hopsital admission)
- It is a diabetic encounter, one during which any kind of diabetes was entered to the system as a diagnosis
- The length of stay ranges between 1 day to 14 days
- The laboratory tests were performed during the encounter
- Medications were administered during the encounter

**The number of outpatient, inpatient and emergency visits are limited to the year before hospitalization!**

**There are 3 columns for diagnoses: diag_1, diag_2 and diag_3. Diag_1 represents the main diagnosis and so on; they are represented by numbers which refer to the ICU9 codes!!**

## 2. Business Case

### The business case is to predict hospital readmissions among diabetic patients.

The objective here is to use the historical data so we could identify the factors associated with early readmission among the patients with diabetes so we could reduce it. When we apply this predictive analysis, it could benefit the hospitals by identifying higher-risk patients and optimize the discharge planning among diabetes patients. To be able to evaluate our models, we should measure our success through readmission rates and  patient outcomes through different metrics such as the accuracy and precision.

## 3. Project Structure

```text
Supervied-learning-Diabetes/
│
├── diabetes.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── fairness.py
│   └── predict.py
│
├── models/
│   └── .gitkeep
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```
The diabetes notebook consists of the EDA analysis along with the feature engineering, model selection and hyperparameter tuning.

## 4. Installation

```text
pip install -r requirements.txt
```
