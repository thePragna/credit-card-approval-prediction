# Credit Card Approval Prediction

A machine learning system that predicts whether a credit card application is likely to be **approved** or **rejected**, based on applicant financial and demographic details. The best-performing model is integrated into a Flask web application for real-time predictions.

## Problem Statement

Banks and financial institutions receive thousands of credit card applications daily. A significant portion are rejected due to factors such as high existing loan balances, insufficient income, or excessive credit inquiries. Manual review is slow and error-prone. This project automates the approval decision using historical applicant data.

## Project Workflow

1. **Data Collection** — Applicant records (`application\\\_record.csv`) and credit history records (`credit\\\_record.csv`) sourced from Kaggle.
2. **Data Visualization \& Analysis** — Univariate and multivariate analysis of applicant demographics and financial attributes.
3. **Data Pre-processing**

   * Dropped `OCCUPATION\\\_TYPE` (\~30% missing values)
   * Removed duplicate applicant records
   * Converted `DAYS\\\_BIRTH` / `DAYS\\\_EMPLOYED` to positive values; handled the unemployed/pensioner placeholder value
   * Converted the multi-class credit `STATUS` column into a binary target (0 = approved, 1 = rejected)
   * Aggregated credit history per applicant and merged with applicant details
   * Engineered a `FAMILY\\\_SIZE` feature from `CNT\\\_CHILDREN` + `CNT\\\_FAM\\\_MEMBERS`
   * Label-encoded all categorical features
4. **Model Building** — Trained and compared four classification algorithms:

   * Logistic Regression
   * Decision Tree
   * Random Forest
   * XGBoost
5. **Application Building** — The best model (Random Forest) is served through a Flask web app with three pages: a landing page, a prediction form, and a results page.

## Model Performance

|Model|Precision (Rejected)|Recall (Rejected)|F1-score (Rejected)|Accuracy|
|-|-|-|-|-|
|Logistic Regression|0.13|0.50|0.21|0.56|
|Decision Tree|0.31|0.66|0.42|0.79|
|**Random Forest (selected)**|**0.37**|0.54|**0.44**|**0.84**|
|XGBoost|0.27|0.58|0.37|0.76|

**Random Forest** was selected as the final model based on its F1-score for the minority (rejected) class, which is the more meaningful metric given the \~88/12 class imbalance in the dataset.

## Tech Stack

* **Language:** Python 3.10
* **Data Analysis:** pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** scikit-learn, XGBoost
* **Web Framework:** Flask
* **Environment:** Google Colab (model training), local Flask server (deployment)

## Project Structure

```
credit-card-approval/
├── app.py                         # Flask backend
├── model.pkl                      # Trained Random Forest model
├── credit\\\_card\\\_approval.ipynb     # Data analysis \\\& model training notebook
└── templates/
    ├── home.html                  # Landing page
    ├── index.html                 # Prediction form
    └── result.html                # Prediction result page
```

## How to Run Locally

1. Clone this repository:

```
   git clone https://github.com/YOUR-USERNAME/credit-card-approval-prediction.git
   cd credit-card-approval-prediction
   ```

2. Create and activate a virtual environment:

```
   python -m venv venv
   venv\\\\Scripts\\\\activate
   ```

3. Install dependencies:

```
   pip install flask scikit-learn numpy
   ```

4. Run the app:

```
   python app.py
   ```

5. Open `http://127.0.0.1:5000/` in your browser.

## Demo

A short walkthrough video demo is available here: **\[https://drive.google.com/file/d/1Hobzn46vpTnxX8ieK0yBJCpuMBvnm1eL/view?usp=drive_link]**

## Conclusion

This project demonstrates how machine learning can automate and enhance the credit card approval process in the banking sector. By analyzing applicant details such as income type, employment status, family status, and housing type alongside historical credit behavior, the system predicts approval likelihood with reasonable accuracy. The pipeline covers the complete ML lifecycle — data collection, preprocessing, feature engineering, model comparison, and deployment through a user-friendly Flask web application — providing hands-on experience in applied machine learning and financial risk assessment.



