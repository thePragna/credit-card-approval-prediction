from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
# (No scaler is used: the saved model was trained on unscaled features,
# since Random Forest does not require feature scaling.)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---- Encoding maps ----
# These match sklearn's LabelEncoder alphabetical ordering, confirmed
# directly against the training data (see project notes).
GENDER_MAP = {'F': 0, 'M': 1}
YN_MAP = {'N': 0, 'Y': 1}
INCOME_TYPE_MAP = {
    'Commercial associate': 0,
    'Pensioner': 1,
    'State servant': 2,
    'Student': 3,
    'Working': 4,
}
EDUCATION_MAP = {
    'Academic degree': 0,
    'Higher education': 1,
    'Incomplete higher': 2,
    'Lower secondary': 3,
    'Secondary / secondary special': 4,
}
FAMILY_STATUS_MAP = {
    'Civil marriage': 0,
    'Married': 1,
    'Separated': 2,
    'Single / not married': 3,
    'Widow': 4,
}
HOUSING_TYPE_MAP = {
    'Co-op apartment': 0,
    'House / apartment': 1,
    'Municipal apartment': 2,
    'Office apartment': 3,
    'Rented apartment': 4,
    'With parents': 5,
}

# FLAG_MOBIL is constant (always 1) in the training data, so it's fixed here
FLAG_MOBIL = 1

# Feature order MUST match X_train.columns.tolist() from training exactly:
# ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
#  'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
#  'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
#  'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
#  'CNT_FAM_MEMBERS', 'FAMILY_SIZE']


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form

    gender = GENDER_MAP[form['gender']]
    own_car = YN_MAP[form['own_car']]
    own_realty = YN_MAP[form['own_realty']]
    children = int(form['children'])
    income = float(form['income'])
    income_type = INCOME_TYPE_MAP[form['income_type']]
    education = EDUCATION_MAP[form['education']]
    family_status = FAMILY_STATUS_MAP[form['family_status']]
    housing_type = HOUSING_TYPE_MAP[form['housing_type']]

    # Convert user-friendly years into the DAYS_* format the model expects
    age_years = int(form['age'])
    years_employed = int(form['years_employed'])
    days_birth = age_years * 365
    days_employed = years_employed * 365  # 0 if unemployed/pensioner

    work_phone = int(form['work_phone'])
    phone = int(form['phone'])
    email = int(form['email'])
    fam_members = int(form['fam_members'])

    family_size = children + fam_members

    features = np.array([[
        gender, own_car, own_realty, children, income,
        income_type, education, family_status, housing_type,
        days_birth, days_employed, FLAG_MOBIL, work_phone,
        phone, email, fam_members, family_size
    ]])

    prediction = int(model.predict(features)[0])

    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
