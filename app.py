import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64


def get_binary_file_downloader_html(df):
    csv= df.to_csv(index=False)
    b64= base64.b64encode(csv.encode()).decode()
    href= f'<a href= "data:file/csv;base64,{b64}" download="predictions.csv"> Download Preductions CSV</a>'
    return href


st.title("Heart Disease Predictor")
tab1,tab2,tab3 = st.tabs(['Predict','Bulk Predict','Model Info'])

with tab1:
    age =st.number_input("Age(years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex",['male','female'])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
 
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['decisiontree.pkl', 'LogisticR.pkl', 'Randomforest.pkl', 'SVM.pkl']

    predictions = []

    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            results = model.predict(data)
            predictions.append(results)
        return predictions

    if st.button("Submit"):
        st.subheader('results...')
        st.markdown('---------')

        result = predict_heart_disease(input_data)

        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0]==0:
                st.write("no heart disease detected")
            else:
                st.write("heart disease detected")
            st.markdown('---------')

with tab2:
    st.title("Upload CSV File")

    uploaded_file =  st.file_uploader("upload csv file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model= pickle.load(open('LogisticR.pkl','rb'))

    # Ensure that the input DataFrame matches the expected columns and format
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(input_data.columns):
            input_data['Prediction LR']=''
            for i in range(len(input_data)):
                arr= input_data.iloc[i,:-1].values
                input_data['Prediction LR'][i]=model.predict([arr])[0]
            input_data.to_csv('PredictedHeartLR.csv')

            st.subheader("Predictions:")
            st.write(input_data)

            st.markdown(get_binary_file_downloader_html(input_data),unsafe_allow_html=True)
        
        else:
            st.warning("Please make sure the csv has correct columns")
    else:
        st.info("Upload a CSV file to get predictions")
with tab3:
    import plotly.express as px

    data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 86.54
    }

    Models = list(data.keys())
    Accuracies = list(data.values())

    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])

    fig = px.bar(df, x='Models', y='Accuracies')

    st.plotly_chart(fig)