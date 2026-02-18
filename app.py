import streamlit as st
import numpy as np
import joblib

# load model and scaler
model=joblib.load("diabetes_model.pkl")
scaler=joblib.load("scaler.pkl")

st.title("Diabetes Prediction System")

preg = st.number_input("Pregnancies",min_value=0)
glucose = st.number_input("Glucose",min_value=0)
bp = st.number_input("Blood Pressure",min_value=0)
skin = st.number_input("Skin Thickness",min_value=0)
insulin = st.number_input("Insulin",min_value=0)
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree function")
age = st.number_input("Age",min_value=1)

if st.button("Predict"):
    data=np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
    data_scaled=scaler.transform(data)
    result=model.predict(data_scaled)

    if result[0]==1:
        st.error("Diabetes")
    else:
        st.success("No Diabetes")


