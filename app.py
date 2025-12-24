import streamlit as st
import joblib

model=joblib.load("model.jlb")
scaler=joblib.load("scaler.jlb")

st.title("Insurance Prediction")

s1=st.number_input("Enter your age: ")

if st.button("predict"):
  age=scaler.transform([[s1]])
  prediction=model.predict(age)
  if prediction[0]==0:
    st.write("No insurance")
  else :
    st.write("Has insurance")
    
  
  
