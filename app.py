import joblib
import time
import streamlit as st



st.title("Welcome to EMOTO\AI (know about your fealings)")
st.title("This model will tell How You Are Felling Right Now: --")
model = joblib.load("coded_first.pkl")
label = joblib.load("label.pkl")
vector = joblib.load("vector.pkl")

Shpd = st.text_input("Enter your emotion\thoughts::", key="g1")


click = st.button("Predict", key="b1")

if click:
    with st.spinner("wait.."):
        time.sleep(7)
        take = vector.transform([Shpd])  # for using to take input as the text from the user and converting into encoded for MODEL
        result = model.predict(take) #predicting the outcome                   
        transform = label.inverse_transform(result) # decoding the result into text from label encoding
        st.write(transform)#printing result