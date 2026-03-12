import joblib
import time
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder



st.title("Welcome to EMOTO\AI (know about your fealings)")
st.title("This model will tell How You Are Felling Right Now: --")
label = LabelEncoder()
model = joblib.load("coded_first.pkl")

Shpd = st.number_input("Enter your emotion\thoughts::", key="g1")


click = st.button("Predict", key="b1")

if click:
    with st.spinner("wait..", show_time=False):
        time.sleep(7)
        user  = str(input("Enter your thought/Felling::")) # input area for the user
        fVectorizer = TfidfVectorizer()
        take = fVectorizer.transform([user])  # for using to take input as the text from the user and converting into encoded for MODEL
        result = model.predict(take) #predicting the outcome                   
        transform = label.inverse_transform(result) # decoding the result into text from label encoding
        st.write(transform)#printing result