# Streamlit app
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🛒 Product Purchase Likelihood Predictor")

df = pd.read_csv("purchase_data.csv")
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

X = df[["Time_Spent","Age","Gender","Ads_Clicked","Previous_Purchases"]]
y = df["Purchased"]

lr = LogisticRegression()
lr.fit(X, y)

dt = DecisionTreeClassifier()
dt.fit(X, y)

st.header("Enter Customer Details")

time_spent = st.slider("Time Spent on Website (Minutes)", 1, 60)
age = st.slider("Customer Age", 18, 70)
gender = st.selectbox("Gender", ["M", "F"])
ads = st.slider("Ads Clicked", 0, 10)
previous = st.slider("Previous Purchases", 0, 10)

gender_encoded = 1 if gender == "M" else 0

input_data = pd.DataFrame([[time_spent, age, gender_encoded, ads, previous]],
                          columns=["Time_Spent","Age","Gender","Ads_Clicked","Previous_Purchases"])

model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])

if st.button("Predict Purchase"):
    if model_choice == "Logistic Regression":
        pred = lr.predict(input_data)[0]
    else:
        pred = dt.predict(input_data)[0]

    if pred == 1:
        st.success("Customer is likely to purchase ✔️")
    else:
        st.error("Customer is not likely to purchase ❌")
