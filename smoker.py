import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Loading GIFs
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#Set page configuration
# st.set_page_config(
#         page_title="Smoker Detection",
#         page_icon="🚬",
#         layout="wide",
#         initial_sidebar_state="expanded",
#         menu_items={
#             'Get Help': 'https://www.extremelycoolapp.com/help',
#             'Report a bug': "https://www.extremelycoolapp.com/bug",
#             'About': "# This is a header. This is an *extremely* cool app!"
#          }
#)

st.subheader("Hi, I am Piyush :wave:")
lottie_coding_2 = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_LGoEHs.json")
st.title("This is a model that will predict if a user is a smoker or not 🚬")
st_lottie(lottie_coding_2, height=300, key="coding")

# Load the dataset from a CSV file
data = pd.read_csv('data/train_dataset.csv')

# Preprocess the data by selecting relevant features and converting categorical variables to numerical
X = data[['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic', 'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 'AST', 'ALT', 'Gtp', 'dental caries']]
y = data['smoking']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model on the training data
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Evaluate the performance of the model on the testing data
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

# Taking data from the user
st.title("Smoker Detection")

lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_lphquaqr.json")

st.subheader("This logistic learning model will detect if a user is a smoker or not based on the following data given to him")
st_lottie(lottie_coding, height=300, key="coding2")

age = st.number_input("Enter age:")
height = st.number_input("Enter height (in cm):")
weight = st.number_input("Enter weight (in kg):")
waist = st.number_input("Enter waist (in cm):")
eyesight_left = st.number_input("Enter eyesight (left):")
eyesight_right = st.number_input("Enter eyesight (right):")
hearing_left = st.number_input("Enter hearing (left):")
hearing_right = st.number_input("Enter hearing (right):")
systolic = st.number_input("Enter systolic:")
relaxation = st.number_input("Enter relaxation:")
fasting_blood_sugar = st.number_input("Enter fasting blood sugar:")
Cholesterol = st.number_input("Enter cholesterol:")
triglyceride = st.number_input("Enter triglyceride:")
HDL = st.number_input("Enter HDL:")
LDL = st.number_input("Enter LDL:")
hemoglobin = st.number_input("Enter hemoglobin:")
Urine_protein = st.number_input("Enter urine protein:")
serum_creatinine = st.number_input("Enter serum creatinine:")
AST = st.number_input("Enter AST:")
ALT = st.number_input("Enter ALT:")
Gtp = st.number_input("Enter GTP:")
Dental_caries = st.number_input("Enter dental caries:")

# Predicting from the given data
prediction = clf.predict([[age, height, weight, waist, eyesight_left, eyesight_right, hearing_left, hearing_right, systolic, relaxation, fasting_blood_sugar, Cholesterol, triglyceride, HDL, LDL, hemoglobin, Urine_protein, serum_creatinine, AST, ALT, Gtp, Dental_caries]])

if prediction == 'yes':
    st.write("This person is a smoker")
elif prediction == 'no':
    st.write("This person is not a smoker")

st.write('Accuracy:', accuracy)
st.write('Precision:', precision)
st.write('Recall:', recall)
st.write('F1 score:', f1)
