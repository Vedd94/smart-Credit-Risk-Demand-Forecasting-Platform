import streamlit as st
import requests

API_URL = "http://backend:8000/predict" 

st.title("Credit Risk classification")
st.markdown("Enter your details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=119, value=30)
sex = st.selectbox("Whats your gender?", options=['male', 'female'])
job = st.number_input("Number of jobs")
housing = st.selectbox("Housing type?", options=['own', 'rent', 'free'])
saving_accounts = st.selectbox("Saving account", options=['little', 'moderate', 'rich', 'quite rich'])
checking_account = st.selectbox("Checking_account?", options=['little', 'moderate', 'rich'])
credit_amount = st.number_input("Credit Amount")
duration = st.number_input("Duration")
purpose =  st.selectbox("Purpose", options=['radio/TV', 'education', 'vacation/others', 'car','furniture/equipment', 'domestic appliances', 'business','repairs'])

if st.button("Classify Risk"):
    input_data = {
        'age':age,
        'sex':sex,
        'job':job,
        'housing':housing,
        'saving_accounts':saving_accounts,
        'checking_account':checking_account,
        'credit_amount':credit_amount,
        'duration':duration,
        'purpose':purpose

    }

    try:
        response = requests.post(API_URL, json=input_data)
        result = response.json()

        if response.status_code == 200:
            st.success(
                f"Credit Risk: **{result['predicted_category']}**"
            )
            

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")