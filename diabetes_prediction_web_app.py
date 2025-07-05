import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open("trained_model.sav", 'rb'))

def diabetes_prediction(input_data):
    with open("scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    to_np_array=np.asarray(input_data)

    reshaped_array=to_np_array.reshape(1,-1)
    std_data=scaler.transform(reshaped_array)

    prediction=loaded_model.predict(std_data)
    print(prediction)

    if prediction[0]==0:
        return 'This person is not diabetic'
    else:
        return 'This person is diabetic'
    
def main():


    st.title('Diabetes Prediction Web App')

    Pregnancies=st.text_input('Number of pregnancies')
    Glucose=st.text_input('Amount of glucose')
    BloodPressure=st.text_input('Blood pression value')
    SkinThickness=st.text_input('Skin thickness value')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes pedigree function value')
    Age=st.text_input('Age of the person')

    diagnosis=''

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
