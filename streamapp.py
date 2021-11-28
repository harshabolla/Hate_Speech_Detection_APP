import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

# Load the Multinomial/XGB Naive Bayes model and CountVectorizer object from disk
filename = 'xgbmodel.pkl'
classifier = pickle.load(open(filename, 'rb'))
#cv = pickle.load(open('tfid -transform.pkl','rb'))



#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(message):
    data = [message]
    #vect = cv.transform(data).toarray()
    my_prediction = classifier.predict(data)
    if my_prediction==1:
        my_prediction = ':-   Oops! its  hate speech' 
        
    else:
        my_prediction  = ":-Great! It's not a hate speech :)"

    return my_prediction
  

def main():
    st.title("            Hate-Speech-Detection ")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Hate Speech  ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    message = st.text_area("message","Type Here")
    
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(message)
    st.success('The output is {}'.format(result))
    
    if st.button("About"):
        
        st.text("Trained with ,XGBclassifier: ROC AUC=0.976 ")
        st.text("f1-score:  0.9159862338889263")
        st.text("built by harsha Teja ")
        st.text("Built with Streamlit")
    

if __name__=='__main__':
    main()
    