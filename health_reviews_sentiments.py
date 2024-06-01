#importing the necessary packages

import streamlit as st 
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pickle
import re

#importing the features and saved model
vect=pickle.load(open('D:/Final_project/features.pkl', 'rb'))
model=pickle.load(open("D:/Final_project/healthcare_model.pkl" ,'rb')) 


# Tokenize, remove stopwords, and lemmatize function
def process_text(sentence):
    tokens = word_tokenize(sentence)
    stop_words = set(stopwords.words("english"))  
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmatized_tokens

def remove_special_symbols(text):
    pattern = r"[^\w\s]"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text
#def function for predicting the feedback
def predict_sentiment(text):
    sent=vect.transform([text])
    pred3=model.predict(sent)
    return pred3

#streamlit page with tabs
st.title(":rainbow[HEALTH CARE FEEDBACK SENTIMNETS ANALYSIS]") 
tab1, tab2,tab3,tab4 = st.tabs(["OVERVIEW", "ANALYSIS","VISUALISATION","LEARNING OUTCOMES"])

#overview tab

with tab1:
    st.header("welcome to the sentiment analysis of feedbacks")
    st.subheader(":orange[**OBJECTIVE OF THE APPLICATION** ]") 
    st.markdown("")
    st.write("Most of the times It is hard to analyse the Reviews of a healthcare companies ")
    st.write("This Application Can Resolve the Ambiguity of the feedbacks")
    st.write("**For the Demonstration of performance it will Provide the Assistance** ")
    st.write("As well as it can contribute in betterment of the progress of a hospital review data ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")

#ANALYSIS TAB

with tab2:
    st.header("Analyse the response with the given fields")
    st.write("Add your own views on the healthcare ")
    input_sentence = st.text_area(":orange[**ENTER YOUR FEEDBACK** :]")
    st.subheader(":blue[RESPONSE OF YOUR FEEDBACK]")
    
    if input_sentence:
        processed_tokens = process_text(input_sentence)
        sent=predict_sentiment(input_sentence) 
        if sent==-1:
            st.warning(":red[****NEGATIVE RESPONSE****]" )
        elif sent==1:
            st.success(":green[****POSITIVE RESPONSE****]")
            st.success("😊          client satisfied")
        elif sent ==0:
            st.write("***😑    NEUTRAL RESPONSE***") 
#visualisation tab 
            
with tab3:
    st.header(":orange[***Import the Feedback Data to Analyse The Clients  Response***] ")
    st.caption("Kindly upload the data below")
    st.caption("Upload the feedback data in the given formats ")
    
    uploaded_file = st.file_uploader(":blue[**Choose the reviews file**]", type=["csv", "xlsx"]) 
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Unsupported file format. Please upload a CSV or Excel file.")
    
        if df is not None:
            
            colmn=df.columns
            selcolmn=st.selectbox("select the feedback column to be analysed",colmn)
            df["stringed"]=df[selcolmn].astype(str)
           
            df["response"]=df["stringed"].apply(predict_sentiment)
            chart=df["response"].value_counts().reset_index()
            chart["response"].replace({1: "POSITIVE", -1: "NEGATIVE",0:"NEUTRAL"}, inplace=True)
            
            st.write("**Here is the visual representation of total feedbacks**")
            

            fig1, ax1 = plt.subplots()
            ax1.pie(chart["count"], labels=chart["response"], autopct='%1.2f%%', shadow=True, startangle=90)
            ax1.axis('equal')
            ax1.legend()

            st.pyplot(fig1)
            
            
            st.write(" ")
            st.write(" ") 
            st.write("Explore the feedback data Efficiently ")
            st.markdown(" 🔍 ")
    else:
        st.warning("Please upload a file.")
        
#my learning outcomes 
with tab4:
    st.header(":grey[LEARNING INSIGHTS OF THE CAPSTONE]")
    st.write("    >         NATURAL LANGUAGE PROCESSING")
    st.write("    >         PYTHON SCRIPTING")
    st.write("    >         MACHINE LEARNING MODELS with sklearn")
    st.write("    >         STREAMLIT")
    st.markdown("")
    st.write(" ")
    st.markdown(" :green[thanks for the support  GUVI TEAM] ")
    
