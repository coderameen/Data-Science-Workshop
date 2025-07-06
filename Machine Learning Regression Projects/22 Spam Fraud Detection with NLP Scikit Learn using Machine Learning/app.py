import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#load model and tfidf_vectorizer
model = pickle.load(open('model.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

#detct spam function
def clean_text(text):
    #1.convert text to lower case
    text = text.lower()
    
    #2.Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]','',text)
    
    #Remove link (optional only if you have links)
    text = re.sub(r'http\S+','',text)
  
    
    #3.Tokenize the text
    words = word_tokenize(text)
    
    #4. Remove stopwords
    stop_words = set(stopwords.words('english'))#{'is','an','this',..........conjunctions}
    filter_stopwords = [word for word in words if word not in stop_words]
    
    
    #5. Stemming (PorterStemmer()) : running->run,loving->love,driven ->drive
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filter_stopwords]
    
    
    #6. Finally, join the stemmed words back into a single string
    cleaned_text = ' '.join(stemmed_words)
    return cleaned_text


def spam_detection(text):
    input_clean_text = clean_text(text)
    #I know clean text has char please convert into vector (numberical)
    input_vectorized_text = tfidf_vectorizer.transform([input_clean_text])
    result = model.predict(input_vectorized_text)
    return result


st.title("SPAM FRAUD DETECTION APPLICATION")

input_text = st.text_input("Enter your text to check Real or Fake")
if st.button("Submit"):
    my_input_text = input_text
    prediction = spam_detection(my_input_text)
    if prediction == 1:
        st.write("Real Message!!!")
    else:
        st.write("Fake Message!!!")