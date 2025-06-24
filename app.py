import streamlit as st
import pickle
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
english_stopwords = ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

def transform_text(text):
    # lower case
    text = text.lower()
    # tokenization
    text = tokenizer.tokenize(text)
    # removing special characters
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text= y.copy()
    y.clear()
    # removing stop words and punctuation
    for i in text:
        if i not in english_stopwords and i not in string.punctuation:
            y.append(i) 

    text= y.copy()
    y.clear()
    # stemming data 
    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    # 1. Preprocess
    transform_sms =transform_text(input_sms)

    # 2. Vectorize
    vector_input =tfidf.transform([transform_sms])
    # 3. Apply model
    result = model.predict(vector_input)[0]
    # 4. Display
    if result== 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
