import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import os
import json
import snscrape.modules.twitter as sntwitter
from langchain.llms import OpenAI
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
import matplotlib.pyplot as plt
from io import StringIO
import csv
import urllib.request
from PIL import Image
import openai
import requests



st.set_page_config(page_title="Sentiment Analysis",)
st.title("Sentiment Analysis on Tweets")

uploaded_file = st.file_uploader("Please upload a csv.file here:")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    pdf = pd.read_csv(uploaded_file)
    st.write(pdf)
else:
    pdf = pd.read_csv("LizAnnSonders_sample.csv")

#@st.cache_data
def get_finbert_result():
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    text_lst = pdf['Text']
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    text_lst = text_lst.to_list()
    result = nlp(text_lst)
    
    label_lst = []
    score_lst = []
    for i in range(0, len(result)):
        label = result[i]['label']
        score = result[i]['score']
        label_lst += [label]
        score_lst.append(score)
    
    label_lower_lst = []
    for label in label_lst:
        label_lower_lst.append(label.lower())

    finbert_pdf = pd.DataFrame({'text':text_lst, 'top_label':label_lower_lst, 'result': score_lst})
    return finbert_pdf

finbert_pdf = get_finbert_result()
labels_array = np.array(['positive', 'negative', 'neutral'])
positive_counts = finbert_pdf.loc[finbert_pdf.top_label == 'positive', 'top_label'].value_counts().reindex(labels_array, fill_value=0).positive
neutral_counts = finbert_pdf.loc[finbert_pdf.top_label == 'neutral', 'top_label'].value_counts().reindex(labels_array, fill_value=0).neutral
negative_counts = finbert_pdf.loc[finbert_pdf.top_label == 'negative', 'top_label'].value_counts().reindex(labels_array, fill_value=0).negative


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'positive', 'neutral', 'negative'
sizes = [positive_counts, neutral_counts, negative_counts]
explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
colors=['darkblue', 'tan', 'lightgrey']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors)
fig1.patch.set_alpha(0.0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#st.pyplot(fig1)

col1, col2 = st.columns(2)
with col1:
    #st.header("Result of FinBERT model")
    st.header("Finbert result")
    st.dataframe(finbert_pdf)

with col2:
    st.header("Count of sentiments")
    st.pyplot(fig1)


#@st.cache_data
def get_roberta_result():
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)
    text_lst = pdf['Text']
    text_lst = text_lst.to_list()
    labels = ['negative', 'neutral', 'positive']
    result_lst = []
    top_result_lst = []
    top_label_lst = []

    for text in text_lst:
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            result = f"{i+1}) {l} {np.round(float(s), 4)}"
            result_lst.append(result)
    
        top_label = labels[ranking[0]]
        top_result = scores[ranking[0]]
        top_result_lst.append(top_result)
        top_label_lst.append(top_label)
    
    roberta_pdf = pd.DataFrame({'text':text_lst, 'top_label':top_label_lst, 'top_result': top_result_lst})
    return roberta_pdf

roberta_pdf = get_roberta_result()
# ro_positive_counts = roberta_pdf.top_label.value_counts().positive
# ro_neutral_counts = roberta_pdf.top_label.value_counts().neutral
# ro_negative_counts = roberta_pdf.top_label.value_counts().negative
ro_positive_counts = roberta_pdf.loc[roberta_pdf.top_label == 'positive', 'top_label'].value_counts().reindex(labels_array, fill_value=0).positive
ro_neutral_counts = roberta_pdf.loc[roberta_pdf.top_label == 'neutral', 'top_label'].value_counts().reindex(labels_array, fill_value=0).neutral
ro_negative_counts = roberta_pdf.loc[roberta_pdf.top_label == 'negative', 'top_label'].value_counts().reindex(labels_array, fill_value=0).negative

sizes = [ro_positive_counts, ro_neutral_counts, ro_negative_counts]
fig2, ax2 = plt.subplots()
ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors)
fig2.patch.set_alpha(0.0)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#st.pyplot(fig1)

ro_col1, ro_col2 = st.columns(2)
with ro_col1:
    #st.header("Result of FinBERT model")
    st.header("Roberta result")
    st.dataframe(roberta_pdf)

with ro_col2:
    st.header("Count of sentiments")
    st.pyplot(fig2)


#@st.cache_data
def run_sentiment_analysis(txt):
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)

    encoded_input = tokenizer(txt, return_tensors='pt')
    output = model(**encoded_input)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    result_lst = []
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        result = (f"{i+1}) {l} {np.round(float(s), 4)}")
        result_lst += [result]
    return result_lst


txt = st.text_area('Please input text to analyze here:', '''Julius Baer offers a broad spectrum of tailored solutions for your investing, financing, and wealth planning needs.''')
st.write('Sentiment:', run_sentiment_analysis(txt))


################

st.title("Sentiment Analysis on News Titles")
with st.expander("ℹ️ - About this app", expanded=True):
    st.write(""" Please make sure "news titles" are in the "title column" """)
    st.markdown("")

uploaded_file = st.file_uploader("Please upload a news titles csv.file here:")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    pdf = pd.read_csv(uploaded_file)
    st.write(pdf)
else:
    pdf = pd.read_csv("check_headline.csv")

#@st.cache_data
def get_finbert_result():
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    text_lst = pdf['title']
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    text_lst = text_lst.to_list()
    result = nlp(text_lst)
    
    label_lst = []
    score_lst = []
    for i in range(0, len(result)):
        label = result[i]['label']
        score = result[i]['score']
        label_lst += [label]
        score_lst.append(score)
    
    label_lower_lst = []
    for label in label_lst:
        label_lower_lst.append(label.lower())

    finbert_pdf = pd.DataFrame({'text':text_lst, 'top_label':label_lower_lst, 'result': score_lst})
    return finbert_pdf

finbert_pdf = get_finbert_result()
labels_array = np.array(['positive', 'negative', 'neutral'])
positive_counts = finbert_pdf.loc[finbert_pdf.top_label == 'positive', 'top_label'].value_counts().reindex(labels_array, fill_value=0).positive
neutral_counts = finbert_pdf.loc[finbert_pdf.top_label == 'neutral', 'top_label'].value_counts().reindex(labels_array, fill_value=0).neutral
negative_counts = finbert_pdf.loc[finbert_pdf.top_label == 'negative', 'top_label'].value_counts().reindex(labels_array, fill_value=0).negative


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'positive', 'neutral', 'negative'
sizes = [positive_counts, neutral_counts, negative_counts]
explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#colors=['wheat', 'rosybrown', 'gray']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors)
fig1.patch.set_alpha(0.0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#st.pyplot(fig1)

col1, col2 = st.columns(2)
with col1:
    #st.header("Result of FinBERT model")
    st.header("Finbert result")
    st.dataframe(finbert_pdf)

with col2:
    st.header("Count of sentiments")
    st.pyplot(fig1)



#@st.cache_data
def get_roberta_result():
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)
    text_lst = pdf['title']
    text_lst = text_lst.to_list()
    labels = ['negative', 'neutral', 'positive']
    result_lst = []
    top_result_lst = []
    top_label_lst = []

    for text in text_lst:
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            result = f"{i+1}) {l} {np.round(float(s), 4)}"
            result_lst.append(result)
    
        top_label = labels[ranking[0]]
        top_result = scores[ranking[0]]
        top_result_lst.append(top_result)
        top_label_lst.append(top_label)
    
    roberta_pdf = pd.DataFrame({'text':text_lst, 'top_label':top_label_lst, 'top_result': top_result_lst})
    return roberta_pdf

roberta_pdf = get_roberta_result()
# ro_positive_counts = roberta_pdf.top_label.value_counts().positive
# ro_neutral_counts = roberta_pdf.top_label.value_counts().neutral
# ro_negative_counts = roberta_pdf.top_label.value_counts().negative
ro_positive_counts = roberta_pdf.loc[roberta_pdf.top_label == 'positive', 'top_label'].value_counts().reindex(labels_array, fill_value=0).positive
ro_neutral_counts = roberta_pdf.loc[roberta_pdf.top_label == 'neutral', 'top_label'].value_counts().reindex(labels_array, fill_value=0).neutral
ro_negative_counts = roberta_pdf.loc[roberta_pdf.top_label == 'negative', 'top_label'].value_counts().reindex(labels_array, fill_value=0).negative

sizes = [ro_positive_counts, ro_neutral_counts, ro_negative_counts]
fig2, ax2 = plt.subplots()
ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors)
fig2.patch.set_alpha(0.0)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#st.pyplot(fig1)

ro_col1, ro_col2 = st.columns(2)
with ro_col1:
    #st.header("Result of FinBERT model")
    st.header("Roberta result")
    st.dataframe(roberta_pdf)

with ro_col2:
    st.header("Count of sentiments")
    st.pyplot(fig2)


#@st.cache_data
def run_sentiment_analysis(txt):
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)

    encoded_input = tokenizer(txt, return_tensors='pt')
    output = model(**encoded_input)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    result_lst = []
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        result = (f"{i+1}) {l} {np.round(float(s), 4)}")
        result_lst += [result]
    return result_lst


txt = st.text_area('Please input news title to analyze here:', '''Julius Baer offers a broad spectrum of tailored solutions for your investing, financing, and wealth planning needs.''')
st.write('Sentiment:', run_sentiment_analysis(txt))


openai.api_key = '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'
PROMPT = st.text_area('Please input text to generate image', '''pig''')
PROMPT = str(PROMPT)
response = openai.Image.create(prompt=PROMPT,n=2,size="256x256")

url_0 = str(response["data"][0]["url"])
im_0 = Image.open(requests.get(url_0, stream=True).raw)

#image = Image.open(url_0)
st.image(url_0, caption='DALL-E Result')