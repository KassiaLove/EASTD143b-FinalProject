#Import libraries
import streamlit as st
import pdfplumber #Reads the PDf file
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import re
import numpy as np
import os
from summarizer import Summarizer,TransformerSummarizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import urllib3
import json



st.set_page_config(
    page_title="Buddhist Text Analysis",
    layout="wide"
)

nlp = spacy.load('en_core_web_sm')
frequent_words = []

st.title("Buddhist Text Analysis")

with st.expander("ℹ️ - About this app - Click to Expand", expanded=False):

    st.write(
        """     
-   The *Buddhist Text Analysis* app is an easy-to-use interface built in Streamlit for summarizing an inputted Buddhist text. This text can be a sutra or a parable.
-   This summarized text can be used for getting the general topic of a piece of Buddhist literature, which can then be used for further analysis.
-   It uses two NLP libraries, Spacy and BERT, to do text summarization.
-   When using the Spacy library, one can choose to run the NLP program on parts of the text and combine the summarized results (Click 'YES' for Segment the Text?) or one can choose to run the NLP program on the text as a whole (Click 'NO' for Segment the Text?). Using the BERT NLP model will default use the entire text for summarization.
-   With this summarized piece of text, one can then do a name entity recognition (NER) with the summarization.
-   With the inputted text, one can choose to display a word cloud.
-   With the inputted text, one can choose to display a piece of artwork from the Harvard Art Museum API (https://api.harvardartmuseums.org/) that relates to the most frequent word in the text within the context of Buddhism.
-   One can input their text by pasting the text in the textbox or uploading a PDF. If you upload a PDF, the text from the PDF will be scraped with the PDFPlumber Python library. To display the text from the PDF file in the textbox area, click 'Summarize'. You can then edit the text that was scraped from the PDF.
-   Once the form is complete to your liking, click 'Summarize' to get your results.
	    """
    )

    st.markdown("")


st.subheader("Paste your document down below")

#Inputting text options are PDF or pasting the text


with st.form(key="my_form"):
    session = requests.Session()

    ce,c1,ce,c2,c3 = st.columns([0.07,1.25,0.07,5,0.07])

    

    with c1:
        model_button = st.radio("Choose Summary Model",['Spacy','BERT'])
        segment_button = st.radio("Segment the Text?",['No','Yes'])
        sum_sent_num = st.slider("Sentence Length of Summary?",4,10,5)
        name_entity_option = st.checkbox('Name Entity Recognition of Summary')
        word_cloud_option = st.checkbox('Word Cloud of Entire Text')
        image_option = st.checkbox('Related Image from Art Museum API')

    with c2:
        uploaded_file = st.file_uploader("Upload File",type=['pdf'])
        text_box_area = st.empty()
        with text_box_area:
            text_input = st.text_area('Paste text below','', height=500)
        raw_text = ""

        #Once they upload and click the analyze button, display the text from the pdf in the textbox
        if uploaded_file is not None:
            #File details
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}

            #Read the text from the file and display on the website
            if uploaded_file.type == "application/pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    total_text = ""
                    for pdf_page in pdf.pages:
                        single_page_text = pdf_page.extract_text()
                        raw_text += single_page_text

                #Preprocess Text
                raw_text = ' '.join(raw_text.splitlines())
                #Use regular expression to get rid of the artifact [#]
                clean_text = re.sub('\[\d+\]','',raw_text)
                text_box_area.empty()
                with text_box_area:
                    text_input = st.text_area('Paste text below',clean_text,height=500)
                
                


            else:
                #Preprocess Text
                text_input = ' '.join(text_input.splitlines())
                #Use regular expression to get rid of the artifact [#]
                text_input = re.sub('\[\d+\]','',text_input)

        submit_button = st.form_submit_button(label="Summarize")

#If the submit button hasn't been pressed yet
if not submit_button:
    st.stop()

#If nothing is in the text box stop here
if len(text_input) == 0:
    st.write('Please Input Text')
    st.stop()

final_sum = ''
ecol, result_col1, ecol, results_col2, ecol2 = st.columns([0.3,2,0.3,2,0.3])

if model_button == "Spacy" and segment_button == "No":
    doc = text_input
    doc = nlp(doc)
    #Find number of sentences in given string
    len(list(doc.sents))

    #Filtering tokens
    keyword = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN','ADJ','NOUN','VERB']
    for token in doc:
        if(token.text in stopwords or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)

    #Top 5 frequent words based on pos_tag
    freq_word = Counter(keyword)
    freq_word.most_common(5)

    keyword2 = []
    stopwords = list(STOP_WORDS)
    for token in doc:
        if(token.text in stopwords or token.text in punctuation):
            continue
        else:
            keyword2.append(token.text)
    #Acutal most frequent words for use in API request
    frequent_words = Counter(keyword2)
    #Need to check if the first entry is an actual word and not a space or punctuation
    for i in range(5):
        if frequent_words.most_common(5)[i][0] == '\n' or frequent_words.most_common(5)[i][0] == '“' or frequent_words.most_common(5)[i][0] == '”':
            continue
        else:
            actual_most_freq_word = frequent_words.most_common(5)[i][0]
            break



    #Normalize by dividing the token's frequencies by the maximum frequency
    #Each sentence will be weighed based on the frequency of the token present in each sentence
    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():
        freq_word[word] = (freq_word[word]/max_freq)
    freq_word.most_common(20)

    #Weighing Sentences
    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]

    #nlargest is used to summarize the string, it takes 3 arguments
    #1: number of data to extract
    #2: an iterable (list, tuple, dictionary)
    #3: condition to be satisfied
    #Below code returns a list containing the top 3 sentences
    #summarized_sentences is a spacy.tokens.span.Span type
    summarized_sentences = nlargest(sum_sent_num, sent_strength, key=sent_strength.get)

    #Convert to a string
    final_sentences = [w.text for w in summarized_sentences]
    final_sum = ' '.join(final_sentences)
    with result_col1:
        st.subheader("Here is your Spacy Summary that was generated by tokenizing and normalizing the entire text as a whole:")
        st.write(final_sum)



if model_button == "Spacy" and segment_button == "Yes":
    doc2 = nlp(text_input)

    keyword2 = []
    stopwords = list(STOP_WORDS)
    for token1 in doc2:
        if(token1.text in stopwords or token1.text in punctuation):
            continue
        else:
            keyword2.append(token1.text)
    #Acutal most frequent words for use in API request
    frequent_words = Counter(keyword2)

    #Need to check if the first entry is an actual word and not a space or punctuation
    for i in range(5):
        if frequent_words.most_common(5)[i][0] == '\n' or frequent_words.most_common(5)[i][0] == '“' or frequent_words.most_common(5)[i][0] == '”':
            continue
        else:
            actual_most_freq_word = frequent_words.most_common(5)[i][0]
            break


    doc = text_input
    doc = nlp(doc)
    #Find number of sentences in given string
    amount_sent = len(list(doc.sents))
    sent_counter = 0
    sent_list = ''
    sent_segment = []

    for i in range(amount_sent):
        sent_list += ' '+str(list(doc.sents)[i])
        sent_counter += 1
        if sent_counter == np.floor(amount_sent/sum_sent_num):
            sent_segment.append(sent_list)
            sent_counter = 0
            sent_list = ''
        if i == amount_sent-1:
            sent_segment.append(sent_list)
    
    summary = ''

    for i in range(len(sent_segment)):

        doc = nlp(sent_segment[i])
        #Filtering tokens
        keyword = []
        stopwords = list(STOP_WORDS)
        pos_tag = ['PROPN','ADJ','NOUN','VERB']
        for token in doc:
            if(token.text in stopwords or token.text in punctuation):
                continue
            if(token.pos_ in pos_tag):
                keyword.append(token.text)

        #Top 5 frequent words
        freq_word = Counter(keyword)
        freq_word.most_common(5)


        #Normalize by dividing the token's frequencies by the maximum frequency
        #Each sentence will be weighed based on the frequency of the token present in each sentence
        max_freq = Counter(keyword).most_common(1)[0][1]
        for word in freq_word.keys():
            freq_word[word] = (freq_word[word]/max_freq)
        freq_word.most_common(5)

        #Weighing Sentences
        sent_strength={}
        for sent in doc.sents:
            for word in sent:
                if word.text in freq_word.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent]+=freq_word[word.text]
                    else:
                        sent_strength[sent]=freq_word[word.text]

        #nlargest is used to summarize the string, it takes 3 arguments
        #1: number of data to extract
        #2: an iterable (list, tuple, dictionary)
        #3: condition to be satisfied
        #Below code returns a list containing the top 3 sentences
        #summarized_sentences is a spacy.tokens.span.Span type
        summarized_sentences = nlargest(1, sent_strength, key=sent_strength.get)

        #Convert to a string
        final_sentences = [w.text for w in summarized_sentences]
        final_sum += ' '+' '.join(final_sentences)

    

    with result_col1:
        st.subheader("Here is your Spacy Summary that was generated by tokenizing and normalizing the entire text as segments:")
        st.write(final_sum)


if model_button == "BERT":
    doc = text_input
    bert_model = Summarizer()
    final_sum = ''.join(bert_model(doc,num_sentences=sum_sent_num))

    keyword2 = []
    stopwords = list(STOP_WORDS)
    for token in nlp(doc):
        if(token.text in stopwords or token.text in punctuation):
            continue
        else:
            keyword2.append(token.text)
    #Acutal most frequent words for use in API request
    frequent_words = Counter(keyword2)

    #Need to check if the first entry is an actual word and not a space or punctuation
    for i in range(5):
        if frequent_words.most_common(5)[i][0] == '\n' or frequent_words.most_common(5)[i][0] == '“' or frequent_words.most_common(5)[i][0] == '”':
            continue
        else:
            actual_most_freq_word = frequent_words.most_common(5)[i][0]
            break



    with result_col1:
        st.subheader("Here is your BERT Summary")
        st.write(final_sum)

if name_entity_option:
    doc = nlp(final_sum)
    html = displacy.render(doc, style='ent')
    html = html.replace("\n\n","\n")
    with results_col2:
        st.subheader("Name Entity Recognition of the Generated Summary:")
        st.markdown(html,unsafe_allow_html=True)

ecol, result_col1, ecol, results_col2, ecol2 = st.columns([1,2,1,2,1])
if word_cloud_option:
    with result_col1:
        st.subheader("Word Cloud of Text:")
        text_wordcloud = WordCloud(width = 1000, height = 1000, background_color='white', min_font_size = 15).generate(text_input)
        fig,ax = plt.subplots()
        ax = plt.imshow(text_wordcloud)
        plt.axis("off")
        plt.show()
        st.pyplot(fig)
        

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

if image_option:
    with results_col2:
        
        if submit_button:
            st.subheader("Image from Art Museum API:")
            data = fetch(session,f"https://api.harvardartmuseums.org/object?apikey=ac378764-8ba4-40e1-b5e5-b04f5bb15e1f&title=Buddha AND {actual_most_freq_word}&size=5&fields=imagepermissionlevel,images,title")
            
            #Check if any records were returned
            if data["info"]["totalrecords"] > 0 : 
                st.write(f"The artwork relating to the most common word in the text, {actual_most_freq_word}, in relation to Buddhism")
                for i in range(5):
                    if data["records"][i]['imagepermissionlevel'] == 0:
                        response = requests.get(data["records"][i]['images'][0]['baseimageurl'])
                        title_image_caption = data["records"][i]['title']
                        break
                    else:
                        continue
                img = Image.open(BytesIO(response.content))
                st.image(img,title_image_caption,use_column_width='auto')
                
            elif data["info"]["totalrecords"] == 0:
                st.write(f"There are no artworks relating to the most common word in the text, {actual_most_freq_word}, in relation to Buddhism")
                st.write("Have an artwork relating to the most common word in the text instead")
                data2 = fetch(session,f"https://api.harvardartmuseums.org/object?apikey=ac378764-8ba4-40e1-b5e5-b04f5bb15e1f&title={actual_most_freq_word}&size=5&fields=imagepermissionlevel,images,title")

                #Check if any records were returned
                if data2["info"]["totalrecords"] > 0 : 
                    for i in range(5):
                        if data2["records"][i]['imagepermissionlevel'] == 0:
                            response = requests.get(data2["records"][i]['images'][0]['baseimageurl'])
                            title_image_caption = data2["records"][i]['title']
                            break
                        else:
                            continue
                    img = Image.open(BytesIO(response.content))
                    st.image(img,title_image_caption,use_column_width='auto')
            else:
                st.write("Unfortunately, I could not find an artwork that is related to the inputted text :(")

