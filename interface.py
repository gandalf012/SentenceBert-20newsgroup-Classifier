""" Testing Scrapping result on cdqa """

import streamlit as st
import time, re, os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

from multiprocessing import freeze_support


HTML_WRAPPER = """<span style = "overflow-x: auto;
                                 color : white;
                                 background-color: rgb(246, 51, 102);
                                 border: 1px solid #e6e9ef;
                                 border-radius: 0.4rem;
                                 padding: 0.2rem;
                                 margin-bottom: 2.5rem">{}</span>"""


HTML_PG_WRAPPER = """<div style = "
                        overflow-y: auto; 
                        background-color: rgba(0, 104, 201, 0.1); 
                        border-radius: 5px; 
                        border: 1px solid #ced7de; 
                        padding:20px; 
                        height: 230px; 
                        max-height: 265px
                        margin: 0 0 1rem;">{}</div>"""


# Load the models out of the main 
@st.cache(allow_output_mutation=True)
def get_fine_tune_model():
    newstrainset = fetch_20newsgroups(subset="train", remove=('headers', 'footers','quotes'))
    newstestset = fetch_20newsgroups(subset='test', remove=('headers', 'footers','quotes'))  

    model = SentenceTransformer('models/fine-TripletLoss-20news-distilbert-base-nli-mean-tokens')
    embedding_cache_path = 'output/20newsgroups-embeddings-fine-tune-sbert.pkl'
    with open(embedding_cache_path, 'rb') as fIn:
        cache_data = pickle.load(fIn)
    return model, cache_data, newstrainset, newstestset

@st.cache(allow_output_mutation=True)
def get_pretrained_model():
    newstrainset = fetch_20newsgroups(subset="train", remove=('headers', 'footers','quotes'))
    newstestset = fetch_20newsgroups(subset='test', remove=('headers', 'footers','quotes'))

    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embedding_cache_path = 'output/20newsgroups-embeddings-pretrained-sbert.pkl'
    with open(embedding_cache_path, 'rb') as fIn:
        cache_data = pickle.load(fIn)
    return model, cache_data, newstrainset, newstestset





if __name__ == "__main__":
    freeze_support()

    st.title("20 Newsgroup sentence classifier")
    st.info("This is a demonstrator of the potential of our sbert based classifier. \
              \n- Select first a **Model** \
              \n- Then, select a **Database** and choose an **article** to admire the result ! \N{bird}")

    # sidebar options    
    st.sidebar.title("Navigation")
    Model = st.sidebar.selectbox("Model", ["Fine-tuned sBert", "Pretrained sBert"])
    source = st.sidebar.selectbox("Database", ["train_set", "test_set"])
    top_k = st.sidebar.slider("top_k_nn", 10, 100)  


    ### MODEL TRAINING SECTION
    s1 = time.time()

    sbert_model = None
    if "Fine-tuned sBert" in Model:
        sbert_model = get_fine_tune_model()[0]
        
        if "train_set" in source:
            df = get_fine_tune_model()[2]
            data = df.data[:500]
            indexes = range(0, len(data))
            mapper = lambda x: data[x].strip().replace('\n', ' ')[:85]+'...'
            ind = st.selectbox("Choose an article", options = indexes, index= 1, format_func = mapper)

            paragraphs_html = ("""<p>{}<p>""".format(data[ind].strip().replace('\n', ' ')))
            st.write(HTML_PG_WRAPPER.format(paragraphs_html), unsafe_allow_html= True)

        else:
            df = get_fine_tune_model()[3]
            data = df.data[:500]
            indexes = range(0, len(data))
            mapper = lambda x: data[x].strip().replace('\n', ' ')[:85]+'...'
            ind = st.selectbox("Choose an article", options = indexes, index= 1, format_func = mapper)

            paragraphs_html = ("""<p>{}<p>""".format(data[ind].strip().replace('\n', ' ')))
            st.write(HTML_PG_WRAPPER.format(paragraphs_html), unsafe_allow_html= True)
    else:
        sbert_model = get_pretrained_model()[0]
            
        if "train_set" in source:
            df = get_fine_tune_model()[2]
            data = df.data[:500]
            indexes = range(0, len(data))
            mapper = lambda x: data[x].strip().replace('\n', ' ')[:85]+'...'
            ind = st.selectbox("Choose an article", options = indexes, index= 1, format_func = mapper)

            paragraphs_html = ("""<p>{}<p>""".format(data[ind].strip().replace('\n', ' ')))
            st.write(HTML_PG_WRAPPER.format(paragraphs_html), unsafe_allow_html= True)

        else:
            df = get_fine_tune_model()[3]
            data = df.data[:500]
            indexes = range(0, len(data))
            mapper = lambda x: data[x].strip().replace('\n', ' ')[:85]+'...'
            ind = st.selectbox("Choose an article", options = indexes, index= 1, format_func = mapper)

            paragraphs_html = ("""<p>{}<p>""".format(data[ind].strip().replace('\n', ' ')))
            st.write(HTML_PG_WRAPPER.format(paragraphs_html), unsafe_allow_html= True)


    t1 = time.time() - s1

    # # Fitting the retriever to the list of documents in the dataframe$
    # s2 = time.time()
    # qa_model.fit_retriever(df)
    # t2 = time.time() - s2

    # # Querying and displaying 
    # query = st.text_input(label="", value=default_query)
    # if st.button("Predict answers") and query != default_query: 
    #     s3 = time.time()
    #     prediction = qa_model.predict(query)
    #     t3 = time.time() - s3
        
    #     st.success(prediction[0]) 

    #     res = prediction[2].replace(prediction[0], HTML_WRAPPER.format(prediction[0]))
    #     st.subheader("Article containing the answer:")
    #     st.write('*{}*\n'.format(res), unsafe_allow_html=True)
    #     st.info('Answering your question required **{} seconds**.'.format(round(t3, 2)))

    # else: 
    #     st.error("You need **ask a question** and **press the button** to predict the answer")

