""" Testing Scrapping result on cdqa """

import streamlit as st
import time, os
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
                        margin: 2rem;">{}</div>"""


# Load the models out of the main 
@st.cache(allow_output_mutation=True)
def get_fine_tune_model():
    model = SentenceTransformer('models/fine-TripletLoss-20news-distilbert-base-nli-mean-tokens')
    embedding_cache_path = 'output/20newsgroups-embeddings-fine-tune-sbert.pkl'
    with open(embedding_cache_path, 'rb') as fIn:
        cache_data = pickle.load(fIn)
    return model, cache_data

@st.cache(allow_output_mutation=True)
def get_pretrained_model():
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embedding_cache_path = 'output/20newsgroups-embeddings-pretrained-sbert.pkl'
    with open(embedding_cache_path, 'rb') as fIn:
        cache_data = pickle.load(fIn)
    return model, cache_data


# Load dataset
@st.cache(allow_output_mutation=True)
def get_trainset():
    return fetch_20newsgroups(subset="train", remove=('headers', 'footers','quotes'))

@st.cache(allow_output_mutation=True)
def get_testset():
    return fetch_20newsgroups(subset="test", remove=('headers', 'footers','quotes'))


@st.cache(allow_output_mutation=True)
def build_index(corpus_embedding, n_cluster = 256, embedding_size = 768, nprobe=4):
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_cluster, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe

    corpus_embeddings = corpus_embedding / np.linalg.norm(corpus_embedding, axis=1)[:, None]
    index.train(corpus_embeddings)
    index.add(corpus_embeddings)
    return index


def predict(query, model, index, top_k_hits, corpus_label, idx2label):
    
    question_embedding = model.encode(query)
    # Query normalization for Faiss inner product
    question_embedding = question_embedding / np.linalg.norm(question_embedding)
    distances, corpus_ids = index.search(question_embedding, top_k_hits)

    hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    # Nearest Neighors label 
    label_idx = corpus_label[hits[0]['corpus_id']]

    return label_idx, idx2label[label_idx]


if __name__ == "__main__":
    freeze_support()

    st.title("20 Newsgroup sentence classifier")
    st.info("This is a demonstrator of our sbert based sentence classifier. The code is available upon request [On Github](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier) \
              \n- First, select a **Model** \
              \n- Then, select a **Database** and choose an **article** to admire the result ! \N{bird}")

    # sidebar options    
    st.sidebar.title("Navigation")
    Model = st.sidebar.selectbox("Model", ["Fine-tuned sBert (acc. 60.45)", "Pretrained sBert (acc. 44.03)"])
    source = st.sidebar.selectbox("Database", ["train_set", "test_set"])
    top_k = st.sidebar.slider("top_k_nn", 10, 100)  

    if "train_set" in source:
        df = get_trainset()
        data = df.data[:1000]
        indexes = range(0, len(data))
        mapper = lambda x: data[x].strip().replace('\n', ' ')[:85]+'...'
        ind = st.selectbox("Choose an article", options = indexes, index= 1, format_func = mapper)

        paragraphs_html = ("""<p>{}<p>""".format(data[ind].strip().replace('\n', ' ')))
        st.write(HTML_PG_WRAPPER.format(paragraphs_html), unsafe_allow_html= True)

    else:
        df = get_testset()
        data = df.data[:500]
        indexes = range(0, len(data))
        mapper = lambda x: data[x].strip().replace('\n', ' ')[:85]+'...'
        ind = st.selectbox("Choose an article", options = indexes, index= 0, format_func = mapper)

        paragraphs_html = ("""<p>{}<p>""".format(data[ind].strip().replace('\n', ' ')))
        st.write(HTML_PG_WRAPPER.format(paragraphs_html), unsafe_allow_html= True)


    ### MODEL RUNNING SECTION
    if "Fine-tuned sBert" in Model:
        sbert_model, cache_data = get_fine_tune_model()
        corpus_embedding = cache_data['embeddings']
        corpus_label = cache_data['targets']
        idx2label = cache_data['label']
        index = build_index(corpus_embedding)

    else:
        sbert_model, cache_data = get_pretrained_model()
        corpus_embedding = cache_data['embeddings']
        corpus_label = cache_data['targets']
        idx2label = cache_data['label']
        index = build_index(corpus_embedding)

    # # Querying and displaying
    st.subheader("")
    if st.button("Predict Label"):
        s1 = time.time()
        prediction = predict(data[ind], sbert_model, index, top_k, corpus_label, idx2label)
        t1 = time.time() - s1
        st.subheader("Predicted label:")
        st.success(prediction[1]) 
        st.info('Finding label took **{} seconds**.'.format(round(t1, 2)))

