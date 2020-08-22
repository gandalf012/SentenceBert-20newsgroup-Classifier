from flask import Flask, request, jsonify
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

import os 
import pickle
import faiss
import numpy as np

app = Flask(__name__)
CORS(app)

model_path = "models/fine-TripletLoss-20news-distilbert-base-nli-mean-tokens"
model = SentenceTransformer(model_path)

embedding_cache_path = 'output/20newsgroups-embeddings-fine-tune-sbert.pkl'
embedding_size = 768

top_k_hits = 10

n_cluster = 256
quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(quantizer, embedding_size, n_cluster, faiss.METRIC_INNER_PRODUCT)
index.nprobe = 4

if not os.path.exists(embedding_cache_path):
    newstrainset = fetch_20newsgroups(subset="train", remove=('headers', 'footers','quotes'))

    corpus_sentences = []
    corpus_label = []
    idx2label = {}
    for idx, label in enumerate(newstrainset.target_names):
        idx2label[idx] = label
    
    for sent, target in zip(newstrainset.data, newstrainset.target):
        corpus_sentences.append(sent)
        corpus_label.append(target)

    corpus_embedding = model.encode(corpus_sentences, convert_to_numpy=True)
    with open(embedding_cache_path, 'wb') as fOut:
        pickle.dump({'embeddings': corpus_embedding, 'targets': corpus_label, 'label': idx2label}, fOut)
else:
    with open(embedding_cache_path, 'rb') as fIn:
        cache_data = pickle.load(fIn)
        corpus_embedding = cache_data['embeddings']
        corpus_label = cache_data['targets']
        idx2label = cache_data['label']

corpus_embedding = corpus_embedding / np.linalg.norm(corpus_embedding, axis=1)[:, None]
index.train(corpus_embedding)
index.add(corpus_embedding)

@app.route("/predict", methods=["GET"])
def api():
    
    query = request.args.get("query")
    question_embedding = model.encode(query)

    # Query normalization for Faiss inner product
    question_embedding = question_embedding / np.linalg.norm(question_embedding)
    distances, corpus_ids = index.search(question_embedding, top_k_hits)

    hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    # Nearest Neighors label 
    label_idx = corpus_label[hits[0]['corpus_id']]
    
    return(jsonify(
        query=query, 
        label=idx2label[label_idx])) 


if __name__ == "__main__":
    app.run(host='0.0.0.0')
