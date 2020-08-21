"""
Build a basic classification pipeline:

    - vectorization of the training set with finetune sBert model

    - index all this vector with the Approximate Nearest Neighbors library (ANN)

    - Build a knn classifier where the new text input get the same labed as that closest index from the index

    - Benchmark the pipeline with the test set

    - Compare the model with the pretrained sBert
"""

from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

import os 
import pickle
import faiss
import torch
import numpy as np
from tqdm import tqdm


def benchmark(model_path: str, name: str, train_set, test_set, embedding_size: int=768, top_k_hits: int=10, n_cluster: int=256, nprobe: int=4):
    
    # load model
    model = SentenceTransformer(model_path)
    embedding_cache_path = 'output/20newsgroups-embeddings-{}.pkl'.format(name)

    # Defining our Faiss index
    # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N)
    # I willn use Inner dot as index. While normalizing our embedding vector, inner product is equal to cosine similarity
    # We choose IndexFlatIP because our dataset size is near of 10000 and what want exact result
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_cluster, faiss.METRIC_INNER_PRODUCT)

    # Number of cluster to explorer at search time. We will choose 4
    index.nprobe = nprobe

    if not os.path.exists(embedding_cache_path):
        corpus_sentences = []
        corpus_label = []
        idx2label = {}
        for idx, label in enumerate(train_set.target_names):
            idx2label[idx] = label
        
        for sent, target in zip(train_set.data, train_set.target):
            corpus_sentences.append(sent)
            corpus_label.append(target)
        
        print("Encode the corpus. This might take a while")
        corpus_embedding = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
        print("Store file on disc")
        with open(embedding_cache_path, 'wb') as fOut:
            pickle.dump({'embeddings': corpus_embedding, 'targets': corpus_label, 'label': idx2label}, fOut)
    else:
        print("Load pre-computed embedding from disc")
        with open(embedding_cache_path, 'rb') as fIn:
            cache_data = pickle.load(fIn)
            corpus_embedding = cache_data['embeddings']
            corpus_label = cache_data['targets']
            idx2label = cache_data['label']
        
    print("Start creating FAISS index")
    corpus_embedding = corpus_embedding / np.linalg.norm(corpus_embedding, axis=1)[:, None]

    # Then we train the index to find a suitable clustering
    index.train(corpus_embedding)

    # Finally we add all embedding to the index 
    index.add(corpus_embedding)
    print("Corpus loaded with {} sentences / embeddings".format(len(corpus_embedding)))

    # Accucacy on the test set
    true_label = list(test_set.target)
    count = 0
    for idx, sent in enumerate(tqdm(test_set.data)):
        pred = predict(sent, model=model, index=index, top_k_hits=top_k_hits, corpus_label=corpus_label, idx2label=idx2label)
        if pred[0] == true_label[idx]:
            count += 1
    accuracy = count / len(true_label)
    print("Approximate Nearest Neighbor precision on 20news group test with {} models: {:.2f}".format(name, accuracy * 100))

    return accuracy


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

# Benchmark pipeline
newstrainset = fetch_20newsgroups(subset="train", remove=('headers', 'footers','quotes'))
newstestset = fetch_20newsgroups(subset='test', remove=('headers', 'footers','quotes'))    

model_paths = ['models/fine-TripletLoss-20news-distilbert-base-nli-mean-tokens', 'distilbert-base-nli-mean-tokens']
model_names = ['fine-tune-sbert', 'pretrained-sbert']

benchs = {}
for model_path, model_name in zip(model_paths, model_names):
    benchs[model_name] = benchmark(model_path, model_path, newstrainset, newstestset)


# Faiss hyperparameters tuning to choose the right nprobe for indexing
# for i in tqdm([1, 2, 4, 8, 16, 32]):
# accuracy = {}
#     index = faiss.IndexIVFFlat(quantizer, embedding_size, n_cluster, faiss.METRIC_INNER_PRODUCT)
#     index.nprobe = i
#     index.train(corpus_embedding)
#     index.add(corpus_embedding)

#     count = 0
#     for idx, sent in enumerate(newstestset.data[:length]):
#         pred = predict(sent, index=index)
#         if pred[0] == true_label[idx]:
#             count += 1
#     accuracy[i] = count / length
#     print(f'{i}, accuracy: {count / length}')

# Number optimal de cluster 4
# accuracy_ann = accuracy[4]
