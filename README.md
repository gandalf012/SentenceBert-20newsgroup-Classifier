# 20 Newsgroups sentence-Bert classifier

API for sklearn 20 Newsgroups classifier with Faiss (Approximate Nearest Neighors library)  
Author: [Arnauld Adjovi](https://github.com/gandalf012)  

Demo:
  - **REST API**: http://68.183.209.70:5000/predict?query= `MY SENTENCE`
  - **WEB APP**: http://68.183.209.70:8501/

## Introduction
This repository fine-tunes DistilBERT models on scikit-learn 20 News groups dataset with a triplet network structure to produce semantically meaningful sentence embeddings that can be used in supervised scenarios: Semantic textual similarity via Facebook Faiss (Approximate Nearest-Neighors library) and label predictions.

We fine-tuned a pretrained `distilbert-base-nli-mean-tokens` with a TripletLoss function for 20 Newsgroups labels prediction.
We choosed facebook `Faiss-IVF`library for semantic search because of his avantages, mainly:
- fast search time (good Recall-Queries per second)
- good accuracy
- low memory footprint per index vector

The final fine-tuned model is available on [Google Drive](https://drive.google.com/uc?export=download&id=1VjYGZasx9sEuJ2u9DCirb8L2wdIYIcsM)

## Results

- **pretrained model**: `44.03` accuracy on 20 Newsgroup test set 
- **our fine-tuning**: `60.45` accuracy on 20 Newsgroup test set 

![pipeline benchmark on test set for Faiss and pretrained Sbert](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/images/pipeline_bench.png)

## How it works

**Installation tutorial** : [INSTALL](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/INSTALL.md)

The following file are the main components of the application:

- the [sbert_tuning.ipynb](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/sbert_tuning.ipynb) contains a custom loading class `Fetch20newsLabelDataset`and a training pipeline to fine-tune distilbert models. It also help to benchmark the sBert model on dev and test with `TripletEvaluator`
- the [pipeline.py](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/pipeline.py) contains a basic prediction code. To reproduce [benchmark results](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/tree/master/images), please follow [INSTALL](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/INSTALL.md) and run `python pipeline.py`
- the [app.py](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/app.py) is the flask app that expose the prediction pipeline through an api on port `5000`. 
- the [interface.py](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/interface.py) is a simple `web app` that leverage the previously created api to perform the classification task. It start by default on the port `8501`
- all these components are containerized into a docker image: [Dockerfile](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/Dockerfile)
- [ANN-Benchmarks-A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.pdf](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/tree/master/Paper) paper help to pick an optimal Approximate Nearest-Neighbor algorithm. For [Link and code: Fair AI Similarity Search], see the [Faiss Repository](https://github.com/facebookresearch/faiss/wiki/Getting-started)

## Using the REST API

After starting the docker image, the api will be available on **localhost:5000/predict**.
To run model prediction on a sentence, you just need to do a GET request with the sentence in the `query` parameter.

```shell
https://localhost:5000/predict?query=It was a 2-door sports car, looked to be from the late 60s early 70s. It was called a Bricklin. The doors were really small.
```
You should get a `json` as output
```json
{
  "label": "rec.autos",
  "query": "It was a 2-door sports car, looked to be from the late 60s early 70s. It was called a Bricklin. The doors were really small."
}
```

## Improvements

The following improvements can help achieve a better performance:

**Sentence-Bert embedding**:
- Use `Albert` instead of distilBert to encode the corpus (Albert `Sentencepiece` yield on better embedding than Bert `Wordpiece`)
- Fine-tune on more labeled data to improve models accuracy and use `BatchSemiHardTripletLoss` function for better training
- Leverage quantization and computational graph optimization with `OnnxRuntime` to improve Albert Inference time

**Approximate Nearest-Neighbors library**:
- Faiss-HNSW is the best option if you have a lot of RAM


## Reference

``` 
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```