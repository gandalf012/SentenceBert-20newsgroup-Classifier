# 20 Newsgroups sentence-Bert classifier

API for sklearn 20 Newsgroups classifier with Faiss (Approximate Nearest Neighors library)
Author: [Arnauld Adjovi](https://github.com/gandalf012)


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

## How it works !

**Install tutorial** : [INSTALL](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/INSTALL.md)

The following are entry points for documentation:

- the [sbert_tuning.ipynb](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/sbert_tuning.ipynb) contains a custom loading class `Fetch20newsLabelDataset`and a training pipeline to fine-tune distilbert models. It also help to benchmark the sBert model on dev and test with `TripletEvaluator`
- the [prediction pipeline](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/pipeline.py) contains a basic prediction code. To reproduce [benchmark results](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/tree/master/images), please follow [INSTALL](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/INSTALL.md) and run `python pipeline.py`
- the [app.py](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/app.py) is the flask app that expose the prediction pipeline through an api on port 5000. 
- the [interface.py](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/interface.py) is a simple `web app` that leverage the previously created api to perform the classification task. It start by default on the port `8501``
- all the systems is containerized into a docker image: [Dockerfile](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/Dockerfile)
- [ANN-Benchmarks-A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.pdf](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/tree/master/Paper) paper help to pick an optimal Approximate Nearest-Neighbor algorithm. For [Link and code: Fair AI Similarity Search], see the [Faiss Repository](https://github.com/facebookresearch/faiss/wiki/Getting-started)

## Improvements

The following improvements can help achieve better performance:

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