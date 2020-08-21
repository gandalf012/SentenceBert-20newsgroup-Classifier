# 20 Newsgroups sentence-Bert classifier

API for sklearn 20 Newsgroups classifier with Faiss (Approximate Nearest Neighors library)

## Introduction


## Results



## Full documentation

The following are entry points for documentation:

- the [sbert_tuning.ipynb](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/sbert_tuning.ipynb) contains custom loading class `Fetch20newsLabelDataset`and training pipeline to fune-tuning distilbert models. It also help to benchmark the sBert model on dev and test with `TripletEvaluator`
- [ANN-Benchmarks-A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.pdf](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/tree/master/Paper) paper help to pick an optimal Approximate Nearest-Neighbor algorithm. For [Link and code: Fair AI Similarity Search], see the [Faiss Repository](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- the [prediction pipeline](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/pipeline.py) contains a basic prediction code. To reproduce [benchmark results](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/tree/master/images), please follow [INSTALL](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/INSTALL.md) and run `python pipeline.py`
- the final [flask app](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/app.py) have been wrapped in a [Dockerfile](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/Dockerfile) and refers to [INSTALL](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/INSTALL.md) to predict.

## Authors

- [Arnauld Adjovi](https://github.com/gandalf012)