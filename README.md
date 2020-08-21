# 20 Newsgroups sentence-Bert classifier

API for sklearn 20 Newsgroups classifier with Faiss (Approximate Nearest Neighors library)

## Introduction
This repository fine-tune DistilBERT models on scikit-learn 20 News groups dataset with a triplet network structure to produce semantically meaningful sentence embeddings that can be used in supervised scenarios: Semantic textual similarity via Facebook Faiss (Approximate Nearest-Neighors library) and label predictions.

We fine-tune a pretrained `distilbert-base-nli-mean-tokens` with a TripletLoss function for 20 Newsgroups labels prediction.
We choose facebook `Faiss-IVF`library for semantic search because of his:
- fast search time (good Recall-Queries per second)
- good accuracy
- low memory usage per index vector
- fast index building time

The final fine-tuned model is available on [Google Drive](https://drive.google.com/file/d/1VjYGZasx9sEuJ2u9DCirb8L2wdIYIcsM/view?usp=sharing)

## Results
Our implementation achieved 60.45 of accuracy for 20 Newsgroup test set, while the pretrained model achieved 44.03 on the same test set.

![pipeline benchmark on test set for Faiss and pretrained Sbert](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/images/pipeline_bench.png)

## Full documentation

The following are entry points for documentation:

- the [sbert_tuning.ipynb](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/sbert_tuning.ipynb) contains custom loading class `Fetch20newsLabelDataset`and training pipeline to fune-tuning distilbert models. It also help to benchmark the sBert model on dev and test with `TripletEvaluator`
- [ANN-Benchmarks-A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.pdf](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/tree/master/Paper) paper help to pick an optimal Approximate Nearest-Neighbor algorithm. For [Link and code: Fair AI Similarity Search], see the [Faiss Repository](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- the [prediction pipeline](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/pipeline.py) contains a basic prediction code. To reproduce [benchmark results](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/tree/master/images), please follow [INSTALL](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/INSTALL.md) and run `python pipeline.py`
- the final [flask app](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/app.py) have been wrapped in a [Dockerfile](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/Dockerfile) and refers to [INSTALL](https://github.com/gandalf012/SentenceBert-20newsgroup-Classifier/blob/master/INSTALL.md) to predict.

## Improvement

The following improvements can help to improve systems global performance:

**Sentence-Bert embedding**:

**Approximate Nearest-Neighbors library**:

**Final Docker image**:


## Authors

Built by [Arnauld Adjovi](https://github.com/gandalf012)

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