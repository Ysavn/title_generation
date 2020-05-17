# title_generation

The aim of the project is to generate title of research papers from their abstract. I have used NIPS and ARXIV dataset available in Kaggle. The architecture of the model used is a Sequence-to-Sequence model [(refer)](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb) involving Encoder-Decoder constructed using 2-Layer LSTM. The model gives a result of **13.08%** on *ROGUE-1* metric. 

**NIPS Dataset** - https://www.kaggle.com/benhamner/nips-papers

**ARXIV Dataset** - https://www.kaggle.com/neelshah18/arxivdataset
