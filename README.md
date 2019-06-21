# Polysemous-Network-Embedding

This project provides an implementation for the [paper](https://arxiv.org/abs/1905.10668): <br>

> **Is a Single Vector Enough? Exploring Node Polysemy for Network Embedding**<br>
Ninghao Liu, Qiaoyu Tan, Yuening Li, Hongxia Yang, Jingren Zhou, Xia Hu<br>
KDD 2019 <br>

![](https://github.com/ninghaohello/Polysemous-Network-Embedding/blob/master/PolyDeepwalk.png)

Current implementation contains of two models:<br>
**PolyDeepwalk**: Extends the Deepwalk model to handle different node aspects for homogeneous networks.<br>
**PolyPTE**: Extends the PTE model to handle different node aspects for hetergeneous networks (bipartite networks in this work).<br>


### Files in the folder
- `data/`
  - `BlogCatelog/`
    - `training.mat`: training data samples in the form of (row, col, val);
    - `testing.pkl`: testing data;
    - `snmf6.mat`: pre-obtained symmetric NMF clustering result on training data graph, following the model in https://github.com/dakuang/symnmf;
  - `movielens/`
    - `training.mat`: training data samples in the form of (row, col, val);
    - `testing.mat`: testing data samples;
    - `sparsemat.npz`: the sparse matrix verion of the training data;
- `src/`: implementations of polysemous embedding models.
