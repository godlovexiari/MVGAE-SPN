This repository restore the code and datasets for "Multi-head Variational Graph Autoencoder Constrained by Sum-product Networks"

## Requirements
* TensorFlow (1.13.1)
* python 3.6

## Run the demo

```bash
python train.py
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the `load_data()` function in `input_data.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/ and here (in a different format): https://github.com/kimiyoung/planetoid

You can specify a dataset as follows:

```bash
python train.py --dataset citeseer
```

(or by editing `train.py`)


## Cite
