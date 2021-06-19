
# Simple KGCN and GraphSAINT for Recommendation System


Dang Lam San - Thesis 2021.

## Video demo for the training phase

[Video Demo](https://husteduvn-my.sharepoint.com/:v:/g/personal/san_dl170111_sis_hust_edu_vn/EcdeT-qPes1Nj3y_6oT-bvQB6QfjUsKKBUXmL_vOwh_5XA?e=iD88Tn).


## Step for install
### Install requirements
`pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html`
### Download data and knowledge graph from [drive](https://husteduvn-my.sharepoint.com/:u:/g/personal/san_dl170111_sis_hust_edu_vn/EY1P9_kezmBHrfHXPjS7p0gBGTFpYmO5A49pqjj9bgKDnw?e=roDqje).

Extract `data.zip` to folder `data` \
`unzip ./data.zip`

### Download movielens-20M and extract.
`wget http://files.grouplens.org/datasets/movielens/ml-20m.zip` \
`unzip ml-20m.zip` \
`mv ml-20m/ratings.csv KGraph/data/movie/`

### Preprocess data
`cd graphsaint/kgraphsaint` \
`python preprocess.py -d music ` \
`python preprocess.py -d movie` \
`cd ../../` \

### Building C++ module for Sampling
`python graphsaint/setup.py build_ext --inplace`

## Training Phase
`python -m graphsaint.kgraphsaint.train --lr 1e-3 --sampler node --l2_weight 1e-5`

## Experimental results
| Model        | MovieLens – 20M         |       MovieLens – 20M              |         Last.FM                |    Last.FM                         |
|--------------|-------------------------|-------------------------|-------------------------|-------------------------|
|              | AUC                     | F1                      | AUC                     | F1                      |
| KGCN         | 0.975                   | 0.928                   | 0.791                   | 0.710                   |
| SKGCN – Node | <strong>0.977</strong>  |  <strong>0.931</strong> | 0.808                   | 0.725                   |
| SKGCN – Edge |  <strong>0.977</strong> |  <strong>0.977</strong> | <strong>0.811</strong> |  <strong>0.731</strong> |
| SKGCN – Rw   | 0.976                   | 0.929                   | 0.810                   | 0.731                   |
