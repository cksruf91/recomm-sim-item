# recomm-sim-item

## Dataset
* [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
```bash
mkdir datasets/movielens
cd datasets/movielens
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

## data preprocess
```shell
python preprocess.py -d 1M
```

## train model
```shell
python train_item2vec.py -d 1M -v 0.1.0 -k 10 -ed 16 -lr 0.001 -bs 256 -op Adam
```
