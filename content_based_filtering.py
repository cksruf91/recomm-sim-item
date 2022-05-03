import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from config import CONFIG


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', help='검색할 아이템 index', type=int, required=True)
    parser.add_argument('-d', '--dataset', default='1M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    parser.add_argument('-k', '--eval_k', default=10, help='', type=int)

    return parser.parse_args()


def multi_hot_encoding(df, col):
    """멀티 핫 인코딩
    """

    values = set()
    for row in df[col]:
        values.update(row)
    values = {g: i for i, g in enumerate(values)}

    multi_hot_encode = np.zeros([len(df), len(values)])
    for i, v in enumerate(df[col]):
        for it in v:
            multi_hot_encode[i, values[it]] = 1

    return pd.concat(
        [df, pd.DataFrame(multi_hot_encode, columns=list(values))], axis=1
    )


if __name__ == '__main__':
    argument = args()

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)
    item_meta = pd.read_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', low_memory=False)

    item_meta['Genres'] = item_meta['Genres'].map(lambda x: set(x.split('|')))
    item_meta = multi_hot_encoding(item_meta, 'Genres')

    genres = [
        'Children\'s', 'War', 'Drama', 'Musical', 'Western', 'Comedy', 'Romance',
        'Adventure', 'Animation', 'Crime', 'Mystery', 'Fantasy', 'Film-Noir',
        'Action', 'Sci-Fi', 'Thriller', 'Documentary', 'Horror'
    ]
    vectors = item_meta[genres].values.astype(np.float32)

    dist = euclidean_distances(
        vectors[argument.index].reshape(1, -1), vectors
    )
    recommend = np.argsort(dist[0])[:argument.eval_k]

    print(
        pd.concat([item_meta.iloc[r:r + 1] for r in recommend])
    )
