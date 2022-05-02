import argparse
import os
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.optim import Adagrad, Adadelta, Adam
from torch.utils.data import Dataset, DataLoader

from config import CONFIG
from model.item2vec import Item2Vector
from model.metrics import Accuracy
from model.callbacks import MlflowLogger


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='1M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    parser.add_argument('-v', '--model_version', required=True, help='모델 버전', type=str)
    parser.add_argument('-k', '--eval_k', default=10, help='', type=int)
    parser.add_argument('-ed', '--embed_dim', default=16, help='embedding size', type=int)
    parser.add_argument('-op', '--optimizer', default='Adam', choices=['Adam', 'Adagrad', 'Adadelta'],
                        help='optimizer', type=str)
    parser.add_argument('-lr', '--learning_rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('-bs', '--batch_size', default=256, help='batch size', type=int)
    parser.add_argument('-cpu', '--cpu', action='store_true', help='')

    return parser.parse_args()


class Iterator(Dataset):

    def __init__(self, center, context, label, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.center = self._to_tensor(center)
        self.context = self._to_tensor(context)
        self.label = self._to_tensor(label, dtype=torch.float32)

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def __getitem__(self, index):
        return self.center[index], self.context[index], self.label[index]

    def __len__(self):
        return len(self.center)


def get_optimizer(model, name: str, lr: float, wd: float = 0.) -> Callable:
    """ get optimizer
    Args:
        model: pytorch model
        name: optimizer name
        lr: learning rate
        wd: weight_decay(l2 regulraization)

    Returns: pytorch optimizer function
    """

    functions = {
        'Adagrad': Adagrad(model.parameters(), lr=lr, eps=0.00001, weight_decay=wd),
        'Adadelta': Adadelta(model.parameters(), lr=lr, eps=1e-06, weight_decay=wd),
        'Adam': Adam(model.parameters(), lr=lr, weight_decay=wd)
    }
    try:
        return functions[name]
    except KeyError:
        raise ValueError(f'optimizer [{name}] not exist, available optimizer {list(functions.keys())}')


def train_progressbar(total: int, i: int, bar_length: int = 50, prefix: str = '', suffix: str = '') -> None:
    """progressbar
    """
    dot_num = int((i + 1) / total * bar_length)
    dot = '■' * dot_num
    empty = ' ' * (bar_length - dot_num)
    sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% {suffix}')


def train(model, epoch, train_dataloader, val_dataloader, loss_func, optim, metrics=[], callback=[]):
    for e in range(epoch):
        # --------- train ---------
        model.train()

        start_epoch_time = time.time()
        total_step = len(train_dataloader)
        train_loss = 0
        y_pred, y_true = [], []
        history = {}

        for step, (centor, context, label) in enumerate(train_dataloader):

            # ------------------ step start ------------------
            if ((step + 1) % 50 == 0) | (step + 1 >= total_step):
                train_progressbar(
                    total_step, step + 1, bar_length=30,
                    prefix=f'train {e + 1:03d}/{epoch} epoch', suffix=f'{time.time() - start_epoch_time:0.2f} sec '
                )
            pred = model(centor, context)
            loss = loss_func(pred, label)

            pred = (pred > 0.5).int()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.zero_grad()
            train_loss += loss.item()

            y_pred.extend(pred.cpu().tolist())
            y_true.extend(label.cpu().tolist())
            # ------------------ step end ------------------

        for func in metrics:
            metrics_value = func(y_pred, y_true)
            history[f'{func}'] = metrics_value
            # result += f' {func} : {metrics_value:3.3f}'

        history['epoch'] = e + 1
        history['time'] = np.round(time.time() - start_epoch_time, 2)
        history['train_loss'] = train_loss / total_step

        sys.stdout.write(f"loss : {history['train_loss']:3.3f}")

        # ------ test  --------
        model.eval()
        val_loss = 0
        y_pred, y_true = [], []
        with torch.no_grad():
            for step, (centor, context, label) in enumerate(val_dataloader):
                pred = model(centor, context)
                loss = loss_func(pred, label)

                pred = (pred > 0.5).int()
                val_loss += loss.item()

                y_pred.extend(pred.cpu().tolist())
                y_true.extend(label.cpu().tolist())

        history['val_loss'] = val_loss / step

        # ------ logging  --------
        for func in metrics:
            metrics_value = func(y_pred, y_true)
            history[f'val_{func}'] = metrics_value

        for func in callback:
            func(model, history)

        print(
            f" Acc : {history['acc']:3.3f} val_loss : {history['val_loss']:3.3f}, val_Acc : {history['val_acc']:3.3f}")


if __name__ == '__main__':
    argument = args()

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)
    train_data = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t')
    val_data = pd.read_csv(os.path.join(save_dir, 'val.tsv'), sep='\t')
    item_meta = pd.read_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', low_memory=False)
    user_meta = pd.read_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if argument.cpu:
        device = torch.device('cpu')

    train_iterator = Iterator(train_data['center_item'], train_data['context_item'], train_data['label'], device=device)
    train_dataloader = DataLoader(train_iterator, batch_size=argument.batch_size)

    val_iterator = Iterator(val_data['center_item'], val_data['context_item'], val_data['label'], device=device)
    val_dataloader = DataLoader(val_iterator, batch_size=argument.batch_size)

    nitem = max(train_data.context_item.max(), train_data.center_item.max(),
                val_data.context_item.max(), val_data.center_item.max()) + 1

    model_params = {
        'optimizer': argument.optimizer, 'learningRate': argument.learning_rate, 'nitem': nitem,
        'emb_dim': argument.embed_dim
    }

    item_to_vector = Item2Vector(nitem, model_params['emb_dim'], device=device)
    param_size = 0
    for param in item_to_vector.parameters():
        param_size += param.nelement() * param.element_size()
    print(f'model size : {param_size / 1024 / 1024:1.5f} mb')
    print(item_to_vector)

    binary_cross_entropy = torch.nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(item_to_vector, model_params['optimizer'], model_params['learningRate'])
    metrics = [Accuracy()]

    model_version = f'item2vec_v{argument.model_version}'
    callback = [
        # ModelCheckPoint(os.path.join(
        #     'result', argument.dataset,
        #     model_version + '-e{epoch:02d}-loss{val_loss:1.3f}-acc{val_acc:1.3f}.zip'),
        #     monitor='val_acc', mode='max'
        # ),
        MlflowLogger(experiment_name=argument.dataset, model_params=model_params, run_name=model_version,
                     log_model=True, model_name='torch_model', monitor='val_acc', mode='max')
    ]

    train(item_to_vector, 25, train_dataloader, val_dataloader, binary_cross_entropy, optimizer, metrics=metrics,
          callback=callback)
