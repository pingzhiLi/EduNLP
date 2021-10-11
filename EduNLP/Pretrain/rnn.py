# coding: utf-8
# 2021/8/3 @ tongshiwei
# todo: finish this module

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from baize import iterwrap
from baize.torch import light_module as lm, Configuration
from baize.torch import fit_wrapper

from EduNLP.Vector import RNNModel, Embedding
from EduNLP.Vector.rnn.elmobilm import ElmoBilm
from EduNLP.ModelZoo import Masker
from EduNLP.ModelZoo import ElmoBilm
from EduNLP.Tokenizer import get_tokenizer


def form_batch(batch, indexer: Embedding, masker: Masker):
    batch_idx, batch_len = indexer.indexing(batch, padding=True)
    masked_batch_idx, masked = masker(batch_idx, batch_len)
    return torch.tensor(masked_batch_idx), torch.tensor(batch_idx), torch.tensor(masked)


@iterwrap()
def etl(items, tokenizer, indexer: Embedding, masker: Masker, params: Configuration):
    batch_size = params.batch_size
    batch = []
    for item in tokenizer(items):
        batch.append(item[:20])
        if len(batch) == batch_size:
            yield form_batch(items, indexer, masker)
            batch = []
    if batch:
        yield form_batch(items, indexer, masker)


class MLMRNN(torch.nn.Module):
    def __init__(self, rnn_type, w2v, vector_size, *args, **kwargs):
        super(MLMRNN, self).__init__()
        self.rnn = RNNModel(rnn_type, w2v, vector_size, *args, **kwargs)
        self.pred_net = nn.Linear(vector_size, self.rnn.embedding.vocab_size)

    def __call__(self, x, *args, **kwargs):
        output, _ = self.rnn(x, *args, **kwargs)
        return F.log_softmax(self.pred_net(output), dim=-1)


@fit_wrapper
def fit_f(_net: RNNModel, batch_data, loss_function, *args, **kwargs):
    masked_seq, seq, mask = batch_data
    pred = _net(masked_seq, indexing=False, padding=False)
    return loss_function(pred, seq) * mask


def train_rnn(items, tokenizer, rnn_type, w2v, vector_size, **kwargs):
    tokenizer = get_tokenizer(tokenizer)
    cfg = Configuration(select="rnn(?!.*embedding)")
    if rnn_type == 'elmo':
        mlm_rnn = Elmo(t2i=w2v)
    else:
        mlm_rnn = MLMRNN(rnn_type, w2v, vector_size, freeze_pretrained=True)
    train_data = etl(items, tokenizer, mlm_rnn.rnn.embedding, Masker(min_mask=1), cfg)
    if rnn_type == 'elmo':
        mlm_rnn.train(batched_item_indices=train_data, epochs=3)
    else:
        lm.train(
            mlm_rnn,
            cfg,
            fit_f=fit_f,
            trainer=None,
            loss_function=torch.nn.CrossEntropyLoss(),
            train_data=train_data,
            initial_net=True,
        )


class Elmo(object):
    """
    Examples
    --------
    >>> t2i={'[PAD]':0, 'I':1, 'am':2, 'a':3, 'robot':4}
    >>> model=Elmo(t2i, emb_size=4, hidden_size=16)
    >>> item_indices=[1, 2, 3, 4]
    >>> model.train([item_indices], epochs=1, lr=1e-2)
    >>> model.get_contextual_emb(item_indices, 0).shape
    torch.Size([8])
    """

    def __init__(self, t2i=None, emb_size: int = 512, hidden_size: int = 4096):
        if t2i is None:
            t2i = {}
        self.t2i = t2i
        i2t = {}
        for t in t2i:
            i2t[t2i[t]] = t
        self.i2t = i2t
        self.Bilm = ElmoBilm(len(t2i), emb_size=emb_size, hidden_size=hidden_size, num_layers=2)

    def __call__(self, item):
        self.Bilm.eval()
        representations = self.Bilm.get_weights()
        return representations

    def train(self, batched_item_indices: list, epochs: int = 1, lr: float = 1e-3):
        self.Bilm.train()
        adam = optim.Adam(self.Bilm.parameters(), lr=lr)
        loss_function = nn.BCELoss(reduction='mean')
        vocab_size = len(self.t2i)
        one_hot = np.eye(vocab_size, dtype=float)
        y = torch.tensor([[one_hot[token_idx] for token_idx in item_indices] for item_indices in batched_item_indices])
        for i in range(epochs):
            adam.zero_grad()
            pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.Bilm(batched_item_indices)
            loss = loss_function(pred_forward[:, :-1].double(), y[:, 1:].double()) + loss_function(
                pred_backward[:, :-1].double(),
                torch.flip(y, [1])[:, 1:].double())
            loss.backward()
            adam.step()

    def save_weights(self, path):
        torch.save(self.Bilm.state_dict(), path)
        return path

    def load_weights(self, path):
        self.Bilm.load_state_dict(torch.load(path))
        return path

    def get_contextual_emb(self, item_indices: list, token_idx: int, scale: int = 1):
        # get contextual embedding of a token, given a sentence containing it
        self.Bilm.eval()
        pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.Bilm.forward([item_indices])
        representations = torch.cat((forward_hiddens[0][0][token_idx], backward_hiddens[0][0][token_idx]),
                                    dim=0).unsqueeze(0)
        for i in range(self.Bilm.num_layers):
            representations = torch.cat((representations, torch.cat(
                (forward_hiddens[i + 1][0][token_idx], backward_hiddens[i + 1][0][token_idx]), 0).unsqueeze(
                0)), dim=0)
        return scale * torch.sum(F.softmax(representations, dim=-1), dim=0)
