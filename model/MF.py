import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time

epsilon = 1e-4


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(MatrixFactorization, self).__init__()

        self.device = args.device
        self.num_users = num_users
        self.num_items = num_items
        self.dim = args.dim

        self.batch_size = args.batch_size

        # self.use_pretrain = args.use_pretrain

        self.user_embeddings = nn.Embedding(self.num_users, args.dim).to(self.device)
        self.item_embeddings = nn.Embedding(self.num_items, args.dim).to(self.device)
        self.user_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.user_embeddings.weight.data)
        self.item_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_embeddings.weight.data)

        # self.agg_item_embeddings = nn.Embedding(self.num_items, args.dim).to(self.device)
        # self.agg_item_embeddings.weight.requires_grad = False

        self.myparameters = [self.user_embeddings.weight, self.item_embeddings.weight]

    @staticmethod
    def bpr_loss(users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1, keepdim=True)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1, keepdim=True)

        tmp = pos_scores - neg_scores

        bpr_loss = -nn.LogSigmoid()(tmp)
        # mf_loss = -torch.sum(maxi)

        return bpr_loss

    def forward(self, user, pos, neg):
        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb = self.user_embeddings(user_id)
        pos_emb = self.item_embeddings(pos_id)
        neg_emb = self.item_embeddings(neg_id)

        batch_loss = torch.sum(self.bpr_loss(user_emb, pos_emb, neg_emb))

        return batch_loss

    def predict(self, user_id):
        user_emb = self.user_embeddings(user_id)
        pred = user_emb.mm(self.item_embeddings.weight.t())

        return pred
