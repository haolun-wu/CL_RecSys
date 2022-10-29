import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helper.utils import index_B_in_A

import time

epsilon = 1e-4


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, time_block,
                 user_emb_cum, item_emb_cum,
                 user_dict_inter_prev, user_dict_inter,
                 item_dict_inter_prev, item_dict_inter,
                 args):
        super(MatrixFactorization, self).__init__()

        self.device = args.device
        self.num_users = num_users
        self.num_items = num_items
        self.dim = args.dim
        self.batch_size = args.batch_size
        self.time_block = time_block
        self.scaling_self_emb = args.scaling_self_emb

        self.user_emb_cum, self.item_emb_cum = user_emb_cum, item_emb_cum
        self.user_dict_inter_prev, self.user_dict_inter = user_dict_inter_prev, user_dict_inter
        self.item_dict_inter_prev, self.item_dict_inter = item_dict_inter_prev, item_dict_inter

        self.user_embeddings = nn.Embedding(self.num_users, args.dim).to(self.device)
        self.item_embeddings = nn.Embedding(self.num_items, args.dim).to(self.device)
        self.user_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.user_embeddings.weight.data)
        self.item_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_embeddings.weight.data)

        if time_block > 0:  # use previous occurred emb for initialization
            self.user_embeddings.weight.data[list(user_dict_inter.values())] = user_emb_cum[
                list(user_dict_inter_prev.values())]
            self.item_embeddings.weight.data[list(item_dict_inter.values())] = item_emb_cum[
                list(item_dict_inter_prev.values())]
            # pass

        self.myparameters = [self.user_embeddings.weight, self.item_embeddings.weight]

    @staticmethod
    def bpr_loss(users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1, keepdim=True)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1, keepdim=True)

        tmp = pos_scores - neg_scores

        bpr_loss = -nn.LogSigmoid()(tmp)
        # bpr_loss = -nn.Sigmoid()(tmp)
        # bpr_loss = torch.log(bpr_loss)

        return bpr_loss

    def self_emb_distill(self, index_list, emb_curr, emb_cum, dict_inter, dict_inter_prev):
        common_values = np.intersect1d(list(set(index_list)), list(dict_inter.values()))
        if len(common_values) != 0:
            # the position of common users
            position_in_all = index_B_in_A(np.array(list(dict_inter.values())), common_values)
            position_in_batch = index_B_in_A(np.array(list(set(index_list))), common_values)

            common_keys = np.array(list(dict_inter.keys()))[position_in_all]  # find the key (real id)
            common_values_prev = np.array(
                [dict_inter_prev[x] for x in common_keys])  # find the corresponding id in prev recorded cum

            distill_loss = emb_curr[position_in_batch] - emb_cum[common_values_prev]
            distill_loss = torch.sqrt(torch.square(distill_loss).sum())
        else:
            distill_loss = 0

        return distill_loss

    def forward(self, user, pos, neg):
        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb = self.user_embeddings(user_id)
        pos_emb = self.item_embeddings(pos_id)
        neg_emb = self.item_embeddings(neg_id)

        batch_loss = torch.sum(self.bpr_loss(user_emb, pos_emb, neg_emb))

        return batch_loss

    def forward_self_emb_distill(self, user, pos, neg):
        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)  # value in dict
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb = self.user_embeddings(user_id)
        pos_emb = self.item_embeddings(pos_id)
        neg_emb = self.item_embeddings(neg_id)

        distill_user_loss = self.self_emb_distill(user, user_emb, self.user_emb_cum, self.user_dict_inter,
                                                  self.user_dict_inter_prev)
        distill_pos_loss = self.self_emb_distill(pos, pos_emb, self.item_emb_cum, self.item_dict_inter,
                                                 self.item_dict_inter_prev)
        # distill_neg_loss = self.self_emb_distill(neg, neg_emb, self.item_emb_cum, self.item_dict_inter,
        #                                          self.item_dict_inter_prev)

        train_loss = torch.sum(self.bpr_loss(user_emb, pos_emb, neg_emb))
        self_loss = distill_user_loss + distill_pos_loss

        return train_loss, self_loss

    #
    #
    #
    # def distill_self_emb(self, user_emb_cum, user_dict_inter_prev, user_dict_inter, item_emb_cum, item_dict_inter_prev,
    #                      item_dict_inter):

    def predict(self, user_id):
        user_emb = self.user_embeddings(user_id)
        pred = user_emb.mm(self.item_embeddings.weight.t())

        return pred
