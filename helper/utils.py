from helper.sampler import negsamp_vectorized_bsearch_preverif
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
import torch
import numpy as np


def neg_item_pre_sampling(train_matrix, num_neg_candidates=500):
    num_users, num_items = train_matrix.shape
    user_neg_items = []
    for user_id in range(num_users):
        pos_items = train_matrix[user_id].indices

        u_neg_item = negsamp_vectorized_bsearch_preverif(pos_items, num_items, num_neg_candidates)
        user_neg_items.append(u_neg_item)

    user_neg_items = np.asarray(user_neg_items)

    return user_neg_items


def generate_pred_list(model, train_matrix, device, topk=20):
    num_users = train_matrix.shape[0]
    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    pred_list = None

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]
        batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(device)

        rating_pred = model.predict(batch_user_ids)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if batchID == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)

    return pred_list


def compute_metrics(test_set, pred_list, topk=20):
    precision, recall, MAP, ndcg = [], [], [], []
    for k in [5, 10, 15, 20]:
        precision.append(round(precision_at_k(test_set, pred_list, k), 6))
        recall.append(round(recall_at_k(test_set, pred_list, k), 6))
        MAP.append(round(mapk(test_set, pred_list, k), 6))
        ndcg.append(round(ndcg_k(test_set, pred_list, k), 6))

    return precision, recall, MAP, ndcg


def dict_extend(dict1, dict2):  # extend the original dict with new real_ids
    new_node = []
    merged_dict = dict1.copy()
    index = list(dict1.values())[-1] + 1
    for key, values in dict2.items():
        if key not in dict1:
            merged_dict[key] = index
            new_node.append(key)
            index += 1
    return merged_dict, new_node


def separete_intersect_dicts(dict1, dict2):
    # return {common_real_id:index} in dict1, {common_real_id:index} in dict2, rest in dict2
    in_dict1, in_dict2, rest_dict2 = {}, {}, {}
    for key, values in dict2.items():
        if key in dict1:
            in_dict1[key] = dict1[key]
            in_dict2[key] = dict2[key]
        else:
            rest_dict2[key] = dict2[key]
    return in_dict1, in_dict2, rest_dict2


def update_emb_table(emb_cum, emb_curr, dict_inter_prev, dict_inter, dict_rest):
    if emb_cum == None:
        emb_cum = emb_curr
    else:
        emb_cum[list(dict_inter_prev.values())] = emb_curr[list(dict_inter.values())]
        emb_cum = torch.cat((emb_cum, emb_curr[list(dict_rest.values())]), 0)
    return emb_cum


def index_B_in_A(A, B):
    if not isinstance(A, (np.ndarray, np.generic)):
        A = np.array(A)
    sort_idx = A.argsort()
    return sort_idx[np.searchsorted(A, B, sorter=sort_idx)]


def select_by_order_B_in_A(A, B):  # A=[3,2,4,7,6], B=[2,6,7], return [2,7,6]
    if not isinstance(A, list):
        A = A.tolist()

    return [x for _, x in sorted(zip([A.index(x) for x in B], B))]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.eval_flag = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        # self.path_ku = path_ku
        # self.path_ki = path_ki
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.eval_flag = True
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.eval_flag = False
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.eval_flag = True
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation recall@20 increased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


if __name__ == '__main__':
    A = np.array([1, 3, 7, 6, 21, 66])
    # A = np.array([3, 6, 21])
    B = np.array([3, 21, 6])

    print(index_B_in_A(A, B))
    # print(select_by_order(A, B))

    # dict1 = {23: 0, 24: 1, 78: 2, 81: 3}
    # dict2 = {27: 0, 91: 3}
    # print(dict_extend(dict1, dict2))
