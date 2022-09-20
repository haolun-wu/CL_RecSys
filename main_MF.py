import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
sys.settrace
import logging
import numpy as np
import torch
import time

from helper.sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from helper import read_data_lastfm_time
from argparse import ArgumentParser
from model.MF import MatrixFactorization
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = ArgumentParser(description="MF")
    parser.add_argument("--data_name", type=str, default="lastfm")
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--user_filter', type=int, default=5)
    parser.add_argument('--item_filter', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--is_logging', type=bool, default=False)
    # neighborhood to use
    parser.add_argument('--n_neg', type=int, default=1, help='the number of negative samples')
    # Seed
    parser.add_argument('--seed', type=int, default=3, help="Seed")
    # Model
    parser.add_argument('--dim', type=int, default=64, help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--wd', type=float, default=1e-3, help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs', type=int, default=1000, help="Number of epoch during training")
    parser.add_argument('--every', type=int, default=50,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch_size")
    parser.add_argument('--topk', type=int, default=20, help="topk")
    parser.add_argument('--sample_every', type=int, default=10, help="sample frequency")

    return parser.parse_args()


def neg_item_pre_sampling(train_matrix, num_neg_candidates=500):
    num_users, num_items = train_matrix.shape
    user_neg_items = []
    for user_id in range(num_users):
        pos_items = train_matrix[user_id].indices

        u_neg_item = negsamp_vectorized_bsearch_preverif(pos_items, num_items, num_neg_candidates)
        user_neg_items.append(u_neg_item)

    user_neg_items = np.asarray(user_neg_items)

    return user_neg_items


def generate_pred_list(model, train_matrix, topk=20):
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
        batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(args.device)

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
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg


def train_model(args):
    # pre-sample a small set of negative samples
    t1 = time.time()
    user_neg_items = neg_item_pre_sampling(train_matrix, num_neg_candidates=500)
    pre_samples = {'user_neg_items': user_neg_items}

    print("Pre sampling time:{}".format(time.time() - t1))

    model = MatrixFactorization(user_size, item_size, args).to(args.device)
    optimizer = torch.optim.Adam(model.myparameters, lr=args.lr, weight_decay=args.wd)

    sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=args.n_neg, n_workers=4)
    num_batches = train_matrix.count_nonzero() // args.batch_size

    try:
        for iter in range(1, args.n_epochs + 1):
            start = time.time()
            model.train()

            loss = 0.

            # Start Training
            for batch_id in range(num_batches):
                # get mini-batch data
                batch_user_id, batch_item_id, neg_samples = sampler.next_batch()
                user, pos, neg = batch_user_id, batch_item_id, np.squeeze(neg_samples)

                batch_loss = model(user, pos, neg)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item()

            logger.info("Epochs:{}".format(iter))
            logger.info('[{:.2f}s] Avg BPR loss:{:.6f}'.format(time.time() - start, loss / num_batches))

            if iter % args.every == 0:
                logger.info("Epochs:{}".format(iter))
                model.eval()
                pred_list = generate_pred_list(model, train_matrix, topk=20)
                if args.val_ratio > 0.0:
                    print("validation:")
                    precision, recall, MAP, ndcg = compute_metrics(val_user_list, pred_list, topk=20)
                    logger.info(', '.join(str(e) for e in recall))
                    logger.info(', '.join(str(e) for e in ndcg))

                print("test:")
                precision, recall, MAP, ndcg = compute_metrics(test_user_list, pred_list, topk=20)
                logger.info(', '.join(str(e) for e in recall))
                logger.info(', '.join(str(e) for e in ndcg))

            if iter % args.sample_every == 0:
                user_neg_items = neg_item_pre_sampling(train_matrix, num_neg_candidates=500)
                pre_samples = {'user_neg_items': user_neg_items}
                sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=args.n_neg,
                                     n_workers=4)

        sampler.close()
    except KeyboardInterrupt:
        sampler.close()
        sys.exit()

    logger.info('Parameters:')
    for arg, value in sorted(vars(args).items()):
        logger.info("%s: %r", arg, value)
    logger.info('\n')
    print("Whole time:{}".format(time.time() - t1))


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)
    if args.is_logging is True:
        handler = logging.FileHandler('./log/' + args.data + '.log')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    print(args)

    print("Data name:", args.data_name)
    data_dir = "/Users/haolunwu/Research_project/CL_RecSys/data/"
    if args.data_name == 'lastfm':
        data_generator = read_data_lastfm_time.Data(data_dir, test_ratio=args.test_ratio,
                                                    val_ratio=args.val_ratio, user_filter=args.user_filter,
                                                    item_filter=args.item_filter, seed=args.seed)

        # try first block
        train_matrix, val_matrix, test_matrix = data_generator.train_matrix[0], data_generator.val_matrix[0], \
                                                data_generator.test_matrix[0]
        train_user_list, val_user_list, test_user_list = data_generator.train_set[0], data_generator.val_set[0], \
                                                         data_generator.test_set[0]
        val_user_list = np.array(val_user_list, dtype=list)
        test_user_list = np.array(test_user_list, dtype=list)
        user_size = train_matrix.shape[0]
        item_size = train_matrix.shape[1]

    # len_list = sorted(np.array(list(map(lambda x: len(x), train_user_list))))
    # print("top_max_len:", len_list[-50:])
    # print("top_min_len:", len_list[:50])

    train_model(args)
