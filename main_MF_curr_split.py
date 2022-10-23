import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys

sys.settrace
import logging
import numpy as np
import torch
import time
import pickle

from helper.sampler import NegSampler
from helper.utils import neg_item_pre_sampling, generate_pred_list, compute_metrics, dict_extend, \
    separete_intersect_dicts, update_emb_table, EarlyStopping
from helper.curr_split import read_data_LF, read_data_ML
from helper.eval_metrics import recall_at_k_list
from argparse import ArgumentParser
from model.MF_time import MatrixFactorization

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = ArgumentParser(description="MF")
    parser.add_argument("--data_name", type=str, default="lastfm")
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--user_filter', type=int, default=5)
    parser.add_argument('--item_filter', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--is_logging', type=bool, default=False)
    parser.add_argument('--strategy', type=str, default='self-emb', choices=['pure', 'self-emb', 'ewc', 'ader'])
    # self-emb
    parser.add_argument('--scaling_self_emb', type=float, default=20.0)
    # neighborhood to use
    parser.add_argument('--n_neg', type=int, default=1, help='the number of negative samples')
    # Seed
    parser.add_argument('--seed', type=int, default=20220921, help="Seed")
    # Model
    parser.add_argument('--dim', type=int, default=64, help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--wd', type=float, default=1e-3, help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs', type=int, default=100, help="Number of epoch during training")
    parser.add_argument('--every', type=int, default=5,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--patience', type=int, default=5, help="patience")
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    parser.add_argument('--topk', type=int, default=20, help="topk")
    parser.add_argument('--sample_every', type=int, default=10, help="sample frequency")

    return parser.parse_args()


def train_model_block(args, time_block=0):
    # pre-sample a small set of negative samples
    t1 = time.time()
    user_neg_items = neg_item_pre_sampling(train_matrix, num_neg_candidates=500)
    pre_samples = {'user_neg_items': user_neg_items}

    print("Pre sampling time:{}".format(time.time() - t1))

    model = MatrixFactorization(user_size, item_size, time_block,
                                user_emb_cum, item_emb_cum,
                                user_dict_inter_prev, user_dict_inter,
                                item_dict_inter_prev, item_dict_inter, args).to(args.device)
    optimizer = torch.optim.Adam(model.myparameters, lr=args.lr, weight_decay=args.wd)

    sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=args.n_neg, n_workers=4)
    num_batches = train_matrix.count_nonzero() // args.batch_size

    saved_model_path = './saved/{}/model/curr/state-{}.pt'.format(args.data_name, args.strategy)
    saved_result_path = './saved/{}/result/curr/'.format(args.data_name)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=saved_model_path)

    try:
        for iter in range(1, args.n_epochs + 1):
            start = time.time()
            model.train()

            loss = 0.
            aux_loss = 0.

            # Start Training
            for batch_id in range(num_batches):
                # get mini-batch data
                batch_user_id, batch_item_id, neg_samples = sampler.next_batch()
                user, pos, neg = batch_user_id, batch_item_id, np.squeeze(neg_samples)

                if args.strategy == 'pure':
                    train_loss = model(user, pos, neg)
                    batch_loss = train_loss
                elif args.strategy == 'self-emb':
                    train_loss, self_loss = model.forward_self_emb_distill(user, pos, neg)
                    batch_loss = train_loss + args.scaling_self_emb * self_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += train_loss.item()
                try:
                    aux_loss += self_loss.item()
                except:
                    pass

            # logger.info("Epochs:{}".format(iter))

            # print(model.myparameters[0].grad.sum())
            # print(model.myparameters[1].grad.sum())

            if iter % args.every == 0:
                logger.info('Avg BPR loss:{:.6f}'.format(loss / num_batches))
                logger.info('Avg aux loss:{:.6f}'.format(aux_loss / num_batches))
                # logger.info("Epochs:{}".format(iter))
                model.eval()
                pred_list = generate_pred_list(model, train_matrix, device=args.device, topk=20)
                # if time_block > 0:
                pred_list = [pred_list[i] for i in list(user_common_curr_id.values())]

                # logger.info("validation")
                precision, recall, MAP, ndcg = compute_metrics(val_set, pred_list, topk=20)
                # logger.info(', '.join(str(e) for e in recall))
                # logger.info(', '.join(str(e) for e in ndcg))

                # early_stopping
                early_stopping(recall[-1], model)

                if early_stopping.early_stop:
                    print("Early stopping")

                    break

            if iter % args.sample_every == 0:
                user_neg_items = neg_item_pre_sampling(train_matrix, num_neg_candidates=500)
                pre_samples = {'user_neg_items': user_neg_items}
                sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=args.n_neg,
                                     n_workers=4)

        sampler.close()
    except KeyboardInterrupt:
        sampler.close()
        sys.exit()

    model.load_state_dict(torch.load(saved_model_path))
    print("Test Item recommendation:")
    pred_list = generate_pred_list(model, train_matrix, device=args.device, topk=20)
    # if time_block > 0:
    pred_list = [pred_list[i] for i in list(user_common_curr_id.values())]
    precision, recall, MAP, ndcg = compute_metrics(test_set, pred_list, topk=20)
    logger.info(', '.join(str(e) for e in recall))
    logger.info(', '.join(str(e) for e in ndcg))

    # save individual recall
    keys = list(user_common_next_id.keys())
    values = recall_at_k_list(test_set, pred_list, topk=20)
    individual_recall = {k: v for k, v in zip(keys, values)}
    with open('./saved/{}/result/curr/recall-{}-block-{}.pkl'.format(args.data_name, args.strategy, i), 'wb') as f:
        pickle.dump(individual_recall, f)

    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)

    # logger.info('Parameters:')
    # for arg, value in sorted(vars(args).items()):
    #     logger.info("%s: %r", arg, value)
    # logger.info('\n')
    print("Whole time:{}".format(time.time() - t1))
    return model.user_embeddings.weight.data, model.item_embeddings.weight.data


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'mps')
    print("device:", args.device)
    if args.is_logging is True:
        handler = logging.FileHandler('./log/' + args.data + '.log')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    print(args)

    print("Data name:", args.data_name)
    data_dir = "/Users/haolunwu/Research_project/CL_RecSys/data/"
    # data_dir = "C:/Users/eq22858/Documents/GitHub/CL_RecSys/data/"
    # data_dir = "C:/Users/31093/Documents/GitHub/CL_RecSys/data/"
    if args.data_name == 'lastfm':
        data_generator = read_data_LF.Data(data_dir, test_ratio=args.test_ratio,
                                           val_ratio=args.val_ratio, user_filter=args.user_filter,
                                           item_filter=args.item_filter, seed=args.seed)
    elif args.data_name == 'ml1m':
        data_generator = read_data_ML.Data(data_dir, test_ratio=args.test_ratio,
                                           val_ratio=args.val_ratio, user_filter=args.user_filter,
                                           item_filter=args.item_filter, seed=args.seed)
    # record appeared nodes
    user_dict_cum_prev, item_dict_cum_prev = {}, {}
    user_dict_cum_ext, item_dict_cum_ext = {}, {}
    user_emb_cum, item_emb_cum = None, None

    for i in range(4):
        print("--------------------")
        print("Data Block:{}".format(i))
        # update appeared nodes for Train. all dict: {real_id: index}
        user_dict_cum_prev, item_dict_cum_prev = user_dict_cum_ext, item_dict_cum_ext
        user_dict, item_dict = data_generator.data_full_dict[i][0], data_generator.data_full_dict[i][1]
        user_dict_cum_ext = dict_extend(user_dict_cum_ext, user_dict) if user_dict_cum_ext != {} else user_dict
        item_dict_cum_ext = dict_extend(item_dict_cum_ext, item_dict) if item_dict_cum_ext != {} else item_dict
        # return the {real_index:id} for intersection users in exsiting cumu and current block ---> for replacement
        user_dict_inter_prev, user_dict_inter, user_dict_rest = separete_intersect_dicts(user_dict_cum_prev, user_dict)
        item_dict_inter_prev, item_dict_inter, item_dict_rest = separete_intersect_dicts(item_dict_cum_prev, item_dict)

        # current block: train+val, next block: test
        train_matrix = data_generator.data_full_dict[i][2]
        test_set = data_generator.data_full_dict[i + 1][6]
        val_set = data_generator.data_full_dict[i][5]

        # only use the test_set and val_set on those users in train_matrix
        user_dict_curr = data_generator.data_full_dict[i][0]
        user_dict_next = data_generator.data_full_dict[i + 1][0]

        user_common_curr_id = {}  # common users, the id in the curr block
        user_common_next_id = {}  # common users, the id in the next block
        for key, values in user_dict_next.items():
            if key in user_dict_curr:
                user_common_curr_id[key] = user_dict_curr[key]
                user_common_next_id[key] = values
        test_set = [test_set[i] for i in list(user_common_next_id.values())]
        val_set = [val_set[i] for i in list(user_common_curr_id.values())]

        # print("user_common_curr_id:", list(user_common_curr_id.keys()))
        # print("user_common_next_id:", list(user_common_next_id.keys()))

        test_set, val_set = np.array(test_set, dtype=list), np.array(val_set, dtype=list)
        user_size, item_size = train_matrix.shape[0], train_matrix.shape[1]
        print("user_size:{}, item_size:{}".format(user_size, item_size))
        print("common users for test:{}".format(len(test_set)))

        # Train model and update embedding table
        user_emb_curr, item_emb_curr = train_model_block(args, time_block=i)
        user_emb_cum = update_emb_table(user_emb_cum, user_emb_curr, user_dict_inter_prev, user_dict_inter,
                                        user_dict_rest)
        item_emb_cum = update_emb_table(item_emb_cum, item_emb_curr, item_dict_inter_prev, item_dict_inter,
                                        item_dict_rest)

        print("user_emb_curr:", user_emb_curr.shape)
        print("item_emb_curr:", item_emb_curr.shape)
        print("user_emb_cum:", user_emb_cum.shape)
        print("item_emb_cum:", item_emb_cum.shape)
