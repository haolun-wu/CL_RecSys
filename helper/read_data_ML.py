import sys
import os

import pandas

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle as pkl
import pandas as pd
import torch
from copy import deepcopy
import time

# [sys.path.append(i) for i in ['.', '..']]
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import pickle


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


def data_statistics(data_list):
    for data in data_list:
        # print("------------")
        print("{}:".format(data[0]))
        print("# interact:{}, # user:{}, # item:{}".format(data[1].shape[0], len(data[1]['user'].unique()),
                                                           len(data[1]['item'].unique())))
        # min_counts = 5
        # counts = data[1]['user'].value_counts()
        # print("# user (interaction > 5):{}".format(counts[counts >= min_counts].index.shape[0]))


class ml1m(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = data_dir

    def load(self):
        # Load data
        df = pd.read_csv(self.fpath,
                         sep='::',
                         engine='python',
                         names=['user', 'item', 'rate', 'time'])
        # https://stackoverflow.com/questions/70149227/split-movielen-100k-dataset-based-on-timestamp
        df['real-time'] = df['time'].apply(lambda x: time.localtime(x)).apply(lambda x: time.strftime("%Y-%m", x))
        df = df.dropna(axis=0, how='any')
        df = df[df['rate'] > 3]
        df = df.drop_duplicates()
        df = df.reset_index().drop(['index'], axis=1)

        return df


class Data(object):
    def __init__(self, data_dir, data_name='ml-1m', val_ratio=None, test_ratio=0.2, user_filter=5, item_filter=5,
                 exact_time=True, seed=0):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.user_filter = user_filter
        self.item_filter = item_filter
        origin_data_path = os.path.join(data_dir, "{}/ratings.dat".format(data_name))
        pro_data_path = os.path.join(data_dir, "preprocessed/{}/overall.csv".format(data_name))
        if exact_time:
            pro_data_block_path = os.path.join(data_dir, "preprocessed/{}/data_t0.csv".format(data_name))
        else:
            pro_data_block_path = os.path.join(data_dir, "preprocessed/{}/data_0.csv".format(data_name))

        # df = ml1m(origin_data_path).load()
        # df = df.sort_values(by=['real-time'])
        # print("df:", df)
        #
        # sys.exit()

        if not os.path.exists(pro_data_path):
            df = ml1m(origin_data_path).load()

            df['rate'] = 1

            print("start generate unique idx")
            df['user'] = df['user'].astype('category').cat.codes
            df['item'] = df['item'].astype('category').cat.codes
            df['time'] = df['time'].astype('category').cat.codes

            df = df.sort_values(by=['time'])
            df = df.reset_index().drop(['index'], axis=1)

            df.to_csv(pro_data_path)
            self.num_user = len(df['user'].unique())
            self.num_item = len(df['item'].unique())
            self.num_interac = df.shape[0]

            print('all_user:{}, all_item:{}, all_interac:{}'.format(self.num_user, self.num_item, self.num_interac))
        else:
            df = pandas.read_csv(pro_data_path)
            if data_name == 'ml-1m':
                statistics = [6038, 3533, 575281]
            elif data_name == 'ml-10m':
                statistics = [69816, 10472, 5885448]
            print('all_user:{}, all_item:{}, all_interac:{}'.format(statistics[0], statistics[1], statistics[2]))

        if exact_time:
            data_full = self.preprocess_data_exact_time(df, data_dir, data_name, pro_data_block_path)
        else:
            data_full = self.preprocess_data_not_exact_time(df, data_dir, data_name, pro_data_block_path)

        self.data_full_dict = {}
        length = len(data_full)
        for i in range(length):
            self.data_full_dict[i] = self.preprocess_data_block(data_full[i], time_block=i)

        for i in range(1, length):
            user_prev = list(self.data_full_dict[i - 1][0].keys())
            user_now = list(self.data_full_dict[i][0].keys())
            print("block {}-{}, #common users:{}".format(i - 1, i, len(list(set(user_prev) & set(user_now)))))

    def preprocess_data_exact_time(self, df, data_dir, data_name, pro_data_block_path):

        if not os.path.exists(pro_data_block_path):
            data_full = []
            time_split = ['2000-04', '2001-01', '2001-04', '2001-07', '2001-10',
                          '2002-01', '2002-04', '2002-07', '2002-10', '2003-01']

            # data_t0 = df[df['real-time'] < time_split[0]]
            # print("data_t0:", data_t0)

            for i in range(9):
                tmp = df[time_split[i] <= df['real-time']]
                tmp = tmp[tmp['real-time'] < time_split[i + 1]].reset_index().drop(['index'], axis=1)
                # print("block:{}, len:{}".format(i, len(tmp)))
                tmp.to_csv(os.path.join(data_dir, "preprocessed/{}/data_t{}.csv".format(data_name, i)))

            # data_t0 = df.loc[0:int(num_interact * 0.6)].reset_index().drop(['index'], axis=1)
            # data_1 = df.loc[int(num_interact * 0.6):int(num_interact * 0.7)].reset_index().drop(['index'], axis=1)
            # data_2 = df.loc[int(num_interact * 0.7):int(num_interact * 0.8)].reset_index().drop(['index'], axis=1)
            # data_3 = df.loc[int(num_interact * 0.8):int(num_interact * 0.9)].reset_index().drop(['index'], axis=1)
            # data_4 = df.loc[int(num_interact * 0.9):].reset_index().drop(['index'], axis=1)
            # data_full = [data_0, data_1, data_2, data_3, data_4]
            # for i in range(5):
            #     data_full[i].to_csv(os.path.join(data_dir, "preprocessed/{}/data_{}.csv".format(data_name, i)))
        else:
            data_full = []
            for i in range(9):
                data_full.append(
                    pandas.read_csv(os.path.join(data_dir, "preprocessed/{}/data_t{}.csv".format(data_name, i))))

        return data_full

    def preprocess_data_not_exact_time(self, df, data_dir, data_name, pro_data_block_path):
        if not os.path.exists(pro_data_block_path):
            if data_name == 'ml-1m':
                num_interact = 575281
            data_0 = df.loc[0:int(num_interact * 0.6)].reset_index().drop(['index'], axis=1)
            data_1 = df.loc[int(num_interact * 0.6):int(num_interact * 0.7)].reset_index().drop(['index'], axis=1)
            data_2 = df.loc[int(num_interact * 0.7):int(num_interact * 0.8)].reset_index().drop(['index'], axis=1)
            data_3 = df.loc[int(num_interact * 0.8):int(num_interact * 0.9)].reset_index().drop(['index'], axis=1)
            data_4 = df.loc[int(num_interact * 0.9):].reset_index().drop(['index'], axis=1)
            data_full = [data_0, data_1, data_2, data_3, data_4]
            for i in range(5):
                data_full[i].to_csv(os.path.join(data_dir, "preprocessed/{}/data_{}.csv".format(data_name, i)))
        else:
            data_full = []
            for i in range(5):
                data_full.append(
                    pandas.read_csv(os.path.join(data_dir, "preprocessed/{}/data_{}.csv".format(data_name, i))))

        return data_full

    def preprocess_data_block(self, data_block, time_block):
        data_block = data_block[["user", "item", "rate"]].drop_duplicates()
        if time_block == 0:
            user_filter, item_filter = 5, 5
        else:
            user_filter, item_filter = 3, 5
        data_block = self.remove_infrequent_users(data_block, user_filter)
        data_block = self.remove_infrequent_items(data_block, item_filter)
        data_block = self.remove_infrequent_users(data_block, user_filter)
        data_block, user_dict = self.convert_unique_idx(data_block, "user")
        data_block, item_dict = self.convert_unique_idx(data_block, "item")
        num_user = data_block['user'].max() + 1
        num_item = data_block['item'].max() + 1
        print("# user:{}, # item:{}, # interact:{}".format(num_user, num_item, data_block.shape[0]))
        UI_matrix = scipy.sparse.csr_matrix(
            (np.array(data_block['rate']), (np.array(data_block['user']), np.array(data_block['item']))))

        self.val_ratio = 0
        self.test_ratio = 0.2

        train_matrix, test_matrix, val_matrix, train_set, test_set, val_set \
            = self.create_train_test_split(UI_matrix, test_ratio=self.test_ratio, val_ratio=self.val_ratio, seed=0)

        all_set = [i + j for i, j in zip(train_set, test_set)]

        # print("# train:{}, # val:{}, # test:{}".format(np.sum([len(x) for x in train_set]),
        #                                                np.sum([len(x) for x in val_set]),
        #                                                np.sum([len(x) for x in test_set])))
        # density = float(
        #     np.sum([len(x) for x in train_set]) + np.sum([len(x) for x in val_set]) + np.sum(
        #         [len(x) for x in test_set])) / num_user / num_item
        # print("density:{:.2%}".format(density))

        return user_dict, item_dict, train_matrix, test_matrix, train_set, test_set, all_set

    def generate_inverse_mapping(self, data_list):
        ds_matrix_mapping = dict()
        for inner_id, true_id in enumerate(data_list):
            ds_matrix_mapping[true_id] = inner_id
        return ds_matrix_mapping

    def split_data_randomly(self, user_records, val_ratio, test_ratio, seed=0):
        train_set = []
        test_set = []
        val_set = []
        for user_id, item_list in enumerate(user_records):
            tmp_train_sample, tmp_test_sample = train_test_split(item_list, test_size=test_ratio, random_state=seed)

            if val_ratio:
                tmp_train_sample, tmp_val_sample = train_test_split(tmp_train_sample, test_size=val_ratio,
                                                                    random_state=seed)

            if val_ratio:
                train_sample = []
                for place in item_list:
                    if place not in tmp_test_sample and place not in tmp_val_sample:
                        train_sample.append(place)

                val_sample = []
                for place in item_list:
                    if place not in tmp_test_sample and place not in tmp_train_sample:
                        val_sample.append(place)

                test_sample = []
                for place in tmp_test_sample:
                    if place not in tmp_train_sample and place not in tmp_val_sample:
                        test_sample.append(place)

                train_set.append(train_sample)
                val_set.append(val_sample)
                test_set.append(test_sample)

            else:
                train_sample = []
                for place in item_list:
                    if place not in tmp_test_sample:
                        train_sample.append(place)

                test_sample = []
                for place in tmp_test_sample:
                    if place not in tmp_train_sample:
                        test_sample.append(place)

                train_set.append(train_sample)
                test_set.append(test_sample)

        return train_set, test_set, val_set

    def create_train_test_split(self, rating_df, val_ratio=None, test_ratio=0.1, seed=0):
        data_set = []
        for item_ids in rating_df:
            data_set.append(item_ids.indices.tolist())
        # data_set = self.data_set

        train_set, test_set, val_set = self.split_data_randomly(data_set, val_ratio=val_ratio, test_ratio=test_ratio,
                                                                seed=seed)
        train_matrix = self.generate_rating_matrix(train_set, rating_df.shape[0], rating_df.shape[1])
        test_matrix = self.generate_rating_matrix(test_set, rating_df.shape[0], rating_df.shape[1])
        val_matrix = self.generate_rating_matrix(val_set, rating_df.shape[0], rating_df.shape[1])

        return train_matrix, test_matrix, val_matrix, train_set, test_set, val_set

    def generate_rating_matrix(self, train_matrix, num_users, num_items):
        # three lists are used to construct sparse matrix
        row = []
        col = []
        data = []
        # triplet = []
        for user_id, article_list in enumerate(train_matrix):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)
                # triplet.append((int(user_id), int(article), float(1)))

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

        return rating_matrix

    def convert_to_inner_index(self, user_records, user_mapping, item_mapping):
        inner_user_records = []
        user_inverse_mapping = self.generate_inverse_mapping(user_mapping)
        item_inverse_mapping = self.generate_inverse_mapping(item_mapping)

        for user_id in range(len(user_mapping)):
            real_user_id = user_mapping[user_id]
            item_list = list(user_records[real_user_id])
            for index, real_item_id in enumerate(item_list):
                item_list[index] = item_inverse_mapping[real_item_id]
            inner_user_records.append(item_list)

        return inner_user_records, user_inverse_mapping, item_inverse_mapping

    def convert_to_lists(self, user_records):
        inner_user_records = []
        user_records = user_records.A

        for user_id in range(user_records.shape[0]):
            item_list = np.where(user_records[user_id] != 0)[0].tolist()
            inner_user_records.append(item_list)

        return inner_user_records

    def remove_infrequent_items(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['item'].value_counts()
        df = df[df['item'].isin(counts[counts >= min_counts].index)]

        # print("items with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def remove_infrequent_users(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['user'].value_counts()
        df = df[df['user'].isin(counts[counts >= min_counts].index)]

        # print("users with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def remove_infrequent_tags(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['tag'].value_counts()
        df = df[df['tag'].isin(counts[counts >= min_counts].index)]

        print("tags with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def convert_unique_idx(self, df, column_name):
        column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
        df[column_name] = df[column_name].apply(column_dict.get)
        df[column_name] = df[column_name].astype('int')
        assert df[column_name].min() == 0
        assert df[column_name].max() == len(column_dict) - 1
        return df, column_dict


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='ml-10m')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--user_filter', type=int, default=5)
    parser.add_argument('--item_filter', type=int, default=5)
    parser.add_argument('--exact_time', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    parser = parser_args()
    data_dir = "/data/"
    # data_dir = "/data/"
    data_generator = Data(data_dir, data_name=parser.data_name, test_ratio=parser.test_ratio,
                          val_ratio=parser.val_ratio,
                          user_filter=parser.user_filter, item_filter=parser.item_filter, exact_time=parser.exact_time)
