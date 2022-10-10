import sys
import os

import pandas

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle as pkl
import pandas as pd
import torch
from copy import deepcopy

[sys.path.append(i) for i in ['.', '..']]
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
        df = df.dropna(axis=0, how='any')
        df = df[df['rate'] > 3]
        df = df.drop_duplicates()
        df = df.reset_index().drop(['index'], axis=1)

        return df


class Data(object):
    def __init__(self, data_dir, val_ratio=None, test_ratio=0.2, user_filter=5, item_filter=5, seed=0):
        data_name = 'ml-1m'
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.user_filter = user_filter
        self.item_filter = item_filter
        origin_data_path = os.path.join(data_dir, "{}/ratings.dat".format(data_name))
        pro_data_path = os.path.join(data_dir, "preprocessed/{}/overall.csv".format(data_name))
        pro_data_block_path = os.path.join(data_dir, "preprocessed/{}/data_0.csv".format(data_name))

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
            print('all_user:{}, all_item:{}, all_interac:{}'.format(6038, 3533, 575281))

        if not os.path.exists(pro_data_block_path):
            num_interact = 575281
            data_0 = df.loc[0:int(num_interact * 0.6)].reset_index().drop(['index'], axis=1)
            data_1 = df.loc[int(num_interact * 0.6):int(num_interact * 0.7)].reset_index().drop(['index'], axis=1)
            data_2 = df.loc[int(num_interact * 0.7):int(num_interact * 0.8)].reset_index().drop(['index'], axis=1)
            data_3 = df.loc[int(num_interact * 0.8):int(num_interact * 0.9)].reset_index().drop(['index'], axis=1)
            data_4 = df.loc[int(num_interact * 0.9):].reset_index().drop(['index'], axis=1)
            data_0.to_csv(os.path.join(data_dir, "preprocessed/{}/data_0.csv".format(data_name)))
            data_1.to_csv(os.path.join(data_dir, "preprocessed/{}/data_1.csv".format(data_name)))
            data_2.to_csv(os.path.join(data_dir, "preprocessed/{}/data_2.csv".format(data_name)))
            data_3.to_csv(os.path.join(data_dir, "preprocessed/{}/data_3.csv".format(data_name)))
            data_4.to_csv(os.path.join(data_dir, "preprocessed/{}/data_4.csv".format(data_name)))
        else:
            data_0 = pandas.read_csv(os.path.join(data_dir, "preprocessed/{}/data_0.csv".format(data_name)))
            data_1 = pandas.read_csv(os.path.join(data_dir, "preprocessed/{}/data_1.csv".format(data_name)))
            data_2 = pandas.read_csv(os.path.join(data_dir, "preprocessed/{}/data_2.csv".format(data_name)))
            data_3 = pandas.read_csv(os.path.join(data_dir, "preprocessed/{}/data_3.csv".format(data_name)))
            data_4 = pandas.read_csv(os.path.join(data_dir, "preprocessed/{}/data_4.csv".format(data_name)))

        data_full = [data_0, data_1, data_2, data_3, data_4]

        self.data_full_dict = {}
        for i in range(len(data_full)):
            self.data_full_dict[i] = self.preprocess_data_block(data_full[i], time_block=i)

        user0 = list(self.data_full_dict[0][0].keys())
        user1 = list(self.data_full_dict[1][0].keys())
        user2 = list(self.data_full_dict[2][0].keys())
        user3 = list(self.data_full_dict[3][0].keys())
        user4 = list(self.data_full_dict[4][0].keys())
        print("common:", len(list(set(user0) & set(user1))))
        print("common:", len(list(set(user1) & set(user2))))
        print("common:", len(list(set(user2) & set(user3))))
        print("common:", len(list(set(user3) & set(user4))))

    def preprocess_data_block(self, data_block, time_block):
        data_block = data_block[["user", "item", "rate"]].drop_duplicates()
        data_block = self.remove_infrequent_users(data_block, self.user_filter)
        data_block, user_dict = self.convert_unique_idx(data_block, "user")
        data_block, item_dict = self.convert_unique_idx(data_block, "item")
        print("# user:{}, # item:{}, # interact:{}".format(data_block['user'].max() + 1,
                                                           data_block['item'].max() + 1, data_block.shape[0]))
        UI_matrix = scipy.sparse.csr_matrix(
            (np.array(data_block['rate']), (np.array(data_block['user']), np.array(data_block['item']))))

        if time_block == 0:
            if self.val_ratio:
                self.val_ratio = self.val_ratio / (1 - self.test_ratio)
        else:
            if self.val_ratio:
                self.val_ratio = self.val_ratio / (1 - self.test_ratio)
            # self.val_ratio = 0
            # self.test_ratio = 0.5

        train_matrix, test_matrix, val_matrix, train_set, test_set, val_set \
            = self.create_train_test_split(UI_matrix, test_ratio=self.test_ratio, val_ratio=self.val_ratio, seed=0)

        # self.user_dict.append(user_dict)
        # self.item_dict.append(item_dict)
        # self.train_matrix.append(train_matrix)
        # self.test_matrix.append(test_matrix)
        # self.val_matrix.append(val_matrix)
        # self.train_set.append(train_set)
        # self.test_set.append(test_set)
        # self.val_set.append(val_set)

        return user_dict, item_dict, train_matrix, test_matrix, val_matrix, train_set, test_set, val_set

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

        print("items with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def remove_infrequent_users(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['user'].value_counts()
        df = df[df['user'].isin(counts[counts >= min_counts].index)]

        print("users with < {} interactoins are removed".format(min_counts))
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
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--user_filter', type=int, default=5)
    parser.add_argument('--item_filter', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    parser = parser_args()
    # data_dir = "/Users/haolunwu/Research_project/CL_RecSys/data/"
    data_dir = "C:/Users/eq22858/Documents/GitHub/CL_RecSys/data/"
    data_generator = Data(data_dir, test_ratio=parser.test_ratio, val_ratio=parser.val_ratio,
                          user_filter=parser.user_filter, item_filter=parser.item_filter)
