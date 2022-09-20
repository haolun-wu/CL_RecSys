import sys
import os
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


def load_pickle(path, name):
    with open(path + name, 'rb') as f:
        return pkl.load(f, encoding='latin1')


# def remove_infrequent_items(data, min_user, min_item):
#     df = deepcopy(data)
#
#     counts = df['user_id'].value_counts()
#     df = df[df["user_id"].isin(counts[counts >= min_user].index)]
#     print("users with < {} interactions are removed".format(min_user))
#
#     counts = df['item_id'].value_counts()
#     df = df[dremove_infrequent_itemsf['item_id'].isin(counts[counts >= min_item].index)]
#     print('items with < {} interactions are removed'.format(min_item))
#
#     return df


def similarity(matrix):
    sim_u = cosine_similarity(matrix, dense_output=False)
    sim_i = cosine_similarity(matrix.T, dense_output=False)

    return sim_u, sim_i


class Data(object):

    def __init__(self, path_list, val_ratio=None, test_ratio=0.2, seed=0, small=False):

        # input file
        UI_path = path_list[0]  # userID-itemID                 len: user_num
        TI_path = path_list[1]  # tag-item          len: item_num
        tag_name_path = path_list[2]  # all tags                       len: tag_num
        item_content_path = path_list[3]  # bag of words for each article  len: item_len

        user_number = 0
        item_number = 0
        tag_number = 0

        tags_name = []
        df = pd.read_csv(tag_name_path, header=None, sep=',')
        for number, tag in enumerate(df.values):
            tags_name.append(tag)

        statistic_dict = {}

        """
        New tag id and name
        """
        tags = []
        number = []
        tag_id_dict = {}
        tag_id_name_dict = {}
        item_id_dict = {}
        id_item_dict = {}

        df = pd.read_csv(TI_path, header=None, sep=',')
        for tag, line in enumerate(df.values):
            tags.append(tag)
            number.append(int(line[0].split(' ')[0]))

        # print("Use the top 500 most frequent tags.")
        if small == True:
            index = np.argsort(number)[:-501:-1]  # choose the most frequent 500 tags
        else:
            index = np.argsort(number)[:-5001:-1]  # choose the most frequent 5000 tags
        # index = np.argsort(number)
        name_tags_filtered = np.array(tags)[index]  # original tag id lists, len: 500
        for t in name_tags_filtered:
            tag_id_dict[int(t)] = tag_number
            tag_number += 1
        # pickle.dump(tag_id_dict, open('tag_id_dict', 'wb'))
        # print("tag_id_dict:", tag_id_dict)
        for key in tag_id_dict.keys():
            tag_id_name_dict[tag_id_dict[key]] = tags_name[key][0]
        # pickle.dump(tag_id_name_dict, open('tag_id_name_dict', 'wb'))
        # print("tag_id_name_dict:", tag_id_name_dict)

        """
        New item id
        """
        item_tags_dict_tmp = dict()
        name_items_filtered = []
        df = pd.read_csv(TI_path, header=None, sep=',')
        for tag, line in enumerate(df.values):
            if tag in name_tags_filtered:
                items = line[0].split(' ')[1:]
                items = [int(i) for i in items]
                for i in items:
                    if i not in item_tags_dict_tmp.keys():
                        item_tags_dict_tmp[i] = [tag]
                    else:
                        item_tags_dict_tmp[i].append(tag)
        item_tags_dict = dict()  # 原始item: 原始tag
        for item, tags in item_tags_dict_tmp.items():
            if len(tags) > 4:
                item_tags_dict[item] = tags
                name_items_filtered.append(item)
        for t in name_items_filtered:
            item_id_dict[int(t)] = item_number
            id_item_dict[item_number] = int(t)
            item_number += 1
        pickle.dump(item_id_dict, open('item_id_dict', 'wb'))
        # print("item_id_dict:", item_id_dict, len(item_id_dict))
        # print("item_tags_dict:", item_tags_dict, len(item_tags_dict))

        """
        New user id
        """
        user_id_dict = {}
        name_users_filtered = []
        df = pd.read_csv(UI_path, header=None, sep=',')
        for user, line in enumerate(df.values):
            items = line[0].split(' ')[1:]
            items = [int(i) for i in items]
            cross = [i for i in items if i in name_items_filtered]
            if len(cross) > 4:
                name_users_filtered.append(user)
        for t in name_users_filtered:
            user_id_dict[int(t)] = user_number
            user_number += 1
        pickle.dump(user_id_dict, open('user_id_dict', 'wb'))
        # print("user_id_dict:", user_id_dict, len(user_id_dict))

        self.num_user = user_number
        self.num_item = item_number
        self.num_tag = tag_number
        print('num_user', self.num_user)
        print('num_item', self.num_item)
        print('num_tag', self.num_tag)

        """
        Filtered UI matrix
        """
        df = pd.read_csv(UI_path, header=None, sep=',').reset_index()
        df.columns = ['user', 'item']
        df['item'] = df['item'].map(lambda x: x.split(' '))
        df = df.explode('item')
        df['item'] = df['item'].astype(int)

        # filter userID and itemID
        df = df[df['user'].isin(user_id_dict.keys()) & df['item'].isin(item_id_dict.keys())]
        # map original index to new id
        df['user'] = df['user'].map(user_id_dict)
        df['item'] = df['item'].map(item_id_dict)
        df['rate'] = 1
        UI_df = df.reset_index(drop=True)

        # print("UI matrix:\n", UI_df)
        self.UI_matrix = scipy.sparse.csr_matrix(
            (np.array(UI_df['rate']), (np.array(UI_df['user']), np.array(UI_df['item']))))

        # print("UI_matrix:", self.UI_matrix)

        """
        Filtered IT matrix
        """
        df = pd.read_csv(TI_path, header=None, sep=',').reset_index()
        df.columns = ['tag', 'item']
        df['item'] = df['item'].map(lambda x: x.split(' '))
        df = df.explode('item')
        df['item'] = df['item'].astype(int)
        # change order of column
        df = df[['item', 'tag']]

        # filter tagID and itemID
        df = df[df['tag'].isin(tag_id_dict.keys()) & df['item'].isin(item_id_dict.keys())]
        # map original index to new ud
        df['tag'] = df['tag'].map(tag_id_dict)
        df['item'] = df['item'].map(item_id_dict)
        df['rate'] = 1
        IT_df = df.reset_index(drop=True)


        # tag_count.columns = ['item', 'tag_num']
        # print("tag_count:", tag_count['item'].to_list())
        # target_item =
        # print("number of items < 10 tags:", (np.array(tag_count)<10).sum())

        self.IT_matrix = scipy.sparse.csr_matrix(
            (np.array(IT_df['rate']), (np.array(IT_df['item']), np.array(IT_df['tag']))))

        # print("IT matrix:\n", IT_df)
        # print(len(set(list(IT_df['tag']))))
        # print(len(set(list(IT_df['item']))))

        if val_ratio:
            val_ratio = val_ratio / (1 - test_ratio)

        self.train_matrix_UI, self.test_matrix_UI, self.val_matrix_UI, \
        self.train_set_UI, self.test_set_UI, self.val_set_UI = self.create_train_test_split(self.UI_matrix,
                                                                                            test_ratio=test_ratio,
                                                                                            val_ratio=val_ratio, seed=0)

        self.train_matrix_IT, self.test_matrix_IT, self.val_matrix_IT, \
        self.train_set_IT, self.test_set_IT, self.val_set_IT = self.create_train_test_split(self.IT_matrix,
                                                                                            test_ratio=test_ratio,
                                                                                            val_ratio=val_ratio, seed=0)

        tag_count_train = np.asarray(self.train_matrix_IT.sum(1)).flatten()
        # item_list = np.array(item_tag_count.index.to_list())
        # tag_count = np.array(item_tag_count.item.to_list())
        # index = np.where(tag_count >= 10)[0][-1]
        self.poor_items = np.where(tag_count_train < 10)[0]
        self.poor_items_tags = tag_count_train[self.poor_items]
        print("poor_items:(tag<10 in train)", self.poor_items, len(self.poor_items))

        # self.u_adj_list, self.v_adj_list = self.create_adj_matrix(self.train_matrix, pad=pad)
        print("U-I:")
        print(np.sum([len(x) for x in self.train_set_UI]),
              np.sum([len(x) for x in self.val_set_UI]),
              np.sum([len(x) for x in self.test_set_UI]))
        density = float(
            np.sum([len(x) for x in self.train_set_UI]) + np.sum([len(x) for x in self.val_set_UI]) + np.sum(
                [len(x) for x in self.test_set_UI])) / self.num_user / self.num_item
        print("density:{:.2%}".format(density))

        print("I-T:")
        print(np.sum([len(x) for x in self.train_set_IT]),
              np.sum([len(x) for x in self.val_set_IT]),
              np.sum([len(x) for x in self.test_set_IT]))
        density = float(
            np.sum([len(x) for x in self.train_set_IT]) + np.sum([len(x) for x in self.val_set_IT]) + np.sum(
                [len(x) for x in self.test_set_IT])) / self.num_tag / self.num_item
        print("density:{:.2%}".format(density))

        avg_deg = (np.sum([len(x) for x in self.train_set_UI]) + np.sum([len(x) for x in self.val_set_UI]) + np.sum(
            [len(x) for x in self.test_set_UI])) / self.num_user
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree U-I graph: ', avg_deg)

        avg_deg = (np.sum([len(x) for x in self.train_set_IT]) + np.sum([len(x) for x in self.val_set_IT]) + np.sum(
            [len(x) for x in self.test_set_IT])) / self.num_item
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree I-T graph: ', avg_deg)

        """
        Distribution
        """
        # from plt.plot_distribution import draw_distribution
        # draw_distribution(UI_df, 'user', 'item', "CUL")
        # draw_distribution(IT_df, 'item', 'tag', "CUL")
        # draw_distribution(IT_df, 'tag', 'item', "CUL")

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

    def create_train_test_split(self, rating_df, val_ratio=None, test_ratio=0.2, seed=0):
        data_set = []
        for item_ids in rating_df:
            data_set.append(item_ids.indices.tolist())
        # data_set = self.data_set

        train_set, test_set, val_set = self.split_data_randomly(data_set, val_ratio=val_ratio,
                                                                test_ratio=test_ratio, seed=seed)
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


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    return parser.parse_args()


if __name__ == '__main__':
    parser = parser_args()
    path_list = ['../data/citeulike-t/users.dat', '../data/citeulike-t/tag-item.dat',
                 '../data/citeulike-t/tags.dat', '../data/citeulike-t/mult.dat']
    data_generator = Data(path_list, test_ratio=parser.test_ratio, val_ratio=parser.val_ratio, small=False)

    print("data_genrator:", data_generator)
