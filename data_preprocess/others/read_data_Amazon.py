import os
import gzip
import json
import math
import random
import pickle
import pprint
import argparse
from collections import Counter
import scipy
import time
import sys
import re

[sys.path.append(i) for i in ['.', '..']]
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import pickle

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import cosine_similarity
from tqdm.std import trange
from copy import deepcopy

from argparse import ArgumentParser
import torch

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

import re
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)


def generate_IT_matrix(df):
    df_genre = df[['item', 'tag']]
    df_genre = df_genre.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True)
    # print("df_genre:", df_genre)
    # df_IT = pd.concat([pd.Series(row['item'], row['genre'].split(',')) for _, row in df_genre.iterrows()]).reset_index()
    df_IT = pd.DataFrame([[i, t] for i, T in df_genre.values for t in T], columns=df_genre.columns)
    # c = df_IT.columns
    # df_IT[[c[0], c[1]]] = df_IT[[c[1], c[0]]]
    df_IT.columns = ['item', 'tag']

    return df_IT


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def pre_process(text):
    text = text.str.split().replace('\d+', '')
    text = text.apply(lambda x: [re.sub(r'[0-9]+', '', item) for item in x])

    t2 = time.time()
    # remove stopwords
    print("remove stopwords...")
    stop_words = set(stopwords.words("english"))
    text = text.apply(lambda x: [item for item in x if item.lower() not in stop_words])
    t3 = time.time()
    # print("text:", text)
    print("time:", t3 - t2)

    # remove_punctuations
    print("remove punctuations and digits...")
    text = text.apply(lambda x: [item for item in x if not item.isdigit()])
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    t4 = time.time()
    print("time:", t4 - t3)

    # text = text.apply(' '.join)

    # stemming
    print("stemming...")
    # text = [[stemmer.stem(word.lower()) for word in line] for line in text]
    text = text.apply(lambda x: [stemmer.stem(y) for y in x])
    # print("text:", text)
    t5 = time.time()
    print("time:", t5 - t4)

    # Lemmatizing
    print("lemmatizing...")
    text = text.apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    # print("text:", text)
    t6 = time.time()
    print("time:", t6 - t5)

    # extract key-words and remove others
    print("extracting key words...")
    key_words = extract_keywords(text)
    text = pd.Series(text)
    text = text.apply(lambda x: [item for item in x if item.lower() in key_words])
    t7 = time.time()
    print("time:", t7 - t6)

    return text


def extract_keywords(text):
    text = [' '.join(line) for line in text]
    # print("text:", text[:5])
    tfidf = TfidfVectorizer(min_df=10, max_df=0.8, max_features=3000, lowercase=True, ngram_range=(1, 1),
                            stop_words='english')
    tf_idf_matrix = tfidf.fit_transform(text)

    feature_names = tfidf.get_feature_names()
    print("features_names:", feature_names[:10])
    # text = pd.Series(text)
    # print("text:", text)
    # text = text.apply(lambda x: [item for item in x if item.lower() not in feature_names])

    return feature_names


# def generate_pred_list(model, train_matrix, topk=20):
#     num_users = train_matrix.shape[0]
#     batch_size = 1024
#     num_batches = int(num_users / batch_size) + 1
#     user_indexes = np.arange(num_users)
#     pred_list = None
#
#     for batchID in range(num_batches):
#         start = batchID * batch_size
#         end = start + batch_size
#
#         if batchID == num_batches - 1:
#             if start < num_users:
#                 end = num_users
#             else:
#                 break
#
#         batch_user_index = user_indexes[start:end]
#         batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(args.device)
#
#         rating_pred = model.predict(batch_user_ids)
#         rating_pred = rating_pred.cpu().data.numpy().copy()
#         rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0
#
#         # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
#         ind = np.argpartition(rating_pred, -topk)
#         ind = ind[:, -topk:]
#         arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
#         arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
#         batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
#
#         if batchID == 0:
#             pred_list = batch_pred_list
#         else:
#             pred_list = np.append(pred_list, batch_pred_list, axis=0)
#
#     return pred_list


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


# class AmazonBeauty(DatasetLoader):
#     def __init__(self, data_dir):
#         self.fpath = os.path.join(data_dir, 'All_Beauty_5.json')
#
#     def load(self):
#         # Load data
#         df = pd.read_json(self.fpath, lines=True)
#         df = df[['reviewerID', 'asin', 'overall', 'reviewText']].rename(
#             columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rate'}).dropna()
#         df = df.sort_values(by='user').reset_index().drop(['index'], axis=1)
#         df = df[df['rate'] >= 3]
#
#         # df = self.remove_infrequent_items(df, 5)
#         # df = self.remove_infrequent_users(df, 5)
#
#         df = df[['user', 'item', 'rate', 'reviewText']]
#         df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', ' ')
#         df['reviewText'] = df['reviewText'].str.replace('\d+', '')
#
#         return df
#
#
# class AmazonCD(DatasetLoader):
#     def __init__(self, data_dir):
#         self.fpath = os.path.join(data_dir, 'CDs_and_Vinyl_5.json')
#
#     def load(self):
#         # Load data
#         df = pd.read_json(self.fpath, lines=True)
#         df = df[['reviewerID', 'asin', 'overall', 'reviewText']].rename(
#             columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rate'}).dropna()
#         df = df.sort_values(by='user').reset_index().drop(['index'], axis=1)
#         df = df[df['rate'] >= 3]
#
#         # df = self.remove_infrequent_items(df, 5)
#         # df = self.remove_infrequent_users(df, 5)
#
#         df = df[['user', 'item', 'rate', 'reviewText']]
#         df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', ' ')
#         df['reviewText'] = df['reviewText'].str.replace('\d+', '')
#
#         return df


class AmazonBeauty_new(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'All_Beauty_5.json')

    def load(self):
        # Load data
        df = pd.read_json(self.fpath, lines=True)
        df = df[['reviewerID', 'asin', 'overall', 'reviewText']].rename(
            columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rate'}).dropna()
        df = df.sort_values(by='user').reset_index().drop(['index'], axis=1)
        df = df[df['rate'] >= 3]

        # df = self.remove_infrequent_items(df, 5)
        # df = self.remove_infrequent_users(df, 5)

        df_rate = df[['user', 'item', 'rate']]

        df_review = df[['item', 'reviewText']]
        df_review = df_review.groupby(['item'])['reviewText'].apply(list)
        df_review = df_review.reset_index()
        print("df_review:", df_review)
        print("df_rate:", df_rate)
        # print("df_review:", df_review)
        df_review['reviewText'] = df_review['reviewText'].map(lambda x: x[0])
        df_review['reviewText'] = df_review['reviewText'].str.replace('[^\w\s]', ' ')
        df_review['reviewText'] = df_review['reviewText'].str.replace('\d+', '')
        # df_review['reviewText'].stack().groupby(level=0).apply(''.join)
        print("df_review:", df_review)

        return df_rate, df_review


class AmazonCD_new(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'CDs_and_Vinyl_5.json')

    def load(self):
        # Load data
        df = pd.read_json(self.fpath, lines=True)
        df = df[['reviewerID', 'asin', 'overall', 'reviewText']].rename(
            columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rate'}).dropna()
        df = df.sort_values(by='user').reset_index().drop(['index'], axis=1)
        df = df[df['rate'] >= 3]

        # df = self.remove_infrequent_items(df, 5)
        # df = self.remove_infrequent_users(df, 5)

        df_rate = df[['user', 'item', 'rate']]

        df_review = df[['item', 'reviewText']]
        df_review = df_review.groupby(['item'])['reviewText'].apply(list)
        df_review = df_review.reset_index()
        print("df_rate:", df_rate)
        # print("df_review:", df_review)
        df_review['reviewText'] = df_review['reviewText'].map(lambda x: x[0])
        df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', ' ')
        import re
        # df['reviewText'] = re.sub(r'[0-9]+', '', df['reviewText'].str)
        df['reviewText'] = df['reviewText'].replace('\d+', '', regex=True)
        # df_review['reviewText'].stack().groupby(level=0).apply(''.join)
        print("df_review:", df_review)

        return df_rate, df_review


class AmazonCell_new(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'Cell_Phones_and_Accessories_5.json')

    def load(self):
        # Load data
        df = pd.read_json(self.fpath, lines=True)
        df = df[['reviewerID', 'asin', 'overall', 'reviewText']].rename(
            columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rate'}).dropna()
        df = df.sort_values(by='user').reset_index().drop(['index'], axis=1)
        df = df[df['rate'] >= 3]

        # df = self.remove_infrequent_items(df, 5)
        # df = self.remove_infrequent_users(df, 5)

        df_rate = df[['user', 'item', 'rate']]

        df_review = df[['item', 'reviewText']]
        df_review = df_review.groupby(['item'])['reviewText'].apply(list)
        df_review = df_review.reset_index()
        print("df_rate:", df_rate)
        # print("df_review:", df_review)
        df_review['reviewText'] = df_review['reviewText'].map(lambda x: x[0])
        df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', ' ')
        import re
        # df['reviewText'] = re.sub(r'[0-9]+', '', df['reviewText'].str)
        df['reviewText'] = df['reviewText'].replace('\d+', '', regex=True)
        # df_review['reviewText'].stack().groupby(level=0).apply(''.join)
        print("df_review:", df_review)

        return df_rate, df_review


class AmazonElec_new(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'Electronics_5.json')

    def load(self):
        # Load data
        df = pd.read_json(self.fpath, lines=True)
        df = df[['reviewerID', 'asin', 'overall', 'reviewText']].rename(
            columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rate'}).dropna()
        df = df.sort_values(by='user').reset_index().drop(['index'], axis=1)
        df = df[df['rate'] >= 3]

        # df = self.remove_infrequent_items(df, 5)
        # df = self.remove_infrequent_users(df, 5)

        df_rate = df[['user', 'item', 'rate']]

        df_review = df[['item', 'reviewText']]
        df_review = df_review.groupby(['item'])['reviewText'].apply(list)
        df_review = df_review.reset_index()
        print("df_rate:", df_rate)
        # print("df_review:", df_review)
        df_review['reviewText'] = df_review['reviewText'].map(lambda x: x[0])
        df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', ' ')
        import re
        # df['reviewText'] = re.sub(r'[0-9]+', '', df['reviewText'].str)
        df['reviewText'] = df['reviewText'].replace('\d+', '', regex=True)
        # df_review['reviewText'].stack().groupby(level=0).apply(''.join)
        print("df_review:", df_review)

        return df_rate, df_review


class Data(object):
    def __init__(self, data_dir, val_ratio=None, test_ratio=0.2, user_filter=5, item_filter=5, seed=0):

        file_path = os.path.join(data_dir, 'preprocessed_df_{}_{}.pkl'.format(str(user_filter), str(item_filter)))

        if os.path.exists(file_path):
            df = pd.read_pickle(file_path)
        else:
            if "Beauty" in data_dir:
                df_rate, df_review = AmazonBeauty_new(data_dir).load()
            elif "CD" in data_dir:
                df_rate, df_review = AmazonCD_new(data_dir).load()
            elif "Cell" in data_dir:
                df_rate, df_review = AmazonCell_new(data_dir).load()
            elif "Elec" in data_dir:
                df_rate, df_review = AmazonElec_new(data_dir).load()

            tag_file = os.path.join(data_dir, 'keywords.pkl')
            if os.path.exists(tag_file):
                print("loading extracted keywords...")
                df_review = pd.read_pickle(tag_file)
            else:
                df_review['tag'] = pre_process(df_review['reviewText'])
                df_review = df_review.drop('reviewText', 1)
                # df_review.to_pickle(tag_file)
                print("finish extracting and saving keywords.")

            # print("df_review:", df_review['tag'])
            # return

            print("merging...")
            df = pd.merge(df_rate, df_review, on='item')
            df = df.dropna(axis=0, how='any')
            df = df[df['rate'] > 3]
            df.drop_duplicates(subset=['user', 'item'], keep='last', inplace=True)
            df = self.remove_infrequent_items(df, item_filter)
            df = self.remove_infrequent_users(df, user_filter)
            df = df.reset_index().drop(['index'], axis=1)
            print("df-medium:", df)

            df = df[df['tag'].apply(lambda x: len(x) >= 5)]  # make sure each item has 5 tags
            df = self.remove_infrequent_items(df, item_filter)
            df = self.remove_infrequent_users(df, user_filter)
            # df.drop_duplicates()
            df.drop_duplicates(['user', 'item', 'rate'], keep='last')
            df = df.reset_index().drop(['index'], axis=1)
            print("df:", df)

            # unique_data = df.groupby('user')['item'].nunique()
            # df = df.loc[df['user'].isin(unique_data[unique_data >= 5].index)]

            print("start generate unique idx")
            # df, user_mapping = self.convert_unique_idx(df, 'user')
            # df, item_mapping = self.convert_unique_idx(df, 'item')
            df['user'] = df['user'].astype('category').cat.codes
            df['item'] = df['item'].astype('category').cat.codes
            df = df.reset_index().drop(['index'], axis=1)
            df.loc[df['rate'] > 0, 'rate'] = 1
            print("Complete assigning unique index to user and item:\n", df[['user', 'item', 'rate']])

            # if "Beauty" in data_dir:
            #     file_name = "../data/Amazon_Beauty/preprocessed_df_5.pkl"
            # elif "CD" in data_dir:
            #     file_name = "../data/Amazon_CD/preprocessed_df_5_smaller.pkl"

            # df.to_pickle(file_path)

        IT_df = generate_IT_matrix(df)
        # IT_df, tag_mapping = self.convert_unique_idx(IT_df, 'tag')
        IT_df['tag'] = IT_df['tag'].astype('category').cat.codes
        print("IT_df:", IT_df)
        IT_df['rate'] = 1
        print("Complete assigning unique index to user and item:\n", df[['user', 'item', 'rate']])

        self.num_user = len(df['user'].unique())
        self.num_item = len(df['item'].unique())
        self.num_tag = len(IT_df['tag'].unique())

        print('num_user', self.num_user)
        print('num_item', self.num_item)
        print('num_tag', self.num_tag)

        self.UI_matrix = scipy.sparse.csr_matrix(
            (np.array(df['rate']), (np.array(df['user']), np.array(df['item']))))

        self.IT_matrix = scipy.sparse.csr_matrix(
            (np.array(IT_df['rate']), (np.array(IT_df['item']), np.array(IT_df['tag']))))

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

        avg_deg = np.sum([len(x) for x in self.train_set_UI]) / self.num_user
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree U-I graph: ', avg_deg)

        avg_deg = np.sum([len(x) for x in self.train_set_IT]) / self.num_item
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree I-T graph: ', avg_deg)

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


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="Beauty",
                        choices=['Beauty', 'CD', 'Cell', 'Elec'])
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.0)
    parser.add_argument('--user_filter', type=int, default=2)
    parser.add_argument('--item_filter', type=int, default=2)

    return parser.parse_args()


if __name__ == '__main__':
    parser = parser_args()
    data_dir = '../data/Amazon_{}'.format(parser.dataset)

    data_generator = Data(data_dir, test_ratio=parser.test_ratio, val_ratio=parser.val_ratio,
                          user_filter=parser.user_filter, item_filter=parser.item_filter)
