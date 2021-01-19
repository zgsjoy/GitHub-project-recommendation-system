#!/usr/bin/env python
# -*- coding: utf-8 -*-
# dao.py

import h5py
import numpy as np
from collections import defaultdict
from processing.handler import Handler


class DAO:

    def __init__(self, train_set, test_set, alldata_set):
        self.user = {}  # used to store the order of users in the training set
        self.item = {}  # used to store the order of items in the training set
        self.id2user = {}
        self.id2item = {}
        self.all_user = {}
        self.all_item = {}
        self.user_means = {}  # used to store the mean values of users's ratings
        self.item_means = {}  # used to store the mean values of items's ratings
        self.global_mean = 0
        self.train_u = defaultdict(dict)
        self.train_i = defaultdict(dict)
        self.test_u = defaultdict(dict)  # used to store the test set by hierarchy user: [item, rating]
        self.test_i = defaultdict(dict)  # used to store the test set by hierarchy item: [user, rating]
        self.rScale = list()

        self.train_data = train_set
        self.test_data = test_set
        self.all_data = alldata_set

        self.__generate_set()

        self.__compute_item_mean()
        self.__compute_user_mean()
        self.__global_average()

    def __generate_set(self):
        scale = set()

        for entry in self.train_data:
            user_name, item_name, rating = entry
            scale.add(rating)
        # find the maximum rating and minimum value
        self.rScale = list(scale)
        self.rScale.sort()

        for i, entry in enumerate(self.train_data):
            user_name, item_name, rating = entry
            # makes the rating within the range [0, 1].
            rating = Handler.normalize(float(rating), self.rScale[-1], self.rScale[0])
            self.train_data[i][2] = rating
            # order the user
            if user_name not in self.user:
                self.user[user_name] = len(self.user)
                self.id2user[self.user[user_name]] = user_name
            # order the item
            if item_name not in self.item:
                self.item[item_name] = len(self.item)
                self.id2item[self.item[item_name]] = item_name
                # userList.append
            self.train_u[user_name][item_name] = rating
            self.train_i[item_name][user_name] = rating

        self.all_user.update(self.user)
        self.all_item.update(self.item)
        for entry in self.test_data:
            user_name, item_name, rating = entry
            # order the user
            if user_name not in self.user:
                self.all_user[user_name] = len(self.all_user)
            # order the item
            if item_name not in self.item:
                self.all_item[item_name] = len(self.all_item)

            self.test_u[user_name][item_name] = rating
            self.test_i[item_name][user_name] = rating

    def __global_average(self):
        total = sum(self.user_means.values())
        if total == 0:
            self.global_mean = 0
        else:
            self.global_mean = total / len(self.user_means)

    def __compute_user_mean(self):
        for u in self.user:
            self.user_means[u] = sum(self.train_u[u].values()) / float(len(self.train_u[u]))

    def __compute_item_mean(self):
        for c in self.item:
            self.item_means[c] = sum(self.train_i[c].values()) / float(len(self.train_i[c]))

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]
        else:
            return -1

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]
        else:
            return -1

    def train_size(self):
        return [len(self.user), len(self.item), len(self.train_data)]

    def test_size(self):
        return [len(self.test_u), len(self.test_i), len(self.test_data)]

    def contains(self, u, i):
        # whether user u rated item i
        if u in self.user and i in self.train_u[u]:
            return True
        else:
            return False

    def contains_user(self, u):
        # whether user is in training set
        if u in self.user:
            return True
        else:
            return False

    def contains_item(self, i):
        # whether item is in training set
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return self.train_u[u].keys(), self.train_u[u].values()

    def item_rated(self, i):
        return self.train_i[i].keys(), self.train_i[i].values()

    def s_row(self, u):
        return self.train_u[u]

    def s_col(self, c):
        return self.train_i[c]

    def rating(self, u, c):
        if self.contains(u, c):
            return self.train_u[u][c]
        return -1

    def rating_scale(self):
        return [self.rScale[0], self.rScale[-1]]

    def element_count(self):
        return len(self.train_data)
