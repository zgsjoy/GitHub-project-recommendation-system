#!/usr/bin/env python
# -*- coding: utf-8 -*-
# recommender.py

import numpy as np
from processing.dao import DAO


class Recommender:

    def __init__(self, train_set, test_set, alldata_set, k, threshold, is_binary_data):
        self.dao = DAO(train_set, test_set,alldata_set)
        self.k = k
        self.threshold = threshold
        self.n_user = len(self.dao.user)
        self.n_item = len(self.dao.item)
        self.R = np.zeros([self.n_user, self.n_item])

        self.loss = 0
        self.last_loss = 0

        self.is_binary_data = is_binary_data
        self.negatives = list()

        self.__generate_rating_matrix()
        self.__generate_negatives()

    def __generate_rating_matrix(self):
        for row in self.dao.train_data:
            u_name, i_name, rating = row
            uid = self.dao.user[u_name]
            iid = self.dao.item[i_name]
            self.R[uid, iid] = rating

    def __generate_negatives(self):
        if self.is_binary_data:
            # generate negative samples
            all_set = set()
            for i in range(self.n_item):
                all_set.add(i)
            for i in range(self.n_user):
                positive = set(self.dao.train_u[i].keys())
                negative = all_set - positive
                self.negatives.append(negative)
        else:
            print("Data is not binary, do not need negative samples.")

    def __predict_for_rating(self, u, i):
        pass

    def __predict_for_ranking(self, u):
        pass

    def rating_performance(self, output_res=False):
        pass

    def ranking_performance(self, top):
        pass

    def check_rating_boundary(self, prediction):
        if prediction > self.dao.rScale[-1]:
            return self.dao.rScale[-1]
        elif prediction < self.dao.rScale[0]:
            return self.dao.rScale[0]
        else:
            return round(prediction, 3)

    def is_converged(self):
        from math import isnan
        if isnan(self.loss):
            print("Loss = NaN or Infinity: current settings does not fit the recommender!")
            print("Change the settings and try again!")
            exit(-1)

        converged = abs(self.last_loss - self.loss) < self.threshold  # check if converged
        self.last_loss = self.loss

        return converged

    def info(self):
        print("-" * 80)
        print("user {}, item {}, rating {}".format(self.n_user, self.n_item, self.dao.element_count()))
        print("k {}".format(self.k))
        print("convergence threshold {}".format(self.threshold))
        print("-" * 80)
