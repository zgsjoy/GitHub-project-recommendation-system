#!/usr/bin/env python
# -*- coding: utf-8 -*-
# dmf_rec.py

import numpy as np
from model.base.recommender import Recommender
from processing.handler import Handler
from processing.measure import Measure
from datetime import datetime


class DMFRec(Recommender):

    def __init__(self, train_set, test_set, alldata_set, k, threshold, is_binary_data):
        Recommender.__init__(self, train_set, test_set, alldata_set,k, threshold, is_binary_data)
        self.P = np.zeros([self.n_user, self.k])
        self.Q = np.zeros([self.n_item, self.k])

    def __predict_for_rating(self, u, i):
        if self.dao.contains_user(u) and self.dao.contains_item(i):
            return self.P[self.dao.user[u]].dot(self.Q[self.dao.item[i]]) / (
                    np.linalg.norm(self.P[self.dao.user[u]]) * np.linalg.norm(self.Q[self.dao.item[i]]))
        elif self.dao.contains_user(u) and not self.dao.contains_item(i):
            return self.dao.user_means[u]
        elif not self.dao.contains_user(u) and self.dao.contains_item(i):
            return self.dao.item_means[i]
        else:
            return self.dao.global_mean

    def __predict_for_ranking(self, u):
        # used to rank all the items for the user
        if self.dao.contains_user(u):
            return self.Q.dot(self.P[self.dao.user[u]]) / (
                    np.linalg.norm(self.Q) * np.linalg.norm(self.P[self.dao.user[u]]))
        else:
            return [self.dao.global_mean] * self.n_item

    def rating_performance(self, output_res=False):
        res = list()  # used to contain the text of the result
        # predict
        for entry in self.dao.all_data:
            user_name, item_name, rating = entry

            # predict
            prediction = self.__predict_for_rating(user_name, item_name)
            # denormalize
            prediction = Handler.denormalize(prediction, self.dao.rScale[-1], self.dao.rScale[0])
            prediction = self.check_rating_boundary(prediction)
            # add prediction in order to measure
            res.append([user_name, item_name, rating, prediction])
        print("rating predict finished")
        # print(res)

        # rating prediction result
        # measure = Measure.rating_measure(res)
        # measure1 = Measure.precision_recall_at_k(res, [3, 5, 10, 50], threshold=8)
        # measure2 = Measure.multi_precision(res)
        measure3 = Measure.precision_recall_topn(res, [3,5,10], self.dao.test_u)
        print(measure3)

        # with open('result/{}_rating-measure_{}.txt'.format(
        #         self.__class__.__name__,
        #         datetime.now().strftime('%Y%m%d_%H%M%S')
        # ), 'w') as f:
        #     f.writelines(measure)
        #
        # if output_res:
        #     with open('result/{}_rating-result_{}.txt'.format(
        #             self.__class__.__name__,
        #             datetime.now().strftime('%Y%m%d_%H%M%S')
        #     ), 'w') as f:
        #         for row in res:
        #             f.write("{}\n".format(row))

        with open('result/google/{}_150k_{}.txt'.format(
                self.__class__.__name__,
                datetime.now().strftime('%Y%m%d_%H%M%S')
        ), 'w') as f:
            f.writelines(measure3)

    def ranking_performance(self, top):
        N = top[-1]

        # predict
        rec_list = {}
        test_sample = {}
        for user in self.dao.test_u:
            test_sample[user] = self.dao.test_u[user]
        user_count = len(test_sample)
        for i, user in enumerate(test_sample):
            item_set = {}
            predicted_items = self.__predict_for_ranking(user)
            for index, rating in enumerate(predicted_items):
                item_set[self.dao.id2item[index]] = rating

            rated_list, _ = self.dao.user_rated(user)
            for item in rated_list:
                del item_set[item]

            recommendations = []
            for item in item_set:
                recommendations.append((item, item_set[item]))
            rec_list[user] = sorted(recommendations, key=lambda x: x[1], reverse=True)[:N]

            if i % 100 == 0:
                print("progress:{}/{}".format(i, user_count))

        # evaluation result
        measure = Measure.ranking_measure(test_sample, rec_list, top)

        with open('result/{}_ranking-measure_{}.txt'.format(
                self.__class__.__name__,
                datetime.now().strftime('%Y%m%d_%H%M%S')
        ), 'w') as f:
            f.writelines(measure)
