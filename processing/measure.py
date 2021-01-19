#!/usr/bin/env python
# -*- coding: utf-8 -*-
# measure.py

import math
from collections import defaultdict
from collections import defaultdict
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from operator import itemgetter


class Measure(object):

    def __init__(self):
        pass

    @staticmethod
    def rating_measure(res):
        measure = []
        mae = Measure.MAE(res)
        measure.append('MAE:' + str(mae) + '\n')
        rmse = Measure.RMSE(res)
        measure.append('RMSE:' + str(rmse) + '\n')

        return measure

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = origin[user].keys()
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def ranking_measure(origin, res, N):
        measure = []
        for n in N:
            predicted = {}
            for user in origin.keys():
                predicted[user] = res[user][:n]
            indicators = []
            if len(origin) != len(predicted):
                print("The Lengths of test set and predicted set are not match!")
                exit(-1)
            hits = Measure.hits(origin, predicted)
            prec = Measure.precision(hits, n)
            indicators.append("Precision:" + str(prec) + "\n")
            recall = Measure.recall(hits, origin)
            indicators.append("Recall:" + str(recall) + "\n")
            F1 = Measure.F1(prec, recall)
            indicators.append("F1:" + str(F1) + "\n")
            MAP = Measure.MAP(origin, predicted, n)
            indicators.append("MAP:" + str(MAP) + "\n")
            NDCG = Measure.NDCG(origin, predicted, n)
            indicators.append('NDCG:' + str(NDCG) + '\n')
            measure.append("Top " + str(n) + "\n")
            measure += indicators
        return measure

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return float(prec) / (len(hits) * N)

    @staticmethod
    def MAP(origin, res, N):
        sum_prec = 0
        for user in res:
            hits = 0
            precision = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user].keys():
                    hits += 1
                    precision += hits / (n + 1.0)
            sum_prec += precision / (min(len(origin[user]), N) + 0.0)
        return sum_prec / (len(res))

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            # 1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / (len(res))

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[-2] - entry[-1])
            count += 1
        if count == 0:
            return error
        return error / count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[-2] - entry[-1]) ** 2
            count += 1
        if count == 0:
            return error
        return math.sqrt(error / count)

    @staticmethod
    def precision_recall_at_k(predictions, N, threshold=5):
        """Return precision and recall at k metrics for each user"""
        measure = []
        for n in N:
            indicators = []
            # First map the predictions to each user.
            user_est_true = defaultdict(list)
            print(predictions)
            for uid, _, true_r, est in predictions:
                user_est_true[uid].append((est, true_r))
            # print(user_est_true)
            precisions = dict()
            recalls = dict()
            accuracys = dict()
            tp=fp=fn=tn=0
            hit=0
            for uid, user_ratings in user_est_true.items():
                # print(type(user_ratings))

                # Sort user ratings by estimated value
                user_ratings.sort(key=lambda x: x[0], reverse=True)

                # Number of relevant items
                n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

                # Number of recommended items in top k
                n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:n])

                # Number of relevant and recommended items in top k
                n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                      for (est, true_r) in user_ratings[:n])

                # hit += n_rel_and_rec_k
                n_true = len(user_ratings)

                accuracys[uid] = n_rel_and_rec_k / n_true if n_true != 0 else 0
                # Precision@K: Proportion of recommended items that are relevant
                # When n_rec_k is 0, Precision is undefined. We here set it to 0.

                precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

                # Recall@K: Proportion of relevant items that are recommended
                # When n_rel is 0, Recall is undefined. We here set it to 0.

                recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

                # ########try one try
                # tp += sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:n])
                # fp += sum(((true_r < threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:n])
                # tn += sum(((true_r < threshold) and (est < threshold)) for (est, true_r) in user_ratings[:n])
                # fn += sum(((true_r >= threshold) and (est < threshold)) for (est, true_r) in user_ratings[:n])

            # #Precision and recall can then be averaged over all users
            accuracy = sum(acc for acc in accuracys.values()) / len(accuracys)
            indicators.append('accuracy ' + str(accuracy) + '\n')
            pre = sum(prec for prec in precisions.values()) / len(precisions)
            indicators.append('pre ' + str(pre) + '\n')
            recall = sum(rec for rec in recalls.values()) / len(recalls)
            indicators.append('recall ' + str(recall) + '\n')
            F1 = 2 * pre * recall / (pre + recall)
            indicators.append('F1 ' + str(F1) + '\n')

            measure.append("Top " + str(n) + "\n")
            measure += indicators

            # ##########
            # print('评价', tp, fp, tn, fn)
            # accuracy = tp / (tp + fp + tn + fn)
            # indicators.append('accuracy ' + str(accuracy) + '\n')
            # pre = tp / (1.0 * (tp + fp))
            # indicators.append('pre ' + str(pre) + '\n')
            # recall = tp / (1.0 * (tp + fn))
            # indicators.append('recall ' + str(recall) + '\n')
            # F1 = 2 * pre * recall / (pre + recall)
            # indicators.append('F1 ' + str(F1) + '\n')
            # measure.append("Top " + str(n) + "\n")
            # measure += indicators

        return measure

    @staticmethod
    def precision_score_topn(predictions, N, testset):
        """Return precision and recall at k metrics for each user"""
        measure = []

        user_est_true = defaultdict(list)

        for uid, pid, true_r, est in predictions:
            user_est_true[uid].append((est, true_r, pid))
        # print('用户数1', len(user_est_true))

        # 加入测试集的条目
        for n in N:
            # First map the predictions to each user.
            top_n = defaultdict(list)
            for uid, user_ratings in user_est_true.items():
                # for pid in testset[uid]:
                #     user_ratings.append((testset[uid][pid], testset[uid][pid], pid))
                user_ratings.sort(key=lambda x: x[0], reverse=True)
                top_n[uid] = user_ratings[:n]

            # Print the recommended items for each user
            hit = accuracy_hit = 0
            test_count = 0
            rec_count = 0
            # print(len(top_n))
            # print(top_n)
            # print(testset)
            for uid, user_ratings in top_n.items():
                accuracy_tag = 0
                for true_pid in testset[uid]:
                    for est, true_r, pid in user_ratings:
                        if pid == true_pid:
                            # print(uid, pid, '=>', est, testset[uid][true_pid])
                            hit += 1
                            accuracy_tag = 1
                accuracy_hit += accuracy_tag
                # print(len(user_ratings))
                rec_count += n
                test_count += len(testset[uid])
                # print(testset[uid])
            print(n, '=>', hit, accuracy_hit, rec_count, test_count)
            indicators = []
            # Precision and recall can then be averaged over all users
            accuracy = accuracy_hit / len(testset)
            indicators.append('accuracy ' + str(accuracy) + '\n')
            pre = hit / (1.0 * rec_count)
            indicators.append('pre ' + str(pre) + '\n')
            recall = hit / (1.0 * test_count)
            indicators.append('recall ' + str(recall) + '\n')
            F1 = 2 * pre * recall / (pre + recall)
            indicators.append('F1 ' + str(F1) + '\n')

            measure.append("Top " + str(n) + "\n")
            measure += indicators
        return measure

    @staticmethod
    def precision_recall_topn(predictions, N, testset):
        """Return precision and recall at k metrics for each user"""
        measure = []

        user_est_true = defaultdict(list)

        for uid, pid, true_r, est in predictions:
            user_est_true[uid].append((est, true_r, pid))
        # print('用户数1', len(user_est_true))

        #加入测试集的条目
        for n in N:
            # First map the predictions to each user.
            top_n = defaultdict(list)
            for uid, user_ratings in user_est_true.items():
                for pid in testset[uid]:
                    user_ratings.append((testset[uid][pid], testset[uid][pid], pid))
                user_ratings.sort(key=lambda x: x[0], reverse=True)
                top_n[uid] = user_ratings[:n]

            # Print the recommended items for each user
            hit = accuracy_hit = 0
            test_count = 0
            rec_count = 0
            # print(len(top_n))
            # print(top_n)
            # print(testset)
            for uid, user_ratings in top_n.items():
                accuracy_tag = 0
                for true_pid in testset[uid]:
                    for est, true_r, pid in user_ratings:
                        if pid == true_pid:
                            # print(uid, pid, '=>', est, testset[uid][true_pid])
                            hit += 1
                            accuracy_tag = 1
                accuracy_hit += accuracy_tag
                # print(len(user_ratings))
                rec_count += n
                test_count += len(testset[uid])
                # print(testset[uid])
            print(n, '=>', hit, accuracy_hit, rec_count, test_count)
            indicators = []
            # Precision and recall can then be averaged over all users
            accuracy = accuracy_hit / len(testset)
            indicators.append('accuracy ' + str(accuracy) + '\n')
            pre = hit / (1.0 * rec_count)
            indicators.append('pre ' + str(pre) + '\n')
            recall = hit / (1.0 * test_count)
            indicators.append('recall ' + str(recall) + '\n')
            F1 = 2 * pre * recall / (pre + recall)
            indicators.append('F1 ' + str(F1) + '\n')

            measure.append("Top " + str(n) + "\n")
            measure += indicators
        return measure

    @staticmethod
    def multi_precision(predictions):
        true_y = [float(3.0), float(5.0), float(8.0), float(10.0), float(13.0)]

        user_est_true = defaultdict(list)
        user_ey_true = defaultdict(list)
        for uid, _, true_r, est in predictions:
            user_est_true[uid].append((est, true_r))

        for uid, user_ratings in user_est_true.items():
            for est, true_r in user_ratings:
                arr = []
                for i in true_y:
                    arr.append((abs(est - i), i))
                arr.sort(key=lambda x: x[0])
                est_y = arr[0][1]
                user_ey_true[uid].append((est_y, true_r))

        # print(user_ey_true)

        y_true = []
        y_pred = []
        for uid, user_ratings in user_ey_true.items():
            for est, true_r in user_ratings:
                y_true.append(float(true_r))
                y_pred.append(float(est))

        # mcm = multilabel_confusion_matrix(y_true, y_pred,labels = true_y)
        # mcm = confusion_matrix(y_true, y_pred, labels=true_y)
        # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
        # print(mcm)

        t = classification_report(y_true, y_pred, target_names=['3.0', '5.0', '8.0', '10.0', '13.0'])

        return t