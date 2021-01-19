#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ncf.py

import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.contrib import slim
from processing.handler import Handler
from processing.measure import Measure
from model.base.recommender import Recommender


class NCF(Recommender):

    def __init__(self, train_set, test_set, k, threshold, is_binary_data, is_training_continued,
                 batch_size=256, learning_rate=0.01, display_step=1):
        Recommender.__init__(self, train_set, test_set, k, threshold, is_binary_data)
        self.is_training_continued = is_training_continued
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.display_step = display_step

        tf.reset_default_graph()
        self.__build_inputs()
        self.__build_nets()
        self.__build_cost()
        self.__build_optimizer()

        self.saver = tf.train.Saver()

    def __build_inputs(self):
        with tf.name_scope('inputs'):
            self.u = tf.placeholder(tf.int32, shape=[None, 1], name='batch_users')
            self.v = tf.placeholder(tf.int32, shape=[None, 1], name='batch_items')
            self.y_ = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

    def __build_nets(self):
        def fc_bn(x, dropout, out_size, activation_fn=None, norm=False):
            """
            :param x: input tensor
            :param dropout: dropout rate, which should be keep_prob at current version
            :param out_size: the number of neurons of current fully connected layer
            :param activation_fn: assigned activation function (ReLU, Tanh, Sigmoid)
            :param norm: should use batch normalization or not
            :return: return the output tensor of this layer collection
            """
            layer = slim.dropout(x, dropout)
            layer = slim.fully_connected(layer, out_size, activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                         biases_initializer=tf.truncated_normal_initializer(stddev=0.1))
            if norm:
                layer = slim.batch_norm(layer, scale=True, updates_collections=None, is_training=self.is_training)
            if activation_fn is not None:
                layer = activation_fn(layer)
            return layer

        with tf.name_scope('nets'):
            u_embed = tf.Variable(tf.random_normal([self.n_user, self.k]), name='user_embedding')
            p = tf.nn.embedding_lookup(u_embed, self.u)
            p = tf.reshape(p, [-1, self.k])

            v_embed = tf.Variable(tf.random_normal([self.n_item, self.k]), name='item_embedding')
            q = tf.nn.embedding_lookup(v_embed, self.v)
            q = tf.reshape(q, [-1, self.k])

            net = tf.concat([p, q], axis=1, name='concat')
            net = fc_bn(net, self.keep_prob, 1024, activation_fn=tf.nn.tanh, norm=False)
            net = fc_bn(net, self.keep_prob, 256, activation_fn=tf.nn.relu, norm=False)
            net = fc_bn(net, self.keep_prob, 32, activation_fn=tf.nn.tanh, norm=False)
            self.y = fc_bn(net, self.keep_prob, 1, activation_fn=tf.nn.sigmoid, norm=False)

    def __build_cost(self):
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(-self.y_ * tf.log(self.y + 1e-10) - (1 - self.y_) * tf.log(1 - self.y + 1e-10))

    def __build_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def __generate_training_batch(self, data, step):
        b_us = np.zeros([self.batch_size, 1])
        b_vs = np.zeros([self.batch_size, 1])
        b_ys = np.zeros([self.batch_size, 1])

        batch_data = data[step * self.batch_size:(step + 1) * self.batch_size]

        for index in range(self.batch_size):
            u, v, r = batch_data[index]

            b_us[index] = self.dao.user[u]
            b_vs[index] = self.dao.item[v]
            b_ys[index] = r

        return b_us, b_vs, b_ys

    def train(self):
        with tf.Session() as sess:
            print("session start")
            sess.run(tf.global_variables_initializer())

            if self.is_training_continued:
                self.saver.restore(sess, 'save/NCF.ckpt')
                print("model loaded", datetime.now())

            print("start optimization", datetime.now())
            epoch = 1
            is_finished = False
            while not is_finished:
                # training data
                from random import shuffle
                shuffle(self.dao.train_data)
                training_iter = len(self.dao.train_data) // self.batch_size
                # every batch
                for step in range(training_iter):
                    # generate batch data
                    us, vs, ys = self.__generate_training_batch(self.dao.train_data, step)
                    sess.run(self.optimizer, feed_dict={self.u: us, self.v: vs, self.y_: ys, self.keep_prob: 0.5,
                                                        self.is_training: True})
                    self.loss = sess.run(self.cost, feed_dict={self.u: us, self.v: vs, self.y_: ys, self.keep_prob: 1,
                                                               self.is_training: True})

                    if step % self.display_step == 0:
                        print("epoch {}, iter {}/{}, loss {:.6f}".format(
                            epoch, step + self.display_step, training_iter, self.loss))

                    if self.is_converged():
                        is_finished = True
                        print("model converged, loss {:.6f}".format(self.loss))
                        break

                epoch += 1
                print("-" * 80)

            print("optimization finished", datetime.now())

            # save model
            self.saver.save(sess, 'save/NCF.ckpt')
            print("model saved", datetime.now())

        print("session close", datetime.now())
        print("-" * 80)

    def transform(self):
        print("NCF model do not need transform procedure")
        print("-" * 80)

        # with h5py.File('save/h5/NCF_{}.h5'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), 'w')) as h5f:
        #     h5f.create_dataset('P', data=self.P)

    def rating_performance(self, output_res=False):
        with tf.Session() as sess:
            self.saver.restore(sess, 'save/NCF.ckpt')
            print("model loaded", datetime.now())

            res = list()  # used to contain the text of the result
            # predict
            max_len = len(self.dao.test_data)
            for i, entry in enumerate(self.dao.test_data):
                user_name, item_name, rating = entry

                # predict
                if self.dao.contains_user(user_name) and self.dao.contains_item(item_name):
                    uid = np.reshape(self.dao.user[user_name], [1, 1])
                    vid = np.reshape(self.dao.item[item_name], [1, 1])
                    prediction = sess.run(self.y, feed_dict={self.u: uid, self.v: vid, self.keep_prob: 1,
                                                             self.is_training: False})
                    prediction = np.reshape(prediction, [])
                elif self.dao.contains_user(user_name) and not self.dao.contains_item(item_name):
                    prediction = self.dao.user_means[user_name]
                elif not self.dao.contains_user(user_name) and self.dao.contains_item(item_name):
                    prediction = self.dao.item_means[item_name]
                else:
                    prediction = self.dao.global_mean

                # denormalize
                prediction = Handler.denormalize(prediction, self.dao.rScale[-1], self.dao.rScale[0])
                prediction = self.check_rating_boundary(prediction)
                # add prediction in order to measure
                res.append([user_name, item_name, rating, prediction])

                # if i % 1000 == 0:
                #     print("progress {}/{}".format(i, max_len))
            print("rating predict finished")

            # rating prediction result
            measure = Measure.rating_measure(res)

            with open('result/{}_rating-measure_{}.txt'.format(
                    self.__class__.__name__,
                    datetime.now().strftime('%Y%m%d_%H%M%S')
            ), 'w') as f:
                f.writelines(measure)

            if output_res:
                with open('result/{}_rating-result_{}.txt'.format(
                        self.__class__.__name__,
                        datetime.now().strftime('%Y%m%d_%H%M%S')
                ), 'w') as f:
                    for row in res:
                        f.write("{}\n".format(row))

    # def load(self, path):
    #     # self.session = tf.Session()
    #     self.saver.restore(self.session, path)
    #     print("model loaded from {}".format(path), datetime.now())
