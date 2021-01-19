#!/usr/bin/env python
# -*- coding: utf-8 -*-
# dmf.py

import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib import slim
from model.base.dmf_rec import DMFRec


class DMF(DMFRec):

    def __init__(self, train_set, test_set, alldata_set , k, threshold, is_binary_data, is_training_continued,
                 batch_size=256, learning_rate=0.01, display_step=1):
        DMFRec.__init__(self, train_set, test_set, alldata_set, k, threshold, is_binary_data)
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
            self.u = tf.placeholder(tf.float32, shape=[None, self.n_item], name='batch_users')
            self.v = tf.placeholder(tf.float32, shape=[None, self.n_user], name='batch_items')
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
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            if norm:
                layer = slim.batch_norm(layer, scale=True, updates_collections=None, is_training=self.is_training)
            if activation_fn is not None:
                layer = activation_fn(layer)
            return layer

        with tf.name_scope('nets'):
            u_net = fc_bn(self.u, self.keep_prob, 1024, activation_fn=tf.nn.relu, norm=False)
            self.p = fc_bn(u_net, self.keep_prob, self.k, activation_fn=None, norm=False)

            v_net = fc_bn(self.v, self.keep_prob, 1024, activation_fn=tf.nn.relu, norm=False)
            self.q = fc_bn(v_net, self.keep_prob, self.k, activation_fn=None, norm=False)

    def __build_cost(self):
        with tf.name_scope('cost'):
            # rating prediction, use cosine similarity to predict
            numerator = tf.reduce_sum(self.p * self.q, 1, keepdims=True)
            norm_p = tf.sqrt(tf.reduce_sum(tf.square(self.p), 1, keepdims=True))
            norm_q = tf.sqrt(tf.reduce_sum(tf.square(self.q), 1, keepdims=True))
            denominator = norm_p * norm_q
            cosine = numerator / denominator
            y = tf.nn.relu(cosine)  # use ReLU to remove negative values
            self.cost = tf.reduce_mean(-self.y_ * tf.log(y + 1e-10) - (1 - self.y_) * tf.log(1 - y + 1e-10))

    def __build_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def __generate_training_batch(self, data, step):
        b_us = np.zeros([self.batch_size, self.n_item])
        b_vs = np.zeros([self.batch_size, self.n_user])
        b_ys = np.zeros([self.batch_size, 1])

        batch_data = data[step * self.batch_size:(step + 1) * self.batch_size]

        for index in range(self.batch_size):
            u, v, r = batch_data[index]

            b_us[index] = self.R[self.dao.user[u]]
            b_vs[index] = self.R.T[self.dao.item[v]]
            b_ys[index] = r

        return b_us, b_vs, b_ys

    def __generate_transform_batch(self, data, step):
        batch_data = data[step * self.batch_size:(step + 1) * self.batch_size]

        return batch_data

    def train(self):
        with tf.Session() as sess:
            print("session start")
            sess.run(tf.global_variables_initializer())

            if self.is_training_continued:
                self.saver.restore(sess, 'save/DMF.ckpt')
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
                    sess.run(self.optimizer, feed_dict={self.u: us, self.v: vs, self.y_: ys, self.keep_prob: 0.8,
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
            self.saver.save(sess, 'save/DMF.ckpt')
            print("model saved", datetime.now())

        print("session close", datetime.now())
        print("-" * 80)

    def transform(self):
        with tf.Session() as sess:
            self.saver.restore(sess, 'save/DMF.ckpt')
            print("model loaded", datetime.now())

            # transform R to P (user latent)
            transform_iter = self.n_user // self.batch_size
            for step in range(transform_iter):
                # generate batch
                us = self.__generate_transform_batch(self.R, step)
                ps = sess.run(self.p, feed_dict={self.u: us, self.keep_prob: 1, self.is_training: False})

                self.P[step * self.batch_size:(step + 1) * self.batch_size] = ps

            # last batch's size is not batch_size
            last_us = self.R[transform_iter * self.batch_size:]
            last_ps = sess.run(self.p, feed_dict={self.u: last_us, self.keep_prob: 1, self.is_training: False})
            self.P[transform_iter * self.batch_size:] = last_ps
            print("P transformation finished")

            # transform R.T to Q (item latent)
            transform_iter = self.n_item // self.batch_size
            for step in range(transform_iter):
                # generate batch
                vs = self.__generate_transform_batch(self.R.T, step)
                qs = sess.run(self.q, feed_dict={self.v: vs, self.keep_prob: 1, self.is_training: False})

                self.Q[step * self.batch_size:(step + 1) * self.batch_size] = qs

            # last batch's size is not batch_size
            last_vs = self.R.T[transform_iter * self.batch_size:]
            last_qs = sess.run(self.q, feed_dict={self.v: last_vs, self.keep_prob: 1, self.is_training: False})
            self.Q[transform_iter * self.batch_size:] = last_qs
            print("Q transformation finished")

            print("transform finished", datetime.now())
            print("-" * 80)

        # with h5py.File('save/h5/DMF_{}.h5'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), 'w')) as h5f:
        #     h5f.create_dataset('P', data=self.P)

    # def load(self, path):
    #     # self.session = tf.Session()
    #     self.saver.restore(self.session, path)
    #     print("model loaded from {}".format(path), datetime.now())