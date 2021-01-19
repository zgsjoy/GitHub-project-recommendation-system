#!/usr/bin/env python
# -*- coding: utf-8 -*-
# main.py

import tensorflow as tf
from model.dmf import DMF
from model.ncf import NCF
from processing.handler import Handler

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('train_path', './data/google/translated_train1.csv', 'training set path')
tf.flags.DEFINE_string('test_path', './data/google/translated_test1.csv', 'testing set path')
tf.flags.DEFINE_string('alldata_path', './data/google/data_without_inter.csv', 'alldata set path')
# tf.flags.DEFINE_string('train_path', './data/ml-100k/ub.base', 'training set path')
# tf.flags.DEFINE_string('test_path', './data/ml-100k/ub.test', 'testing set path')
tf.flags.DEFINE_integer('k', 150, 'low dimension rank')
tf.flags.DEFINE_float('threshold', 1e-4, 'convergence threshold')
tf.flags.DEFINE_boolean('binary', False, 'entry value is binary or not')
tf.flags.DEFINE_boolean('continued', False, 'continue training from the last one or not')
tf.flags.DEFINE_integer('batch_size', 100, 'training batch size')                          #256
tf.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.flags.DEFINE_integer('display_step', 10, 'display for every n batch')


def main(_):
    train_set = Handler.get_data(FLAGS.train_path)
    test_set = Handler.get_data(FLAGS.test_path)
    alldata_set=Handler.get_data(FLAGS.alldata_path)

    model = DMF(train_set, test_set, alldata_set,
                k=FLAGS.k,
                threshold=FLAGS.threshold,
                is_binary_data=FLAGS.binary,
                is_training_continued=FLAGS.continued,
                learning_rate=FLAGS.learning_rate,
                display_step=FLAGS.display_step)

    model.info()
    model.train()
    model.transform()
    model.rating_performance()
    # model.ranking_performance([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


if __name__ == '__main__':
    tf.app.run()
