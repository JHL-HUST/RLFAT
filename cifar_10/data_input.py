from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils import np_utils
import sys
import tensorflow as tf
version = sys.version_info

from keras.datasets import cifar10
import numpy as np


def _load_data(one_hot=False):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(X_train.shape)
    print(y_test.shape)
    X_train = X_train.reshape(X_train.shape[0],
                              32,
                              32, 3)

    X_test = X_test.reshape(X_test.shape[0],
                            32,
                            32, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    print("Loaded CIFAR10 data.")

    if one_hot:
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, 10).astype(np.float32)
        y_test = np_utils.to_categorical(y_test, 10).astype(np.float32)

    return X_train, y_train, X_test, y_test


class Data(object):

    def __init__(self, one_hot=True):
        train_images, train_labels, eval_images, eval_labels = _load_data(one_hot)
        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)


class AugmentedData(object):
    """
    Data augmentation wrapper over a loaded dataset.

    Inputs to constructor
    =====================
        - raw_cifar10data: the loaded CIFAR10 dataset, via the CIFAR10Data class
        - sess: current tensorflow session
        - model: current model (needed for input tensor)
    """
    def __init__(self, raw_data, sess, model):
        assert isinstance(raw_data, Data)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), self.x_input_placeholder)
        self.augmented = flipped

        self.train_data = AugmentedDataSubset(raw_data.train_data, sess,
                                              self.x_input_placeholder,
                                              self.augmented)
        self.eval_data = AugmentedDataSubset(raw_data.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)


class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += batch_size
        return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                       reshuffle_after_pass)
        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                    raw_batch[0]}), raw_batch[1]


