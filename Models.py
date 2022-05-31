import os
import sys
from numpy.testing._private.nosetester import run_module_suite
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.keras import layers
"""
sys.path.clear()
sys.path.append('E:\\anaconda\\envs\\python37\\python37.zip')
sys.path.append('E:\\anaconda\\envs\\python37\\DLLs')
sys.path.append('E:\\anaconda\\envs\\python37\\lib')
sys.path.append('E:\\anaconda\\envs\\python37')
sys.path.append('E:\\anaconda\\envs\\python37\\lib\\site-packages')
"""
sys.path.append('E:\\feb\\feb\\test\\test\\test')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from DatasSet import dataset
from constant import constant
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data

class models(object):
    def __init__(self, modelName, learning_rate):
        self.model_name = modelName
        self.lr = learning_rate
        if self.model_name == 'mnist_2nn':
            self.mnist_2nn_construct()
        elif self.model_name == 'mnist_cnn'or self.model_name == 'fmnist_cnn':
            self.mnist_cnn_construct()
        elif self.model_name == 'cifar10_cnn':
            self.cifar10_cnn_construct()

    
    def mnist_2nn_construct(self):
        inputs = tf.keras.Input(shape=(784))
        fc1 = layers.Dense(200, activation='relu')(inputs)
        fc2 = layers.Dense(200, activation='relu')(fc1)
        self.outputs = layers.Dense(10, activation='softmax')(fc2)
        self.model = tf.keras.Model(inputs=inputs, outputs=self.outputs)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(self.lr),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
        #self.model.summary()

    def mnist_cnn_construct(self):
        inputs = tf.keras.Input(shape=(28, 28, 1))
        cov1 = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                  activation='relu')(inputs)
        pool1 = layers.MaxPool2D(pool_size=(2, 2))(cov1)
        cov2 = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                  activation='relu')(pool1)
        pool2 = layers.MaxPool2D(pool_size=(2, 2))(cov2)
        trans_pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
        fc1 = layers.Dense(512, activation='relu')(trans_pool2)
        self.outputs = layers.Dense(10, activation='softmax')(fc1)
        self.model = tf.keras.Model(inputs=inputs, outputs=self.outputs)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(self.lr),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
        #self.model.summary()

    def cifar10_cnn_construct(self):
        inputs = tf.keras.Input(shape=(24, 24, 3))
        cov1 = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                  activation='relu')(inputs)
        pool1 = layers.MaxPool2D(pool_size=(2, 2))(cov1)
        cov2 = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                  activation='relu')(pool1)
        pool2 = layers.MaxPool2D(pool_size=(2, 2))(cov2)
        trans_pool2 = tf.reshape(pool2, [-1, 6 * 6 * 64])
        fc1 = layers.Dense(768, activation='relu')(trans_pool2)
        fc2 = layers.Dense(384, activation='relu')(fc1)
        fc3 = layers.Dense(192, activation='relu')(fc2)
        self.outputs = layers.Dense(10, activation='softmax')(fc3)
        self.model = tf.keras.Model(inputs=inputs, outputs=self.outputs)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(self.lr),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
        #self.model.summary()


    def conv2d(self, x, w, b, strides=1):
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(self, x, weights, biases, dropout):
        x= tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = self.maxpool2d(conv1, k=2)
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.maxpool2d(conv2, k=2)
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    """

    def mnist_2nn_construct(self):
       return 0

    def cifar10_cnn_construct(self):
        inputs = tf.keras.Input(shape=(24, 24, 3))
        cov1 = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                  activation='relu')(inputs)
        pool1 = layers.MaxPool2D(pool_size=(2, 2))(cov1)
        cov2 = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                  activation='relu')(pool1)
        pool2 = layers.MaxPool2D(pool_size=(2, 2))(cov2)
        trans_pool2 = tf.reshape(pool2, [-1, 6 * 6 * 64])
        fc1 = layers.Dense(768, activation='relu')(trans_pool2)
        fc2 = layers.Dense(384, activation='relu')(fc1)
        fc3 = layers.Dense(192, activation='relu')(fc2)
        self.outputs = layers.Dense(10, activation='softmax')(fc3)
        self.model = tf.keras.Model(inputs=inputs, outputs=self.outputs)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(self.lr),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
        self.model.summary()
        
        learning_rate = 0.1
        self.batch_size = constant.NODE_TRAIN_BATCH_SIZE
        n_input = 3*32*32
        self.n_classes = 10
        self.dropout = 0.85
        self.weights = {
            'wc1' : tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'wc2' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'wd1' : tf.Variable(tf.random_normal([7*7*64, 1024])),
            'out' : tf.Variable(tf.random_normal([1024, self.n_classes]))
            }

        self.biases = {
            'bc1' : tf.Variable(tf.random_normal([32])),
            'bc2' : tf.Variable(tf.random_normal([64])),
            'bd1' : tf.Variable(tf.random_normal([1024])),
            'out' : tf.Variable(tf.random_normal([self.n_classes]))
            }
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.pred = self.conv_net(self.x, self.weights, self.biases, self.keep_prob)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.init = tf.global_variables_initializer()
        

    def mnist_cnn_construct(self):
        learning_rate = 0.001
        training_iters = 500
        display_step = 10
        n_input = 784
        self.n_classes = 10
        self.dropout = 0.85
        self.weights = {
            'wc1' : tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'wc2' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'wd1' : tf.Variable(tf.random_normal([7*7*64, 1024])),
            'out' : tf.Variable(tf.random_normal([1024, self.n_classes]))
            }

        self.biases = {
            'bc1' : tf.Variable(tf.random_normal([32])),
            'bc2' : tf.Variable(tf.random_normal([64])),
            'bd1' : tf.Variable(tf.random_normal([1024])),
            'out' : tf.Variable(tf.random_normal([self.n_classes]))
            }
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.pred = self.conv_net(self.x, self.weights, self.biases, self.keep_prob)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.init = tf.global_variables_initializer()

    """


    def init_model(self):
             with tf.Session() as sess:
                 sess.run(self.init)

    def zero_model(self,sess):
        weights = {
            'wc1' : tf.zeros([5, 5, 1, 32]),
            'wc2' : tf.zeros([5, 5, 32, 64]),
            'wd1' : tf.zeros([7*7*64, 1024]),
            'out' : tf.zeros([1024, self.n_classes])
            }

        biases = {
            'bc1' : tf.zeros([32]),
            'bc2' : tf.zeros([64]),
            'bd1' : tf.zeros([1024]),
            'out' : tf.zeros([self.n_classes])
            }
        self.set_model(weights, biases, sess)

    def add_model(self,weights,biases,sess):
        #print("wei\self %s" % sess.run(weights['wc1'][0][0]))
        #print("new\self %s" % sess.run(self.weights['wc1'][0][0]))
        sess.run(tf.assign(self.weights['wc1'],tf.add(self.weights['wc1'],weights['wc1'])))
        sess.run(tf.assign(self.weights['wc2'],tf.add(self.weights['wc2'],weights['wc2'])))
        sess.run(tf.assign(self.weights['wd1'],tf.add(self.weights['wd1'],weights['wd1'])))
        sess.run(tf.assign(self.weights['out'],tf.add(self.weights['out'],weights['out'])))

        sess.run(tf.assign(self.biases['bc1'],tf.add(self.biases['bc1'],biases['bc1'])))
        sess.run(tf.assign(self.biases['bc2'],tf.add(self.biases['bc2'],biases['bc2'])))
        sess.run(tf.assign(self.biases['bd1'],tf.add(self.biases['bd1'],biases['bd1'])))
        sess.run(tf.assign(self.biases['out'],tf.add(self.biases['out'],biases['out'])))
    
    def set_model(self, weights, biases, sess):
        
        #wc1=tf.zeros([5, 5, 1, 32])

        sess.run(tf.assign(self.weights['wc1'],weights['wc1']))
        sess.run(tf.assign(self.weights['wc2'],weights['wc2']))
        sess.run(tf.assign(self.weights['wd1'],weights['wd1']))
        sess.run(tf.assign(self.weights['out'],weights['out']))

        sess.run(tf.assign(self.biases['bc1'],biases['bc1']))
        sess.run(tf.assign(self.biases['bc2'],biases['bc2']))
        sess.run(tf.assign(self.biases['bd1'],biases['bd1']))
        sess.run(tf.assign(self.biases['out'],biases['out']))

        """
        print("new\wei %s" % sess.run(wc1[0][0]))
        print("new\self %s" % sess.run(self.weights['wc1'][0][0]))
        print("***********")
        """

    def new_model(self,sess):
        sess.run(tf.assign(self.weights['wc1'],tf.divide(self.weights['wc1'],constant.CLIENT_NUMBER)))
        sess.run(tf.assign(self.weights['wc2'],tf.divide(self.weights['wc2'],constant.CLIENT_NUMBER)))
        sess.run(tf.assign(self.weights['wd1'],tf.divide(self.weights['wd1'],constant.CLIENT_NUMBER)))
        sess.run(tf.assign(self.weights['out'],tf.divide(self.weights['out'],constant.CLIENT_NUMBER)))

        sess.run(tf.assign(self.biases['bc1'],tf.divide(self.biases['bc1'],constant.CLIENT_NUMBER)))
        sess.run(tf.assign(self.biases['bc2'],tf.divide(self.biases['bc2'],constant.CLIENT_NUMBER)))
        sess.run(tf.assign(self.biases['bd1'],tf.divide(self.biases['bd1'],constant.CLIENT_NUMBER)))
        sess.run(tf.assign(self.biases['out'],tf.divide(self.biases['out'],constant.CLIENT_NUMBER)))

    def run_model(self, max_batch_x, max_batch_y, ses):
            step = 1
            while step <= (constant.NODE_TRAIN_NUMBER):
                data_init = (step-1)*self.batch_size
                ses.run(self.optimizer, feed_dict = {self.x:max_batch_x[data_init :data_init + self.batch_size], self.y:max_batch_y[data_init :data_init + self.batch_size], self.keep_prob: self.dropout})
                step += 1
            data_init = (step-1)*self.batch_size
            loss_train, acc_train = ses.run([self.cost, self.accuracy], feed_dict = {self.x:max_batch_x[data_init :data_init + self.batch_size], self.y:max_batch_y[data_init :data_init + self.batch_size], self.keep_prob: 1.})
            #self.train_loss.append(loss_train)
            #self.train_acc.append(acc_train)
            return loss_train

    def test_model(self, step, batch_x, batch_y, test_x, test_y, ses):
            loss_train, acc_train = ses.run([self.cost, self.accuracy], feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: 1.})
            print("Iter "+ str(step)+ ", Minibatch Loss= "+ str(loss_train)+ ", Training Accuracy= "+ str(acc_train))
            acc_test = ses.run(self.accuracy, feed_dict = {self.x: test_x, self.y: test_y, self.keep_prob: 1.})
            print(", Testing Accuracy= "+ str(acc_test))
            #self.train_loss.append(loss_train)
            #self.train_acc.append(acc_train)
            self.test_acc.append(acc_test)
