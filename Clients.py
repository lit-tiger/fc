import os
import sys
"""
sys.path.clear()
sys.path.append('E:\\anaconda\\envs\\python37\\python37.zip')
sys.path.append('E:\\anaconda\\envs\\python37\\DLLs')
sys.path.append('E:\\anaconda\\envs\\python37\\lib')
sys.path.append('E:\\anaconda\\envs\\python37')
sys.path.append('E:\\anaconda\\envs\\python37\\lib\\site-packages')
"""
sys.path.append('E:\\feb\\feb\\test\\test\\test')
import tensorflow as tf
import numpy as np
import math
from DatasSet import dataset
from constant import constant
from Models import models
from Noise import noise

#from tensorflow.examples.tutorials.mnist import input_data

class clients(object):
    def __init__(self, localData, localLabel, client_name, is_Non_IID, model_name = "mnist"):
        self.model_name = model_name
        self.name = client_name
        self.train_label = localLabel
        self.train_data_size = localData.shape[0]
        self._index_in_train_epoch = 0
        self.parameters = {}
        self.train_data = localData
        self.model = models(model_name, constant.ETA)
        self.old_weight = None
        self.new_weight = None
        self.noise = noise(is_Non_IID, model_name)
        self.max_norm = None
        self.self_norm = None

    def poor_quilty(self):
        for i in range(len(self.train_label)):
            for j in range(len(self.train_label[i])):
                if self.train_label[i][j] == 1:
                    self.train_label[i][j] = 0
                    if j < len(self.train_label[i]) - 1:
                        self.train_label[i][j+1] = 1
                    else:
                        self.train_label[i][0] = 1

    
    def set_weights(self, weights):
        self.old_weight = weights
        self.model.model.set_weights(weights)

    def get_self_norm(self):
        Tw_norm = 0
        self.new_weight = self.model.model.get_weights()
        for i in range(len(self.old_weight)):
            Tw_norm += pow(np.linalg.norm(self.old_weight[i] - self.new_weight[i]), 2)
        self.self_norm = math.sqrt(Tw_norm)
        return self.self_norm

    def get_weights(self, noise_flag, max_weights):
        if noise_flag == 0:
            return self.model.model.get_weights()
        self.new_weight = self.model.model.get_weights()
        noise_weight = self.noise.get_noise(self.self_norm,self.get_max_norm(max_weights))
        weights = self.model.model.get_weights()
        for i in range(len(weights)):
            weights[i] += noise_weight[i]
        return weights

    def get_max_norm(self, max_weights):
        Tw_norm = 0
        self.new_weight = self.model.model.get_weights()
        for i in range(len(self.new_weight)):
            Tw_norm += pow(np.linalg.norm(self.new_weight[i] - max_weights[i]), 2)
        self.max_norm = math.sqrt(Tw_norm)
        return self.max_norm

    def next_batch(self, batch_size):
        start = self._index_in_train_epoch
        self._index_in_train_epoch += batch_size
        if self._index_in_train_epoch > self.train_data_size:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]
            start = 0
            self._index_in_train_epoch = batch_size
            """
            print("clint %d" %self.name)
            print(batch_size)
            print(self.train_data_size)
            """
            assert batch_size <= self.train_data_size
        end = self._index_in_train_epoch
        #print(type(self.train_data[0][0][0][0]))
        #print(type(self.train_label[0][0]))
        return self.train_data[start: end], self.train_label[start: end]

