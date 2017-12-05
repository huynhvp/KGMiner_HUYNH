import argparse
import math
import os.path
import timeit

import numpy as np
import tensorflow as tf

class ProjE:
    @property
    def trainable_variables(self):
        return self.__trainable

    @property
    def pred_embedding(self):
        return self.__ent_embedding


    def __init__(self, data_dir, embed_dim=100, dropout=0.5, neg_weight=0.5):

        self.__embed_dim = embed_dim
        self.__initialized = False

        self.__trainable = list()
        self.__dropout = dropout

        with open(".csv", 'r', encoding='utf-8') as f_triple:
           self.__train_predpath = 0 # need to update
           self.__hrt_triples = 0 # need to update
        self.__n_predicate = self.__train_predpath.shape[1]
        print("N_TRAIN_TRIPLES: %d" % self.__train_predpath.shape[0])
        

        bound = 6 / math.sqrt(embed_dim)

        with tf.device('/cpu'):
            self.__pred_embedding = tf.get_variable("pred_embedding", [self.__n_predicate, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                             maxval=bound,
                                                                                             seed=345))
            self.__trainable.append(self.__ent_embedding)   
            self.__pred_bias = tf.get_variable("pred_bias",
                                                         initializer=tf.zeros([embed_dim]))
            self.__trainable.append(self.__pred_bias)

    @staticmethod
    def __l1_normalize(x, dim, epsilon=1e-12, name=None):
        square_sum = tf.reduce_sum(tf.abs(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        return tf.mul(x, x_inv_norm, name=name)

    @staticmethod
    def sampled_softmax(tensor, weights):
        max_val = tf.reduce_max(tensor , 1, keep_dims=True)
        tensor_rescaled = tensor - max_val
        tensor_exp = tf.exp(tensor_rescaled)
        tensor_sum = tf.reduce_sum(tensor_exp, 1, keep_dims=True)

        return (tensor_exp / tensor_sum) * tf.abs(weights)  # all ignored elements will have a prob of 0.
    
    def train(self, inputs, regularizer_weight=1., scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.__initialized:
                scp.reuse_variables()

            pred_list, pred_weight = inputs


            pred_res = tf.nn.dropout(tf.tanh(tf.matmul(pred_list,
                                                          self.__pred_embedding) + self.__pred_bias),
                                        self.__dropout)
        
            self.pred_softmax = pred_res_softmax = self.sampled_softmax(hrt_res, pred_weight)

            pred_loss = -tf.reduce_sum(
                tf.log(tf.clip_by_value(pred_res_softmax, 1e-10, 1.0)) * hr_tlist_weight)

          
            return pred_loss
