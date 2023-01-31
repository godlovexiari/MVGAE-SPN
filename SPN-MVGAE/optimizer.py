import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from input_data import load_data
import time
import tensorflow.contrib.distributions as dists
flags = tf.app.flags
FLAGS = flags.FLAGS
import numpy as np

class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) 

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        print("preds_sub", preds_sub)
        print("labels_sub", labels_sub)
        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        print("model.z_mean", model.z_mean)
        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl
        print("self.cost", self.cost)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        print("self.correct_prediction", self.correct_prediction)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        print("self.accuracy", self.accuracy)

class OptimizerCSPN_GVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, ori_features):
        preds_sub = preds
        labels_sub = labels
        self.y_ph = tf.placeholder(tf.float32, [2708] + [32], name="y_ph")   
        self.y_ph = ori_features
        encoder_y_ph = tf.placeholder(tf.float32, [2708] + [32], name="encoder_y_ph")
        encoder_y_ph = np.zeros((2708, 32))
        self.marginalized = tf.placeholder(tf.float32, tf.reshape(self.y_ph, [2708, -1]).shape, name="marg_ph")
        self.marginalized = np.zeros(self.marginalized.shape)
        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) 
        self.log_q_z = tf.reduce_logsumexp(model.spn.forward(tf.cast(model.z, tf.float32), tf.cast(self.marginalized, tf.float32)), axis=-1) - tf.log(float(32))  
        self.log_q = tf.reduce_mean(self.log_q_z)
        self.k2 = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= 0.003*self.k2 + 0.004*tf.reduce_mean(self.log_q_z)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
