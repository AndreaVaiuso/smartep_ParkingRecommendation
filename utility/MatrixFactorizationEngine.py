# https://www.pluralsight.com/guides/building-a-recommendation-engine-with-tensorflow

import tensorflow
import numpy as np
import os
import pandas as pd

from utility.util import format_print
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf = tensorflow.compat.v1
tf.disable_v2_behavior()


def pq_factor(ratings, latent_vars, learning_rate, epochs):
    # dimensions of the ratings matrix
    n = len(ratings)
    m = len(ratings[0])

    # format_print("nxm: {n} x {m}", "purple")

    # copy of the matrix
    r = ratings

    # random init of user and item latent matrices
    user_matrix = np.random.rand(n, latent_vars)
    items_matrix = np.random.rand(m, latent_vars)

    # placeholder for the ratings matrix
    ratings = tf.placeholder(tf.float32, name="ratings")
    p_matrix = tf.Variable(user_matrix, dtype=tf.float32)
    q_matrix = tf.Variable(items_matrix, dtype=tf.float32)

    # product user and item latent matrices
    p_times_q = tf.matmul(p_matrix, q_matrix, transpose_b=True)

    # compute the initial loss
    squared_error = tf.square(p_times_q - ratings)
    loss = tf.reduce_sum(squared_error) / (n*m)

    # start the optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epochs):
        sess.run(train, {ratings: r})

    # get the final user and item latent matrices
    return sess.run(p_matrix), sess.run(q_matrix)


def recommending_for_user(user_tuples, p_matrix, q_matrix, learning_rate, epochs):
    # unpack the user tuples
    user_i, user_r = zip(*user_tuples)

    # initialize the user latent matrix
    user_init_p = np.zeros((1, len(q_matrix[0])))
    user_p_row = tf.Variable(user_init_p, dtype=tf.float32)
    user_p_times_q = tf.matmul(user_p_row, q_matrix, transpose_b=True)

    # compute the loss
    res = tf.gather(user_p_times_q, user_i, axis=1)
    squared_error = tf.square(user_r - res)
    loss = tf.reduce_sum(squared_error)/(1*len(q_matrix[0]))

    # start the optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    predict = optimizer.minimize(loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epochs):
        sess.run(predict)

    # return the final user latent matrix
    return sess.run(user_p_times_q)  # return also p_matrix to be added to the model (also consider when to update aleeady existent rows


def sml_update(p_matrix, q_matrix, new_data_ratings, new_users, comb_coeff, learning_rate, epochs):
    # append new users to the p matrix
    p_matrix = np.append(p_matrix, np.zeros((len(new_users), len(p_matrix[0]))), axis=0)

    # build P^ and Q^ variables
    p_hat = tf.Variable(np.random.rand(len(p_matrix), len(p_matrix[0])), dtype=tf.float32)
    q_hat = tf.Variable(np.random.rand(len(q_matrix), len(q_matrix[0])), dtype=tf.float32)

    # build placeholder for the new ratings
    new_ratings = tf.placeholder(tf.float32, name="new_ratings")

    product = tf.matmul(p_hat, q_hat, transpose_b=True)

    # compute the loss
    squared_error = tf.square(product - new_ratings)
    loss = tf.reduce_sum(squared_error) / (len(p_matrix)* len(p_matrix[0]))

    # start the optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epochs):
        sess.run(train, {new_ratings: new_data_ratings})
    
    p_hat_final = sess.run(p_hat) 
    q_hat_final = sess.run(q_hat)

    p_ret = comb_coeff * p_matrix + (1-comb_coeff) * p_hat_final #np.sum(np.multiply(comb_coeff, p_matrix), np.multiply(1-comb_coeff, p_hat_final ))
    q_ret = comb_coeff * q_matrix + (1-comb_coeff) * q_hat_final #np.sum(np.multiply(comb_coeff, q_matrix), np.multiply(1-comb_coeff, q_hat_final ))
        
    return p_ret, q_ret


def decoupling_normalization(matrix):
    new_mat = []
    for i in range(0, len(matrix)):
        row = []
        for j in range(0, len(matrix[0])):
            if(matrix[i, j] == 0):
                row.append(0)
                continue
            less_eq = 0
            eq = 0
            for k in range(0, len(matrix[0])):
                if matrix[i, j] >= matrix[i, k]:
                    less_eq += 1
                elif matrix[i, j] == matrix[i, k]:
                    eq += 1
            row.append((less_eq-eq/2) * (1/len(matrix[0])))
        new_mat.append(row)

    return np.array(new_mat)
