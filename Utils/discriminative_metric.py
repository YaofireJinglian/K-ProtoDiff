"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from metric_utils import train_test_divide, extract_time
from tqdm import tqdm  # 导入进度条库

def batch_generator(data, time, batch_size):
    """Mini-batch generator."""
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]
    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)
    return X_mb, T_mb

def discriminative_score_metrics(ori_data, generated_data):
    """Pure TF 1.15 implementation with progress bar."""
    tf.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape    
    
    # Corrected: use generated_data for generated_time
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)  # Fixed
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
     
    # Network parameters
    hidden_dim = int(dim/2)
    iterations = 2000
    batch_size = 128
    
    # Input placeholders
    X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="X")
    X_hat = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="X_hat")
    T = tf.placeholder(tf.int32, [None], name="T")
    T_hat = tf.placeholder(tf.int32, [None], name="T_hat")
    
    def discriminator(x, t):
        """Discriminator function."""
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            d_cell = tf.nn.rnn_cell.GRUCell(
                num_units=hidden_dim, 
                activation=tf.nn.tanh)
            d_outputs, d_last_states = tf.nn.dynamic_rnn(
                d_cell, x, 
                dtype=tf.float32, 
                sequence_length=t)
            y_hat_logit = tf.layers.dense(d_last_states, 1, activation=None)
            y_hat = tf.nn.sigmoid(y_hat_logit)
            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        return y_hat_logit, y_hat, d_vars
    
    # Build models
    y_logit_real, y_pred_real, d_vars = discriminator(X, T)
    y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)
        
    # Loss functions
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_logit_real,
            labels=tf.ones_like(y_logit_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_logit_fake,
            labels=tf.zeros_like(y_logit_fake)))
    d_loss = d_loss_real + d_loss_fake
    
    # Optimizer
    d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
        
    # Train/test split
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Training with progress bar
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # 添加进度条
        for _ in tqdm(range(iterations), desc="Training Discriminator", unit="iter"):
            # Get batches
            X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
            X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
            
            # Train step
            sess.run(d_solver, 
                    feed_dict={
                        X: X_mb, 
                        T: T_mb, 
                        X_hat: X_hat_mb, 
                        T_hat: T_hat_mb
                    })
        
        # Evaluation
        y_pred_real_curr, y_pred_fake_curr = sess.run(
            [y_pred_real, y_pred_fake],
            feed_dict={
                X: test_x,
                T: test_t,
                X_hat: test_x_hat,
                T_hat: test_t_hat
            })
    
    # Calculate score
    y_pred_final = np.squeeze(np.concatenate([y_pred_real_curr, y_pred_fake_curr]))
    y_label_final = np.concatenate([
        np.ones(len(y_pred_real_curr)),
        np.zeros(len(y_pred_fake_curr))
    ])
    acc = accuracy_score(y_label_final, y_pred_final > 0.5)
    return np.abs(0.5 - acc)


data_list = ['Electricity','Energy','ETTh','Exchange','Illness','Stocks','Weather',"EEG",'Traffic',]

for data_name in data_list:
      # root = '/home/user/dyh/ts/PaD-TS/OUTPUT/' + data_name + '_24'
    # ori_path = '/home/user/dyh/ts/PaD-TS/OUTPUT/samples/'+ data_name + '_norm_truth_24_train.npy'
    # fake_path = root + '/ddpm_fake_' + data_name + '_24.npy'

    # ========================================================================
    # root = '/home/user/dyh/ts/Diffusion-TS/OUTPUT/' + data_name
    # ori_path = root + '/samples/' + data_name + '_norm_truth_24_train.npy'
    # fake_path = root + '/ddpm_fake_' + data_name + '.npy'
    # ========================================================================
    # root = '/home/user/dyh/ts/TimeGAN/OUTPUT/' + data_name
    # ori_path = root + '/timegen_truth_'+ data_name + '_24.npy'
    # fake_path = root + '/timegen_fake_'+ data_name + '_24.npy'
    #=========================================================================
    # root = '/home/user/dyh/ts/timeVAE-pytorch-main/OUTPUT/' + data_name
    # ori_path = root + '/timeVAE_truth_'+ data_name + '_24.npy'
    # fake_path = root + '/timeVAE_fake_'+ data_name + '_24.npy'
    #=========================================================================
    # root = '/home/user/dyh/ts/GT-GAN/OUTPUT/' + data_name
    # ori_path = root + '/GT-GAN_truth_'+ data_name + '_24.npy'
    # fake_path = root + '/GT-GAN_fake_'+ data_name + '_24.npy'
    #=========================================================================
    root = '/home/user/dyh/ts/TimeVQVAE/OUTPUT/' + data_name
    ori_path = root + '/TimeVQVAE_truth_'+ data_name + '_24.npy'
    fake_path = root + '/TimeVQVAE_fake_'+ data_name + '_24.npy'

    ori_data = np.load(ori_path)
    fake_data = np.load(fake_path)
    print("===================================="+root+"====================================")
    print(ori_data.shape,fake_data.shape)

    ds = discriminative_score_metrics(ori_data,fake_data)
    result_text = (
        f"discriminative_score: {ds}\n"
  
    )
    with open(os.path.join(root, 'DS_res_tf.txt'), 'w') as file:
      file.write(result_text)

    # 打印输出到控制台
    print(result_text)