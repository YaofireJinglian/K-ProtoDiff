"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from metric_utils import extract_time
from tqdm import tqdm

def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction (TF 1.15 compatible)."""
    tf.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = ori_data.shape
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])
    
    # Network parameters
    hidden_dim = int(dim/2)
    iterations = 5000
    batch_size = 128

    # Input placeholders
    X = tf.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name="X")
    T = tf.placeholder(tf.int32, [None], name="T")
    Y = tf.placeholder(tf.float32, [None, max_seq_len-1, 1], name="Y")
    
    def predictor(x, t):
        """Predictor function."""
        with tf.variable_scope("predictor", reuse=tf.AUTO_REUSE):
            # GRU Cell
            p_cell = tf.nn.rnn_cell.GRUCell(
                num_units=hidden_dim,
                activation=tf.nn.tanh,
                name='p_cell')
            
            # Dynamic RNN
            p_outputs, _ = tf.nn.dynamic_rnn(
                p_cell, x,
                dtype=tf.float32,
                sequence_length=t)
            
            # Output layer
            y_hat_logit = tf.layers.dense(p_outputs, 1, activation=None)
            y_hat = tf.nn.sigmoid(y_hat_logit)
            
            # Collect variables
            p_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="predictor")
        
        return y_hat, p_vars
    
    # Build model
    y_pred, p_vars = predictor(X, T)
    
    # Loss and optimizer
    p_loss = tf.losses.absolute_difference(Y, y_pred)
    p_solver = tf.train.AdamOptimizer().minimize(p_loss, var_list=p_vars)
    
    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Training loop with progress bar
        for _ in tqdm(range(iterations), desc='Training predictor'):
            # Mini-batch sampling
            idx = np.random.permutation(len(generated_data))
            train_idx = idx[:batch_size]
            
            # Prepare batch data
            X_mb = [generated_data[i][:-1, :(dim-1)] for i in train_idx]
            T_mb = [generated_time[i]-1 for i in train_idx]
            Y_mb = [
                np.reshape(generated_data[i][1:, (dim-1)], 
                [-1, 1]) for i in train_idx
            ]
            
            # Train step
            sess.run(p_solver, feed_dict={
                X: X_mb,
                T: T_mb,
                Y: Y_mb
            })
        
        # Testing on original data
        test_idx = np.random.permutation(len(ori_data))[:no]
        
        X_test = [ori_data[i][:-1, :(dim-1)] for i in test_idx]
        T_test = [ori_time[i]-1 for i in test_idx]
        Y_test = [
            np.reshape(ori_data[i][1:, (dim-1)],
            [-1, 1]) for i in test_idx
        ]
        
        # Prediction
        pred_Y = sess.run(y_pred, feed_dict={
            X: X_test,
            T: T_test
        })
        
        # Calculate MAE
        MAE_temp = 0
        for i in range(len(test_idx)):
            MAE_temp += mean_absolute_error(Y_test[i], pred_Y[i])
        
        predictive_score = MAE_temp / len(test_idx)
    
    return predictive_score
    
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

    ps = predictive_score_metrics(ori_data,fake_data)
    result_text = (
        f"Predictive Score (MAE): {ps}\n"
  
    )
    with open(os.path.join(root, 'PS_res_tf.txt'), 'w') as file:
      file.write(result_text)

    # 打印输出到控制台
    print(result_text)