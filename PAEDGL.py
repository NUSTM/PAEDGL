# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb

from utils.tf_funcs import *
from utils.prepare_data import load_w2v, load_data, acc_prf

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', 'data/w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
tf.app.flags.DEFINE_string('pos_trainable', '', 'whether position embedding is trainable')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of sentences per documents')
## model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.app.flags.DEFINE_string('use_position', 'PAE', 'PAE or PEC' )
tf.app.flags.DEFINE_string('use_DGL', '', 'whether use DGL')
tf.app.flags.DEFINE_string('hierachy', '', 'whether use hierachy')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('train_file_path', 'data/clause_keywords.csv', 'training file')
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 10, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
# not easy to tune 
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of samples per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization rate')
tf.app.flags.DEFINE_float('lambda1', 0.1, 'rate for position prediction loss')

def build_model(word_embedding, pos_embedding, x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2,RNN = biLSTM):
    x = tf.nn.embedding_lookup(word_embedding, x)
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    word_dis = tf.nn.embedding_lookup(pos_embedding, word_dis)
    sen_dis = word_dis[:,:,0,:]
    word_dis = tf.reshape(word_dis, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim_pos])
    if FLAGS.use_position == 'PAE':
        print('using PAE')
        inputs = tf.concat([inputs, word_dis], axis=2)
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    sen_len = tf.reshape(sen_len, [-1])
    # inputs shape:        [-1, FLAGS.max_sen_len, FLAGS.embedding_dim + FLAGS.embedding_dim_pos]
    with tf.name_scope('word_encode'):  
        inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'word_layer')
    # inputs shape:        [-1, FLAGS.max_sen_len, 2 * FLAGS.n_hidden]
    with tf.name_scope('word_attention'):
        # average attention
        # s = att_avg(inputs, sen_len)
        # varible attention
        sh2 = 2 * FLAGS.n_hidden
        w1 = get_weight_varible('word_att_w1', [sh2, sh2])
        b1 = get_weight_varible('word_att_b1', [sh2])
        w2 = get_weight_varible('word_att_w2', [sh2, 1])
        s = att_var(inputs,sen_len,w1,b1,w2)
    s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
    n_feature = 2 * FLAGS.n_hidden
    if FLAGS.use_position == 'PEC':
        print('using PEC')
        s = tf.concat([s, sen_dis], axis=2)
        n_feature = 2 * FLAGS.n_hidden + FLAGS.embedding_dim_pos
    # s shape:        [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden]
    if FLAGS.hierachy:
        print('use hierachy')
        s = RNN(s, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_layer')
        n_feature = 2 * FLAGS.n_hidden

    with tf.name_scope('softmax'):
        s = tf.reshape(s, [-1, n_feature])
        s = tf.nn.dropout(s, keep_prob=keep_prob2)
        # postion prediction
        w_p = get_weight_varible('position_w', [n_feature, 102])
        b_p = get_weight_varible('position_b', [102])
        pred_p = tf.matmul(s, w_p) + b_p
        pred_p = tf.nn.softmax(pred_p) 
        pred_p = tf.reshape(pred_p, [-1, FLAGS.max_doc_len, 102])
        # emotion cause prediction
        if FLAGS.use_DGL:
            # emotion cause prediction in training phase
            print('using DGL feature!')
            DGL = tf.reshape(DGL, [-1, FLAGS.max_doc_len])
            s_tr = tf.concat([s, DGL], axis=1)
            w = get_weight_varible('cause_w', [n_feature + FLAGS.max_doc_len, FLAGS.n_class])
            b = get_weight_varible('cause_b', [FLAGS.n_class])
            pred_c_tr = tf.matmul(s_tr, w) + b
            pred_c_tr = tf.nn.softmax(pred_c_tr) 
            pred_c_tr = tf.reshape(pred_c_tr, [-1, FLAGS.max_doc_len, FLAGS.n_class])
            # emotion cause prediction in testing phase
            s = tf.reshape(s, [-1, FLAGS.max_doc_len, n_feature])
            batch = tf.shape(s)[0]
            pred_c_te = []
            dgl = tf.zeros([batch, FLAGS.max_doc_len])
            for i in range(FLAGS.max_doc_len):
                s_i = s[:,i,:]
                feature_i = tf.concat([s_i, tf.identity(dgl)], axis=1)
                pred_c_te_i = tf.nn.softmax(tf.matmul(feature_i, w) + b)
                pred_c_te.append(pred_c_te_i)
                dgl_i = tf.reshape(tf.cast(tf.argmax(pred_c_te_i, axis=1), tf.float32) * 2 - 1, [-1, 1]) 
                dgl = tf.concat([dgl[:,:i], dgl_i, dgl[:,i+1:]], axis=1)
            pred_c_te = tf.transpose(tf.cast(pred_c_te, tf.float32), perm=[1, 0, 2])
        else:
            w = get_weight_varible('cause_w', [n_feature, FLAGS.n_class])
            b = get_weight_varible('cause_b', [FLAGS.n_class])
            pred_c_tr = tf.matmul(s, w) + b
            pred_c_tr = tf.nn.softmax(pred_c_tr) 
            pred_c_tr = tf.reshape(pred_c_tr, [-1, FLAGS.max_doc_len, FLAGS.n_class])
            pred_c_te = pred_c_tr

    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b) + tf.nn.l2_loss(w_p) + tf.nn.l2_loss(b_p)
    return pred_c_tr, pred_c_te, pred_p, reg

def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2, y, y_p, batch_size, test=False):
    for index in batch_index(len(y), batch_size, test):
        feed_list = [x[index], word_dis[index], DGL[index], sen_len[index], doc_len[index], keep_prob1, keep_prob2, y[index], y_p[index]]
        yield feed_list, len(index)

def run():
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    tf.reset_default_graph()
    # Model Code Block
    word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, FLAGS.train_file_path, FLAGS.w2v_file)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    if FLAGS.pos_trainable:
        print('pos_embedding trainable!')
        pos_embedding = tf.Variable(pos_embedding, dtype=tf.float32, name='pos_embedding')
    else:
        pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    print('build model...')
    
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    DGL = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.max_doc_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_p = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, 102])
    placeholders = [x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2, y, y_p]
    
    
    pred_c_tr, pred_c_te, pred_p, reg = build_model(word_embedding, pos_embedding, x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2)
    valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
    loss_cause = - tf.reduce_sum(y * tf.log(pred_c_tr)) / valid_num
    loss_position = - tf.reduce_sum(y_p * tf.log(pred_p)) / valid_num
    loss_op = loss_cause + loss_position * FLAGS.lambda1 + reg * FLAGS.l2_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)
    print('lambda1: {}'.format(FLAGS.lambda1))
    
    
    true_y_op = tf.argmax(y, 2)
    pred_y_op = tf.argmax(pred_c_tr, 2)
    pred_y_op_te = tf.argmax(pred_c_te, 2)
    print('build model done!\n')
    # Data Code Block
    y_p_data, y_data, x_data, sen_len_data, doc_len_data, word_distance, DGL_data = load_data(FLAGS.train_file_path, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
    # with tf.Session() as sess:
        kf, fold = KFold(n_splits=10), 1
        p_list, r_list, f1_list = [], [], []
        for train, test in kf.split(x_data):
            tr_x, tr_y, tr_y_p, tr_sen_len, tr_doc_len, tr_word_dis, tr_DGL = map(lambda x: x[train],
                [x_data, y_data, y_p_data, sen_len_data, doc_len_data, word_distance, DGL_data])
            te_x, te_y, te_y_p, te_sen_len, te_doc_len, te_word_dis, te_DGL = map(lambda x: x[test],
                [x_data, y_data, y_p_data, sen_len_data, doc_len_data, word_distance, DGL_data])
            
            sess.run(tf.global_variables_initializer())
            print('############# fold {} ###############'.format(fold))
            fold += 1
            max_f1 = 0.0
            print('train docs: {}    test docs: {}'.format(len(tr_y), len(te_y)))
            for i in xrange(FLAGS.training_iter):
                start_time = time.time() 
                step = 1
                # train
                for train, _ in get_batch_data(tr_x, tr_word_dis, tr_DGL, tr_sen_len, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_y, tr_y_p, FLAGS.batch_size):
                    _, loss, loss_c, loss_p, pred_y, true_y, doc_len_batch = sess.run(
                        [optimizer, loss_op, loss_cause, loss_position, pred_y_op, true_y_op, doc_len], feed_dict=dict(zip(placeholders, train)))
                    acc, p, r, f1 = acc_prf(pred_y, true_y, doc_len_batch)
                    print('step {}: loss {:.4f} loss_cause {:.4f} loss_position {:.4f} acc {:.4f} \np {:.4f} r {:.4f} f1 {:.4f}'.format(step, loss, loss_c, loss_p, acc, p, r, f1 ))
                    step = step + 1
                # test
                test = [te_x, te_word_dis, te_DGL, te_sen_len, te_doc_len, 1., 1., te_y, te_y_p]
                loss, pred_y_te, true_y, doc_len_batch = sess.run(
                        [loss_op, pred_y_op_te, true_y_op, doc_len], feed_dict=dict(zip(placeholders, test)))
                acc, p, r, f1 = acc_prf(pred_y_te, true_y, doc_len_batch)
                if f1 > max_f1:
                    max_acc, max_p, max_r, max_f1 = acc, p, r, f1
                print('\nepoch {}: loss {:.4f} acc {:.4f}\np {:.4f} r {:.4f} f1 {:.4f} max_f1 {:.4f}'.format(i, loss, acc, p, r, f1, max_f1 ))
                print("cost time: {:.1f}s\n".format(time.time()-start_time))
            print 'Optimization Finished!\n'
            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)  
        print_training_info()
        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])
        print("f1_score in 10 fold: {}\naverage : p {} r {} f1 {}\n".format(np.array(f1_list).reshape(-1,1), p, r, f1))


def main(_):
    FLAGS.use_DGL = ''
    FLAGS.train_file_path = 'data/clause_keywords.csv'
    FLAGS.log_file_name = 'PAE.log'
    FLAGS.scope = 'PAE'
    run()

    FLAGS.use_DGL = 'use'
    FLAGS.train_file_path = 'data/reordered_clause_keywords.csv'
    FLAGS.log_file_name = 'PAEDGL.log'
    FLAGS.scope = 'PAEDGL'
    run()


if __name__ == '__main__':
    tf.app.run()