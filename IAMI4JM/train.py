'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''

import os, sys, shutil

from time import time
import numpy as np
import tensorflow as tf
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #ignore the warnings 

from Logging import Logging

def start(conf, data, model, evaluate):
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # define log name 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(os.getcwd(), 'log/%s_%s_[r_feature_%s_s_feature_%s].log' % (conf.data_name, conf.model_name, conf.r_feature, conf.s_feature))

    np.random.seed(conf.seed)
    tf.set_random_seed(conf.seed)

    # start to prepare data for training and evaluating
    data.initializeRankingHandle()

    d_train_rating, d_val_rating, d_test_rating, d_test_rating_eva = data.train_rating, data.val_rating, data.test_rating, data.test_rating_eva
    d_train_social, d_val_social, d_test_social, d_test_social_eva = data.train_social, data.val_social, data.test_social, data.test_social_eva

    print('System start to load rating data...')
    t0 = time()
    d_train_rating.initializeRankingTrain()
    d_val_rating.initializeRankingVT()
    d_test_rating.initializeRankingVT()
    d_test_rating_eva.initalizeRankingEva()
    t1 = time()
    print('Rating Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    print('System start to load social data...')
    t0 = time()
    d_train_social.initializeSocialTrain()
    d_val_social.initializeSocialVT()
    d_test_social.initializeSocialVT()
    d_test_social_eva.initalizeSocialEva()
    t1 = time()
    print('Social Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    # prepare model necessary data.
    data_dict = d_train_rating.prepareModelSupplement(model)
    s_data_dict = d_train_social.prepareModelSocialSupplement(model)
    data_dict.update(s_data_dict)
    model.inputSupply(data_dict)
    model.startConstructGraph()

    # standard tensorflow running environment initialize
    tf_conf = tf.ConfigProto()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    tf_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_conf)
    sess.run(model.init)

    # saver = tf.train.Saver()
    # save_path = os.path.join(os.getcwd(),'model/','model1.ckpt')
    # model_path = saver.save(sess, save_path)
    # print("Model saved in path: %s" % model_path)

    # if conf.pretrain_flag == 1:
    #     saver.restore(sess, save_path)

    # set debug_flag=0, doesn't print any results
    log = Logging(log_path)
    print()
    log.record('Following will output the evaluation of the model:')

    log.record("========== Configuration ==========")
    for section in conf.conf.sections():
        for (key, value) in conf.conf.items(section):
            log.record("%s = %s" % (key, value))
    log.record("==================================")

    # Start Training !!!
    for epoch in range(1, conf.epochs+1):
        # optimize model with training data and compute train loss
        tmp_train_loss = []
        tmp_r_loss = []
        tmp_s_loss = []
        t0 = time()

        #tmp_total_list = []
        while d_train_rating.terminal_flag:
            d_train_rating.getTrainRankingBatch()
            d_train_rating.linkedMap()

            d_train_social.getTrainSocialBatch()
            d_train_social.linkedMapSocial()

            train_feed_dict = {}
            for (key, value) in model.map_dict['r_train'].items():
                train_feed_dict[key] = d_train_rating.data_dict[value]
            for (key, value) in model.map_dict['s_train'].items():
                train_feed_dict[key] = d_train_social.data_dict[value]

            [sub_train_loss, r_loss, s_loss, r_three_loss, s_three_loss, _] = sess.run(\
                [model.map_dict['out']['total_loss'],model.map_dict['out']['r_train'],model.map_dict['out']['s_train'],model.map_dict['out']['r_three_loss'], model.map_dict['out']['s_three_loss'], model.opt], feed_dict=train_feed_dict)
            tmp_train_loss.append(sub_train_loss)
            tmp_r_loss.append(r_loss)
            tmp_s_loss.append(s_loss)
        train_loss = np.mean(tmp_train_loss)

        log.record("rating three part loss:%s"%(r_three_loss))
        log.record("social three part loss:%s"%(s_three_loss))
        log.record("rating loss:%f"%(np.mean(tmp_r_loss)))
        log.record("social loss:%f"%(np.mean(tmp_s_loss)))
        t1 = time()

        # compute rating val loss and test loss
        d_val_rating.getVTRankingOneBatch()
        d_val_rating.linkedMap()
        r_val_feed_dict = {}
        for (key, value) in model.map_dict['r_val'].items():
            r_val_feed_dict[key] = d_val_rating.data_dict[value]
        r_val_loss = sess.run(model.map_dict['out']['r_val'], feed_dict=r_val_feed_dict)

        d_test_rating.getVTRankingOneBatch()
        d_test_rating.linkedMap()
        r_test_feed_dict = {}
        for (key, value) in model.map_dict['r_test'].items():
            r_test_feed_dict[key] = d_test_rating.data_dict[value]
        r_test_loss = sess.run(model.map_dict['out']['r_test'], feed_dict=r_test_feed_dict)

        # compute social val loss and test loss
        d_val_social.getVTSocialOneBatch()
        d_val_social.linkedMapSocial()
        s_val_feed_dict = {}
        for (key, value) in model.map_dict['s_val'].items():
            s_val_feed_dict[key] = d_val_social.data_dict[value]
        s_val_loss = sess.run(model.map_dict['out']['s_val'], feed_dict=s_val_feed_dict)

        d_test_social.getVTSocialOneBatch()
        d_test_social.linkedMapSocial()
        s_test_feed_dict = {}
        for (key, value) in model.map_dict['s_test'].items():
            s_test_feed_dict[key] = d_test_social.data_dict[value]
        s_test_loss = sess.run(model.map_dict['out']['s_test'], feed_dict=s_test_feed_dict)
        t2 = time()

        # start evaluate model performance, hr and ndcg
        def getPositiveRatingPredictions():
            d_test_rating_eva.getEvaPositiveBatch()
            d_test_rating_eva.linkedRankingEvaMap()
            eva_feed_dict = {}
            for (key, value) in model.map_dict['r_eva'].items():
                eva_feed_dict[key] = d_test_rating_eva.data_dict[value]
            positive_predictions = sess.run(
                model.map_dict['out']['r_eva'],
                feed_dict=eva_feed_dict
            )
            return positive_predictions

        def getNegativeRatingPredictions():
            negative_predictions = {}
            terminal_flag = 1
            while terminal_flag:
                batch_user_list, terminal_flag = d_test_rating_eva.getEvaRankingBatch()
                d_test_rating_eva.linkedRankingEvaMap()
                eva_feed_dict = {}
                for (key, value) in model.map_dict['r_eva'].items():
                    eva_feed_dict[key] = d_test_rating_eva.data_dict[value]
                index = 0
                tmp_negative_predictions = np.reshape(
                    sess.run(
                        model.map_dict['out']['r_eva'],
                        feed_dict=eva_feed_dict
                    ),
                    [-1, conf.num_evaluate])
                for u in batch_user_list:
                    negative_predictions[u] = tmp_negative_predictions[index]
                    index = index + 1
            return negative_predictions


        def getPositiveSocialPredictions():
            d_test_social_eva.getEvaSocialPositiveBatch()
            d_test_social_eva.linkedSocialEvaMap()
            eva_feed_dict = {}
            for (key, value) in model.map_dict['s_eva'].items():
                eva_feed_dict[key] = d_test_social_eva.data_dict[value]
            positive_predictions = sess.run(
                model.map_dict['out']['s_eva'],
                feed_dict=eva_feed_dict
            )
            return positive_predictions

        def getNegativeSocialPredictions():
            negative_predictions = {}
            terminal_flag = 1
            while terminal_flag:
                batch_user_list, terminal_flag = d_test_social_eva.getEvaSocialBatch()
                d_test_social_eva.linkedSocialEvaMap()
                eva_feed_dict = {}
                for (key, value) in model.map_dict['s_eva'].items():
                    eva_feed_dict[key] = d_test_social_eva.data_dict[value]
                index = 0
                tmp_negative_predictions = np.reshape(
                    sess.run(
                        model.map_dict['out']['s_eva'],
                        feed_dict=eva_feed_dict
                    ),
                    [-1, conf.num_social_evaluate])
                for u in batch_user_list:
                    negative_predictions[u] = tmp_negative_predictions[index]
                    index = index + 1
            return negative_predictions

        tt2 = time()

        r_index_dict = d_test_rating_eva.eva_index_dict
        s_index_dict = d_test_social_eva.eva_social_index_dict

        r_positive_predictions = getPositiveRatingPredictions()
        r_negative_predictions = getNegativeRatingPredictions()
        s_positive_predictions = getPositiveSocialPredictions()
        s_negative_predictions = getNegativeSocialPredictions()

        d_test_rating_eva.index = 0 # !!!important, prepare for new batch
        d_test_social_eva.index = 0

        hr_10, ndcg_10, pre_10, rec_10, f1_10 = evaluate.evaluateRankingPerformance(\
            r_index_dict, r_positive_predictions, r_negative_predictions, conf.topk, conf.num_procs)
        hr_5, ndcg_5, pre_5, rec_5, f1_5 = evaluate.evaluateRankingPerformance(\
            r_index_dict, r_positive_predictions, r_negative_predictions, conf.top5, conf.num_procs)
        hr_15, ndcg_15, pre_15, rec_15, f1_15 = evaluate.evaluateRankingPerformance(\
            r_index_dict, r_positive_predictions, r_negative_predictions, conf.top15, conf.num_procs)

        s_hr_10, s_ndcg_10, s_pre_10, s_rec_10, s_f1_10 = evaluate.evaluateRankingPerformance(\
            s_index_dict, s_positive_predictions, s_negative_predictions, conf.topk, conf.num_procs)
        s_hr_5, s_ndcg_5, s_pre_5, s_rec_5, s_f1_5 = evaluate.evaluateRankingPerformance(\
            s_index_dict, s_positive_predictions, s_negative_predictions, conf.top5, conf.num_procs)
        s_hr_15, s_ndcg_15, s_pre_15, s_rec_15, s_f1_15 = evaluate.evaluateRankingPerformance(\
            s_index_dict, s_positive_predictions, s_negative_predictions, conf.top15, conf.num_procs)
        tt3 = time()
                
        # print log to console and log_file
        log.record('Epoch:%d, compute loss cost:%.4fs, train loss:%.4f, rating test loss:%.4f, social test loss:%.4f' % \
            (epoch, (t2-t0), train_loss, r_test_loss, s_test_loss))
        log.record('Evaluate cost:%.4fs \n \
                    "Rating: \t\t Social: \n \
                    "Top5: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f \t sTop5: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f  \n \
                    "Top10: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f \t sTop10: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f \n \
                    "Top15: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f \t sTop15: hr:%.4f, ndcg:%.4f, pre:%.4f, recall:%.4f f1:%.4f ' % ((tt3-tt2), hr_5, ndcg_5, pre_5, rec_5, f1_5, s_hr_5, s_ndcg_5, s_pre_5, s_rec_5, s_f1_5, \
                                                                                                                                                         hr_10, ndcg_10, pre_10, rec_10, f1_10, s_hr_10, s_ndcg_10, s_pre_10, s_rec_10, s_f1_10,\
                                                                                                                                                         hr_15, ndcg_15, pre_15, rec_15, f1_15, s_hr_15, s_ndcg_15, s_pre_15, s_rec_15, s_f1_15))
        log.record('------------------------------------------------------------------')

        ## reset train data pointer, and generate new negative data
        d_train_rating.generateTrainNegative()
        d_train_social.generateSocialTrainNegative()
