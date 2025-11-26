'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''

import os
from time import time
from DataModule import DataModule

class DataUtil():
    def __init__(self, conf):
        self.conf = conf
        #print('DataUtil, Line12, test- conf data_dir:%s' % self.conf.data_dir)

    def initializeRankingHandle(self):
        #t0 = time()
        self.createTrainHandle()
        self.createEvaluateHandle()
        #t1 = time()
        #print('Prepare data cost:%.4fs' % (t1 - t0))
    
    def createTrainHandle(self):
        data_dir = self.conf.data_dir
        #rating
        train_rating_filename = "%s/%s.train.rating" % (data_dir, self.conf.data_name)
        val_rating_filename = "%s/%s.val.rating" % (data_dir, self.conf.data_name)
        test_rating_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)

        train_rating_pop_path = "%s/%s_rating_pop_train.npy" % (data_dir, self.conf.data_name)
        val_rating_pop_path = "%s/%s_rating_pop_val.npy" % (data_dir, self.conf.data_name)
        test_rating_pop_path = "%s/%s_rating_pop_test.npy" % (data_dir, self.conf.data_name)

        #social
        train_link_filename = "%s/%s.train.link" % (data_dir, self.conf.data_name)
        val_link_filename = "%s/%s.val.link" % (data_dir, self.conf.data_name)
        test_link_filename = "%s/%s.test.link" % (data_dir, self.conf.data_name)

        train_link_pop_path = "%s/%s_link_pop_train.npy" % (data_dir, self.conf.data_name)
        val_link_pop_path = "%s/%s_link_pop_val.npy" % (data_dir, self.conf.data_name)
        test_link_pop_path = "%s/%s_link_pop_test.npy" % (data_dir, self.conf.data_name)

        self.train_rating = DataModule(self.conf, train_rating_filename, train_rating_pop_path)
        self.val_rating = DataModule(self.conf, val_rating_filename, val_rating_pop_path)
        self.test_rating = DataModule(self.conf, test_rating_filename, test_rating_pop_path)
        self.train_social = DataModule(self.conf, train_link_filename, train_link_pop_path)
        self.val_social = DataModule(self.conf, val_link_filename, val_link_pop_path)
        self.test_social = DataModule(self.conf, test_link_filename, test_link_pop_path)

    def createEvaluateHandle(self):
        data_dir = self.conf.data_dir
        test_rating_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)
        test_link_filename = "%s/%s.test.link" % (data_dir, self.conf.data_name)

        test_rating_pop_path = "%s/%s_rating_pop_test.npy" % (data_dir, self.conf.data_name)
        test_link_pop_path = "%s/%s_link_pop_test.npy" % (data_dir, self.conf.data_name)

        self.test_rating_eva = DataModule(self.conf, test_rating_filename, test_rating_pop_path)
        self.test_social_eva = DataModule(self.conf, test_link_filename, test_link_pop_path)
