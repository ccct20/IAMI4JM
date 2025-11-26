#coding=utf-8
import math
import numpy as np
from ipdb import set_trace

class Evaluate():
    def __init__(self, conf):
        self.conf = conf

    def getIdcg(self, length):
        idcg = 0.0
        for i in range(length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        return idcg

    def getDcg(self, value):
        dcg = math.log(2) / math.log(value + 2) #排名越前值越大
        return dcg

    def getHr(self, value):
        hit = 1.0
        return hit

    def evaluateRankingPerformance(self, evaluate_index_dict, evaluate_real_rating_matrix, \
        evaluate_predict_rating_matrix, topK, num_procs, exp_flag=0, sp_name=None, result_file=None):
        user_list = list(evaluate_index_dict.keys())
        batch_size = len(user_list) / num_procs

        hr_list, ndcg_list = [], []
        precision_list, recall_list, f1_list = [], [], []
        index = 0
        for _ in range(num_procs):
            if index + batch_size < len(user_list):
                batch_user_list = user_list[index:index+batch_size]
                index = index + batch_size
            else:
                batch_user_list = user_list[index:len(user_list)]

            tmp_hr_list, tmp_ndcg_list, tmp_precision_list, tmp_recall_list, tmp_f1_list = self.getHrNdcgProc(evaluate_index_dict, evaluate_real_rating_matrix, \
                evaluate_predict_rating_matrix, topK, batch_user_list)
            #set_trace()
            hr_list.extend(tmp_hr_list)
            ndcg_list.extend(tmp_ndcg_list)
            precision_list.extend(tmp_precision_list)
            recall_list.extend(tmp_recall_list)
            f1_list.extend(tmp_f1_list)
        
        return np.mean(hr_list), np.mean(ndcg_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

    def getHrNdcgProc(self, 
        evaluate_index_dict, 
        evaluate_real_rating_matrix,
        evaluate_predict_rating_matrix, 
        topK, 
        user_list):

        tmp_hr_list, tmp_ndcg_list = [], []
        tmp_precision_list, tmp_recall_list, tmp_f1_list = [], [], []
        for u in user_list:
            real_item_index_list = evaluate_index_dict[u] #evaluate_index_dict包含有user对应的正样本在hash_data中的索引
            real_item_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_item_index_list])) #list(prediction(u-i))
            positive_length = len(real_item_rating_list)
            target_length = min(positive_length, topK)
           
            predict_rating_list = evaluate_predict_rating_matrix[u]
            real_item_rating_list.extend(predict_rating_list) # real + preidct
            sort_index = np.argsort(real_item_rating_list) #返回的是数组元素排序后的索引：从小到大
            sort_index = sort_index[::-1] #逆序取:从大到小

            user_hr_list = []
            user_ndcg_list = []
            user_precision_list = []
            user_recall_list = []
            user_f1_list = []
            hits_num = 0

            for idx in range(topK):
                ranking = sort_index[idx]
                if ranking < positive_length:
                    hits_num += 1
                    user_hr_list.append(self.getHr(idx))
                    user_ndcg_list.append(self.getDcg(idx))
                if ranking <= positive_length:
                    user_precision_list.append(self.getHr(idx))
                    user_recall_list.append(self.getHr(idx))

            idcg = self.getIdcg(target_length) #前target_length最大值

            tmp_hr = np.sum(user_hr_list) / target_length #前topk内正样本比例
            tmp_ndcg = np.sum(user_ndcg_list) / idcg #在前target_length理想最大值中正样本排位值占比
            tmp_precision = np.sum(user_precision_list) / topK   # precision
            #tmp_recall = np.sum(user_recall_list) / target_length
            tmp_recall = np.sum(user_recall_list) / positive_length
            #set_trace()
            if (tmp_precision + tmp_recall)==0:
                tmp_f1=0
            else:
                tmp_f1 = (2*tmp_precision*tmp_recall)/(tmp_precision+tmp_recall)
            
            
            
            tmp_hr_list.append(tmp_hr)
            tmp_ndcg_list.append(tmp_ndcg)
            tmp_precision_list.append(tmp_precision)
            tmp_recall_list.append(tmp_recall)
            tmp_f1_list.append(tmp_f1)

        return tmp_hr_list, tmp_ndcg_list, tmp_precision_list, tmp_recall_list, tmp_f1_list

   