#coding=utf-8
'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''
import tensorflow as tf
import numpy as np
from ipdb import set_trace
# import tensorflow_probability as tfp


class IAMI4JM():
    def __init__(self, conf):
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX'
        )
        
        # 实验配置标志
        self.use_cross_attention = True      # 是否使用特征交叉注意力


    def startConstructGraph(self):
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()

    def inputSupply(self, data_dict):
        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']

        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']

        # prepare sparse matrix, in order to compute user's embedding from social neighbors and consumed items
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)

        self.social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input,
            dense_shape=self.consumed_items_dense_shape
        )
    
    #free embedding如何生成的？
    #作用？-转换成标准正态分布
    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1]) #计算均值和方差
        y = (x - mean) * 0.2 / tf.sqrt(var)
        return y

    def generateUserEmbeddingFromSocialNeighbors(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.social_neighbors_sparse_matrix, current_user_embedding # M*M x M*K = M*K
        )
        return user_embedding_from_social_neighbors
    
    def generateUserEmebddingFromConsumedItems(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.consumed_items_sparse_matrix, current_item_embedding # M*N x N*K = M*K
        )
        return user_embedding_from_consumed_items

    def initializeNodes(self):
        ##### Input #####
        ### Rating ###
        self.pop_item_input = tf.placeholder("int32", [None, 1]) # Get item embedding from the core_item_input
        self.pop_user_input = tf.placeholder("int32", [None, 1]) # Get user embedding from the core_user_input
        self.pop_labels_input = tf.placeholder("float32", [None, 1])

        self.unpop_item_input = tf.placeholder("int32", [None, 1])
        self.unpop_user_input = tf.placeholder("int32", [None, 1])
        self.unpop_labels_input = tf.placeholder("float32", [None, 1])

        self.eva_item_input = tf.placeholder("int32", [None, 1])
        self.eva_user_input = tf.placeholder("int32", [None, 1])

        ### Social ###
        self.pop_user1_input = tf.placeholder("int32", [None, 1])
        self.pop_user2_input = tf.placeholder("int32", [None, 1])
        self.pop_s_labels_input = tf.placeholder("float32", [None, 1])

        self.unpop_user1_input = tf.placeholder("int32", [None, 1])
        self.unpop_user2_input = tf.placeholder("int32", [None, 1])
        self.unpop_s_labels_input = tf.placeholder("float32", [None, 1])

        self.eva_user1_input = tf.placeholder("int32", [None, 1])
        self.eva_user2_input = tf.placeholder("int32", [None, 1])




        ##### Free Embedding #####
        ### Pop ###
        self.user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding')
        self.item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding')
        self.user_social_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_social_embedding')

        ### Unpop ###
        self.unpop_user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='unpop_user_embedding')
        self.unpop_item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='unpop_item_embedding')

        self.unpop_user_social_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='unpop_user_social_embedding')


        # self.user_review_vector_matrix = tf.constant(\
        #     np.load(self.conf.user_review_vector_matrix), dtype=tf.float32) #shape=(17237, 150)
        # self.item_review_vector_matrix = tf.constant(\
        #     np.load(self.conf.item_review_vector_matrix), dtype=tf.float32) #shape=(38342, 150)
        
        self.reduce_dimension_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='reduce_dimension_layer')
        self.item_fusion_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='item_fusion_layer')
        self.user_fusion_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='user_fusion_layer')



    def feature_cross_attention(self, embedding1, embedding2):
        # 拼接两个特征向量
        concat_embedding = tf.concat([embedding1, embedding2], axis=-1)
        # 计算第一个任务的注意力权重
        attention_weights = tf.nn.softmax(
            tf.layers.dense(concat_embedding, units=2), 
            axis=1
        )
        # 扩展维度以便广播
        weight_1 = tf.expand_dims(attention_weights[:, 0], axis=1)
        weight_2 = tf.expand_dims(attention_weights[:, 1], axis=1)
        # 计算加权组合
        fusion_vector = weight_1 * embedding1 + weight_2 * embedding2
        
        return fusion_vector




    def info_nce_loss(self, logu):

        # 联合项：f(x[i], y[i]) 的平均
        joint_term = tf.reduce_mean(tf.linalg.diag_part(logu), axis=-1)

        # 边缘项：log(sum_j exp(f(x[i], y[j]))) 的平均，再减去 log(batch_size)
        batch_size = tf.cast(tf.shape(logu)[-1], logu.dtype)
        logsumexp = tf.reduce_logsumexp(logu, axis=-1)  # 每行做 logsumexp
        marginal_term = tf.reduce_mean(logsumexp, axis=-1) - tf.math.log(batch_size)

        # 最终 InfoNCE 下界（注意不是负号）
        return joint_term - marginal_term




    def constructTrainGraph(self):
        # handle review information, map the origin review into the new space and 
        # first_user_review_vector_matrix = self.convertDistribution(self.user_review_vector_matrix)
        # first_item_review_vector_matrix = self.convertDistribution(self.item_review_vector_matrix)
        # self.user_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_user_review_vector_matrix)
        # self.item_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_item_review_vector_matrix)
        # second_user_review_vector_matrix = self.convertDistribution(self.user_reduce_dim_vector_matrix)
        # second_item_review_vector_matrix = self.convertDistribution(self.item_reduce_dim_vector_matrix)

        # compute item embedding
        self.pop_final_item_embedding = self.fusion_item_embedding \
                             = self.item_embedding #+ second_item_review_vector_matrix

        self.unpop_final_item_embedding = self.fusion_item_embedding \
                             = self.unpop_item_embedding #+ second_item_review_vector_matrix

        # compute user rating embedding
        user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems(self.pop_final_item_embedding)

        self.fusion_user_embedding = self.user_embedding #+ second_user_review_vector_matrix
        first_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(self.fusion_user_embedding)
        second_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(first_gcn_user_embedding)
        self.pop_final_user_embedding = first_gcn_user_embedding + second_gcn_user_embedding + user_embedding_from_consumed_items 
        # self.pop_final_user_embedding = self.fusion_user_embedding

        
        self.unpop_fusion_user_embedding = self.unpop_user_embedding #+ second_user_review_vector_matrix
        unpop_first_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(self.unpop_fusion_user_embedding)
        unpop_second_gcn_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(unpop_first_gcn_user_embedding)
        self.unpop_final_user_embedding = unpop_first_gcn_user_embedding + unpop_second_gcn_user_embedding + user_embedding_from_consumed_items 
        # self.unpop_final_user_embedding = self.unpop_fusion_user_embedding
        
        
        # compute user social embedding
        self.fusion_social_user_embedding = self.user_social_embedding
        first_gcn_social_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(self.fusion_social_user_embedding)
        second_gcn_social_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(first_gcn_social_user_embedding)
        self.pop_final_user_social_embedding = self.fusion_social_user_embedding + second_gcn_social_user_embedding
        # self.pop_final_user_social_embedding = self.fusion_social_user_embedding

        self.unpop_fusion_social_user_embedding = self.unpop_user_social_embedding
        unpop_first_gcn_social_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(self.unpop_fusion_social_user_embedding)
        unpop_second_gcn_social_user_embedding = self.generateUserEmbeddingFromSocialNeighbors(unpop_first_gcn_social_user_embedding)
        self.unpop_final_user_social_embedding = self.unpop_fusion_social_user_embedding + unpop_second_gcn_social_user_embedding
        # self.unpop_final_user_social_embedding = self.unpop_fusion_social_user_embedding

        # concate
        if self.use_cross_attention:
            self.final_item_embedding = self.feature_cross_attention(self.pop_final_item_embedding, self.unpop_final_item_embedding)
            self.final_user_embedding = self.feature_cross_attention(self.pop_final_user_embedding, self.unpop_final_user_embedding)
            self.final_user_social_embedding = self.feature_cross_attention(self.pop_final_user_social_embedding, self.unpop_final_user_social_embedding)
        else:
            # 简单平均或者直接使用pop版本
            self.final_item_embedding = (self.pop_final_item_embedding + self.unpop_final_item_embedding) * 0.5
            self.final_user_embedding = (self.pop_final_user_embedding + self.unpop_final_user_embedding) * 0.5
            self.final_user_social_embedding = (self.pop_final_user_social_embedding + self.unpop_final_user_social_embedding) * 0.5
        

        # ############ trick:特征交叉 ##############
        # fusion_user_embedding = tf.concat([self.final_user_embedding, self.final_user_social_embedding], axis=1)

        # pref_attention_weights = tf.nn.softmax(
        #     tf.layers.dense(fusion_user_embedding, units=2, name='attention_pref'), axis=1
        # )
        # social_attention_weights = tf.nn.softmax(
        #     tf.layers.dense(fusion_user_embedding, units=2, name='attention_social'), axis=1
        # )
        # pref_weight_1 = tf.expand_dims(pref_attention_weights[:, 0], axis=1)
        # pref_weight_2 = tf.expand_dims(pref_attention_weights[:, 1], axis=1)
        # social_weight_1 = tf.expand_dims(social_attention_weights[:, 0], axis=1)
        # social_weight_2 = tf.expand_dims(social_attention_weights[:, 1], axis=1)

        # self.user_pref_task_vector = pref_weight_1 * self.final_user_embedding + pref_weight_2 * self.final_user_social_embedding
        # self.user_social_task_vector = social_weight_1 * self.final_user_embedding + social_weight_2 * self.final_user_social_embedding
        # #########################################

        
        ############ trick:特征交叉 ##############
        r_feature = self.conf.r_feature
        s_feature = self.conf.s_feature


        self.user_pref_task_vector = r_feature * self.final_user_embedding + (1 - r_feature) * self.final_user_social_embedding
        self.user_social_task_vector = s_feature * self.final_user_social_embedding + (1 - s_feature) * self.final_user_embedding
        #########################################

        # ############ trick2:轻量稳定的特征交叉（替换之前方案） ##############
        # # 设计目标：
        # # 1) 减少参数与深度，降低过拟合与梯度震荡风险
        # # 2) 显式分解为 shared + diff 结构，便于两个任务各取所需
        # # 3) 加入低秩双线性交互捕捉协同模式，但限制秩避免噪声放大
        # # 4) 维度级 gating 控制“差异”注入强度；双任务对称更新，保持稳定
        # # 5) 保留可学习缩放系数，便于训练自动调节交互占比
        # with tf.variable_scope('light_feature_cross', reuse=tf.AUTO_REUSE):
        #     d = self.conf.dimension
        #     # shared / diff 分解
        #     shared = 0.5 * (self.final_user_embedding + self.final_user_social_embedding)          # 公共偏好
        #     diff   = self.final_user_embedding - self.final_user_social_embedding                  # 差异信息

        #     # 维度级 gating：输入 [shared, diff]，决定差异在各维度注入强度
        #     gating_input = tf.concat([shared, diff], axis=1)                                       # [N, 2d]
        #     gate = tf.layers.dense(gating_input, units=d, activation=tf.nn.sigmoid, name='dim_gate')  # [N, d]

        #     # 低秩双线性交互：捕捉协同模式  (u P) ⊙ (s P) P^T  （秩 r << d）
        #     rank = max(8, d // 16)
        #     P = tf.get_variable('bilinear_P', shape=[d, rank], initializer=tf.glorot_uniform_initializer())
        #     u_proj = tf.matmul(self.final_user_embedding, P)       # [N, r]
        #     s_proj = tf.matmul(self.final_user_social_embedding, P)# [N, r]
        #     bilinear_core = u_proj * s_proj                        # [N, r]
        #     bilinear = tf.matmul(bilinear_core, P, transpose_b=True)  # 回到 [N, d]

        #     # 可学习缩放，限制初始影响
        #     bilinear_scale = tf.get_variable('bilinear_scale', shape=[], initializer=tf.constant_initializer(0.2))
        #     diff_scale     = tf.get_variable('diff_scale',     shape=[], initializer=tf.constant_initializer(1.0))

        #     # 任务向量构造： shared ± gate*(diff*diff_scale) + bilinear_scale*bilinear
        #     pref_vec_raw   = shared + gate * (diff * diff_scale) + bilinear_scale * bilinear
        #     social_vec_raw = shared - gate * (diff * diff_scale) + bilinear_scale * bilinear

        #     # Layer Norm（简化实现：对每个向量做均值方差归一再线性缩放）——比直接 L2 更稳定尺度
        #     def layer_norm(x, name):
        #         with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #             mean, var = tf.nn.moments(x, axes=[1], keep_dims=True)
        #             norm = (x - mean) / tf.sqrt(var + 1e-6)
        #             gamma = tf.get_variable('gamma', shape=[d], initializer=tf.ones_initializer())
        #             beta  = tf.get_variable('beta',  shape=[d], initializer=tf.zeros_initializer())
        #             return norm * gamma + beta
        #     pref_vec_ln   = layer_norm(pref_vec_raw, 'ln_pref')
        #     social_vec_ln = layer_norm(social_vec_raw, 'ln_social')

        #     # 最终再做一次 L2，保证与原后续点积尺度兼容
        #     self.user_pref_task_vector   = tf.nn.l2_normalize(pref_vec_ln, axis=1)
        #     self.user_social_task_vector = tf.nn.l2_normalize(social_vec_ln, axis=1)
        # #########################################


        ############ EVA ##############
        ### Rating ###
        eva_latest_item_latent = tf.gather_nd(self.final_item_embedding, self.eva_item_input)
        eva_latest_user_latent = tf.gather_nd(self.user_pref_task_vector, self.eva_user_input)
        r_eva_predict_vector = tf.multiply(eva_latest_item_latent, eva_latest_user_latent)
        self.r_eva_prediction = tf.sigmoid(tf.reduce_sum(r_eva_predict_vector, -1, keepdims=True))

        ### Social ###
        eva_latest_user1_latent = tf.gather_nd(self.user_social_task_vector, self.eva_user1_input)
        eva_latest_user2_latent = tf.gather_nd(self.user_social_task_vector, self.eva_user2_input)
        s_eva_predict_vector = tf.multiply(eva_latest_user1_latent, eva_latest_user2_latent)
        self.s_eva_prediction = tf.sigmoid(tf.reduce_sum(s_eva_predict_vector, -1, keepdims=True))
        ###############################



        ##### compute interest loss and comformity loss
        ### Rating ###
        int_latest_user_latent = tf.gather_nd(self.pop_final_user_embedding, self.pop_user_input)
        int_latest_item_latent = tf.gather_nd(self.pop_final_item_embedding, self.pop_item_input)
        com_latest_user_latent = tf.gather_nd(self.unpop_final_user_embedding, self.unpop_user_input)
        com_latest_item_latent = tf.gather_nd(self.unpop_final_item_embedding, self.unpop_item_input)
        
        r_int_predict_vector = tf.multiply(int_latest_user_latent, int_latest_item_latent)
        r_com_predict_vector = tf.multiply(com_latest_user_latent, com_latest_item_latent)
        self.r_int_prediction = tf.sigmoid(tf.reduce_sum(r_int_predict_vector, 1, keepdims=True))
        self.r_com_prediction = tf.sigmoid(tf.reduce_sum(r_com_predict_vector, 1, keepdims=True))
        self.r_int_loss = tf.nn.l2_loss(self.pop_labels_input - self.r_int_prediction)
        self.r_com_loss = tf.nn.l2_loss(self.unpop_labels_input - self.r_com_prediction)


        ### Social ###
        int_latest_user1_latent = tf.gather_nd(self.pop_final_user_social_embedding, self.pop_user1_input)
        int_latest_user2_latent = tf.gather_nd(self.pop_final_user_social_embedding, self.pop_user2_input)
        com_latest_user1_latent = tf.gather_nd(self.unpop_final_user_social_embedding, self.unpop_user1_input)
        com_latest_user2_latent = tf.gather_nd(self.unpop_final_user_social_embedding, self.unpop_user2_input)
        
        s_int_predict_vector = tf.multiply(int_latest_user1_latent, int_latest_user2_latent)
        s_com_predict_vector = tf.multiply(com_latest_user1_latent, com_latest_user2_latent)
        self.s_int_prediction = tf.sigmoid(tf.reduce_sum(s_int_predict_vector, 1, keepdims=True))
        self.s_com_prediction = tf.sigmoid(tf.reduce_sum(s_com_predict_vector, 1, keepdims=True))
        self.s_int_loss = tf.nn.l2_loss(self.pop_s_labels_input - self.s_int_prediction)
        self.s_com_loss = tf.nn.l2_loss(self.unpop_s_labels_input - self.s_com_prediction)


        ############ 新增: 剔除相同正样本 ##############
        ### Rating
        negative_mask = tf.not_equal(self.unpop_labels_input, 1)  # 假设1表示正样本

        filtered_unpop_user_input = tf.expand_dims(tf.boolean_mask(self.unpop_user_input, negative_mask), 1)
        filtered_unpop_item_input = tf.expand_dims(tf.boolean_mask(self.unpop_item_input, negative_mask), 1)
        filtered_unpop_labels_input = tf.expand_dims(tf.boolean_mask(self.unpop_labels_input, negative_mask), 1)

        total_item_list = tf.concat([self.pop_item_input, filtered_unpop_item_input], axis=0)
        total_user_list = tf.concat([self.pop_user_input, filtered_unpop_user_input], axis=0)
        total_label_list = tf.concat([self.pop_labels_input, filtered_unpop_labels_input], axis=0)

        ### Social
        s_negative_mask = tf.not_equal(self.unpop_s_labels_input, 1)  # 假设1表示正样本

        filtered_unpop_user1_input = tf.expand_dims(tf.boolean_mask(self.unpop_user1_input, s_negative_mask), 1)
        filtered_unpop_user2_input = tf.expand_dims(tf.boolean_mask(self.unpop_user2_input, s_negative_mask), 1)
        filtered_unpop_s_labels_input = tf.expand_dims(tf.boolean_mask(self.unpop_s_labels_input, s_negative_mask), 1)

        total_user1_list = tf.concat([self.pop_user1_input, filtered_unpop_user1_input], axis=0)
        total_user2_list = tf.concat([self.pop_user2_input, filtered_unpop_user2_input], axis=0)
        total_s_label_list = tf.concat([self.pop_s_labels_input, filtered_unpop_s_labels_input], axis=0)


        ##### compute rating click loss and social link loss
        latest_user_latent = tf.gather_nd(self.user_pref_task_vector, total_user_list)
        latest_item_latent = tf.gather_nd(self.final_item_embedding, total_item_list)
        latest_user1_latent = tf.gather_nd(self.user_social_task_vector, total_user1_list)
        latest_user2_latent = tf.gather_nd(self.user_social_task_vector, total_user2_list)

        r_predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
        self.r_prediction = tf.sigmoid(tf.reduce_sum(r_predict_vector, 1, keepdims=True))
        s_predict_vector = tf.multiply(latest_user1_latent, latest_user2_latent)
        self.s_prediction = tf.sigmoid(tf.reduce_sum(s_predict_vector, 1, keepdims=True))
        
        self.click_loss = tf.nn.l2_loss(total_label_list - self.r_prediction)
        self.link_loss = tf.nn.l2_loss(total_s_label_list - self.s_prediction)


        # compute rating discrepency loss
        unique_item_all = tf.unique(tf.reshape(total_item_list,[-1]))[0]
        item_int = tf.gather(self.item_embedding, unique_item_all)
        item_com = tf.gather(self.unpop_item_embedding, unique_item_all)
        unique_user_all = tf.unique(tf.reshape(total_user_list,[-1]))[0]
        user_int = tf.gather(self.user_embedding, unique_user_all)
        user_com = tf.gather(self.unpop_user_embedding, unique_user_all)
        # self.r_discrepency_loss = - tf.nn.l2_loss(item_int - item_com) - tf.nn.l2_loss(user_int - user_com)

        # compute social discrepency loss
        total_social_user_all = tf.concat([total_user1_list, total_user2_list], axis=0)
        unique_social_user_all = tf.unique(tf.reshape(total_social_user_all,[-1]))[0]
        social_user_int = tf.gather(self.user_social_embedding, unique_social_user_all)
        social_user_com = tf.gather(self.unpop_user_social_embedding, unique_social_user_all)
        # self.s_discrepency_loss = -tf.nn.l2_loss(social_user_int - social_user_com)


        # 归一化
        item_int = tf.nn.l2_normalize(item_int, axis=1)
        item_com = tf.nn.l2_normalize(item_com, axis=1)
        user_int = tf.nn.l2_normalize(user_int, axis=1)
        user_com = tf.nn.l2_normalize(user_com, axis=1)

        # 构造打分矩阵（点积）
        def compute_logu(x, y):
            return tf.matmul(x, y, transpose_b=True)

        item_logu = compute_logu(item_int, item_com)
        user_logu = compute_logu(user_int, user_com)

        self.r_discrepency_loss = self.info_nce_loss(item_logu) + self.info_nce_loss(user_logu)





        # 归一化
        social_user_int = tf.nn.l2_normalize(social_user_int, axis=1)
        social_user_com = tf.nn.l2_normalize(social_user_com, axis=1)

        social_logu = compute_logu(social_user_int, social_user_com)

        self.s_discrepency_loss = self.info_nce_loss(social_logu)



        # 计算 rating 用户嵌入和 social 用户嵌入的互信息 InfoNCE loss
        total_r_s_user_all = tf.concat([unique_user_all, unique_social_user_all], axis=0)
        unique_r_s_user_all = tf.unique(tf.reshape(total_r_s_user_all,[-1]))[0]
        r_gather_user = tf.gather(self.final_user_embedding, unique_r_s_user_all)
        s_gather_user = tf.gather(self.final_user_social_embedding, unique_r_s_user_all)

        user_feature = tf.nn.l2_normalize(r_gather_user, axis=1)
        social_user_feature = tf.nn.l2_normalize(s_gather_user, axis=1)

        rating_social_user_logu = tf.matmul(user_feature, social_user_feature, transpose_b=True)
        self.user_discrepency_loss = self.info_nce_loss(rating_social_user_logu)




        alpha = self.conf.alpha
        beta = self.conf.beta

        r_weight = self.conf.r_weight
        # s_weight = self.conf.s_weight
        s_weight = 1.0 - r_weight

        self.r_loss = self.click_loss + alpha*(self.r_int_loss + self.r_com_loss) + beta*self.r_discrepency_loss 
        self.s_loss = self.link_loss + alpha*(self.s_int_loss + self.s_com_loss) + beta*self.s_discrepency_loss

        self.loss = r_weight*self.r_loss + s_weight*self.s_loss + 0.0001*self.user_discrepency_loss
        self.opt_loss = r_weight*self.r_loss + s_weight*self.s_loss + 0.0001*self.user_discrepency_loss
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.user_embedding.op.name] = self.user_embedding
        variables_dict[self.unpop_user_embedding.op.name] = self.unpop_user_embedding
        variables_dict[self.item_embedding.op.name] = self.item_embedding
        variables_dict[self.unpop_item_embedding.op.name] = self.unpop_item_embedding
        variables_dict[self.user_social_embedding.op.name] = self.user_social_embedding
        variables_dict[self.unpop_user_social_embedding.op.name] = self.unpop_user_social_embedding

        for v in self.reduce_dimension_layer.variables:
            variables_dict[v.op.name] = v
                
        self.saver = tf.train.Saver(variables_dict)
        ############################# Save Variables #################################
    
    def defineMap(self):
        map_dict = {}
        # Rating
        map_dict['r_train'] = {
            self.pop_user_input: 'POP_USER_LIST', 
            self.pop_item_input: 'POP_ITEM_LIST', 
            self.pop_labels_input: 'POP_LABEL_LIST',
            self.unpop_user_input: 'UNPOP_USER_LIST', 
            self.unpop_item_input: 'UNPOP_ITEM_LIST', 
            self.unpop_labels_input: 'UNPOP_LABEL_LIST'
        }
        
        map_dict['r_val'] = {
            self.pop_user_input: 'POP_USER_LIST', 
            self.pop_item_input: 'POP_ITEM_LIST', 
            self.pop_labels_input: 'POP_LABEL_LIST',
            self.unpop_user_input: 'UNPOP_USER_LIST', 
            self.unpop_item_input: 'UNPOP_ITEM_LIST', 
            self.unpop_labels_input: 'UNPOP_LABEL_LIST'
        }

        map_dict['r_test'] = {
            self.pop_user_input: 'POP_USER_LIST', 
            self.pop_item_input: 'POP_ITEM_LIST', 
            self.pop_labels_input: 'POP_LABEL_LIST',
            self.unpop_user_input: 'UNPOP_USER_LIST', 
            self.unpop_item_input: 'UNPOP_ITEM_LIST', 
            self.unpop_labels_input: 'UNPOP_LABEL_LIST'
        }

        map_dict['r_eva'] = {
            self.eva_user_input: 'EVA_USER_LIST', 
            self.eva_item_input: 'EVA_ITEM_LIST'
        }
        # Social 
        map_dict['s_train'] = {
            self.pop_user1_input: 'POP_USER1_LIST', 
            self.pop_user2_input: 'POP_USER2_LIST', 
            self.pop_s_labels_input: 'POP_SOCIAL_LABEL_LIST',
            self.unpop_user1_input: 'UNPOP_USER1_LIST', 
            self.unpop_user2_input: 'UNPOP_USER2_LIST', 
            self.unpop_s_labels_input: 'UNPOP_SOCIAL_LABEL_LIST'
        }
        
        map_dict['s_val'] = {
            self.pop_user1_input: 'POP_USER1_LIST', 
            self.pop_user2_input: 'POP_USER2_LIST', 
            self.pop_s_labels_input: 'POP_SOCIAL_LABEL_LIST',
            self.unpop_user1_input: 'UNPOP_USER1_LIST', 
            self.unpop_user2_input: 'UNPOP_USER2_LIST', 
            self.unpop_s_labels_input: 'UNPOP_SOCIAL_LABEL_LIST'
        }

        map_dict['s_test'] = {
            self.pop_user1_input: 'POP_USER1_LIST', 
            self.pop_user2_input: 'POP_USER2_LIST', 
            self.pop_s_labels_input: 'POP_SOCIAL_LABEL_LIST',
            self.unpop_user1_input: 'UNPOP_USER1_LIST', 
            self.unpop_user2_input: 'UNPOP_USER2_LIST', 
            self.unpop_s_labels_input: 'UNPOP_SOCIAL_LABEL_LIST'
        }

        map_dict['s_eva'] = {
            self.eva_user1_input: 'EVA_USER1_LIST', 
            self.eva_user2_input: 'EVA_USER2_LIST', 
        }

        map_dict['out'] = {
            'r_train': self.r_loss,
            'r_val': self.r_loss,
            'r_test': self.r_loss,
            'r_eva': self.r_eva_prediction,
            's_train': self.s_loss,
            's_val': self.s_loss,
            's_test': self.s_loss,
            's_eva': self.s_eva_prediction,
            'total_loss': self.loss,
            'r_three_loss': [self.click_loss, self.r_int_loss, self.r_com_loss, self.r_discrepency_loss],
            's_three_loss': [self.link_loss, self.s_int_loss, self.s_com_loss, self.s_discrepency_loss]
        }

        self.map_dict = map_dict
