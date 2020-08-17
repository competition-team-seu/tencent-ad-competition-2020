#!/usr/bin/env python
# coding: utf-8
import gc
import os
import time

import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from gensim.models import FastText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

data_list = pd.read_pickle('all_log_agg_wide_semi_input_1.pkl')
data_list.replace('\\N', '0', inplace=True)

train_labels = pd.read_csv('./train_preliminary/user.csv')


# Tokenizer 序列化文本
def set_tokenizer(docs, split_char=' ', max_len=100):
    '''
    输入
    docs:文本列表
    split_char:按什么字符切割
    max_len:截取的最大长度

    输出
    X:序列化后的数据
    word_index:文本和数字对应的索引
    '''
    tokenizer = Tokenizer(lower=False, char_level=False, split=split_char)
    tokenizer.fit_on_texts(docs)
    X = tokenizer.texts_to_sequences(docs)
    X = pad_sequences(X, maxlen=max_len, value=0)
    word_index = tokenizer.word_index
    return X, word_index


class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''

    def __init__(self, savedir, save_name="word2vector.model"):
        self.save_path = savedir + save_name
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss()  # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" %
              (self.epoch, epoch_loss, time_taken // 60, time_taken % 60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)
        self.pre_loss = cum_loss
        self.since = time.time()


# 得到embedding矩阵
def get_embedding_matrix(word_index, embed_size=256, Emed_path="w2v_256.txt"):
    embeddings_index = gensim.models.Word2Vec.load(Emed_path)
    nb_words = len(word_index) + 1
    embedding_matrix = np.zeros((nb_words, embed_size))
    count = 0
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            print('exception')
            embedding_vector = np.zeros(embed_size)
            count += 1
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("null cnt", count)
    return embedding_matrix


text_1_list = list(data_list["advertiser_id_agg_docs"])
text_2_list = list(data_list["ad_id_agg_docs"])
text_3_list = list(data_list['creative_id_agg_docs'])

text_4_list = list(data_list["product_id_agg_docs"])
text_6_list = list(data_list['industry_agg_docs'])

print('开始序列化')
x1, index_1 = set_tokenizer(text_1_list, split_char=' ', max_len=180)
x2, index_2 = set_tokenizer(text_2_list, split_char=' ', max_len=180)
x3, index_3 = set_tokenizer(text_3_list, split_char=' ', max_len=180)

x4, index_4 = set_tokenizer(text_4_list, split_char=' ', max_len=180)
x6, index_6 = set_tokenizer(text_6_list, split_char=' ', max_len=180)
print('序列化完成')
gc.collect()

emb1 = get_embedding_matrix(index_1, embed_size=256, Emed_path='./semi_model/advertiser_w2v_256.bin')
gc.collect()
print(emb1[1000])

emb2 = get_embedding_matrix(index_2, embed_size=256, Emed_path='./semi_model/ad_w2v_256.bin')
gc.collect()
print(emb2[1000])

emb3 = get_embedding_matrix(index_3, embed_size=256, Emed_path='./semi_model/creative_w2v_256.bin')
gc.collect()
print(emb3[1000])

emb4 = get_embedding_matrix(index_4, embed_size=256, Emed_path='./semi_model/product_w2v_256.bin')
gc.collect()
print(emb4[1000])

emb6 = get_embedding_matrix(index_6, embed_size=256, Emed_path='./semi_model/industry_w2v_256.bin')
gc.collect()
print(emb6[100])

# emb1_1 = get_embedding_matrix(index_1, embed_size=256, Emed_path='./semi_model/advertiser_w2v_256_v2.bin')
# gc.collect()
# print(emb1_1[1000])
#
# emb2_1 = get_embedding_matrix(index_2, embed_size=256, Emed_path='./semi_model/ad_w2v_256_v2.bin')
# gc.collect()
# print(emb2_1[1000])
#
# emb3_1 = get_embedding_matrix(index_3, embed_size=256, Emed_path='./semi_model/creative_w2v_256_v2.bin')
# gc.collect()
# print(emb3_1[1000])
#
# emb4_1 = get_embedding_matrix(index_4, embed_size=256, Emed_path='./semi_model/product_w2v_256_v2.bin')
# gc.collect()
# print(emb4_1[1000])

print('id嵌入载入完毕')


def tokenizer_space(text):
    return text.split(" ")


def convert_list_to_str(input_list):
    return " ".join(input_list)


cv = CountVectorizer(tokenizer=tokenizer_space, max_features=500)
product_category_train_data = [convert_list_to_str(t) for t in data_list['product_category']]
model = cv.fit(product_category_train_data)
product_category_train_words = model.transform(product_category_train_data)
product_category_train_features = product_category_train_words.toarray()

cv = CountVectorizer(tokenizer=tokenizer_space, max_features=100)
click_times_data = [convert_list_to_str(t) for t in data_list['click_times'].astype("str")]
model = cv.fit(click_times_data)
click_times_data_words = model.transform(click_times_data)
click_times_data_features = click_times_data_words.toarray()

data_list['click_times_sum'] = data_list['click_times'].apply(lambda x: [np.sum(x)])
click_times_sum = np.array(list(data_list['click_times_sum']))

hin_features = np.concatenate([product_category_train_features, click_times_data_features, click_times_sum], axis=1)
ss = StandardScaler()
ss.fit(hin_features)
hin_input = ss.transform(hin_features)
num_features_input = hin_input.shape[1]

print(hin_input)

from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects


def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})

from tensorflow.keras import *


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(int(input_shape[-1]),),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(int(input_shape[1]),),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


# def comprehensive_model_conv_v12(emb1, emb2, emb3, emb4, emb6, emb1_1, emb2_1, emb3_1, emb4_1, num_features_input):
def comprehensive_model_conv_v12(emb1, emb2, emb3, emb4, emb6, num_features_input):
    K.clear_session()

    emb_layer_1 = Embedding(
        input_dim=emb1.shape[0],
        output_dim=emb1.shape[1],
        weights=[emb1],
        input_length=180,
        trainable=False
    )
    emb_layer_2 = Embedding(
        input_dim=emb2.shape[0],
        output_dim=emb2.shape[1],
        weights=[emb2],
        input_length=180,
        trainable=False
    )
    emb_layer_3 = Embedding(
        input_dim=emb3.shape[0],
        output_dim=emb3.shape[1],
        weights=[emb3],
        input_length=180,
        trainable=False
    )
    emb_layer_4 = Embedding(
        input_dim=emb4.shape[0],
        output_dim=emb4.shape[1],
        weights=[emb4],
        input_length=180,
        trainable=False
    )
    emb_layer_6 = Embedding(
        input_dim=emb6.shape[0],
        output_dim=emb6.shape[1],
        weights=[emb6],
        input_length=180,
        trainable=False
    )

    # emb_layer_1_1 = Embedding(
    #     input_dim=emb1_1.shape[0],
    #     output_dim=emb1_1.shape[1],
    #     weights=[emb1_1],
    #     input_length=180,
    #     trainable=False
    # )
    # emb_layer_2_1 = Embedding(
    #     input_dim=emb2_1.shape[0],
    #     output_dim=emb2_1.shape[1],
    #     weights=[emb2_1],
    #     input_length=180,
    #     trainable=False
    # )
    # emb_layer_3_1 = Embedding(
    #     input_dim=emb3_1.shape[0],
    #     output_dim=emb3_1.shape[1],
    #     weights=[emb3_1],
    #     input_length=180,
    #     trainable=False
    # )
    # emb_layer_4_1 = Embedding(
    #     input_dim=emb4_1.shape[0],
    #     output_dim=emb4_1.shape[1],
    #     weights=[emb4_1],
    #     input_length=180,
    #     trainable=False
    # )

    seq1 = Input(shape=(180,))
    seq2 = Input(shape=(180,))
    seq3 = Input(shape=(180,))

    seq4 = Input(shape=(180,))
    seq6 = Input(shape=(180,))

    x1 = emb_layer_1(seq1)
    x2 = emb_layer_2(seq2)
    x3 = emb_layer_3(seq3)

    x4 = emb_layer_4(seq4)
    x6 = emb_layer_6(seq6)

    # x1_1 = emb_layer_1_1(seq1)
    # x2_1 = emb_layer_2_1(seq2)
    # x3_1 = emb_layer_3_1(seq3)
    # x4_1 = emb_layer_4_1(seq4)

    sdrop = SpatialDropout1D(rate=0.1)

    x1 = sdrop(x1)
    x2 = sdrop(x2)
    x3 = sdrop(x3)

    x4 = sdrop(x4)
    x6 = sdrop(x6)

    # x1_1 = sdrop(x1_1)
    # x2_1 = sdrop(x2_1)
    # x3_1 = sdrop(x3_1)
    # x4_1 = sdrop(x4_1)

    query = Dense(256, activation=custom_gelu)(x1)
    query = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(query)
    query = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(query)
    key = Dense(256, activation=custom_gelu)(x1)
    key = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(key)
    key = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(key)
    value = Dense(256, activation=custom_gelu)(x1)
    value = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(value)
    value = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(value)
    qk = Multiply()([query, key])
    attention = Dense(64, activation='softmax')(qk)
    a_value = Multiply()([attention, value])
    tran_out = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(a_value)
    tran_out = Lambda(lambda x: tf.reshape(x, (-1, 180, 256)))(tran_out)

    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(tran_out))
    att_1 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x))
    att_2 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_2_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x1_all = concatenate([att_1, merged_1, merged_1_avg, att_2, merged_2, merged_2_avg])

    query = Dense(256, activation=custom_gelu)(x2)
    query = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(query)
    query = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(query)
    key = Dense(256, activation=custom_gelu)(x2)
    key = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(key)
    key = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(key)
    value = Dense(256, activation=custom_gelu)(x2)
    value = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(value)
    value = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(value)
    qk = Multiply()([query, key])
    attention = Dense(64, activation='softmax')(qk)
    a_value = Multiply()([attention, value])
    tran_out = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(a_value)
    tran_out = Lambda(lambda x: tf.reshape(x, (-1, 180, 256)))(tran_out)

    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(tran_out))
    att_1 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x))
    att_2 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_2_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x2_all = concatenate([att_1, merged_1, merged_1_avg, att_2, merged_2, merged_2_avg])

    query = Dense(256, activation=custom_gelu)(x3)
    query = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(query)
    query = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(query)
    key = Dense(256, activation=custom_gelu)(x3)
    key = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(key)
    key = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(key)
    value = Dense(256, activation=custom_gelu)(x3)
    value = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(value)
    value = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(value)
    qk = Multiply()([query, key])
    attention = Dense(64, activation='softmax')(qk)
    a_value = Multiply()([attention, value])
    tran_out = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(a_value)
    tran_out = Lambda(lambda x: tf.reshape(x, (-1, 180, 256)))(tran_out)

    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(tran_out))
    att_1 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x))
    att_2 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_2_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x3_all = concatenate([att_1, merged_1, merged_1_avg, att_2, merged_2, merged_2_avg])

    query = Dense(256, activation=custom_gelu)(x4)
    query = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(query)
    query = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(query)
    key = Dense(256, activation=custom_gelu)(x4)
    key = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(key)
    key = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(key)
    value = Dense(256, activation=custom_gelu)(x4)
    value = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(value)
    value = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(value)
    qk = Multiply()([query, key])
    attention = Dense(64, activation='softmax')(qk)
    a_value = Multiply()([attention, value])
    tran_out = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(a_value)
    tran_out = Lambda(lambda x: tf.reshape(x, (-1, 180, 256)))(tran_out)

    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(tran_out))
    att_1 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x))
    att_2 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_2_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x4_all = concatenate([att_1, merged_1, merged_1_avg, att_2, merged_2, merged_2_avg])

    query = Dense(256, activation=custom_gelu)(x6)
    query = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(query)
    query = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(query)
    key = Dense(256, activation=custom_gelu)(x6)
    key = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(key)
    key = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(key)
    value = Dense(256, activation=custom_gelu)(x6)
    value = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(value)
    value = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(value)
    qk = Multiply()([query, key])
    attention = Dense(64, activation='softmax')(qk)
    a_value = Multiply()([attention, value])
    tran_out = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(a_value)
    tran_out = Lambda(lambda x: tf.reshape(x, (-1, 180, 256)))(tran_out)

    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(tran_out))
    att_1 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x))
    att_2 = Attention(180)(x)
    semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    merged_2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    merged_2_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    x6_all = concatenate([att_1, merged_1, merged_1_avg, att_2, merged_2, merged_2_avg])

    # query = Dense(256, activation=custom_gelu)(x1_1)
    # query = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(query)
    # query = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(query)
    # key = Dense(256, activation=custom_gelu)(x1_1)
    # key = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(key)
    # key = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(key)
    # value = Dense(256, activation=custom_gelu)(x1_1)
    # value = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(value)
    # value = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(value)
    # qk = Multiply()([query, key])
    # attention = Dense(64, activation='softmax')(qk)
    # a_value = Multiply()([attention, value])
    # tran_out = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(a_value)
    # tran_out = Lambda(lambda x: tf.reshape(x, (-1, 180, 256)))(tran_out)
    #
    # x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(tran_out))
    # att_1 = Attention(180)(x)
    # semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    # merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    # merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    # x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x))
    # att_2 = Attention(180)(x)
    # semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    # merged_2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    # merged_2_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    # x1_1_all = concatenate([att_1, merged_1, merged_1_avg, att_2, merged_2, merged_2_avg])
    #
    # query = Dense(256, activation=custom_gelu)(x2_1)
    # query = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(query)
    # query = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(query)
    # key = Dense(256, activation=custom_gelu)(x2_1)
    # key = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(key)
    # key = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(key)
    # value = Dense(256, activation=custom_gelu)(x2_1)
    # value = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(value)
    # value = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(value)
    # qk = Multiply()([query, key])
    # attention = Dense(64, activation='softmax')(qk)
    # a_value = Multiply()([attention, value])
    # tran_out = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(a_value)
    # tran_out = Lambda(lambda x: tf.reshape(x, (-1, 180, 256)))(tran_out)
    # x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(tran_out))
    # att_1 = Attention(180)(x)
    # semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    # merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    # merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    # x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x))
    # att_2 = Attention(180)(x)
    # semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    # merged_2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    # merged_2_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    # x2_1_all = concatenate([att_1, merged_1, merged_1_avg, att_2, merged_2, merged_2_avg])
    #
    # query = Dense(256, activation=custom_gelu)(x3_1)
    # query = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(query)
    # query = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(query)
    # key = Dense(256, activation=custom_gelu)(x3_1)
    # key = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(key)
    # key = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(key)
    # value = Dense(256, activation=custom_gelu)(x3_1)
    # value = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(value)
    # value = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(value)
    # qk = Multiply()([query, key])
    # attention = Dense(64, activation='softmax')(qk)
    # a_value = Multiply()([attention, value])
    # tran_out = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(a_value)
    # tran_out = Lambda(lambda x: tf.reshape(x, (-1, 180, 256)))(tran_out)
    # x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(tran_out))
    # att_1 = Attention(180)(x)
    # semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    # merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    # merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    # x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x))
    # att_2 = Attention(180)(x)
    # semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    # merged_2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    # merged_2_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    # x3_1_all = concatenate([att_1, merged_1, merged_1_avg, att_2, merged_2, merged_2_avg])
    #
    # query = Dense(256, activation=custom_gelu)(x4_1)
    # query = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(query)
    # query = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(query)
    # key = Dense(256, activation=custom_gelu)(x4_1)
    # key = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(key)
    # key = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(key)
    # value = Dense(256, activation=custom_gelu)(x4_1)
    # value = Lambda(lambda x: tf.reshape(x, (-1, 180, 4, 64)))(value)
    # value = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(value)
    # qk = Multiply()([query, key])
    # attention = Dense(64, activation='softmax')(qk)
    # a_value = Multiply()([attention, value])
    # tran_out = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(a_value)
    # tran_out = Lambda(lambda x: tf.reshape(x, (-1, 180, 256)))(tran_out)
    # x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(tran_out))
    # att_1 = Attention(180)(x)
    # semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    # merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    # merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    # x = Dropout(0.1)(Bidirectional(CuDNNLSTM(256, return_sequences=True))(x))
    # att_2 = Attention(180)(x)
    # semantic = TimeDistributed(Dense(128, activation=custom_gelu))(x)
    # merged_2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    # merged_2_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(128,))(semantic)
    # x4_1_all = concatenate([att_1, merged_1, merged_1_avg, att_2, merged_2, merged_2_avg])

    hin = Input(shape=(num_features_input,))
    htime = Dropout(0.1)(Dense(128, activation=custom_gelu)(hin))

    merged12 = Multiply()([x1_all, x2_all])
    merged12 = Dropout(0.1)(merged12)
    merged13 = Multiply()([x1_all, x3_all])
    merged13 = Dropout(0.1)(merged13)
    merged23 = Multiply()([x2_all, x3_all])
    merged23 = Dropout(0.1)(merged23)
    merged34 = Multiply()([x3_all, x4_all])
    merged34 = Dropout(0.1)(merged34)
    merged41 = Multiply()([x4_all, x1_all])
    merged41 = Dropout(0.1)(merged41)
    merged36 = Multiply()([x3_all, x6_all])
    merged36 = Dropout(0.1)(merged36)

    # merged11_1 = Multiply()([x1_all, x1_1_all])
    # merged11_1 = Dropout(0.1)(merged11_1)
    # merged22_1 = Multiply()([x2_all, x2_1_all])
    # merged22_1 = Dropout(0.1)(merged22_1)
    # merged33_1 = Multiply()([x3_all, x3_1_all])
    # merged33_1 = Dropout(0.1)(merged33_1)
    # merged44_1 = Multiply()([x4_all, x4_1_all])
    # merged44_1 = Dropout(0.1)(merged44_1)

    x = concatenate(
        [x1_all, x2_all, x3_all, x4_all, x6_all, merged12, merged13, merged23, merged34, merged41, merged36, htime])
    x = Dropout(0.1)(Activation(activation=custom_gelu)(BatchNormalization()(Dense(2048)(x))))
    x = Activation(activation=custom_gelu)(BatchNormalization()(Dense(512)(x)))
    pred = Dense(20, activation='softmax')(x)

    model = Model(inputs=[seq1, seq2, seq3, seq4, seq6, hin], outputs=pred)
    adam = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    return model


gc.collect()

train_semi_labels = pd.read_csv("train_semi_final/user.csv")

train_labels = pd.concat([train_labels, train_semi_labels], axis=0, ignore_index=True)

train_size = len(train_labels)
train_x1 = x1[:train_size]
test_x1 = x1[train_size:]

train_x2 = x2[:train_size]
test_x2 = x2[train_size:]

train_x3 = x3[:train_size]
test_x3 = x3[train_size:]

train_x4 = x4[:train_size]
test_x4 = x4[train_size:]

train_x6 = x6[:train_size]
test_x6 = x6[train_size:]

train_x7 = hin_input[:train_size]
test_x7 = hin_input[train_size:]

labels = (train_labels['gender'] - 1) * 10 + train_labels['age']
category_label = to_categorical(labels - 1)

gc.collect()

skf = StratifiedKFold(n_splits=10, random_state=1234, shuffle=True)
count = 0

for i, (train_index, val_index) in enumerate(skf.split(train_x1, labels)):
    print("FOLD | ", count + 1)
    print("###" * 35)
    gc.collect()
    K.clear_session()
    filepath = "semi_model/comprehensive_nn_0720_5_3_180_model_20_trans_v30_%d.h5" % count
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_acc', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    earlystopping = EarlyStopping(
        monitor='val_acc', min_delta=0.00001, patience=6, verbose=1, mode='max')
    callbacks = [checkpoint, reduce_lr, earlystopping]
    # comprehensive_model = comprehensive_model_conv_v12(emb1, emb2, emb3, emb4, emb6, emb1_1, emb2_1, emb3_1, emb4_1,
    #                                                    num_features_input)
    comprehensive_model = comprehensive_model_conv_v12(emb1, emb2, emb3, emb4, emb6, num_features_input)
    if count == 0:
        comprehensive_model.summary()
    x1_tr, x1_va = np.array(train_x1)[train_index], np.array(train_x1)[val_index]
    x2_tr, x2_va = np.array(train_x2)[train_index], np.array(train_x2)[val_index]
    x3_tr, x3_va = np.array(train_x3)[train_index], np.array(train_x3)[val_index]
    x4_tr, x4_va = np.array(train_x4)[train_index], np.array(train_x4)[val_index]
    x6_tr, x6_va = np.array(train_x6)[train_index], np.array(train_x6)[val_index]
    x7_tr, x7_va = np.array(train_x7)[train_index], np.array(train_x7)[val_index]

    y_tr, y_va = category_label[train_index], category_label[val_index]

    hist = comprehensive_model.fit([x1_tr, x2_tr, x3_tr, x4_tr, x6_tr, x7_tr], y_tr, batch_size=128, epochs=30,
                                   validation_data=([x1_va, x2_va, x3_va, x4_va, x6_va, x7_va], y_va),
                                   callbacks=callbacks, verbose=1, shuffle=True)

    # best_model = comprehensive_model_conv_v12(emb1, emb2, emb3, emb4, emb6, emb1_1, emb2_1, emb3_1, emb4_1,
    #                                           num_features_input)
    best_model = comprehensive_model_conv_v12(emb1, emb2, emb3, emb4, emb6, num_features_input)
    best_model.load_weights(filepath)
    comprehensive_pred = best_model.predict([test_x1, test_x2, test_x3, test_x4, test_x6, hin_input], batch_size=512)

    np.save('./submission/comprehensive_nn_0720_5_3_180_model_20_trans_v30_%d_pred.npy' % count, comprehensive_pred)
    count = count + 1
