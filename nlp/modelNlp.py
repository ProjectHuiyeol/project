"""가사 예측 모델 로드

모듈
====

2개의 Pre-trained Electra 모델 로드
1번 모델(cls_model) -> 가사의 감정 3분류(부정, 중립, 긍정)
2번 모델(neg_model) -> 가사의 부정감정 5분류(분노, 슬픔, 공포, 놀람, 싫음)

사용 라이브러리
====

numpy==1.21.6
pandas==1.3.5
tqdm==4.64.1
tensorflow==2.10.0
transformers==4.24.0

"""

import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import tensorflow as tf
from transformers import TFElectraModel

tf.random.set_seed(42)
np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)

CLASS_NUMBER = 3
NEG_CLASS_NUMBER = 5
MAX_LEN = 40
MAX_WORD = 20
BERT_CKPT = './data_out/'
DATA_IN_PATH = '../metadata/'
DATA_OUT_PATH = './data_out/'

# Load Electra Tokenizer
with open(DATA_OUT_PATH+'tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
# Load 3classes(neg, neut, pos) LabelEncoder
with open(DATA_OUT_PATH+'3classEncoder.pickle', 'rb') as handle:
    le = pickle.load(handle)
# Load 5classes(angry, dislike, fear, sad, surprise) LabelEncoder
with open(DATA_OUT_PATH+'negClassEncoder.pickle', 'rb') as handle:
    neg_le = pickle.load(handle)

# Fuctiong for Tokenizing Sentence
def electra_tokenizer(sent, MAX_LEN):
    encoded_dict = loaded_tokenizer.encode_plus(
        text=sent,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id

class TFElectraClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super().__init__()

        self.bert = TFElectraModel.from_pretrained(model_name, cache_dir=dir_path, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(num_class, name='classifier', activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range))

    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs[0]
        last_hidden_state = self.flatten(last_hidden_state)
        last_hidden_state = self.dropout(last_hidden_state, training=training)
        logits = self.classifier(last_hidden_state)

        return logits

# ElectraClassifier for classifying 3classes(neg, neut, pos)
cls_model = TFElectraClassifier(model_name='monologg/koelectra-base-v3-discriminator', dir_path=os.path.join(BERT_CKPT, 'model'), num_class=CLASS_NUMBER)
# ElectraClassifier for classifying 5classes(angry, dislike, fear, sad, surprise)
neg_model = TFElectraClassifier(model_name='monologg/koelectra-base-v3-discriminator', dir_path=os.path.join(BERT_CKPT, 'model'), num_class=NEG_CLASS_NUMBER)
# For Loading Weights
a, b, c = electra_tokenizer('안녕하세요', MAX_LEN)
cls_model.call((np.array(a).reshape(1,-1), np.array(b).reshape(1,-1), np.array(c).reshape(1,-1)))
cls_model.built = True
cls_model.load_weights(DATA_OUT_PATH+'tf2_electra_plutchik_hs_6.h5')
# For Loading Weights
a, b, c = electra_tokenizer('안녕하세요', MAX_LEN)
neg_model.call((np.array(a).reshape(1,-1), np.array(b).reshape(1,-1), np.array(c).reshape(1,-1)))
neg_model.built = True
neg_model.load_weights(DATA_OUT_PATH+'tf2_electra_plutchik_hs_10.h5')
# Compile
optimizer = tf.keras.optimizers.Adam(3e-6)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

def preprocessing(x):
    temp = x.split('\n')
    temp = ' '.join(temp)
    return temp

def slicing(x):
    x = x.split()
    res = []
    for i in range(0, len(x)+1, MAX_WORD):
        if len(x)-i-MAX_WORD < MAX_WORD//2:
            temp = x[i:]
            res.append(temp)
            break
        temp = x[i:i+MAX_WORD]
        res.append(temp)
    return res

def predict(lyrics):
    temp = []
    lyrics = preprocessing(lyrics)
    lyrics = slicing(lyrics)
    for lyric in lyrics:
        try:
            txt = ' '.join(lyric)
            temp.append(txt)
        except:
            continue

    input_ids = []
    attention_masks = []
    token_type_ids = []

    for lyric in temp:
        try:
            input_id, attention_mask, token_type_id = electra_tokenizer(lyric, MAX_LEN)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)

        except Exception as e:
            print(e)
            pass
    prediction = cls_model.predict(input_ids)

    return prediction

def transform(array):
    result = le.inverse_transform([np.argmax(array)])
    return result

def neg_predict(lyrics):
    temp = []
    lyrics = preprocessing(lyrics)
    lyrics = slicing(lyrics)
    for lyric in lyrics:
        try:
            txt = ' '.join(lyric)
            temp.append(txt)
        except:
            continue

    input_ids = []
    attention_masks = []
    token_type_ids = []

    for lyric in temp:
        try:
            input_id, attention_mask, token_type_id = electra_tokenizer(lyric, MAX_LEN)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)

        except Exception as e:
            print(e)
            pass
    prediction = neg_model.predict(input_ids)

    return prediction

def neg_transform(array):
    result = neg_le.inverse_transform([np.argmax(array)])
    return result