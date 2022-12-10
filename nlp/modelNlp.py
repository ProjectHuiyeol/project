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
scikit-learn==1.0.2
torch==1.13.0+cu117

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
BERT_CKPT = '../project/nlp/data_out/'
DATA_IN_PATH = '../project/metadata/'
DATA_OUT_PATH = '../project/nlp/data_out/'

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

class EmotionClassfier():
    def __init__(self,encoder1, encoder2):

        self.model1 = TFElectraClassifier(model_name='monologg/koelectra-base-v3-discriminator', dir_path=os.path.join(BERT_CKPT, 'model'), num_class=CLASS_NUMBER)
        self.model2 = TFElectraClassifier(model_name='monologg/koelectra-base-v3-discriminator', dir_path=os.path.join(BERT_CKPT, 'model'), num_class=NEG_CLASS_NUMBER)
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def call(self):
        # For Loading Weights
        a, b, c = electra_tokenizer('안녕하세요', MAX_LEN)
        self.model1.call((np.array(a).reshape(1,-1), np.array(b).reshape(1,-1), np.array(c).reshape(1,-1)))
        self.model1.built = True
        self.model1.load_weights(DATA_OUT_PATH+'tf2_electra_plutchik_hs_6.h5')
        # For Loading Weights
        a, b, c = electra_tokenizer('안녕하세요', MAX_LEN)
        self.model2.call((np.array(a).reshape(1,-1), np.array(b).reshape(1,-1), np.array(c).reshape(1,-1)))
        self.model2.built = True
        self.model2.load_weights(DATA_OUT_PATH+'tf2_electra_plutchik_hs_10.h5')

    # Function for Preprocessing 3classes classification(neg, neut, pos)
    def preprocessing(self, lyric):
        temp = lyric.split('\n')
        temp = ' '.join(temp)
        return temp

    def slicing(self, lyric):
        lyric = lyric.split()
        res = []
        for i in range(0, len(lyric)+1, MAX_WORD):
            if len(lyric)-i-MAX_WORD < MAX_WORD//2:
                temp = lyric[i:]
                res.append(temp)
                break
            temp = lyric[i:i+MAX_WORD]
            res.append(temp)
        return res

    def predict3classes(self, lyrics):
        temp = []
        lyrics = self.preprocessing(lyrics)
        lyrics = self.slicing(lyrics)
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
        prediction = self.model1.predict(input_ids, verbose=0)

        return prediction

    # Function for Preprocessing 5classes classification(angry, dislike, fear, sad, surprise)
    def predict_neg(self, lyrics):
        temp = []
        lyrics = self.preprocessing(lyrics)
        lyrics = self.slicing(lyrics)
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
        prediction = self.model2.predict(input_ids, verbose=0)

        return prediction

    def transform_neg(self, array):
        result = self.encoder2.inverse_transform([np.argmax(array)])
        return result

    def transform3classes(self, array):
        result = self.encoder1.inverse_transform([np.argmax(array)])
        return result

    def predictAll(self, lyrics):
        # 데이터 입력
        data = lyrics[['SONG_ID', 'LYRICS']]
        song_ids = lyrics['SONG_ID']
        # 3분류
        pred = dict()
        for id in tqdm(song_ids):
            lyric = data[data['SONG_ID']==id]['LYRICS'].values[0]
            prediction = self.predict3classes(lyric)
            score = sum(prediction)/len(prediction)
            senti = self.transform3classes(score)
            pred[id] = (score, senti)
        data['score'] = ''
        data['senti'] = ''
        for id in song_ids:
            sc, se = pred[id]
            data.loc[data[data['SONG_ID']==id].index, 'score'] = str(sc)
            data.loc[data[data['SONG_ID']==id].index, 'senti'] = se
        # 5분류
        neg_data = data[data['senti']=='neg']
        pos_data = data[data['senti']=='pos']
        neg_ids = neg_data['SONG_ID']
        pos_ids = pos_data['SONG_ID']
        pos_pred = dict()
        neg_pred = dict()
        # 대분류가 neg인 경우
        for id in tqdm(neg_ids):
            lyric = data[data['SONG_ID']==id]['LYRICS'].values[0]
            prediction = self.predict_neg(lyric)
            score = sum(prediction)/len(prediction)
            senti = self.transform_neg(score)
            neg_pred[id] = (score, senti)
        data['final_score'] = ''
        data['final_senti'] = ''
        data['total'] = ''
        for key in neg_ids:
            sc, se = neg_pred[key]
            data.loc[data[data['SONG_ID']==key].index, 'final_score'] = str(sc)
            data.loc[data[data['SONG_ID']==key].index, 'final_senti'] = se
        # neg인 경우의 감정 지표 계산
        for key in neg_ids:
            neg_sc = np.fromstring(data.loc[data[data['SONG_ID']==key].index, 'final_score'].tolist()[0][1:-1], dtype=float, sep=' ')
            pos_sc = np.fromstring(data.loc[data[data['SONG_ID']==key].index, 'score'].tolist()[0][1:-1], dtype=float, sep=' ')
            ratio = np.array(((pos_sc[int(le.transform(['neg'])[0])]+pos_sc[1]*0.5), (pos_sc[int(le.transform(['pos'])[0])]+pos_sc[1]*0.5)))
            rated_sc = np.append(np.max(neg_sc)*(ratio[1]/ratio[0]), neg_sc)
            total_sc = rated_sc/(sum(rated_sc))
            data.loc[data[data['SONG_ID']==key].index, 'total'] = str(total_sc)
        # 대분류가 pos인 경우
        for id in tqdm(pos_ids):
            lyric = data[data['SONG_ID']==id]['LYRICS'].values[0]
            prediction = self.predict_neg(lyric)
            score = sum(prediction)/len(prediction)
            pos_pred[id] = (score, 'happy')
        for key in pos_ids:
            sc, se = pos_pred[key]
            data.loc[data[data['SONG_ID']==key].index, 'final_score'] = str(sc)
            data.loc[data[data['SONG_ID']==key].index, 'final_senti'] = se
        # pos인 경우의 감정 지표 계산
        for key in pos_ids:
            neg_sc = np.fromstring(data.loc[data[data['SONG_ID']==key].index, 'final_score'].tolist()[0][1:-1], dtype=float, sep=' ')
            pos_sc = np.fromstring(data.loc[data[data['SONG_ID']==key].index, 'score'].tolist()[0][1:-1], dtype=float, sep=' ')
            ratio = np.array(((pos_sc[int(le.transform(['neg'])[0])]+pos_sc[1]*0.5), (pos_sc[int(le.transform(['pos'])[0])]+pos_sc[1]*0.5)))
            rated_sc = np.append(ratio[1], ratio[0]*neg_sc)
            total_sc = rated_sc/(sum(rated_sc))
            data.loc[data[data['SONG_ID']==key].index, 'total'] = str(total_sc)
        # 중립 제거
        data = data.drop(data[data['senti']=='neut'].index)
        return data
# 모델 실행
model = EmotionClassfier(le,neg_le)
model.call()