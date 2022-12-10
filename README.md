# 1. 음원가사 감정 분류 기준

## 1. Robert Plutchik 감정모델(Electra)

</br>
<p align="center"><img src=./img/Plutchik-wheel.png width=50% title="Robert Plutchik's Emotional Wheel"></p>

- 감정 분류
  - joy
  - trust
  - fear
  - surprise
  - sadness
  - disgust
  - anger
  - anticipation

### A. Base

- 감정 분류 그대로 사용
- 정확도 약 40%

### B. + Neutrality

- 기존 감정 분류에 중립 감정 추가
- 정확도 약 44%

### C. - Anticipation

- 기존 감정 분류에 중립 감정을 추가하고 ‘기대’ 감정 제거
- 정확도 약 47%

## 2. Paul Ekman 감정모델(Electra)

</br>
<p align="center"><img src=./img/paul_ekman.png width=50% title="Paul Ekman's Emotional Model"></p>

- 감정 분류
  - angry
  - fear
  - happy
  - sad
  - surprsie
  - contempt

### A. AI hub 데이터 사용

- 일상 대화 데이터 약 8만개 사용
- 기존 분류에 dislike, 중립 감정 추가
- 정확도 약 50%

### B. - contempt

- contempt 감정 → dislike로 분류
- 정확도 약 54%

### C. + complex

- 복합 감정 태그 추가
- 정확도 약 50%

## 3. 긍정, 중립, 부정 3분류

- AI hub 데이터를 사용하여 긍정, 부정, 중립 감정으로 라벨링
- 정확도 70%

### A. 부정 감정 세부 분류

- 긍정 + 중립 데이터를 제외한 데이터로 모델 학습 및 예측
- angry, dislike(dislike + contempt), fear, sad, surprise 5가지로 세부 분류

# 2. 음원가사 전처리

### A. 어절 단위 120개 슬라이싱

- 학습할 수 있는 최대 토큰 개수
- Electra Tokenizer가 자동으로 토큰화하기 때문에 토큰 개수가 120개 보다 많아져서 문장이 잘리는 문제 확인

### B. 어절 단위 60개 슬라이싱

- 문장이 잘리는 문제를 해결하기 위해 반토막 냄
- 문장이 길어 한 문장에 여러 감정이 포함되는 문제가 생김
- 슬라이싱 후 남은 짧은 문장들이 발생

### C. 어절 단위 20개 슬라이싱

- 한 문장에 여러 감정이 포함되는 문제 해결
- 슬라이싱 후 남은 짧은 문장을 그 전 문장에 합침

# 3. 프로세스

### 사용 라이브러리

```python
numpy==1.21.6
pandas==1.3.5
tqdm==4.64.1
tensorflow==2.10.0
transformers==4.24.0
```

### 필요 라이브러리 import 및 변수 설정

```python
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import tensorflow as tf
from transformers import TFElectraModel
```

```python
# 시드 고정 및 넘파이배열 소수점 표기법 고정
tf.random.set_seed(42)
np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)
```

```python
CLASS_NUMBER = 3
NEG_CLASS_NUMBER = 5
MAX_LEN = 40
MAX_WORD = 20
BERT_CKPT = './data_out/'
DATA_IN_PATH = '../metadata/'
DATA_OUT_PATH = './data_out/'
```

### 토크나이저, 라벨인코더 로드

```python
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
```

### 모델 정의

```python
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
```

```python
# ElectraClassifier for classifying 3classes(neg, neut, pos)
cls_model = TFElectraClassifier(model_name='monologg/koelectra-base-v3-discriminator', dir_path=os.path.join(BERT_CKPT, 'model'), num_class=CLASS_NUMBER)
# ElectraClassifier for classifying 5classes(angry, dislike, fear, sad, surprise)
neg_model = TFElectraClassifier(model_name='monologg/koelectra-base-v3-discriminator', dir_path=os.path.join(BERT_CKPT, 'model'), num_class=NEG_CLASS_NUMBER)
```

```python
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
```

```python
# Compile
optimizer = tf.keras.optimizers.Adam(3e-6)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
```

### 예측 준비

```python
# Function for Preprocessing 3classes classification(neg, neut, pos)
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

# Function for Preprocessing 5classes classification(angry, dislike, fear, sad, surprise)
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
```

### 예측

```python
lyrics = pd.read_csv(DATA_IN_PATH+'db에넣을노래들.csv')
# 데이터 입력
data = lyrics[['SONG_ID', 'SONG_TITLE', 'LYRICS']]
song_ids = lyrics['SONG_ID']
# 3분류
pred = dict()
for id in song_ids:
    lyric = data[data['SONG_ID']==id]['LYRICS'].values[0]
    prediction = predict(lyric)
    score = sum(prediction)/len(prediction)
    senti = transform(score)
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
for id in neg_ids:
    lyric = data[data['SONG_ID']==id]['LYRICS'].values[0]
    prediction = neg_predict(lyric)
    score = sum(prediction)/len(prediction)
    senti = neg_transform(score)
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
for id in pos_ids:
    lyric = data[data['SONG_ID']==id]['LYRICS'].values[0]
    prediction = neg_predict(lyric)
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

data = data.drop(data[data['senti']=='neut'].index)
```

### 결과 예시

|     | SONG_ID | SONG_TITLE       | LYRICS | score                        | senti | final_score                                    | final_senti | total                                                   |
| --- | ------- | ---------------- | ------ | ---------------------------- | ----- | ---------------------------------------------- | ----------- | ------------------------------------------------------- |
| 0   | 9270    | 매직 카펫 라이드 | -      | [0.121995 0.293892 0.584113] | pos   | [0.148379 0.245044 0.067424 0.509761 0.029392] | happy       | [0.731059 0.039905 0.065902 0.018133 0.137096 0.007905] |
| 1   | 19807   | 벌써 일년        | -      | [0.319238 0.256552 0.424209] | pos   | [0.167217 0.091865 0.019813 0.712458 0.008647] | happy       | [0.552486 0.074832 0.041111 0.008867 0.318835 0.00387 ] |
| 2   | 32616   | 3!4!             | -      | [0.069013 0.425081 0.505906] | pos   | [0.054421 0.044274 0.028501 0.866249 0.006555] | happy       | [0.718446 0.015322 0.012465 0.008025 0.243895 0.001846] |
| 3   | 33133   | DOC와 춤을       | -      | [0.38072  0.380664 0.238616] | neg   | [0.138274 0.582804 0.094649 0.175393 0.00888 ] | dislike     | [0.304481 0.096172 0.405351 0.06583  0.121989 0.006176] |

### 클래스화

```python
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
        data = lyrics[['SONG_ID', 'SONG_TITLE', 'LYRICS']]
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
```

```python
lyrics = pd.read_csv(DATA_IN_PATH+'test.csv')
model = EmotionClassfier(le,neg_le)
model.call()
data = model.predictAll(lyrics)
data.to_csv(DATA_IN_PATH+'predictTest.csv', encoding='utf-8-sig')
```
