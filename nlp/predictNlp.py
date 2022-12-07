"""가사 예측

모듈
====

modelNlp 모듈 사용
가사 감정 3분류 진행(부정, 중립, 긍정)
긍정==행복으로 태깅
부정과 긍정으로 예측된 가사 감정 5분류 진행(분노, 슬픔, 공포, 놀람, 싫음)
중립으로 예측된 가사는 사용X
최종적으로 행복, 분노, 슬픔, 공포, 놀람, 싫음 총 6가지의 감정으로 분류하고
거기에 대한 확률값을 계산하여 반환한다.

Input: 가사 데이터(lyrics)
Output: 감정(6분류), 감정지표(softmax)

"""

import numpy as np
import pandas as pd
import pickle
from modelNlp import *

# 데이터 입력
lyrics = pd.read_csv(DATA_IN_PATH+'db에넣을노래들.csv')
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
for key in neg_ids:
    neg_sc = np.fromstring(data.loc[data[data['SONG_ID']==key].index, 'final_score'].tolist()[0][1:-1], dtype=float, sep=' ')
    pos_sc = np.fromstring(data.loc[data[data['SONG_ID']==key].index, 'score'].tolist()[0][1:-1], dtype=float, sep=' ')
    ratio = np.array(((pos_sc[int(le.transform(['neg'])[0])]+pos_sc[1]*0.5), (pos_sc[int(le.transform(['pos'])[0])]+pos_sc[1]*0.5)))
    rated_sc = np.append(np.max(neg_sc)*(ratio[1]/ratio[0]), neg_sc)
    total_sc = rated_sc/(sum(rated_sc))
    data.loc[data[data['SONG_ID']==key].index, 'total'] = str(total_sc)
for id in pos_ids:
    lyric = data[data['SONG_ID']==id]['LYRICS'].values[0]
    prediction = neg_predict(lyric)
    score = sum(prediction)/len(prediction)
    pos_pred[id] = (score, 'happy')
for key in pos_ids:
    sc, se = pos_pred[key]
    data.loc[data[data['SONG_ID']==key].index, 'final_score'] = str(sc)
    data.loc[data[data['SONG_ID']==key].index, 'final_senti'] = se
for key in pos_ids:
    neg_sc = np.fromstring(data.loc[data[data['SONG_ID']==key].index, 'final_score'].tolist()[0][1:-1], dtype=float, sep=' ')
    pos_sc = np.fromstring(data.loc[data[data['SONG_ID']==key].index, 'score'].tolist()[0][1:-1], dtype=float, sep=' ')
    ratio = np.array(((pos_sc[int(le.transform(['neg'])[0])]+pos_sc[1]*0.5), (pos_sc[int(le.transform(['pos'])[0])]+pos_sc[1]*0.5)))
    rated_sc = np.append(ratio[1], ratio[0]*neg_sc)
    total_sc = rated_sc/(sum(rated_sc))
    data.loc[data[data['SONG_ID']==key].index, 'total'] = str(total_sc)
data = data.drop(data[data['senti']=='neut'].index)