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
from modelNlp import *

lyrics = pd.read_csv(DATA_IN_PATH+'concatSongs.csv')
model = EmotionClassfier(le,neg_le)
model.call()
data = model.predictAll(lyrics)
data.to_csv(DATA_IN_PATH+'predictTotal.csv', encoding='utf-8-sig')