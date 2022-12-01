import numpy as np
import pandas as pd
import pickle
import modelElectra

CLASS_NUMBER = 8
BERT_CKPT = '../project/nlp/data_out/'
DATA_IN_PATH = '../project/metadata/'
DATA_OUT_PATH = '../project/nlp/data_out/'
MAX_LEN = 40

data = pd.read_csv(DATA_IN_PATH+'Song_Lyrics2.csv')
lyric = data.loc[3076, 'LYRICS']

prediction = modelElectra.predict(lyric)
for pred in prediction:
    print(modelElectra.transform([0,1,2,3,4,5]))
    print(pred)