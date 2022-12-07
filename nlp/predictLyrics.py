import numpy as np
import pandas as pd
import pickle
import modelElectra

CLASS_NUMBER = 6
BERT_CKPT = '../project/nlp/data_out/'
DATA_IN_PATH = '../project/metadata/'
DATA_OUT_PATH = '../project/nlp/data_out/'
MAX_LEN = 40

data = pd.read_csv(DATA_IN_PATH+'concatSongs.csv')
id = input()
lyric = data.loc[id, 'LYRICS']

prediction = modelElectra.predict(lyric)
print(modelElectra.transform([0,1,2,3,4,5]))