import pandas as pd
import numpy as np
from time import *

lyrics = pd.read_csv('/project_huiyeol/project/metadata/Song_Info.csv')
st_tag = ['화나요', '행복해요', '편안해요', '불안', '슬픔', '신남']
lyrics['new_tag'] = ''

tags = dict()
for key, value in enumerate(st_tag):
    tags[key+1] = value

for idx in range(len(lyrics)):
    print(lyrics.loc[idx, 'LYRICS'], '\n')
    for i in range(1, len(tags)+1):
        print('{}:{}'.format(i, tags[i]), end=' ')
    key = int(input())
    if key not in tags.keys():
        key = int(input())
    else:
        lyrics.loc[idx, 'new_tag'] = tags[key]
        sleep(1)

lyrics.to_csv('/project_huiyeol/project/metadata/Song_Lyrics_Labeling.csv')