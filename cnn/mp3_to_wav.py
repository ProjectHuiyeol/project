import os
from tqdm import tqdm
from pydub import AudioSegment

mp3_path = os.getcwd() + '/download'
wav_path = os.getcwd() + '/data_wav/'

mp3_lst = os.listdir(mp3_path)

for file in tqdm(mp3_lst):
    new_name = file[:-4] + '.wav'
    temp = AudioSegment.from_mp3(mp3_path+'/'+file)
    temp.export(wav_path+new_name, format='wav')