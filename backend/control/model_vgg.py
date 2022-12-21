import os
import numpy as np
import tensorflow as tf

class VGG():
    def __init__(self):
        path = './static/weights/1205_best_128.h5'
        self.model = tf.keras.models.load_model(path)
        self.data_path = './static/weights/melspectrogram_data_plus/'
        
    def infer(self, song_id):
        # print(self.data_path)
        mel_data = np.load(self.data_path+str(song_id)+'.npy')
        sample = []
        for i in range(len(mel_data[0])):
            if 128+(64*i) >= len(mel_data[0]):
                break
            sample.append(mel_data[:,64*i:128+(64*i)])
        # with tf.device('/CPU:0'):
        result = self.model.predict(np.array(sample), verbose=1)
        inference = np.sum(result, axis=0) / len(sample)
        
        return inference
        
    def voting(self, song_ids):
        length = len(song_ids)
        inference = []
        # print(os.getcwd())
        for song_id in song_ids:
            inference.append(self.infer(song_id[0]))
        
        return np.sum(np.array(inference), axis=0) / length
        