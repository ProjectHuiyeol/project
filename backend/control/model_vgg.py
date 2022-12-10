import os
import numpy as np
import tensorflow as tf

class VGG():
    def __init__(self):
        path = '../static/weights/1205_best_128.h5'
        self.model = tf.keras.models.load_model(path)
        self.data_path = '../static/weights/melsepctrogram_data_plus/'
        
    def infer(self, song_id):
        mel_data = np.load(self.data_path+str(song_id)+'.npy')
        sample = []
        for i in range(len(mel_data[0])):
            if 128+(64*i) >= len(mel_data[0]):
                break
            sample.append(mel_data[:,64*i:128+(64*i)])
        result = self.model.predict(np.array(sample))
        inference = np.sum(result, axis=0) / len(sample)
        
        return inference
        
    def voting(self, song_ids):
        length = len(song_ids)
        inference = []
        for song_id in song_ids:
            inference.append(self.infer(song_id))
        
        return np.sum(np.array(inference), axis=0) / length
        