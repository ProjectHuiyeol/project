import os
import pickle
import numpy as np
import tensorflow as tf
from transformers import TFElectraModel

tf.random.set_seed(42)
np.random.seed(42)

CLASS_NUMBER = 6
BERT_CKPT = '../project/nlp/data_out/'
DATA_IN_PATH = '../project/metadata/'
DATA_OUT_PATH = '../project/nlp/data_out/'
MAX_LEN = 40 #95% - >60개   90% ->40개
MAX_WORD = 20

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

cls_model = TFElectraClassifier(model_name='monologg/koelectra-base-v3-discriminator', dir_path=os.path.join(BERT_CKPT, 'model'), num_class=CLASS_NUMBER)

optimizer = tf.keras.optimizers.Adam(3e-6)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

with open(DATA_OUT_PATH+'bert_tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

with open(DATA_OUT_PATH+'encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

def bert_tokenizer(sent, MAX_LEN):
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

a, b, c = bert_tokenizer('안녕하세요', MAX_LEN)
cls_model.call((np.array(a).reshape(1,-1), np.array(b).reshape(1,-1), np.array(c).reshape(1,-1)))
cls_model.built = True
cls_model.load_weights(DATA_OUT_PATH+'tf2_electra_plutchik_hs_10.h5')

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
            input_id, attention_mask, token_type_id = bert_tokenizer(lyric, MAX_LEN)
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