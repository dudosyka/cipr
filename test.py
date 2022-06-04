import os
import datetime

import tensorflow as tf
import numpy as np
import mysql.connector as mysql

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

db = mysql.connect(
    host='localhost',
    user='dudosyka',
    passwd='123',
    database='fts'
)

cursor = db.cursor()


def printTime():
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))


# print('Request count')
# printTime()
# cursor.execute("SELECT COUNT(*) FROM `vocabulary`")
# VOCABULARY_SIZE = cursor.fetchall()[0][0]
# print('Request count')
# printTime()
# encoder = tf.keras.layers.TextVectorization(max_tokens=VOCABULARY_SIZE)
# cursor.execute("SELECT * FROM `vocabulary`")
# data = cursor.fetchall()
# print('Request vocabulary end')
# printTime()
# encoder.adapt(list(map(lambda el: el[1], data)))
#
#
# model = tf.keras.Sequential([
#     encoder,
#     tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1)
# ])
#
#
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(1e-4),
#               metrics=['accuracy'])
#
#
# printTime()
# model.summary()
# model.save('./models/empty')
model = tf.keras.models.load_model('/home/dudosyka/Documents/HackITMO/neuro/models/empty')


def predict(sample_text):
    for item in os.scandir('./models/fts'):
        i = 0
        if item.is_dir():
            model.load_weights('/home/dudosyka/Documents/HackITMO/neuro/models/fts/' + item.name)
            predictions = model.predict(np.array([sample_text]))
            print("Category: ", item.name, " - ", predictions)


# while True:
#     line = sys.stdin.readline()
#     print(line)
predict(('ЛЕКАРСТВЕННОЕ СРЕДСТВО ЭНЗИСТАЛ'))


''' 
{
    1207: -0.21
    0811: -0.20
    1904: -0.46
    0810: -1.63
    0603: -2.65
    0604: -2.20
    0713: -0.017
    0407: -0.077
    0807: -0.21
    0808: -1.23
    1517: -0.25
    0703: -1.28
    1805: -0.24
    0806: -2.16
    0805: -1.57
    2005: -0.50
    1704: -0.98
    1604: -1.81
    0809: -2.20
    2008: -1.51
    2201: -0.17
    1806: -1.57
    3004: 0.42
    0901: -1.78
    0712: -0.15
}
'''