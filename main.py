import csv
import re
import datetime

import tensorflow as tf
import mysql.connector as mysql


db = mysql.connect(
    host='localhost',
    user='dudosyka',
    passwd='123',
    database='fts'
)

cursor = db.cursor()


def GetNGram(text):
    text = re.sub('\W+', ' ', text)
    res = []
    for item in text.split(" "):
        if len(item) > 1:
            res.append(item)
    return res


def TextToVector(text, vocabulary):
    base = len(vocabulary)
    vector = [0] * base
    textNGram = GetNGram(text)
    for word in textNGram:
        if vocabulary.__contains__(word):
            vector[vocabulary.index(word)] = 1

    return vector


CLASSES = []


def GenerateData(file, category, maxLen):
    data = []
    keys = []
    validation_x = []
    validation_y = []
    with open(file, 'r', newline='', encoding='windows-1251') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        i = 0
        iCategory = 0
        iOther = 0
        iTrainCategory = 0
        iTrainOther = 0
        for row in reader:
            text = re.sub('\W+', ' ', row[1])
            if i >= maxLen*4:
                break
            if iOther < maxLen and not (row[0] == category):
                keys.append(0)
                iOther += 1
                data.append(text)
                i += 1
                CLASSES.append(row[1])
            elif iCategory < maxLen and row[0] == category:
                keys.append(1)
                iCategory += 1
                i += 1
                data.append(text)
                CLASSES.append(row[1])
            elif iTrainCategory < maxLen and row[0] == category:
                validation_x.append(text)
                validation_y.append(1)
                i += 1
                iTrainCategory += 1
            elif iTrainOther < maxLen and not (row[0] == category):
                validation_x.append(text)
                validation_y.append(0)
                i += 1
                iTrainOther += 1

    return data, keys, validation_x, validation_y


def getCategoryCount(category_id):
    cursor.execute("SELECT * FROM `category_vector` WHERE `category_id` = %s", [ (category_id) ])
    return int(cursor.fetchall()[0][2])


cursor.execute("SELECT COUNT(*) FROM `vocabulary`")
VOCABULARY_SIZE = cursor.fetchall()[0][0]
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCABULARY_SIZE)
cursor.execute("SELECT * FROM `vocabulary`")
data = cursor.fetchall()
encoder.adapt(list(map(lambda el: el[1], data)))


def Learn():
    cursor.execute('SELECT * FROM `categories`')
    categories = cursor.fetchall()
    for category in categories:
        # if not (int(category[0]) == 3004):
        #     continue
        now = datetime.datetime.now()
        print("Start work on category: ", category[0], " ", now.strftime("%Y-%m-%d %H:%M:%S"))
        categoryCount = getCategoryCount(category[0])
        maxLen = 5000
        if categoryCount < 2500:
            maxLen = categoryCount // 2
        print('Category max:', maxLen)
        x, y, val_x, val_y = GenerateData("./dataset_20211126.csv", category[0], maxLen)
        print('Category data generated.')
        # print(x, y)
        #

        model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])

        print('Model compiled.')

        model.fit(x, y, epochs=10, validation_data=(val_x, val_y), shuffle=True)

        model.save('../models/fts/' + category[0])
        now = datetime.datetime.now()
        print('Model saved. ',  now.strftime("%Y-%m-%d %H:%M:%S"))


Learn()
