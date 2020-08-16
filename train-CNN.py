# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 19:59:47 2018

@author: xzf0724
"""

from keras.layers import merge, concatenate, add, Conv1D, Embedding, Dense, Flatten, Input, Concatenate
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import os
# Set Tensorflow Messages Type
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1337)

# start = time.process_time()
# print("start time: ", start)

distances = []  # TrainSet
labels = []  # 0/1
texts = []  # ClassName And MethodName
MAX_SEQUENCE_LENGTH = 15
EMBEDDING_DIM = 200  # Dimension of word vector

print('\nIndexing word vectors...')

embeddings_index = {}
f = open('word2vecNocopy.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# print('Found %s word vectors.' % len(embeddings_index))

print('\nFinding train_distnaces...')

with open('Data/2#Fold/train-projectName/train_Distances.txt', 'r') as file_to_read:
    # Sample Data: (0.9444444444444 , 1.0 , 0)
    for line in file_to_read.readlines():
        values = line.split()
        distance = values[:2]
        distances.append(distance)
        label = values[2:]
        labels.append(label)

with open('Data/2#Fold/train-projectName/train_Names.txt', 'r') as file_to_read:
    for line in file_to_read.readlines():
        texts.append(line)

# print('Found %s train_distances.' % len(distances))

# Convert Class names and Method names to Sequences
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

distances = np.asarray(distances)
labels = to_categorical(np.asarray(labels))

# print('\nShape of data array:', data.shape)
# print('Shape of labels array:', labels.shape)


x_train_dis = distances
# print('\nShape of x_train_dis array:', x_train_dis.shape)
x_train_dis = np.expand_dims(x_train_dis, axis=2)
# print('\nShape of x_train_dis array after expanding:', x_train_dis.shape)

x_train_names = data
# print('\nShape of x_train_names array:', x_train_names.shape)

x_train = []
x_train.append(x_train_names)
x_train.append(np.array(x_train_dis))

y_train = np.array(labels)

nb_words = len(word_index)
embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('\nBuilding model...')

# print(tf.is_tensor(x_train[0]))
# print(x_train[0].shape)

# Model_RIGHT -> Embeddings using text and nemes
input_right = Input(shape=(MAX_SEQUENCE_LENGTH,))
model_right = Embedding(nb_words + 1,
                        EMBEDDING_DIM,
                        input_length=MAX_SEQUENCE_LENGTH,
                        weights=[embedding_matrix],
                        trainable=False)(input_right)
model_right = Conv1D(128, 1, padding="same", activation='tanh')(model_right)
model_right = Conv1D(128, 1, activation='tanh')(model_right)
model_right = Conv1D(128, 1, activation='tanh')(model_right)
model_right = Flatten()(model_right)

# Model_LEFT -> Distances
input_left = Input(shape=(2, 1))
# model_left = Dense(128, activation='tanh')(input_left)
# model_left = Dense(128, activation='sigmoid')(model_left)

model_left = Conv1D(128, 1, input_shape=(2, 1),
                    padding="same", activation='tanh')(input_left)
model_left = Conv1D(128, 1, activation='tanh')(model_left)
model_left = Conv1D(128, 1, activation='tanh')(model_left)
model_left = Flatten()(model_left)

# MODEL _ MERGING
merged = concatenate([model_left, model_right])

# MODEL _ AFTER MERGING
model = Dense(128, activation='tanh')(merged)
output = Dense(2, activation='sigmoid')(merged)
model = Model(inputs=[input_right, input_left], outputs=output)
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

# TRAINING MODEL
print("\ntraining model:"+time.strftime("%Y/%m/%d  %H:%M:%S"))
history = model.fit([x_train[0], x_train[1]], y_train, epochs=6)

# EVALUATING MODEL
score = model.evaluate([x_train[0], x_train[1]], y_train, verbose=0)
print('train loss:', score[0])
print('train accuracy:', score[1])

# Timing
# end = time.process_time()
# print("end time:", end)
# print('Running time: %s Seconds' % (end-start))

# SAVING MODEL
json_string = model.to_json()
open('Results/2_my_model2.json', 'w').write(json_string)
model.save_weights('Results/2_my_model_weights.h5')

# PLOT_HISTORY AND PLOTS
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# PLOT_LOSS
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
