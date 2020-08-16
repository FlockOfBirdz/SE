# -*- coding: utf-8 -*-

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import time
import os

# Set Tensorflow Messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1337)
FOLD = '1'
MAX_SEQUENCE_LENGTH = 15

TESTPATH = 'Data/'+FOLD+'#Fold/test-projectName/'
MODELPATH = 'Results/'
FILENAME = 'Data/'+FOLD+'#Fold/test-projectName/test_ClssId.txt'
TARGETPATH = 'Data/'+FOLD+'#Fold/test-projectName/targetClasses.txt'
values = []
predsTargetClassNames = []
print('***************************************************************************************************')
print("start time:"+time.strftime("%Y/%m/%d  %H:%M:%S"))
start = time.process_time()

f = open(TARGETPATH, 'r', encoding='utf-8')
for line in f:
    predsTargetClassName = line.split()
    predsTargetClassNames.append(predsTargetClassName)

f = open(FILENAME, 'r', encoding='utf-8')
for line in f:
    value = line.split()
    values.append(value)
# print('\n', values, '\n')

TP = 0
FN = 0
FP = 0
TN = 0
NUM_CORRECT = 0
TOTAL = 0

model = model_from_json(
    open(MODELPATH + FOLD + '_my_model.json').read())
model.load_weights(MODELPATH + FOLD + '_my_model_weights.h5')

print('\nLoading Files...\n')
ii = 0
for sentence in values:
    print('~ii:', ii)
    ii = ii+1
    test_distances = []
    test_labels = []
    test_texts = []
    targetClassNames = []
    classId = sentence[0]  # Class ID of sentence # in values
    label = sentence[1]  # 1 or 0 for each classID in values

    if(os.path.exists(TESTPATH + 'test_Distances'+classId+'.txt')):
        # print('1.ospath:', TESTPATH + 'test_Distances' + classId + '.txt'+'\n')
        with open(TESTPATH + 'test_Distances'+classId+'.txt', 'r') as file_to_read:
            lineNumber = 1
            for line in file_to_read.readlines():
                # print('Line Number:', lineNumber)
                lineNumber += lineNumber
                values = line.split()
                test_distance = values[:2]
                test_distances.append(test_distance)
                # print('test_distance:', test_distance)
                test_label = values[2:]
                test_labels.append(test_label)
                # print('test_label:', test_label, '\n')

        # print('test_distances:', test_distances)
        # print('test_labels:', test_labels, '\n')

        with open(TESTPATH + 'test_Names'+classId+'.txt', 'r') as file_to_read:
            # print('2.ospath:', TESTPATH + 'test_Names' + classId + '.txt'+'\n')
            lineNumber = 1
            for line in file_to_read.readlines():
                # print('Line Number:', lineNumber)
                lineNumber += lineNumber
                test_texts.append(line)
                # print('test_text:', line)
                line = line.split()
                targetClassNames.append(line[10:])
                # print('targetClassName:', line[10:])

        # print('test_texts:', test_texts)
        # print('targetClassNames:', targetClassNames)

        print('Tokenizing...')
        tokenizer1 = Tokenizer(num_words=None)
        tokenizer1.fit_on_texts(test_texts)
        test_sequences = tokenizer1.texts_to_sequences(test_texts)
        test_word_index = tokenizer1.word_index
        test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        test_distances = np.asarray(test_distances)
        test_labels1 = test_labels
        test_labels = np.asarray(test_labels)
        # print('test_data:\n', test_data)
        # print('test_distances:\n', test_distances)
        # print('test_labels:\n', test_labels)

        x_val = []
        x_val_names = test_data
        x_val_dis = test_distances
        x_val_dis = np.expand_dims(x_val_dis, axis=2)
        x_val.append(x_val_names)
        x_val.append(np.array(x_val_dis))
        y_val = np.array(test_labels)

        preds = model.predict(x_val)
        print('raw preds:\n', preds)
        # print('shape', preds.shape)

        preds = np.argmax(preds, axis=1)
        # print('preds after argmax:', preds)
        # print('shape after argmax', preds.shape)

        preds_double = model.predict(x_val)
        print('predictions after argmax:', preds, '\n')
        # print('shape', preds_double.shape)

        NUM_ZERO = 0
        NUM_ONE = 0
        MAX = 0
        for i in range(len(preds)):
            if(preds[i] == 0):
                NUM_ZERO += 1
            else:
                NUM_ONE += 1
        if(len(preds) != 0 and label == '1'):
            TOTAL += 1
            # print('TOTAL_ONEs--------', TOTAL)
        if(label == '1' and NUM_ONE == 0):
            FN += 1
        if(label == '1' and NUM_ONE != 0):
            TP += 1
            correctTargets = []
            for i in range(len(preds_double)):
                if(preds_double[i][0] >= MAX):
                    MAX = preds_double[i][0]
            for i in range(len(preds_double)):
                if(preds_double[i][0] == MAX):
                    correctTargets.append(targetClassNames[i])
            for i in range(len(correctTargets)):
                if(correctTargets[i] == predsTargetClassNames[TOTAL-1]):
                    NUM_CORRECT += 1
                    break
        if(label == '0' and NUM_ONE == 0):
            TN += 1
        if(label == '0' and NUM_ONE != 0):
            FP += 1
print('TOTAL_ONEs--------', TOTAL)
print('TP----------------', TP)
print('TN----------------', TN)
print('FP----------------', FP)
print('FN----------------', FN)
print('NUM_ZERO----------', NUM_ZERO)
print('NUM_ONE-----------', NUM_ONE)
print('NUM_CORRECT-------', NUM_CORRECT)
print('Target Accuracy---', NUM_CORRECT/TP)

if(TP+FP != 0):
    print('Test Precision----', TP/(TP+FP))
else:
    print('Test Precision----', 0)
if(TP+FN != 0):
    print('Test Recall-------', TP/(TP+FN))
else:
    print('Test Recall-------', 0)

# print("end time: "+time.strftime("%Y/%m/%d  %H:%M:%S"))
end = time.process_time()

print('\nTotal process time: %s Seconds' % (end-start))
