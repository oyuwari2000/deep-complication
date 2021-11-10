# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:41:44 2020

@author: fukuyo ryosuke
"""

import os
import keras
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import roc_curve , roc_auc_score
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical
from keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime



def positive(y_true,y_pred):
    positives = K.sum((y_true * [0,1]))
    return positives

def negative(y_true, y_pred):
    return K.sum(K.cast((y_true * [1,0]), K.floatx()))

def true_positive(y_true, y_pred):
    return K.sum(K.cast(K.greater((y_true + y_pred) * [0,1], 1.5), K.floatx()))

def true_negative(y_true, y_pred):
    return K.sum(K.cast(K.greater((y_true + y_pred) * [1,0], 1.5), K.floatx()))

def sensitivity(y_true, y_pred):
    return K.sum(K.cast(K.greater((y_true + y_pred) * [0,1], 1.5), K.floatx()))/K.sum((y_true * [0,1]))

def speci(y_true, y_pred):
    return K.sum(K.cast(K.greater((y_true + y_pred) * [1,0], 1.5), K.floatx()))/K.sum((y_true * [1,0]))

def pred_positive(y_true, y_pred):
    return K.sum(K.cast(K.greater((y_pred * [0,1]),0.5), K.floatx()))

def roc_auc(y_true, y_pred):
    roc_auc = tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    return roc_auc

dt= datetime.datetime.now()
time = dt.strftime('%Y%m%d%H%M')

batch_size = 1000
epochs = 500

data = pd.read_csv('dataset.csv')
#print(data.shape[1])
ans_index = 'grade3'
# comp2,comp3はそれぞれ腹腔内合併症G2,G3を指す。
#　gradeが全合併症のCD分類
#　grade3はCD3以上の全合併症
model_name = 'model{}.h5'.format(time)
CSV_name = 'model_ans.csv'



train_data = data.loc[:,data.columns.str.endswith('T')]
mean = train_data.describe()[1:2].values
std = train_data.describe()[2:3].values

x_train = (train_data.values-mean)/std
y_train = data[ans_index].values

y_trains = to_categorical(y_train)

model=Sequential()
model.add(Dense(16, input_shape=(x_train.shape[1], )))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_regularizer=keras.regularizers.l2(0.05)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_regularizer=keras.regularizers.l2(0.05)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_regularizer=keras.regularizers.l2(0.05)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_regularizer=keras.regularizers.l2(0.05)))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', sensitivity, positive, true_positive, pred_positive, negative, true_negative, speci, roc_auc])

file_name = 'models{}'.format(time)
os.makedirs(file_name, exist_ok=True)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(file_name, 'model_{epoch:03d}_{val_loss:.2f}.h5'),
    monitor='val_loss',
    period = 10,
    save_best_only = False,
    save_weights_only = False,
    mode = 'min',
    verbose=0)

csv_logger = CSVLogger('log{}.log'.format(time))

hist = model.fit(x_train, y_trains,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=0,
                 validation_split=0.3,
                 class_weight={0:1 ,1:20},
                 callbacks=[csv_logger, model_checkpoint])

pred=model.predict(x_train)
pred=pd.DataFrame(pred,columns=['N','P'])

pred_ans = pd.concat([data[ans_index],pred],axis=1)
pred_ans.to_csv(CSV_name, index=False, encoding='utf-8')


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
sensitivity = hist.history['sensitivity']
val_sensitivity = hist.history['val_sensitivity']
speci = hist.history['speci']
val_speci = hist.history['val_speci']
auc = hist.history['roc_auc']
val_auc = hist.history['val_roc_auc']

epochs = len(loss)

plt.figure(figsize=[15,10])

plt.subplot(5,1,1)
plt.plot(range(epochs), loss, marker='.', label='loss')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(5,1,2)
plt.plot(range(epochs), acc, marker='.', label='accuracy')
plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc(%)')

plt.subplot(5,1,3)
plt.plot(range(epochs), sensitivity, marker='.', label='sensitivity')
plt.plot(range(epochs), val_sensitivity, marker='.', label='val_sens')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('sensitivity(%)')

plt.subplot(5,1,4)
plt.plot(range(epochs),speci,marker='.',label='speci')
plt.plot(range(epochs),val_speci,marker='.',label='val_speci')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('specificity')

plt.subplot(5,1,5)
plt.plot(range(epochs),auc,marker='.',label='roc_auc')
plt.plot(range(epochs),val_auc,marker='.',label='val_aoc_auc')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('roc_auc')

plt.show()

db = pd.read_csv(CSV_name)
y_true = db[ans_index].values
y_pred = db['P'].values


roc = roc_curve(y_true, y_pred)

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.show

print('ROC={}'.format(roc_auc_score(y_true, y_pred)))
print('study_date={}'.format(time))