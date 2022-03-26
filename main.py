import os
import random

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def ShowImage(images, labels, total=10):
    plt.gcf().set_size_inches(18, 20)
    if total >= 25:
        total = 25
    for i in range(0, total):
        num = random.randint(0, len(labels) - 1)
        img_show = plt.subplot(5, 5, i + 1)
        Img = images[num]
        img_show.imshow(Img)
    plt.show()


ShowImage(X_train, y_train, total=10)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(X_train.shape, X_test.shape)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(y_train.shape, y_test.shape)


# model，請依照自己的方式建立模型
def build_model():
    # input layer
    inputs = Input(shape=(X_train.shape[-1],))

    # dense layer
    dense = Dense(units='512', activation='relu', name='Layer1')(inputs)
    dense = Dense(units='1024', activation='relu', name='layer2')(dense)
    dense = Dense(units='512', activation='relu', name='layer3')(dense)
    dense = Dense(units='128', activation='relu', name='layers4')(dense)

    # output layer
    outputs = Dense(units='10', activation='softmax', name='output')(dense)

    return Model(inputs, outputs)


model = build_model()
# 查看模型結構
model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              metrics=['acc'],
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

modelcheckpoint = ModelCheckpoint(filepath="./classification.h5",
                                  monitor="val_acc",
                                  verbose=0,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode="max", )

earlystop = EarlyStopping(monitor="val_loss",
                          patience=5,
                          verbose=0,
                          mode="min", )

reduced = ReduceLROnPlateau(monitor="val_loss",
                            factor=0.5,
                            patience=10,
                            verbose=0,
                            mode="min",
                            min_delta=0.0001,
                            cooldown=1, )

callbacks = [modelcheckpoint, earlystop, reduced]

history = model.fit(x=X_train,
                    y=y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    callbacks=callbacks)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']


def Show_Train_flow(dnnmodel, Show='acc', Title='Training accuracy comparison'):
    plt.plot(dnnmodel.history[Show])
    plt.title(Title)
    plt.ylabel(Show)
    plt.xlabel('Epoch')
    plt.legend(['dnn'])
    plt.show()


Show_Train_flow(history, Show='acc', Title='Training accuracy comparison')
Show_Train_flow(history, Show='val_acc', Title='Validation accuracy comparison')
Show_Train_flow(history, Show='loss', Title='Training loss comparison')
Show_Train_flow(history, Show='val_loss', Title='Validation loss comparison')
