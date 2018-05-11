import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

def get_cnn_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    return model


# if __name__ == '__main__':
#     model = get_cnn_model((20, 20, 1), 10)
#
#     print(model.summary())

''' OUTPUT 

Last Epoch - 
Epoch 50/50

  64/1215 [>.............................] - ETA: 5s - loss: 0.0280 - acc: 1.0000
 128/1215 [==>...........................] - ETA: 5s - loss: 0.0431 - acc: 0.9844
 192/1215 [===>..........................] - ETA: 4s - loss: 0.0330 - acc: 0.9896
 256/1215 [=====>........................] - ETA: 4s - loss: 0.0277 - acc: 0.9922
 320/1215 [======>.......................] - ETA: 4s - loss: 0.0255 - acc: 0.9938
 384/1215 [========>.....................] - ETA: 3s - loss: 0.0290 - acc: 0.9896
 448/1215 [==========>...................] - ETA: 3s - loss: 0.0291 - acc: 0.9888
 512/1215 [===========>..................] - ETA: 3s - loss: 0.0375 - acc: 0.9863
 576/1215 [=============>................] - ETA: 3s - loss: 0.0367 - acc: 0.9878
 640/1215 [==============>...............] - ETA: 2s - loss: 0.0347 - acc: 0.9875
 704/1215 [================>.............] - ETA: 2s - loss: 0.0335 - acc: 0.9872
 768/1215 [=================>............] - ETA: 2s - loss: 0.0314 - acc: 0.9883
 832/1215 [===================>..........] - ETA: 1s - loss: 0.0303 - acc: 0.9892
 896/1215 [=====================>........] - ETA: 1s - loss: 0.0329 - acc: 0.9888
 960/1215 [======================>.......] - ETA: 1s - loss: 0.0313 - acc: 0.9896
1024/1215 [========================>.....] - ETA: 0s - loss: 0.0300 - acc: 0.9902
1088/1215 [=========================>....] - ETA: 0s - loss: 0.0338 - acc: 0.9890
1152/1215 [===========================>..] - ETA: 0s - loss: 0.0331 - acc: 0.9896
1215/1215 [==============================] - 6s 5ms/step - loss: 0.0317 - acc: 0.9901 - val_loss: 0.2954 - val_acc: 0.9185
             precision    recall  f1-score   support

          0       1.00      0.84      0.91        19
          1       0.87      0.87      0.87        15
          2       1.00      1.00      1.00        23
          3       0.91      1.00      0.95        10
          4       1.00      1.00      1.00        10
          5       1.00      1.00      1.00        23
          6       1.00      1.00      1.00        13
          7       0.93      1.00      0.96        13
          8       1.00      1.00      1.00        14
          9       0.91      1.00      0.95        10

avg / total       0.97      0.97      0.97       150

'''