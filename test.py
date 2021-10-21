import keras
from sklearn.metrics import classification_report
from utils import wav2mfcc, model, get_data
import numpy as np
import tensorflow as tf

def check_preds(X, y):

    trained_model = keras.models.load_model('trained_model.h5')
    predictions = trained_model.predict(X)
    classes_x = np.argmax(predictions, axis=1)
    print(classification_report(y, tf.keras.utils.to_categorical(classes_x)))


if __name__ == '__main__':
    _, X_test, _, y_test, _ = get_data.get_all()

    check_preds(X_test, y_test)