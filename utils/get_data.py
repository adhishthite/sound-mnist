import utils
from sklearn.model_selection import train_test_split

def get_all():
    mfccs, labels = utils.wav2mfcc.get_data()

    dim_1 = mfccs.shape[1]
    dim_2 = mfccs.shape[2]
    channels = 1
    classes = 10

    X = mfccs
    X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
    y = labels

    input_shape = (dim_1, dim_2, channels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    model = utils.model.get_cnn_model(input_shape, classes)

    return X_train, X_test, y_train, y_test, model