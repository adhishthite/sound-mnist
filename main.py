import comet_ml
from comet_ml import Experiment
experiment = Experiment(api_key="S1ApsUzOKM7jwtsLLcT1drdZ0", project_name="soundmnist")

from utils import model, wav2mfcc, get_data
import test

X_train, X_test, y_train, y_test, cnn_model = get_data.get_all()

print(cnn_model.summary())

cnn_model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1, validation_split=0.1)

cnn_model.save('trained_model.h5')

test.check_preds(X_test, y_test)