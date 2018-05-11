import numpy as np
import librosa
import os
from keras.utils import to_categorical

def wav2mfcc(file_path, max_pad_len=20):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=8000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def get_data():

    labels = []
    mfccs = []

    for f in os.listdir('./recordings'):
        if f.endswith('.wav'):
            # MFCC
            mfccs.append(wav2mfcc('./recordings/' + f))

            # List of labels
            label = f.split('_')[0]
            labels.append(label)

    return np.asarray(mfccs), to_categorical(labels)

# if __name__ == '__main__':
#     mfccs, labels = get_data()
#     print(mfccs.shape)
#     print(labels.shape)