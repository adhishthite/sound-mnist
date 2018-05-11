# Digit Recognition from Sound
#### A simple neural network (CNN) to classify spoken digits (0-9).
---

Dataset: <a href='https://github.com/Jakobovski/free-spoken-digit-dataset'>free-spoken-digit-dataset</a> (FSDD)

## Step 1 - Data Preprocessing

The data is provided as 50 audio samples (WAV files) of each digit per person, and 3 people have contributed to the official project.

Total data = 1500 audio samples in *.wav* format.

### Possible approaches to this problem -
* Simple Neural Network<br>

	1. Load the *wav* file as a NUMPY array, and feed this numpy array to a simple _multi-layer perceptron_.
	2. When we convert a WAV file to a NUMPY array, the data gets stored as a 1-D matrix. The length of this array is not specific, andit is highly dependent on the data.
	3. Due to this, the first layer of the MLP will have ~10000 neurons. Further calculations will add extreme complexity.

* Spectrogram<br>
	
	1. Convert the *wav* data into a spectrogram (image file) of size (64*64)
	2. Feed the image file to a simple Neural Network with 4096 neurons in the first layer.
	3. This is a good approach, but the number of neurons are large, and it does not seem logical to flatten out an image and feed it to a simple NN. We can do better.
	4. Based on point 3 above, we can feed this 64*64 image to a simple Convolutional Neural Network.
	5. Every audio willl be converted into a simple 2-D image, and this image will be fed to a CNN. This will speed up the training, and as CNNs are flawless in simple image recognition, we will definitely get a good output.
	
