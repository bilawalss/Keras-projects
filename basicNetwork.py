# Validation Accuracy = 85%
# Test Accuracy = 84.22%
# Training_samples = 1393
# Validation_samples = 697
# channels_first with theano
# channels_last with tensorflow

from __future__ import division

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models  import Sequential
from keras.layers import *
from keras import backend as K
from scipy.misc import imread
import sys, os
from sklearn.preprocessing import LabelEncoder

###### Configuration #######

img_width, img_height = 256, 342 # These are default image dimensions
train_data_dir = './data/all_years_342x256/train/'
val_data_dir = './data/all_years_342x256/val/'
test_data_dir = './data/all_years_342x256/test/'
batch_size = 16 # increase it depending on how fast the gpu runs
epochs = 40

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

		
# Loads the data from the directory, will try to make a separate class for it
# Returns (X_train, y_train), (X_test, y_test)s

def _load_data(data_dir):
	
	labels = os.listdir(data_dir)
	label_counter = 0
	label_lst = list()
	img_lst = list()

	# time to return the images and the labels

	encoder = LabelEncoder()
	for label in labels: 	
		if (not label.startswith('.')):
			img_dir = data_dir + str(label)+"/"
			images = os.listdir(img_dir)
			for img in images:
				if (not img.startswith('.')):
					img2 = imread((img_dir + img)[:])
					img_lst.append(img2)
					label_lst.append(label)

	transformed_label = encoder.fit_transform(label_lst)

	X_data = np.asarray(img_lst)
	X_data = np.transpose(X_data, (0, 3, 1, 2))
	y_data = np.asarray(transformed_label, dtype=np.uint8)

	return (X_data, y_data)


# load the data
(X_train, y_train) = _load_data(train_data_dir)
(X_test, y_test) = _load_data(test_data_dir)
(X_val, y_val) = _load_data(val_data_dir)

# Preprocess the data

X_train = X_train.astype('float32')
X_train /= 255
y_train = keras.utils.to_categorical(y_train, 14)

X_test = X_test.astype('float32')
X_test /= 255
y_test = keras.utils.to_categorical(y_test, 14)

X_val = X_val.astype('float32')
X_val /= 255
y_val = keras.utils.to_categorical(y_val, 14)

# create the neural network architecture

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(3, 256, 342)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='softmax'))


model.compile(loss='categorical_crossentropy', # can change to categorical_crossentropy
				optimizer = 'rmsprop', # can use adagrad instead
				metrics = ['accuracy'])


model.fit(x=X_train, y=y_train, batch_size=batch_size, validation_data=(X_val, y_val), epochs = epochs, verbose=1)

score = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
print score

# model.save_weights('Saved_Weights.h5')




