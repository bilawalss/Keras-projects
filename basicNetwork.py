from __future__ import division

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models  import Sequential
from keras.layers import *
from keras import backend as K
from scipy.misc import imread
import sys, os
from sklearn.preprocessing import LabelEncoder

###### preprocessing image #######

# dimensions of our images
img_width, img_height = 256, 342 # These are default image dimensions
train_data_dir = '/home/bilawal/Summer2017/myNetwork/data/all_years_342x256/train/'
val_data_dir = '/home/bilawal/Summer2017/myNetwork/data/all_years_342x256/val/'
test_data_dir = '/home/bilawal/Summer2017/myNetwork/data/all_years_342x256/test/'
batch_size = 8 # increase it depending on how fast the gpu runs


if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

		
# Loads the data from the directory, will try to make a separate class for it
# Returns (X_train, y_train), (X_test, y_test)s

def _load_data_train(data_dir=train_data_dir):
	
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
	y_data = np.asarray(transformed_label, dtype=np.uint8)
	#X_data = X_data.transpose(0,3,1,2)

	return (X_data, y_data)

def _load_data_test(data_dir=test_data_dir):
	
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
	y_data = np.asarray(transformed_label, dtype=np.uint8)
	#X_data = X_data.transpose(0,3,1,2)

	return (X_data, y_data)	

def _load_data_val(data_dir=val_data_dir):
	
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
	y_data = np.asarray(transformed_label, dtype=np.uint8)
	#X_data = X_data.transpose(0,3,1,2)

	return (X_data, y_data)		



(X_train, y_train) = _load_data_train()
(X_test, y_test) = _load_data_test()
(X_val, y_val) = _load_data_val()

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
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 342, 3)))
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

"""

model.fit_generator (
		(X_data, y_data),
		steps_per_epoch = X_data.shape[0] // batch_size,
		epochs = epochs,
		validation_data = (X_data, y_data),
		validaton_steps = X_data.shape[0] // batch_size)


"""

model.fit(x=X_train, y=y_train, batch_size=8, validation_data=(X_val, y_val), epochs = 10, verbose=1)

score = model.evaluate(X_test, y_test, batch_size = 8, verbose = 1)
print score

# model.save_weights('Saved_Weights.h5')




