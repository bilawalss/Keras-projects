from __future__ import division

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models  import Sequential
from keras.layers import *
from keras import backend as K
from scipy.misc import imread
import sys, os
from sklearn.preprocessing import LabelEncoder

###### preprocessing image #######

# configuration
img_width, img_height = 225, 225 # These are default image dimensions
train_data_dir = './data/all_years_342x256/train/'
val_data_dir = './data/all_years_342x256/val/'
test_data_dir = './data/all_years_342x256/test/'
batch_size = 32 # increase it depending on how fast the gpu runs
epochs = 10

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
y_train = keras.utils.to_categorical(y_train, 14)

X_test = X_test.astype('float32')
y_test = keras.utils.to_categorical(y_test, 14)

X_val = X_val.astype('float32')
y_val = keras.utils.to_categorical(y_val, 14)

# create the neural network architecture

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape= input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics = ['accuracy'])



train_datagen = ImageDataGenerator(

			rescale = 1./255,
			fill_mode = 'nearest',
			horizontal_flip = True,)

test_datagen = ImageDataGenerator(

			rescale = 1./255)

train_datagen.fit(X_train)

train_generator = train_datagen.flow_from_directory(

				train_data_dir,
				target_size = (225, 225),
				batch_size = batch_size,
				data_format = "channels_first")

validation_generator = test_datagen.flow_from_directory(

				val_data_dir,
				target_size=(225, 225),
				batch_size = batch_size,
				data_format = "channels_first")

model.fit_generator(

				train_generator,
				steps_per_epoch = 1393 // batch_size,
				epochs = epochs,
				validation_data = validation_generator,
				validation_steps = 697 // batch_size)


score = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
print score

# model.save_weights('Saved_Weights.h5')



