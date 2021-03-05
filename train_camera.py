from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras_preprocessing.image import ImageDataGenerator

model=Sequential()

model.add(Convolution2D(32,(3,3),input_shape=(64,64,1),activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(32,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128,activation = 'relu'))
model.add(Dense(5,activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen=ImageDataGenerator(1/255.0,validation_split=0.2)

training_set=train_datagen.flow_from_directory("Dataset/fingers",
                                         target_size=(64,64),
                                         class_mode='categorical',
                                         batch_size=1,
                                         color_mode='grayscale',
                                         subset='training')


test_set = test_datagen.flow_from_directory("Dataset/fingers",
                                        target_size = (64,64),
                                        batch_size = 1,
                                        color_mode='grayscale',
                                        class_mode = 'categorical',
                                        subset='validation')

model.fit_generator(generator=training_set,
                    steps_per_epoch=len(training_set),
                    epochs=12,
                    validation_data=test_set,
                    validation_steps=len(test_set))

testing_model = model.evaluate_generator(test_set, len(test_set), verbose=1)
print('Percentage of accuracy: ' + str(int(testing_model[1] * 10000) / 100) + '%')

model.save('Model/hand_gestures2.h5') # %99 accuracy