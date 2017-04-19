import os
from keras.preprocessing.image import ImageDataGenerator

from cnn.network import Architecture

batch_size = 16
img_height = 150
img_width = 150

## path to the datasets and training model
training_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..',"resources","training"))
validation_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..',"resources","validation"))
trained_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"model"))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    training_dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    color_mode='grayscale')

#print train_generator.samples
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    color_mode='grayscale')


model = Architecture.build_model(img_height, img_width)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


model.fit_generator(
    train_generator,
    steps_per_epoch=1600 // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)


model.save_weights(os.path.join(trained_model_path,'trained_model_50.h5'), overwrite=True)
del model