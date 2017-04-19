import os

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from cnn.network import Architecture


img_height = 150
img_width = 150
batch_size = 16


#path to dataset and training model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"model","trained_model_50.h5"))
test_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..',"resources","test"))

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    color_mode='grayscale')


model = Architecture.build_model(img_height, img_width)

model.load_weights(model_path,by_name='true')

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
scores = model.evaluate_generator(test_generator,40)

print "accuracy: " + str(scores[1])

