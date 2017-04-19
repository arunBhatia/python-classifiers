from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Conv2D
from keras.models import Sequential


class Architecture:

    @staticmethod
    def build_model(img_height, img_width):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(1, img_height, img_width)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(40))
        model.add(Activation('softmax'))


        # return the constructed network architecture
        return model