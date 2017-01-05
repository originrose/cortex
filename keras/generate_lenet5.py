from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Activation
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD


def mnist_model():
    """Cortex MNIST impl inspired model:

       (layers/convolutional 5 0 1 20)
       (layers/max-pooling 2 0 2)
       (layers/relu)
       (layers/convolutional 5 0 1 50)
       (layers/max-pooling 2 0 2)
       (layers/relu)
       (layers/convolutional 1 0 1 50)
       (layers/relu)
       (layers/linear->relu 1000)
       (layers/dropout 0.5)
       (layers/linear->softmax 10)])"""
    model = Sequential()
    model.add(Convolution2D(20, 5, 5, border_mode='same', input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(50, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(50, 1, 1, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def modern_model():
    """Modern MNIST architecture."""
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def train(model, X_train, y_train, epochs, validation_data=None):
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, nb_epoch=epochs, verbose=1,
              validation_data=validation_data)
    return model


def save(model, output_pre):
    with open(output_pre + '.json', 'w') as jsonf:
        jsonf.write(model.to_json())
    model.save_weights(output_pre + ".h5")

# import MNIST dataset from Keras, reshape to add gray channel dimension
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# train a modern and lenet inspired model, save them.
model = modern_model()
model = train(model, X_train, y_train, 8, validation_data=(X_test, y_test))
model2 = mnist_model()
model2 = train(model2, X_train, y_train, 8, validation_data=(X_test, y_test))

save(model, "models/modern_mnist")
save(model2, "models/cortex_mnist")
