from tensorflow import keras
from tensorflow.keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


model = keras.models.load_model('cifar-model.h5')

model.evaluate(x_test, y_test)
