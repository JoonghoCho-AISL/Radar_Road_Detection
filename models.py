import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class basemodel(Model):
    def __init__(self):
        super(basemodel, self).__init__()
        self.nl1 = layers.Normalization(axis = -1)
        self.fc1 = layers.Dense(units = 10, activation = 'relu')
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(units = 5, activation = 'relu')
        self.out = layers.Dense(units = 5, activation = 'softmax')
    
    def call(self, x, training = False, mask = None):
        out = self.nl1(x)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.out(out)
        return out

class acconeer(Model):
    def __init__(self):
        super(acconeer, self).__init__()
        self.fc1 = layers.Dense(units = 24, activation = 'relu')
        self.fc2 = layers.Dense(units = 12, activation = 'relu')
        self.out = layers.Dense(units = 5, activation = 'softmax')
    def call(self, x , training = False, mask = None):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.out(out)
        return out

def create_model():
    model = tf.keras.models.Sequential([
        layers.Normalization(axis = -1),
        layers.Dense(units = 10, activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dense(units=5, activation = 'relu'),
        layers.Dense(units = 1, activation = 'relu')
        ])
    return model