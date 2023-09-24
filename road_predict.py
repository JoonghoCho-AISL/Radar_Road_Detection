import numpy as np
import tensorflow as tf

class road_detection():
    def __init__(self, path='model/basemodel/concat_mean_var_norm_pca'):
        self.idx2label_Dict = {
                0 : 'asphalt',
                # 1 : 'bicycle',
                1 : 'brick',
                2 : 'tile',
                3 : 'sandbed',
                4 : 'urethane',
            }
        self.model = tf.keras.models.load_model(path)
    
    def reload(self,path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, data):
        y_pred = self.model.predict(data)
        idx = np.argmax(y_pred)
        return self.idx2label_Dict[idx]
