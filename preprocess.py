import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

import os

from acconeer.exptool import a121

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from tqdm import tqdm

import pickle as pkl

label2idx_Dict = {
                'asphalt' : 0,
                # 'bicycle' : 1,
                'brick' : 1,
                'tile' : 2,
                'sandbed' : 3,
                'urethane' : 4,
            }

idx2label_Dict = {
    0 : 'asphalt',
    # 1 : 'bicycle',
    1 : 'brick',
    2 : 'tile',
    3 : 'sandbed',
    4 : 'urethane',
}
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

class processing():
    def __init__(self):
        self.init_data = {
            'asphalt' : a121.load_record('./road_data/asphalt.h5').frames,
            'brick' : a121.load_record('./road_data/block.h5').frames,
            'tile' : a121.load_record('./road_data/floor.h5').frames,
            'sandbed' : a121.load_record('./road_data/ground.h5').frames,
            'urethane' : a121.load_record('./road_data/urethane.h5').frames,
        }

        self.random_seed = 33

        self.abs_data = dict()

        for i in tqdm(self.init_data) :
            self.abs_data[i] = self.make_abs(self.init_data[i])

        self.MAX_ABS = None

    def make_abs(self, arr):
        return np.abs(arr)
        # return np.abs(arr[:,:,:11])
    
    def make_mean_frame(self):
        self.mean_data = dict()
        for i in tqdm(self.abs_data):
            self.mean_data[i] = np.mean(self.abs_data[i], axis = 1)
    
    def make_var_frame(self):
        self.var_data = dict()
        for i in tqdm(self.abs_data):
            self.var_data[i] = np.var(self.abs_data[i], axis = 1)
    
    def make_norm_mean_frame(self):
        self.make_mean_frame()
        
        mean_frame = list()
        for i in self.mean_data:
            mean_frame.append(self.mean_data[i])

        self.MAX_MEAN = np.max(mean_frame)

        self.norm_mean_data = dict()
        for i in self.mean_data:
            self.norm_mean_data[i] = self.mean_data[i] / self.MAX_MEAN
        
        # return self.norm_mean_frame

    def make_mean_of_norm_data(self):
        temp = list()
        for i in self.abs_data:
            temp.append(self.abs_data[i])
        
        # MAX value for normalization of data
        self.MAX_ABS = np.max(temp)

        self.mean_of_norm_data = dict()
        for i in self.abs_data:
            self.mean_of_norm_data[i] = np.mean(self.abs_data[i] / self.MAX_ABS, axis = 1)

    def make_var_of_norm_data(self):
        """
        Get variance of Normalized Data
        """
        if self.MAX_ABS is None:
            temp = list()
            for i in self.abs_data:
                temp.append(self.abs_data[i])
        
        # MAX value for normalization of data
            self.MAX_ABS = np.max(temp)

        self.var_of_norm_data = dict()
        for i in self.abs_data:
            self.var_of_norm_data[i] = np.var(self.abs_data[i] / self.MAX_ABS, axis = 1)

        # return self.var_of_norm_data
    def make_norm_of_var_data(self):
        temp = list()
        self.make_var_frame()
        for i in self.var_data:
            temp.append(self.var_data[i])
        var_max = np.max(temp)
        self.norm_of_var_data = dict()
        for i in self.var_data:
            self.norm_of_var_data[i] = self.var_data[i] / var_max


    def make_frame_diff(self):
        """
        Get diffential of sweeps
        """
        self.diff_of_frame = dict()
        for i in self.abs_data:
            self.diff_frame[i] = self.diff_frame(self.abs_data[i])
        # return self.diff_frame

    def mean_of_frame_diff(self):
        self.diff_frame_mean = dict()
        self.make_frame_diff()
        for i in self.diff_of_frame:
            self.diff_frame_mean[i] = np.mean(self.diff_of_frame[i], axis = 1)
        # return self.diff_frame_mean
    
    def var_of_frame_diff(self):
        self.diff_frame_var = dict()
        self.make_frame_diff()
        for i in self.diff_of_frame:
            self.diff_frame_var[i] = np.var(self.diff_of_frame[i], axis = 1)
        # return self.diff_frame_var

    def diff(self, arr1, arr2):
        if arr1.shape == ():
            return arr2 - arr1
        else :
            tmp = list()
            for i in range(len(arr1)):
                tmp.append(arr2[i] - arr1[i])
            return np.array(tmp)
        
    def diff_sweep(self, arr):
        tmp = list()
        for i in range(len(arr) - 1):
            tmp.append(self.diff(arr[i], arr[i + 1]))
        return np.array(tmp)
    
    def diff_frame(self, arr):
        tmp = list()
        for i in range(len(arr)):
            tmp.append(self.diff_sweep(arr[i]))
        return np.array(tmp)
    
    def diff_dist_sweep(self, arr):
        tmp = list()
        for i in range(len(arr) - 1):
            tmp.append(arr[i + 1] - arr[i])
        return np.array(tmp)

    def diff_dist_frame(self, arr):
        tmp = list()
        for i in range(len(arr)):
            tmp.append(self.diff_dist_sweep(arr[i]))
        return np.array(tmp)

    def diff_dist(self):
        self.dist = dict()
        for i in self.abs_data:
            self.dist[i] = np.argmax(self.abs_data[i], axis = 2)
        self.diff_of_dist = dict()
        for i in self.dist:
            # for j in range(len(self.dist[i])):
            self.diff_of_dist[i] = (self.diff_dist_frame(self.dist[i]))
        # return self.diff_of_dist
    
    def diff_mean_dist(self):
        self.diff_dist()
        self.diff_mean_of_dist = dict()
        for i in self.diff_of_dist:
            self.diff_mean_of_dist[i] = np.mean(self.diff_of_dist[i], axis = 1)

    def diff_var_dist(self):
        self.diff_dist()
        self.diff_var_of_dist = dict()
        for i in self.diff_of_dist:
            self.diff_var_of_dist[i] = np.mean(self.diff_of_dist[i], axis = 1)

    def add_label(self, arr, label):
        # label = list()
        # for i in idx2label_Dict:
        #     label.append(idx2label_Dict[i])
        label_list = [label2idx_Dict[label] for j in range(len(arr))]
        label_list = np.array(label_list)
        label_list = np.reshape(label_list, (len(arr), 1))
        # print(label_list.shape)
        labeled_arr = np.concatenate((arr, label_list), axis = 1)
        return labeled_arr

    def data_concat(self, dict1, dict2):
        temp = dict()
        for i in dict1:
            temp[i] = np.concatenate((dict1[i], dict2[i]), axis = -1)
        return temp

    def autocovariance(self, Xi, N, k, Xs):
        autoCov = 0
        for i in range(N-k):
            autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        # print(Xi.shape)
        return (1/(N-1))*autoCov

    def autocovar(self, Xi, Xs):
        temp = list()
        # Xs = np.mean(arr, axis = 1)
        N = Xi.shape[1]
        # print(Xi.shape)
        for i in range(len(Xi)):
            temp.append(self.autocovariance(Xi[i], N = N, k = 4, Xs = Xs[i]))
        # print(np.array(temp).shape)
        return np.array(temp)

    def autocovdict(self):
        temp = dict()
        self.make_mean_frame()
        for i in self.abs_data:
            temp[i] = self.autocovar(self.abs_data[i], self.mean_data[i])
        return temp

    def concat(self, arr : list):
        temp = arr[0]
        for i in range(1, len(arr)):
            temp = np.concatenate((temp, arr[i]), axis = 0)
        return temp

    def pca_data(self, data_dict, path):
        temp = list()
        self.pca_dict = dict()
        for i in data_dict:
            temp.append(data_dict[i])

        pre_data = self.concat(temp)
        # pca = PCA(n_components = 'mle')
        pca = PCA()
        pca.fit(pre_data)

        for i in data_dict:
            self.pca_dict[i] = pca.transform(data_dict[i])
        
        dir_name = os.path.dirname(path)
        createDirectory(dir_name)
        with open(path, 'wb') as pickle_file:
            pkl.dump(pca, pickle_file)

        return self.pca_dict
    
    def make_ds(self, data : dict):
        temp_arr = list()

        for i in data:
            temp_arr.append(self.add_label(data[i], i))

        temp_data = self.concat(temp_arr)

        Y = to_categorical(temp_data[:,-1], num_classes = len(label2idx_Dict))
        X_train, X_test, Y_train, Y_test = train_test_split(temp_data[:,:-1], Y, random_state = self.random_seed)

        return X_train, X_test, Y_train, Y_test