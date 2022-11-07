import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from acconeer.exptool import a121
from sklearn.model_selection import train_test_split
import argparse

import pickle as pkl

from sklearn.decomposition import PCA

from models import basemodel


label2idx_Dict = {
                'asphalt' : 0,
                # 'bicycle' : 1,
                'sidewalk' : 1,
                'floor' : 2,
                'ground' : 3,
            }

idx2label_Dict = {
    0 : 'asphalt',
    # 1 : 'bicycle',
    1 : 'sidewalk',
    2 : 'floor',
    3 : 'ground',
}

def train(
    model, X, Y,
    test_X, test_Y, 
    # callback,
    Epoch = 50,
    learning_rate = 1e-3,
    # devices = ['/gpu:0', '/gpu:1']
    ):

    # strategy = tf.distribute.MirroredStrategy(devices=devices)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    # with strategy.scope():
    model.compile(optimizer = optimizer , loss = loss, 
                metrics = ['accuracy', 'categorical_crossentropy'])
    history = model.fit(X, Y,  epochs = Epoch,
                validation_data = (test_X, test_Y),
                )
    return history

def add_label(arr, label):
    label_list = [label for j in range(arr.shape[0])]
    label_list = np.array(label_list)
    label_list = np.reshape(label_list, (arr.shape[0], 1))
    # print(label_list.shape)
    labeled_arr = np.concatenate((arr, label_list), axis = 1)
    return labeled_arr

def concat(arr : list):
    temp = arr[0]
    for i in range(1, len(arr)):
        temp = np.concatenate((temp, arr[i]), axis = 0)
    return temp

def saveplot(history, plot_name, pca):
    if pca:
        plot_name = './basemodel_diff/' + str(plot_name) + 'pca' + '.png'
    else:
        plot_name = './basemodel_diff/' + str(plot_name) + '.png'
    plt.subplot(4,1,1)
    plt.plot(history['val_accuracy'])
    plt.title('val_accuracy')
    plt.subplot(4,1,2)
    plt.plot(history['val_loss'])
    plt.title('val_loss')
    plt.subplot(4,1,3)
    plt.plot(history['accuracy'])
    plt.title('accuracy')
    plt.subplot(4,1,4)
    plt.plot(history['loss'])
    plt.title('loss')
    plt.savefig(plot_name, dpi = 300)

def diff(arr1, arr2):
    tmp = list()
    for i in range(len(arr1)):
        tmp.append(arr2[i] - arr1[i])
    return np.array(tmp)

def diff_sweep(arr):
    tmp = list()
    for i in range(len(arr) - 1):
        tmp.append(diff(arr[i], arr[i + 1]))
    return np.array(tmp)

def diff_frame(arr):
    tmp = list()
    for i in range(len(arr)):
        tmp.append(diff_sweep(arr[i]))
    return np.array(tmp)

def main():
    
    parser = argparse.ArgumentParser(description = 'selt data and gpu')
    parser.add_argument('-g', '--gpu', action = 'store', default = '3')
    parser.add_argument('-p', '--pca', action = 'store_true')
    parser.add_argument('-d', '--data', action = 'store')
    parser.add_argument('-f', '--diff', action = 'store')
    parser.add_argument('-m', '--model', action = 'store', default = 'base')
    args = parser.parse_args()

    gpu = int(args.gpu)
    data = args.data
    Pca = args.pca
    sel_model = args.model
    diff = args.diff

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # 텐서플로가 첫 번째 GPU에 10GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)
    asphalt_data = a121.load_record('./road_data/asphalt.h5').frames
    sidewalk_data = a121.load_record('./road_data/block.h5').frames
    floor_data = a121.load_record('./road_data/floor.h5').frames
    ground_data = a121.load_record('./road_data/ground.h5').frames

    asphalt_abs = np.abs(asphalt_data)
    sidewalk_abs = np.abs(sidewalk_data)
    floor_abs = np.abs(floor_data)
    ground_abs = np.abs(ground_data)

    asphalt_mean = np.mean(asphalt_abs, axis = 1)
    sidewalk_mean = np.mean(sidewalk_abs, axis = 1)
    floor_mean = np.mean(floor_abs, axis = 1)
    ground_mean = np.mean(ground_abs, axis =1)

    MAX = np.max([np.max(asphalt_mean), np.max(sidewalk_mean), np.max(floor_mean), np.max(ground_mean)])
    MAX_origin = np.max([np.max(asphalt_abs), np.max(sidewalk_abs), np.max(floor_abs), np.max(ground_abs)])

    random_seed = 33

    MAX_ABS = np.max([asphalt_abs, sidewalk_abs, floor_abs, ground_abs])
    asphalt_abs = asphalt_abs / MAX_ABS
    sidewalk_abs = sidewalk_abs / MAX_ABS
    floor_abs = floor_abs / MAX_ABS
    ground_abs = ground_abs / MAX_ABS

    norm_asphalt_mean = asphalt_mean / MAX
    norm_sidewalk_mean = sidewalk_mean / MAX
    norm_floor_mean = floor_mean / MAX
    norm_ground_mean = ground_mean / MAX

    norm_asphalt_var = np.var(asphalt_abs / MAX_origin, axis = 1)
    norm_sidewalk_var = np.var(sidewalk_abs / MAX_origin, axis = 1)
    norm_floor_var = np.var(floor_abs/ MAX_origin, axis = 1)
    norm_ground_var = np.var(ground_abs / MAX_origin, axis = 1)

    asphalt_frame_diff = diff_frame(asphalt_abs)
    sidewalk_frame_diff = diff_frame(sidewalk_abs)
    floor_frame_diff = diff_frame(floor_abs)
    ground_frame_diff = diff_frame(ground_abs)

    asphalt_frame_diff_mean = np.mean(asphalt_frame_diff, axis = 1)
    sidewalk_frame_diff_mean = np.mean(sidewalk_frame_diff, axis = 1)
    floor_frame_diff_mean = np.mean(floor_frame_diff, axis = 1)
    ground_frame_diff_mean = np.mean(ground_frame_diff, axis = 1)

    asphalt_frame_diff_var = np.var(asphalt_frame_diff, axis = 1)
    sidewalk_frame_diff_var = np.var(sidewalk_frame_diff, axis = 1)
    floor_frame_diff_var = np.var(floor_frame_diff, axis = 1)
    ground_frame_diff_var = np.var(ground_frame_diff, axis = 1)

    if data == 'mean':
    
        if not Pca:
            asphalt_X = add_label(norm_asphalt_mean[:,:3], label2idx_Dict['asphalt'])
            floor_X = add_label(norm_floor_mean[:,:3], label2idx_Dict['floor'])
            sidewalk_X = add_label(norm_sidewalk_mean[:,:3], label2idx_Dict['sidewalk'])
            ground_X = add_label(norm_ground_mean[:,:3], label2idx_Dict['ground'])
        
        elif diff == 'mean':
            pre_Data =concat([asphalt_frame_diff_mean, sidewalk_frame_diff_mean, floor_frame_diff_mean, ground_frame_diff_mean])
            pca = PCA(n_components = 'mle')
            pca.fit(pre_Data)
        
            asphalt_X = add_label(asphalt_frame_diff_mean, label2idx_Dict['asphalt'])
            floor_X = add_label(floor_frame_diff_mean, label2idx_Dict['floor'])
            sidewalk_X = add_label(sidewalk_frame_diff_mean, label2idx_Dict['sidewalk'])
            ground_X = add_label(ground_frame_diff_mean, label2idx_Dict['ground'])
            
            pca_path = './pca_diff/mean_pca.pkl'
        
        elif diff == 'var':
            pre_Data =concat([asphalt_frame_diff_var, sidewalk_frame_diff_var, floor_frame_diff_var, ground_frame_diff_var])
            pca = PCA(n_components = 'mle')
            pca.fit(pre_Data)
        
            asphalt_X = add_label(asphalt_frame_diff_var, label2idx_Dict['asphalt'])
            floor_X = add_label(floor_frame_diff_var, label2idx_Dict['floor'])
            sidewalk_X = add_label(sidewalk_frame_diff_var, label2idx_Dict['sidewalk'])
            ground_X = add_label(ground_frame_diff_var, label2idx_Dict['ground'])

            pca_path = './pca_diff/var_pca.pkl'

        elif diff == 'all' :
            asphalt_X = np.concatenate((norm_asphalt_mean, asphalt_frame_diff_var), axis = 1)
            sidewalk_X = np.concatenate((norm_sidewalk_mean, sidewalk_frame_diff_var), axis = 1)
            floor_X = np.concatenate((norm_floor_mean, floor_frame_diff_var), axis = 1)
            ground_X = np.concatenate((norm_ground_mean, ground_frame_diff_var), axis = 1)

            pre_Data =concat([asphalt_X, sidewalk_X, floor_X, ground_X])
            pca = PCA(n_components = 'mle')
            pca.fit(pre_Data)

            asphalt_X = add_label(asphalt_X, label2idx_Dict['asphalt'])
            sidewalk_X = add_label(sidewalk_X, label2idx_Dict['sidewalk'])
            floor_X = add_label(floor_X, label2idx_Dict['floor'])
            ground_X = add_label(ground_X, label2idx_Dict['ground'])
            
            pca_path = './pca_diff/mean_var_pca.pkl'

        Data = concat([asphalt_X, floor_X, sidewalk_X, ground_X])

        Y = tf.one_hot(Data[:,-1], len(label2idx_Dict)).numpy()

    elif data == 'var':

        if not Pca:
            asphalt_var_X = add_label(norm_asphalt_var[:,:4], label2idx_Dict['asphalt'])
            floor_var_X = add_label(norm_floor_var[:,:4], label2idx_Dict['floor'])
            sidewalk_var_X = add_label(norm_sidewalk_var[:,:4], label2idx_Dict['sidewalk'])
            ground_var_X = add_label(norm_ground_var[:,:4], label2idx_Dict['ground'])
        
        else:
            pre_Data =concat([asphalt_frame_diff_var, sidewalk_frame_diff_var, floor_frame_diff_var, ground_frame_diff_var])
            pca = PCA(n_components = 'mle')
            pca.fit(pre_Data)
        
            asphalt_var_X = add_label(asphalt_frame_diff_var, label2idx_Dict['asphalt'])
            floor_var_X = add_label(floor_frame_diff_var, label2idx_Dict['floor'])
            sidewalk_var_X = add_label(sidewalk_frame_diff_var, label2idx_Dict['sidewalk'])
            ground_var_X = add_label(ground_frame_diff_var, label2idx_Dict['ground'])
            pca_path = './pca_diff/var_pca.pkl'
        Data = concat([asphalt_var_X, floor_var_X, sidewalk_var_X, ground_var_X])

        Y = tf.one_hot(Data[:,-1], len(label2idx_Dict)).numpy()

    elif data == 'all':
        
        if not Pca:
            asphalt_X = add_label(np.concatenate((norm_asphalt_mean[:,:3], norm_asphalt_var[:,:4]), axis = 1), label2idx_Dict['asphalt'])
            floor_X = add_label(np.concatenate((norm_floor_mean[:,:3], norm_floor_var[:,:4]), axis = 1), label2idx_Dict['floor'])
            sidewalk_X = add_label(np.concatenate((norm_sidewalk_mean[:,:3], norm_sidewalk_var[:, :4]), axis = 1), label2idx_Dict['sidewalk'])
            ground_X = add_label(np.concatenate((norm_ground_mean[:,:3], norm_ground_var[:,:4]), axis = 1), label2idx_Dict['ground'])
        
        else:
            asphalt = np.concatenate([norm_asphalt_mean, norm_asphalt_var], axis = 1)
            floor = np.concatenate([norm_floor_mean, norm_floor_var], axis = 1)
            sidewalk = np.concatenate([norm_sidewalk_mean, norm_sidewalk_var], axis = 1)
            ground = np.concatenate([norm_ground_mean, norm_ground_var], axis = 1)
            pre_Data = concat([asphalt, floor, sidewalk, ground])

            pca = PCA(n_components='mle')
            pca.fit(pre_Data)

            asphalt_X = add_label(pca.transform(asphalt), label2idx_Dict['asphalt'])
            floor_X = add_label(pca.transform(floor), label2idx_Dict['floor'])
            sidewalk_X = add_label(pca.transform(sidewalk), label2idx_Dict['sidewalk'])
            ground_X = add_label(pca.transform(ground), label2idx_Dict['ground'])
            
            pca_path = './pca_diff/all_pca.pkl'

        Data = concat([asphalt_X, floor_X, sidewalk_X, ground_X])
  
        Y = tf.one_hot(Data[:,-1], len(label2idx_Dict)).numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(Data[:,:-1], Y, random_state = random_seed)

    if pca:
        with open(pca_path, 'wb') as pickle_file:
            pkl.dump(pca, pickle_file)

    if sel_model == 'base':
        model = basemodel()
        history = train(model, X_train, Y_train, X_test, Y_test)
        saveplot(history.history, (data + '_' + diff), Pca)

    elif sel_model == 'rf':
        import tensorflow_decision_forests as tfdf
        model = tfdf.keras.RandomForestModel()
        # history = train(model, X_train, np.argmax(Y_train, axis = 1), X_test, np.argmax(Y_test, axis = 1))
        history = model.fit(X_train, np.argmax(Y_train, axis = 1), validation_data = (X_test, np.argmax(Y_test, axis = 1)))
        model.evaluate(X_test, np.argmax(Y_test, axis = 1))

    else: 
        pass

if __name__ == '__main__':
    main()