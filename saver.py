from tkinter import S
import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
# import time
from tqdm import tqdm
import json
import os
import argparse
from MobiusAPI import http_post_get
import pickle as pkl
# from sklearn.decomposition import PCA
import time
# import tensorflow as tf
# from models import basemodel_rasp
# from pub_sub import publishing
# from sklearn.decomposition import PCA

MAX = 20655.93

class radar_raspi():
    def __init__(self,
                    ip_address='127.0.0.1',
                    start_point = 16,
                    num_points = 32,
                    hwaas = 8,
                    sweep_per_frame = 10,
                    frame_rate = 10
                    ):
        self.client = a121.Client(ip_address = ip_address)

        self.client.connect()
        #start_distance = start_point * 2.5mm
        # start_point = 160 # 400mm
        #end_distance = start_point * 2.5mm + num_points * 2.5mm 
        # num_points = (end_dis - start_point * 2.5) / 2.5 + 1
        # num_points = 301 # 1,150mm

        sensor_config = a121.SensorConfig(
            subsweeps=[
                a121.SubsweepConfig(
                    start_point = start_point,
                    step_length = 1,
                    num_points = num_points,
                    profile = a121.Profile.PROFILE_1,
                    hwaas = 8,
                ),
            ],
            sweeps_per_frame = 10,
            frame_rate = 10,
        )

        self.client.setup_session(sensor_config)
        self.client.start_session()
    
    def read_data(self):
        raw_data = self.client.get_next()
        frame = raw_data.frame
        data = np.expand_dims(np.abs(np.mean(frame, axis=0)), axis = 0)
        return data
    
    def read_mean_var(self):
        raw_data = self.client.get_next()
        frame = raw_data.frame
        mean = np.mean(frame, axis = 0)
        var = np.var(frame, axis = 0)
        data = np.expand_dims(np.abs(np.concatenate((mean, var), axis = 0)), axis = 0 )
        return data
        
    def disconnect(self):
        self.client.disconnect()

def main():
    parser = argparse.ArgumentParser(description = 'select ip, road name, save or post')
    parser.add_argument('-i', '--ip', action = 'store', default = '192.168.222.144')
    parser.add_argument('-c', '--road_class', action = 'store', default = None)
    parser.add_argument('-s', '--save', action = 'store_true')
    parser.add_argument('-n', '--number', action = 'store', default = None)
    args = parser.parse_args()

    ip = args.ip
    road = args.road_class
    save = args.save
    number = args.number

    MAX = 122299391.16999999
    
    client = radar_raspi(ip_address=ip)
    print('clinet_start')
    if save:
        folder_path = 'e-scooter/'
        file_path = folder_path + road
        temp = list()
        for i in tqdm(range(int(number))):
            data = client.read_mean_var()
            temp.append(data)
        save_data = np.array(temp)
        np.save(file_path, save_data)
    else:
        # pca = pkl.load(open('pca.pkl', 'rb'))
        while True:
            data = client.read_mean_var()
            processed_data = data/MAX
            # processed_data = pca.transform(processed_data)
            send_data = json.dumps(processed_data.tolist())
            URI = '/Mobius/PM_MFBE29/radarSensor/rawData'
            AE_ID = 'PM_MFBE29'
            http_post_get.mobius_post(URI, AE_ID, AE_ID, send_data)
            # print('pub')
            time.sleep(10)
    client.disconnect()

if __name__ == '__main__':
    main()