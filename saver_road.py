# Copyright (c) Acconeer AB, 2022
# All rights reserved

from tkinter import S
import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time
from tqdm import tqdm
import os

def main(data_len):


    parser = a121.ExampleArgumentParser()
    parser.add_argument('--name', '-n', action = 'store')
    args = parser.parse_args()
    et.utils.config_logging(args)
    label = args.name

    filename = ('./road_data/%s.h5'%(label))
    if os.path.exists(filename):
        os.remove(filename)
    h5_recorder = a121.H5Recorder(filename)

    client = a121.Client(ip_address = '127.0.0.1')
    # client.ip_address = '127.0.0.1'
    client.connect()
    #start_distance = start_point * 2.5mm
    start_distance = 100
    start_point = start_distance / 2.5
    #end_distance = start_point * 2.5mm + num_points * 2.5mm
    end_distance = 200
    num_points = (end_distance - start_point * 2.5) / 2.5

    sensor_config = a121.SensorConfig(
        subsweeps=[
            a121.SubsweepConfig(
                start_point = start_point,
                step_length = 1,
                num_points = num_points,
                profile = a121.Profile.PROFILE_1,
                hwaas = 10,
            ),
        ],
        sweeps_per_frame = 30,
        frame_rate = 30,
    )
    client.setup_session(sensor_config)
    client.start_session(h5_recorder)
    
    start = time.time()
    for i in tqdm(range(data_len)):
        client.get_next()
        # print(f"Result {i + 1}/{data_len} was sampled")
    print('time : ', time.time() - start)

    client.stop_session()
    client.disconnect()

if __name__ == '__main__':

    data_len = 15000
    # label = input('Type label : ')
    main(data_len)
