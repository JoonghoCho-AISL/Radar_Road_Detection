# Copyright (c) Acconeer AB, 2022
# All rights reserved

import acconeer.exptool as et
from acconeer.exptool import a121
from multiprocessing import Process, Queue
import argparse
from kafka import KafkaProducer
from json import dumps
import numpy as np
import time
from tqdm import tqdm

def producer(label, q):
    producer = KafkaProducer(
        acks = 'all', 
        compression_type = 'gzip', 
        bootstrap_servers = ['203.250.148.120:20517'],
        # value_serializer = lambda v: dumps(v).encode('utf-8')
        value_serializer = lambda v: v.tobytes(),
        key_serializer = lambda x: dumps(x).encode('utf-8')
    )
    i = 0
    start = time.time()
    while True:
        data = q.get()
        # print(data.shape)
        if (str(type(data)) == "<class 'str'>"):
            producer.send('radar-joongho', value = np.array([0+0j]), key = data)
            print(data)
            break
        value = data
        # print(value.shape)
        # print(value)
        label_num = label + str(i)
        # producer.send('radar-joongho', value = value, key = label)
        producer.send('radar-joongho', value = value, key = label_num)
        print(i)
        i += 1
        
        # time.sleep(0.1)
        # print('send')
    print('kafka time : ', time.time() - start)


def main(data_len, q):
    args = a121.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    # client = a121.Client(**a121.get_client_args(args))
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
        sweeps_per_frame = 20,
        frame_rate = 30,
    )
    # session_config = a121.SessionConfig(
    #     [
    #         {
    #             2: sensor_config,
    #         },
    #     ],
    # )

    client.setup_session(sensor_config)
    # client.setup_session(session_config)
    client.start_session()
    
    start = time.time()
    for i in tqdm(range(data_len)):
        client.get_next()
    print('time : ', time.time() - start)
    client.stop_session()
    client.disconnect()

if __name__ == '__main__':

    data_len = 10

    # parent_conn, child_conn = Pipe()
    queue = Queue()


    label = input('Type label : ')
    # label = label + str(1)
    # print(label)
    # print(type(label))
    # main(data_len, label, producer)
    # p_receiver = Process(target = main, args = (data_len, child_conn))
    p_receiver = Process(target = main, args = (data_len, queue,))
    p_receiver.start()
    # p_sender = Process(target = sender, args = (label, producer, parent_conn))
    p_sender = Process(target = producer, args = (label, queue,))
    p_sender.start()

    p_receiver.join()
    p_sender.join()
    queue.close()
