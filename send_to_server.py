# Copyright (c) Acconeer AB, 2022
# All rights reserved

import acconeer.exptool as et
from acconeer.exptool import a121
from multiprocessing import Process, Queue, Pipe
import argparse
from kafka import KafkaProducer
from json import dumps
import numpy as np
import time

# def sender(label, producer, q):

#     while True:
#         # data = q.get()
#         # encode_data = np.array2string(data)
#         test = {'hello' : 'hello'}
#         producer.send('radar_joongho', value = test)
#         # producer.send('radar_joongho', value = {label : encode_data})
#         # print('send')

def main(data_len, label, producer):
    args = a121.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    client = a121.Client(**a121.get_client_args(args))
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
            # a121.SubsweepConfig(
            #     start_point=75,
            #     step_length=4,
            #     num_points=25,
            #     profile=a121.Profile.PROFILE_3,
            #     hwaas=20,
            # ),
        ],
        sweeps_per_frame = 10,
        frame_rate = 30,
    )
    session_config = a121.SessionConfig(
    [
        {
            2: sensor_config,
        },
    ],
    extended=True,
)

    # client.setup_session(sensor_config)
    client.setup_session(session_config)
    client.start_session()
    
    start = time.time()
    for i in range(data_len):
        result = client.get_next()
        # print(result.frame)
        # q.put(result.frame)
        data = np.array2string(result.frame)
        send_data = {label : data}
        producer.send('radar_joongho', value = send_data)
    print('time : ', time.time() - start)

    client.stop_session()
    client.disconnect()

if __name__ == '__main__':

    data_len = 30

    parent_conn, child_conn = Pipe()
    queue = Queue()
    producer = KafkaProducer(
        acks = 0, 
        compression_type = 'gzip', 
        bootstrap_servers = ['203.250.148.120:20517'],
        value_serializer = lambda v: dumps(v).encode('utf-8')
    )

    label = input('Type label')
    main(data_len, label, producer)
    # p_receiver = Process(target = main, args = (data_len, child_conn))
    # p_receiver = Process(target = main, args = (data_len, queue))
    # p_receiver.start()
    # # p_sender = Process(target = sender, args = (label, producer, parent_conn))
    # p_sender = Process(target = sender, args = (label, producer, queue))
    # p_sender.start()

    # p_receiver.join()
    # p_sender.join()
    queue.close()
