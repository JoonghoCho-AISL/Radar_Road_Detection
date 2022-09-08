# Copyright (c) Acconeer AB, 2022
# All rights reserved

import acconeer.exptool as et
from acconeer.exptool import a121
from multiprocessing import Process, Queue, Pipe
import argparse
from kafka import KafkaProducer
from json import dumps

def sender(producer, conn):
    while True:
        data = conn.recv()
        if data == False:
            break
        else:
            producer.send('radar_joongho', data)

def main(data_len, conn):
    args = a121.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    client = a121.Client(**a121.get_client_args(args))
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
        sweeps_per_frame = 64,
        frame_rate = 30,
    )

    client.setup_session(sensor_config)
    client.start_session()
    label = input('Type label')

    for i in range(data_len):
        result = client.get_next()
        conn.send(result)
        # print(f"\nResult {i + 1} subframes:")
        print(result.subframes)

    client.stop_session()
    client.disconnect()

if __name__ == '__main__':

    data_len = 30000

    parent_conn, child_conn = Pipe()

    producer = KafkaProducer(
    acks = 0, 
    compression_type = 'gzip', 
    bootstrap_servers = ['203.250.148.120:20517'],
    value_serializer = lambda v: dumps(v).encode('utf-8')
    )

    p_receiver = Process(target = main, args = (data_len, child_conn))
    p_receiver.start()
    p_sender = Process(target = sender, args = (producer, parent_conn))
    p_sender.start()