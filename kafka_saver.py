from kafka import KafkaConsumer
from json import loads
import numpy as np
from multiprocessing import Process, Queue
import pandas as pd

SWEEPS_PER_FRAME = 30

# def preprocess(kafka_q, data_q):
#     window = np.hanning(SWEEPS_PER_FRAME)
#     window  /= np.sum(window)
    
#     while True:
#         frame = kafka_q.get()
#         z_ft = np.fft.fftshift(np.fft.fft(frame * window, axis = 0), axes = (0,))
#         abs_z_ft = np.abs(z_ft)
#         mean_frame = np.mean(frame, axis = 0)


def consumer(q):
    consumer = KafkaConsumer(
        # 'radar_joongho',
        'radar-joongho',
        bootstrap_servers = ['203.250.148.120:20517'],
        auto_offset_reset = 'latest',
        # value_deserializer = lambda v: loads(v.decode('utf-8')),
        key_deserializer = lambda x: loads(x.decode('utf-8')),
        value_deserializer = lambda v: np.frombuffer(v, dtype = complex),
        consumer_timeout_ms = 1000
    )
    
    try:
        while True:
            for message in consumer:
                # print('Topic: %s, Key: %s, Value: %s' % (message.topic, message.key, message.value))
                q.put(message)
    except KeyboardInterrupt:
        print('consumer end')


def saver(data_q):
    window = np.hanning(SWEEPS_PER_FRAME)
    window  /= np.sum(window)
    save_data = list()
    try:
        while True:
            data = data_q.get()
            label = np.array(data.key)
            print(label)
            if label == 'end':
                print('break')
                break
            frame = data.value
            frame = np.reshape(frame, (SWEEPS_PER_FRAME,40))
            # print(frame.shape)
            save_data.append(frame)
            # print(label)
            # print(frame.shape)
            # mean_frame = np.mean(frame, axis = 0)
            # var = np.var(frame, axis = 0)
            # z_ft = np.fft.fftshift(np.fft.fft(frame * window, axis = 0), axes = (0,))
            # abs_z_ft = np.abs(z_ft)
            # feature = np.concatenate((frame))
            # df = pd.concat([df, pd.DataFrame(feature.T)])
        print('break')
    except KeyboardInterrupt:
        # df.to_csv('./road_data.csv')
        # df.to_csv('./road_data/{}.csv')
        save_np = np.array(save_data)
        np.save('./road_data/%s.npy'& (label), save_np)
        print('saver end')


if __name__ == '__main__':
    q = Queue()
    a = list()
    df = pd.DataFrame()
    # df = pd.read_csv('./road_data/road_data.csv')
    p_consumer_1 = Process(target = consumer, args =(q,))
    # p_consumer_2 = Process(target = consumer, args = (q,))
    # p_consumer_3 = Process(target = consumer, args = (q,))
    # p_consumer_4 = Process(target = consumer, args = (q,))
    p_saver = Process(target = saver, args = (q,))
    
    try:
        p_consumer_1.start()
        # p_consumer_2.start()
        # p_consumer_3.start()
        # p_consumer_4.start()
        p_saver.start()
    except KeyboardInterrupt:
        p_consumer_1.join()
        # p_consumer_2.join()
        # p_consumer_3.join()
        # p_consumer_4.join()
        p_saver.join()