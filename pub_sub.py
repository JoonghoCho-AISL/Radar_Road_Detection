import paho.mqtt.client as mqtt
import json
import random


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)


def on_disconnect(client, userdata, flags, rc=0):
    print(str(rc))


def on_publish(client, userdata, mid):
    print("In on_pub callback mid= ", mid)

def crt_sub(IP, aename, path, cnt_name):
    rand = str(int(random.random()*100000))
    crt_sub = {
                    "to": path+'/'+cnt_name,
                    "fr": "S"+aename,
                    "op":1,
                    "ty":23,
                    "rqi": rand,
                    "pc":{
                        "m2m:sub": {
                            "rn": cnt_name, 
                            "enc":{"net":[3]},
                            "nu":["mqtt://" + IP + "/"+aename + cnt_name+"?ct=json"]
                            }
                        }
                }
    return crt_sub

def publishing(IP , Port, aename, path, cnt_name):
    # 새로운 클라이언트 생성
    client = mqtt.Client()

    # 콜백 함수 설정 on_connect(브로커에 접속), on_disconnect(브로커에 접속중료), on_publish(메세지 발행)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish

    # address : localhost, port: 1883 에 연결
    client.connect(IP, Port)

    init_sub = json.dumps(crt_sub(IP, aename, path, cnt_name))
    
    # common topic 으로 메세지 발행
    # client.publish('/oneM2M/req/SintelliSensor/Mobius2/json', init_cnt)
    client.publish('/oneM2M/req/S' + aename + '/Mobius2/json', init_sub)

    # 연결 종료
    client.disconnect()

if __name__ == "__main__":

    IP = '203.250.148.120'
    Port = 20516
    aename = 'AISL'
    path = "Mobius/AISL/radarSensor"
    cnt_name = 'target'

    # sub_name = cnt_name

    publishing(IP , Port, aename, path, cnt_name)