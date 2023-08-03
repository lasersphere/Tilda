import random
from paho.mqtt import client as mqtt_client
from Tilda.Interface.MQTTSimpleCounter.Ui_simpleCounterRunnning import Ui_SimpleCounterRunning
from PyQt5 import QtCore, QtGui, QtWidgets


broker = 'broker.emqx.io'
port = 1883
topic = "DACvoltage"
# Generate a Client ID with the subscribe prefix.
client_id = f'subscribe-{random.randint(0, 1000)}'
# username = 'emqx'
# password = 'public'

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        #Ui_SimpleCounterRunning.doubleSpinBox.setProperty("value", msg.payload.decode())

    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()

run()