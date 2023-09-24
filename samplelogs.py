import http.client, urllib.request, urllib.parse
import urllib.request
import requests
import threading
import random
import json
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import pickle
import numpy as np
import math


def thingspeak_post():
    threading.Timer(15, thingspeak_post).start()
    global machinestatus
    machinestatus = 1
    type_list = [0,1,2]
    type = random.choice(type_list)
    airtemp = random.uniform(294,305)
    processtemp = random.uniform(304, 315)
    rotspeed = random.uniform(1160, 2900)#1168 -2886
    torque =random.uniform(3.5, 77)#3.8 - 76.6

    print(type,airtemp,processtemp,rotspeed,torque)
    if(type == 0):
        wear_add = 2
    elif(type == 1):
        wear_add = 3
    else:
        wear_add = 5

    toolwear = read_data() + wear_add #0-253
    if toolwear > 300:
        toolwear = 0
    machine_failure = predict_randomforest(type, airtemp, processtemp, rotspeed, torque,toolwear)
    #print('Machine Failure : ',machine_failure)
    print(machine_failure.tolist())
    machine_failure = machine_failure.tolist()
    tmp = []
    for i in range(1,len(machine_failure)):
        tmp.append(str(int(machine_failure[i])))
    predicted_val = '_'.join(tmp)
    print('Predicted failures : ',predicted_val)

    #ct = datetime.datetime.now()
    #print("current time:-", ct)

    URl = 'https://api.thingspeak.com/update?api_key='
    KEY = 'JHQ5NHT94XAIG20R'
    HEADER = '&field1={}&field2={}&field3={}&field4={}&field5={}&field6={}&field7={}&field8={}'.format(type, airtemp, processtemp, rotspeed, torque, toolwear,predicted_val,machine_failure[0])
    NEW_URL = URl + KEY + HEADER
    print(NEW_URL)
    data = urllib.request.urlopen(NEW_URL)
    print(data)

    #Now Displaying the Uploaded data by rendering on the Dashboard
    #outputdisplay = read_all_data()
    #return render_template('dashboard.html',
     #                      airtmp='{}'.format(outputdisplay[3]),processtmp='{}'.format(outputdisplay[4]))


def predict_randomforest(type, airtemp, processtemp, rotspeed, torque,toolwear):
    airprocessdiff = processtemp - airtemp
    power = torque * rotspeed
    overstrain = toolwear * torque
    final = np.array([type, airtemp, processtemp, rotspeed, torque, toolwear, airprocessdiff, power, overstrain]).reshape(1, 9)
    class_prediced = randomforestModel.predict(final)[0]
    #output = str(class_prediced)
    #if output == 1:
        #return render_template('svm_ui.html',
                               #pred='Machine may fail.\nProbability of failure is {}'.format(output),
                               #bhai="kuch karna hain iska ab?")
    #else:
        #return render_template('svm_ui.html',
                               #pred='Your Machine is safe.\n Probability of failure is {}'.format(output),
                               #bhai="Your Forest is Safe for now")
    #return (output)
    return class_prediced



def load_model():
    global randomforestModel
    randomforestFile = open('RandomForestModel.pkl', 'rb')
    randomforestModel = pickle.load(randomforestFile)
    randomforestFile.close()


def read_data():
    URL = 'https://api.thingspeak.com/channels/2060037/fields/6.json?api_key=XJI52RR42TKU1WT1&results=6'
    KEY = 'XJI52RR42TKU1WT1'
    HEADER = '&results=2'
    NEW_URL = URL+KEY+HEADER
    #print(URL)
    get_data = requests.get(URL).json()
    #print(get_data)
    channel_id = get_data['channel']['id']
    if len(get_data['feeds']) == 0:
        field6 = 0
    else:
        field6 = int(get_data['feeds'][-1]['field6'])
    #print(field6)

    return field6


if __name__ == '__main__':
    #read_data()
    load_model()
    #thingspeak_post()
    #read_all_data()
    #app.run(debug=True)
    thingspeak_post()