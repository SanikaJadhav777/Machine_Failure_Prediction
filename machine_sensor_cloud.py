import urllib.request, urllib.parse
import urllib.request
import requests
import random
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

key = 'JHQ5NHT94XAIG20R'  # copy the write API key

# using datetime module
import datetime

# ct stores current time

uid = 1


def predict_randomforest(type, airtemp, processtemp, rotspeed, torque, toolwear):
    airprocessdiff = processtemp - airtemp
    power = torque * rotspeed
    overstrain = toolwear * torque
    final = np.array(
        [type, airtemp, processtemp, rotspeed, torque, toolwear, airprocessdiff, power, overstrain]).reshape(1, 9)
    class_prediced = randomforestModel.predict(final)[0]
    # output = str(class_prediced)
    # if output == 1:
    # return render_template('svm_ui.html',
    # pred='Machine may fail.\nProbability of failure is {}'.format(output),
    # bhai="kuch karna hain iska ab?")
    # else:
    # return render_template('svm_ui.html',
    # pred='Your Machine is safe.\n Probability of failure is {}'.format(output),
    # bhai="Your Forest is Safe for now")
    # return (output)
    return class_prediced


@app.route('/predict_svm', methods=['POST', 'GET'])
def predict_svm(typee, airtp, processtp, rotrpm, trq, toolwear):
    global svmIrisModel
    svmIrisFile = open('SupportVectorModel.pkl', 'rb')
    svmIrisModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = float(airtp)
    process = float(processtp)
    rpm = float(rotrpm)
    torque = float(trq)
    tool = int(toolwear)
    print(typee)
    if typee == 'L':
        typ = 0
    elif typee == 'M':
        typ = 1
    elif typee == 'H':
        typ = 2

    diff = float(air) - float(process)
    power = float(rpm) * float(torque)
    overstrain = float(torque) * float(tool)
    final = np.array([typ, air, process, rpm, torque, tool, diff, power, overstrain]).reshape(1, 9)
    class_prediced = (svmIrisModel.predict(final)[0])
    output = str(class_prediced)
    tmp1 = [0, 0, 0, 0, 0, 0]
    fail = "No Failure"
    '''if class_prediced == 1 or class_prediced == 2 or class_prediced == 3 or class_prediced == 4 or class_prediced == 5:
        tmp1[0] = 1
        if float(tool) > 200 and float(tool) <= 240:
            fail += "Tool Wear Failure"
            tmp1[1] = 1
        if diff <= 8.66 and float(rpm) <= 1380:
            tmp1[2] = 1
            fail += "Heat Dissipation Failure"
        if power >= 3500 or power <= 9000:
            tmp1[3] = 1
            fail += "Power Failure"
        if overstrain >= 11000 or overstrain <= 13000:
            tmp1[4] = 1
            fail += "Overstrain Failure"
        if class_prediced == 5:
            tmp1[5] = 1'''
    print(class_prediced)
    if class_prediced == 1:
        tmp1[0] = 1
        tmp1[3] = 1
        fail += "Power Failure"
    elif class_prediced == 2:
        tmp1[0] = 1
        tmp1[1] = 1
        fail += "Tool Wear Failure"
    elif class_prediced == 3:
        tmp1[0] = 1
        tmp1[4] = 1
        fail += "Overstrain Failure"
    elif class_prediced == 4:
        tmp1[0] = 1
        tmp1[5] = 1
        fail += "Random Failure"
    elif class_prediced == 5:
        tmp1[0] = 1
        tmp1[2] = 1
        fail += "Heat Dissipation Failure"

    return tmp1


def predict_decisiontree(type, airtemp, processtemp, rotspeed, torque, toolwear):
    global dtModel
    dtModelFile = open('DT.pkl', 'rb')
    dtModel = pickle.load(dtModelFile)
    dtModelFile.close()
    airprocessdiff = processtemp - airtemp
    power = torque * rotspeed
    overstrain = toolwear * torque
    final = np.array(
        [type, airtemp, processtemp, rotspeed, torque, toolwear, airprocessdiff, power, overstrain]).reshape(1, 9)
    class_prediced = dtModel.predict(final)[0]
    # output = str(class_prediced)
    # if output == 1:
    # return render_template('svm_ui.html',
    # pred='Machine may fail.\nProbability of failure is {}'.format(output),
    # bhai="kuch karna hain iska ab?")
    # else:
    # return render_template('svm_ui.html',
    # pred='Your Machine is safe.\n Probability of failure is {}'.format(output),
    # bhai="Your Forest is Safe for now")
    # return (output)
    return class_prediced


@app.route('/predict_decision', methods=['POST', 'GET'])
def predict_decision(typee, airtp, processtp, rotrpm, trq, toolwear):
    global dtModel
    svmIrisFile = open('DecisionTreeModel.pkl', 'rb')
    dtModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = float(airtp)
    process = float(processtp)
    rpm = float(rotrpm)
    torque = float(trq)
    tool = int(toolwear)
    if typee == 'L':
        typ = 0
    elif typee == 'M':
        typ = 1
    elif typee == 'H':
        typ = 2

    diff = float(air) - float(process)
    power = float(rpm) * float(torque)
    overstrain = float(torque) * float(tool)
    final = np.array([typ, air, process, rpm, torque, tool, diff, power, overstrain]).reshape(1, 9)
    class_predicted = dtModel.predict(final)
    output1 = class_predicted.tolist()
    fail = ''
    print(output1)
    tmp = []
    tmp1 = []
    msg = 'Machine is safe'
    for i in output1[0]:
        tmp.append(str(int(i)))
        tmp1.append(int(i))
    output = "_".join(tmp)
    print(tmp1)

    if tmp[0] == '1':
        msg = 'Machine may fail'
        for j in range(1, 5):
            if tmp1[j] == 1 and j == 1:
                fail += "Tool wear Failure"
            elif tmp1[j] == 1 and j == 2:
                fail += " Heat Dissipation Failure"
            elif tmp1[j] == 1 and j == 3:
                fail += " Power Failure"
            elif tmp1[j] == 1 and j == 4:
                fail += " Overstrain Failure"
        # return render_template('sample.html',pred='{}.\nTarget failure is {}.\nFailure Type: {}'.format(msg, output, fail))
    else:
        msg = "Machine is safe"
    print(output)

    return tmp1


@app.route('/predict_lr', methods=['POST', 'GET'])
def predict_lr(typee, airtp, processtp, rotrpm, trq, toolwear):
    global lrModel
    svmIrisFile = open('LogisticsModel.pkl', 'rb')
    lrModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = float(airtp)
    process = float(processtp)
    rpm = float(rotrpm)
    torque = float(trq)
    tool = int(toolwear)
    if typee == 'L':
        typ = 0
    elif typee == 'M':
        typ = 1
    elif typee == 'H':
        typ = 2
    diff = float(air) - float(process)
    power = float(rpm) * float(torque)
    overstrain = float(torque) * float(tool)
    print(type(typee))
    final = np.array([typ, air, process, rpm, torque, tool, diff, power, overstrain]).reshape(1, 9)
    class_prediced = int(lrModel.predict(final)[0])
    output = str(class_prediced)
    tmp1 = [0, 0, 0, 0, 0,0]
    fail = "No Failure"
    '''if class_prediced == 1 or class_prediced == 2 or class_prediced == 3 or class_prediced == 4 or class_prediced == 5:
        tmp1[0] = 1
        if float(tool) > 200 and float(tool) <= 240:
            fail += "Tool Wear Failure"
            tmp1[1] = 1
        if diff <= 8.66 and float(rpm) <= 1380:
            tmp1[2] = 1
            fail += "Heat Dissipation Failure"
        if power >= 3500 or power <= 9000:
            tmp1[3] = 1
            fail += "Power Failure"
        if overstrain >= 11000 or overstrain <= 13000:
            tmp1[4] = 1
            fail += "Overstrain Failure"
        if class_prediced == 5:
            tmp1[5] = 1
            fail += "Random Failure"'''
    print(class_prediced)
    if class_prediced == 1:
        tmp1[0] = 1
        tmp1[3] = 1
        fail += "Power Failure"
    elif class_prediced == 2:
        tmp1[0] = 1
        tmp1[1] = 1
        fail += "Tool Wear Failure"
    elif class_prediced == 3:
        tmp1[0] = 1
        tmp1[4] = 1
        fail += "Overstrain Failure"
    elif class_prediced == 4:
        tmp1[0] = 1
        tmp1[5] = 1
        fail += "Random Failure"
    elif class_prediced == 5:
        tmp1[0] = 1
        tmp1[2] = 1
        fail += "Heat Dissipation Failure"

    return tmp1


def load_model():
    global randomforestModel
    randomforestFile = open('RandomForestModel.pkl', 'rb')
    randomforestModel = pickle.load(randomforestFile)
    randomforestFile.close()


def read_data():
    URL = 'https://api.thingspeak.com/channels/2060037/fields/6.json?api_key=XJI52RR42TKU1WT1&results=6'
    KEY = 'XJI52RR42TKU1WT1'
    HEADER = '&results=2'
    NEW_URL = URL + KEY + HEADER
    # print(URL)
    get_data = requests.get(URL).json()
    # print(get_data)
    channel_id = get_data['channel']['id']
    if len(get_data['feeds']) == 0:
        field6 = 0
    else:
        field6 = int(get_data['feeds'][-1]['field6'])
    # print(field6)

    return field6


def read_all_data():
    URL = 'https://api.thingspeak.com/channels/2060037/feeds.json?api_key=XJI52RR42TKU1WT1&results=1'
    KEY = 'XJI52RR42TKU1WT1'
    HEADER = '&results=2'
    NEW_URL = URL + KEY + HEADER
    print(URL)
    get_data = requests.get(URL).json()
    # print(get_data)
    # channel_id = get_data['channel']['id']
    if len(get_data['feeds']) == 0:
        currenttimestmp = 0
        uid = 0
        field1 = 0
        field2 = 0
        field3 = 0
        field4 = 0
        field5 = 0
        field6 = 0
        field7 = 0
        field8 = 0
    else:
        currenttimestmp = str(get_data['feeds'][-1]['created_at'])
        uid = str(get_data['feeds'][-1]['entry_id'])
        field1 = int(get_data['feeds'][-1]['field1'])
        field2 = float(get_data['feeds'][-1]['field2'])
        field3 = float(get_data['feeds'][-1]['field3'])
        field4 = float(get_data['feeds'][-1]['field4'])
        field5 = float(get_data['feeds'][-1]['field5'])
        field6 = int(get_data['feeds'][-1]['field6'])
        field7 = str(get_data['feeds'][-1]['field7'])
        field8 = int(float(get_data['feeds'][-1]['field8']))
        failtype_list = field7.split("_");
        TWF = int(failtype_list[0])
        HDF = int(failtype_list[1])
        PWF = int(failtype_list[2])
        OSF = int(failtype_list[3])
        RNF = int(failtype_list[4])
        if field1 == 0:
            mach_type = 'L'
        if field1 == 1:
            mach_type = 'M'
        if field1 == 2:
            mach_type = 'H'
        m_uid = mach_type + uid
        # print(currenttimestmp,uid,field1,field2,field3,field4,field5,field6,field7,field8)
        all_cloud_data = [currenttimestmp, m_uid, mach_type, field2, field3, field4, field5, field6, TWF, HDF, PWF, OSF,
                          RNF, field8]
        # print(all_cloud_data)
    return all_cloud_data


global machinestatus
machinestatus = 0


# @app.route('/dashboard', methods=['POST', 'GET'])
def thingspeak_post():
    # threading.Timer(15, thingspeak_post).start()
    global machinestatus
    machinestatus = 1
    type_list = [0, 1, 2]
    type = random.choice(type_list)
    airtemp = random.uniform(294, 305)
    processtemp = random.uniform(304, 315)
    rotspeed = random.uniform(1160, 2900)  # 1168 -2886
    torque = random.uniform(3.5, 77)  # 3.8 - 76.6

    print(type, airtemp, processtemp, rotspeed, torque)
    if (type == 0):
        wear_add = 2
    elif (type == 1):
        wear_add = 3
    else:
        wear_add = 5

    toolwear = read_data() + wear_add  # 0-253
    if toolwear > 260:
        toolwear = 0
    machine_failure = predict_randomforest(type, airtemp, processtemp, rotspeed, torque, toolwear)
    # print('Machine Failure : ',machine_failure)
    # print(machine_failure.tolist())
    machine_failure = machine_failure.tolist()
    tmp = []
    for i in range(1, len(machine_failure)):
        tmp.append(str(int(machine_failure[i])))
    predicted_val = '_'.join(tmp)
    # print('Predicted failures : ',predicted_val)

    ct = datetime.datetime.now()
    # print("current time:-", ct)

    URl = 'https://api.thingspeak.com/update?api_key='
    KEY = 'JHQ5NHT94XAIG20R'
    HEADER = '&field1={}&field2={}&field3={}&field4={}&field5={}&field6={}&field7={}&field8={}'.format(type, airtemp,
                                                                                                       processtemp,
                                                                                                       rotspeed, torque,
                                                                                                       toolwear,
                                                                                                       predicted_val,
                                                                                                       machine_failure[
                                                                                                           0])
    NEW_URL = URl + KEY + HEADER
    print(NEW_URL)
    data = urllib.request.urlopen(NEW_URL)
    # print(data)

    # Now Displaying the Uploaded data by rendering on the Dashboard
    # outputdisplay = read_all_data()
    # return render_template('dashboard.html',
    #                      airtmp='{}'.format(outputdisplay[3]),processtmp='{}'.format(outputdisplay[4]))


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'machine_pred'

mysql = MySQL(app)


@app.route('/')
@app.route('/startup')
def startup():
    return render_template("startup.html", msg="")


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user_info WHERE email = % s AND password = % s', (username, password,))
        account = cursor.fetchone()
        if account:
            msg = 'Logged in successfully !'
            return render_template('sample.html', msg=msg)
        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg=msg)


# @app.route('/logout')
# def logout():
#    return redirect(url_for('login'))

@app.route('/close', methods=['GET', 'POST'])
def close():
    return render_template('startup.html', msg="")


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'fname' in request.form and 'lname' in request.form and 'email' in request.form and 'mobile' in request.form and 'password' in request.form:
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user_info WHERE email = % s', (email,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z]+', fname):
            msg = 'Name must contain only characters'
        elif not re.match(r'[A-Za-z]+', lname):
            msg = 'Name must contain only characters'
        elif not re.match(r'[0-9]+', mobile):
            msg = 'Mobile must contain only digits'
        elif not lname or not fname or not password or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user_info VALUES (%s, % s, % s, % s, %s)',
                           (fname, lname, email, mobile, password))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
            return render_template('login.html', msg=msg)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg=msg)


@app.route('/display_livedata', methods=['POST', 'GET'])
def display_livedata():
    return render_template('display_livedata.html', msg='')


@app.route('/logout')
def logout():
    return redirect(url_for('login'))


@app.route('/home')
def home():
    return render_template("sample.html", msg="")


@app.route('/dashboard')
def dashboard():
    # Now Displaying the Uploaded data by rendering on the Dashboard
    # threading.Timer(15, dashboard).start()
    thingspeak_post()
    # ----------------------------------------------------------

    # alerts------------------------
    infoalerts = ""
    warnalerts = ""
    dangeralerts = ""

    if machinestatus == 0:
        mstatus = 'Not in Work'
    else:
        mstatus = 'Active'

    outputdisplay = read_all_data()
    type = outputdisplay[2]
    airtp = outputdisplay[3]
    processtp = outputdisplay[4]
    rotrpm = outputdisplay[5]
    trq = outputdisplay[6]
    toolwear = outputdisplay[7]
    twf = outputdisplay[8]
    tediff = (outputdisplay[4] - outputdisplay[3])
    powe = outputdisplay[6] * outputdisplay[5]
    overstrain = outputdisplay[7] * outputdisplay[6]

    # ------------------------------------------------------
    # SVM Prediction -----------------------------
    tmp1 = predict_svm(type, airtp, processtp, rotrpm, trq, toolwear)
    # Decision tree------------------------------
    tmp2 = predict_decision(type, airtp, processtp, rotrpm, trq, toolwear)
    # Logistic Regression------------------------
    tmp3 = predict_lr(type, airtp, processtp, rotrpm, trq, toolwear)
    # ------------------------------------------------------

    RUL = 0
    tstatus = ''
    if toolwear < 190 and twf == 0:
        RUL = abs(toolwear - 210)
        tstatus = 'Safe Working Condition'
        infoalerts = "No Tool Failure Expected!"
    elif (toolwear >= 190 and toolwear <= 200) or twf == 1:
        RUL = abs(toolwear - 210)
        tstatus = 'Near to Wearout'
        warnalerts = "Cutting Tool needs to be Replaced....Quality of Workpiece will decrease!"
    elif (toolwear > 200 and toolwear <= 220) or twf == 1:
        RUL = abs(toolwear - 210)
        tstatus = 'Moderate Wearout'
        warnalerts = "Workpiece Quality has Reduced! Cutting Tool needs to be Replaced Immediately"
    elif toolwear > 220 or twf == 1:
        RUL = 0
        tstatus = 'Severe Wearout'
        dangeralerts = "Cutting Tool Needs Urgent Replacement! Workpiece Quality Damaging!"

    if overstrain > 11000 and type == 0 or outputdisplay[11] == 1:
        warnalerts += "\nMachine Type is 'L'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 12000 and type == 1 or outputdisplay[11] == 1:
        warnalerts += "\nMachine Type is 'M'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 13000 and type == 2 or outputdisplay[11] == 1:
        dangeralerts += "\nMachine Type is 'H'. Torque is Exceeding. Workpiece will Damage. Overstrain on Machine"

    if outputdisplay[10] == 1:
        dangeralerts += "\nPower is Fluctuating. Machine will Damage."

    if outputdisplay[9] == 1:
        dangeralerts += "\nThere is a Process Failure due to Heat dissipation failure!"

    return render_template('dashboard.html', uid='{}'.format(outputdisplay[1]), timestmp='{}'.format(outputdisplay[0]),
                           airtmp='{}'.format(outputdisplay[3]),
                           processtmp='{}'.format(outputdisplay[4]), activestatus=mstatus, twearstatus=tstatus,
                           rultool=RUL
                           , rpm='{}'.format(outputdisplay[5]), torque=outputdisplay[6], toolwear=outputdisplay[7],
                           mtype=outputdisplay[2], tdiff=(outputdisplay[4] - outputdisplay[3]),
                           power=(outputdisplay[6] * outputdisplay[5]),
                           overstrain=(outputdisplay[7] * outputdisplay[6]),
                           rffail=outputdisplay[13], rftwf=outputdisplay[8], rfhdf=outputdisplay[9],
                           rfpwf=outputdisplay[10], rfosf=outputdisplay[11], rfrnf=outputdisplay[12], knnfail=tmp3[0],
                           knntwf=tmp3[1], knnhdf=tmp3[2],
                           knnpwf=tmp3[3], knnosf=tmp3[4], knnrnf=0, dtfail=tmp2[0], dttwf=tmp2[1], dthdf=tmp2[2],
                           dtpwf=tmp2[3], dtosf=tmp2[4], dtrnf=0, svmfail=tmp1[0], svmtwf=tmp1[1], svmhdf=tmp1[2],
                           svmpwf=tmp1[3], svmosf=tmp1[4], svmrnf=0,
                           dangeralert=dangeralerts, warnalert=warnalerts, infoalert=infoalerts)


@app.route('/logs')
def logs():
    return render_template('logs.html')


@app.route('/plot_demo')
def plot_demo():
    return render_template('plot_charts.html', msg="")


@app.route('/data_plots')
def data_plots():
    return render_template('data_plots.html', msg='')


@app.route('/sample')
def sample():
    return render_template('sample.html', msg='')


@app.route('/predict_svm_manual', methods=['POST', 'GET'])
def predict_svm_manual():
    global svmIrisModel
    svmIrisFile = open('SupportVectorModel.pkl', 'rb')
    svmIrisModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = request.form['air']
    process = request.form['process']
    rpm = request.form['rpm']
    torque = request.form['torque']
    tool = request.form['tool']
    type = request.form['Type']
    diff = float(air) - float(process)
    power = float(rpm) * float(torque)
    overstrain = float(torque) * float(tool)
    final = np.array([type, air, process, rpm, torque, tool, diff, power, overstrain]).reshape(1, 9)
    class_prediced = int(svmIrisModel.predict(final)[0])
    output = str(class_prediced)
    tmp1 = [0, 0, 0, 0, 0, 0]
    fail = ""
    print(class_prediced)
    if class_prediced == 1:
        tmp1[0] = 1
        tmp1[3] = 1
        fail += "Power Failure"
    elif class_prediced == 2:
        tmp1[0] = 1
        tmp1[1] = 1
        fail += "Tool Wear Failure"
    elif class_prediced == 3:
        tmp1[0] = 1
        tmp1[4] = 1
        fail += "Overstrain Failure"
    elif class_prediced == 4:
        tmp1[0] = 1
        tmp1[5] = 1
        fail += "Random Failure"
    elif class_prediced == 5:
        tmp1[0] = 1
        tmp1[2] = 1
        fail += "Heat Dissipation Failure"

    '''if class_prediced == 1 or class_prediced == 2 or class_prediced == 3 or class_prediced == 4 or class_prediced == 5:
        tmp1[0] = 1
        if float(tool) > 191 and float(tool) <= 240:
            fail += "Tool Wear Failure"
            tmp1[1] = 1
        if diff <= 8.66 and float(rpm) <= 1380:
            tmp1[2] = 1
            fail += "Heat Dissipation Failure"
        if power >= 3500 or power <= 9000:
            tmp1[3] = 1
            fail += "Power Failure"
        if overstrain >= 11000 or overstrain <= 13000:
            tmp1[4] = 1
            fail += "Overstrain Failure"'''

    RUL = 0
    tstatus = ''
    dangeralerts = ''
    warnalerts = ''
    infoalerts = ''
    if float(tool) < 190 and tmp1[1] == 0:
        RUL = abs(float(tool) - 210)
        tstatus = 'Safe Working Condition'
        infoalerts = "No Tool Failure Expected!"
    elif (float(tool) >= 190 and float(tool) <= 200) or tmp1[1] == 1:
        RUL = abs(float(tool) - 210)
        tstatus = 'Near to Wearout'
        warnalerts = "Cutting Tool needs to be Replaced....Quality of Workpiece will decrease!"
    elif (float(tool) > 200 and float(tool) <= 220) or tmp1[1] == 1:
        RUL = abs(float(tool) - 210)
        tstatus = 'Moderate Wearout'
        warnalerts = "Workpiece Quality has Reduced! Cutting Tool needs to be Replaced Immediately"
    elif float(tool) > 220 or tmp1[1] == 1:
        RUL = 0
        tstatus = 'Severe Wearout'
        dangeralerts = "Cutting Tool Needs Urgent Replacement! Workpiece Quality Damaging!"

    if overstrain > 11000 and int(type) == 0 or tmp1[4] == 1:
        warnalerts += "\nMachine Type is 'L'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 12000 and int(type) == 1 or tmp1[4] == 1:
        warnalerts += "\nMachine Type is 'M'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 13000 and int(type) == 2 or tmp1[4] == 1:
        dangeralerts += "\nMachine Type is 'H'. Torque is Exceeding. Workpiece will Damage. Overstrain on Machine"
    if tmp1[3] == 1:
        dangeralerts += "\nPower is Fluctuating. Machine will Damage."
    if tmp1[2] == 1:
        dangeralerts += "\nThere is a Process Failure due to Heat dissipation failure!"
    if class_prediced == 1 or class_prediced == 2 or class_prediced == 3 or class_prediced == 4 or class_prediced == 5:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, fail),
                               svmfail=tmp1[0], svmtwf=tmp1[1], svmhdf=tmp1[2],
                               svmpwf=tmp1[3], svmosf=tmp1[4], svmrnf=tmp1[5], dangeralert=dangeralerts,
                               warnalert=warnalerts,
                               infoalert=infoalerts)
    else:
        return render_template('sample.html',
                               pred='Machine machine is safe.\nTarget failure is {}.\nFailure Type: No Failure'.format(
                                   output),
                               svmfail=tmp1[0], svmtwf=tmp1[1], svmhdf=tmp1[2],
                               svmpwf=tmp1[3], svmosf=tmp1[4], svmrnf=tmp1[5])
    return output


@app.route('/predict_random_manual', methods=['POST', 'GET'])
def predict_random_manual():
    svmIrisFile = open('RandomForestModel.pkl', 'rb')
    RMModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = request.form['air']
    process = request.form['process']
    rpm = request.form['rpm']
    torque = request.form['torque']
    tool = request.form['tool']
    type = request.form['Type']
    diff = float(air) - float(process)
    power = float(rpm) * float(torque)
    overstrain = float(torque) * float(tool)
    final = np.array([type, air, process, rpm, torque, tool, diff, power, overstrain]).reshape(1, 9)
    class_predicted = RMModel.predict(final)
    output1 = class_predicted.tolist()
    fail = ''
    print(output1)
    tmp = []
    tmp1 = []
    msg = 'Machine is safe'
    for i in output1[0]:
        tmp.append(str(int(i)))
        tmp1.append(int(i))
    output = "_".join(tmp)
    print(tmp1)

    if tmp[0] == '1':
        msg = 'Machine may fail'
        for j in range(1, 5):
            if tmp1[j] == 1 and j == 1:
                fail += "Tool wear Failure"
            elif tmp1[j] == 1 and j == 2:
                fail += " Heat Dissipation Failure"
            elif tmp1[j] == 1 and j == 3:
                fail += " Power Failure"
            elif tmp1[j] == 1 and j == 4:
                fail += " Overstrain Failure"
        # return render_template('sample.html',pred='{}.\nTarget failure is {}.\nFailure Type: {}'.format(msg, output, fail))
    else:
        msg = "Machine is safe"
    print(output)
    RUL = 0
    tstatus = ''
    dangeralerts = ''
    warnalerts = ''
    infoalerts = ''
    if float(tool) < 190 and tmp1[1] == 0:
        RUL = abs(float(tool) - 210)
        tstatus = 'Safe Working Condition'
        infoalerts = "No Tool Failure Expected!"
    elif (float(tool) >= 190 and float(tool) <= 200) or tmp1[1] == 1:
        RUL = abs(float(tool) - 210)
        tstatus = 'Near to Wearout'
        warnalerts = "Cutting Tool needs to be Replaced....Quality of Workpiece will decrease!"
    elif (float(tool) > 200 and float(tool) <= 220) or tmp1[1] == 1:
        RUL = abs(float(tool) - 210)
        tstatus = 'Moderate Wearout'
        warnalerts = "Workpiece Quality has Reduced! Cutting Tool needs to be Replaced Immediately"
    elif float(tool) > 220 or tmp1[1] == 1:
        RUL = 0
        tstatus = 'Severe Wearout'
        dangeralerts = "Cutting Tool Needs Urgent Replacement! Workpiece Quality Damaging!"

    if overstrain > 11000 and int(type) == 0 or tmp1[4] == 1:
        warnalerts += "\nMachine Type is 'L'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 12000 and int(type) == 1 or tmp1[4] == 1:
        warnalerts += "\nMachine Type is 'M'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 13000 and int(type) == 2 or tmp1[4] == 1:
        dangeralerts += "\nMachine Type is 'H'. Torque is Exceeding. Workpiece will Damage. Overstrain on Machine"

    if tmp1[3] == 1:
        dangeralerts += "\nPower is Fluctuating. Machine will Damage."
    if tmp1[2] == 1:
        dangeralerts += "\nThere is a Process Failure due to Heat dissipation failure!"
    return render_template('sample.html',
                           pred='{}.\nTarget failure is {}.\nFailure Type: {}'.format(msg, output, fail),
                           rffail=tmp1[0], rftwf=tmp1[1], rfhdf=tmp1[2],
                           rfpwf=tmp1[3], rfosf=tmp1[4], rfrnf=0, dangeralert=dangeralerts, warnalert=warnalerts,
                           infoalert=infoalerts)
    # return 0


@app.route('/predict_decision_manual', methods=['POST', 'GET'])
def predict_decision_manual():
    global dtModel
    svmIrisFile = open('DecisionTreeModel.pkl', 'rb')
    dtModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = request.form['air']
    process = request.form['process']
    rpm = request.form['rpm']
    torque = request.form['torque']
    tool = request.form['tool']
    type = request.form['Type']
    diff = float(air) - float(process)
    power = float(rpm) * float(torque)
    overstrain = float(torque) * float(tool)
    final = np.array([type, air, process, rpm, torque, tool, diff, power, overstrain]).reshape(1, 9)
    class_predicted = dtModel.predict(final)
    output1 = class_predicted.tolist()
    fail = ''
    print(output1)
    tmp = []
    tmp1 = []
    msg = 'Machine is safe'
    for i in output1[0]:
        tmp.append(str(int(i)))
        tmp1.append(int(i))
    output = "_".join(tmp)
    print(tmp1)

    if tmp[0] == '1':
        msg = 'Machine may fail'
        for j in range(1, 5):
            if tmp1[j] == 1 and j == 1:
                fail += "Tool wear Failure"
            elif tmp1[j] == 1 and j == 2:
                fail += " Heat Dissipation Failure"
            elif tmp1[j] == 1 and j == 3:
                fail += " Power Failure"
            elif tmp1[j] == 1 and j == 4:
                fail += " Overstrain Failure"
        # return render_template('sample.html',pred='{}.\nTarget failure is {}.\nFailure Type: {}'.format(msg, output, fail))
    else:
        msg = "Machine is safe"
    print(output)
    RUL = 0
    tstatus = ''
    dangeralerts = ''
    warnalerts = ''
    infoalerts = ''
    if float(tool) < 190 and tmp1[1] == 0:
        RUL = abs(float(tool) - 210)
        tstatus = 'Safe Working Condition'
        infoalerts = "No Tool Failure Expected!"
    elif (float(tool) >= 190 and float(tool) <= 200) or tmp1[1] == 1:
        RUL = abs(float(tool) - 210)
        tstatus = 'Near to Wearout'
        warnalerts = "Cutting Tool needs to be Replaced....Quality of Workpiece will decrease!"
    elif (float(tool) > 200 and float(tool) <= 220) or tmp1[1] == 1:
        RUL = abs(float(tool) - 210)
        tstatus = 'Moderate Wearout'
        warnalerts = "Workpiece Quality has Reduced! Cutting Tool needs to be Replaced Immediately"
    elif float(tool) > 220 or tmp1[1] == 1:
        RUL = 0
        tstatus = 'Severe Wearout'
        dangeralerts = "Cutting Tool Needs Urgent Replacement! Workpiece Quality Damaging!"

    if overstrain > 11000 and int(type) == 0 or tmp1[4] == 1:
        warnalerts += "\nMachine Type is 'L'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 12000 and int(type) == 1 or tmp1[4] == 1:
        warnalerts += "\nMachine Type is 'M'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 13000 and int(type) == 2 or tmp1[4] == 1:
        dangeralerts += "\nMachine Type is 'H'. Torque is Exceeding. Workpiece will Damage. Overstrain on Machine"

    if tmp1[3] == 1:
        dangeralerts += "\nPower is Fluctuating. Machine will Damage."
    if tmp1[2] == 1:
        dangeralerts += "\nThere is a Process Failure due to Heat dissipation failure!"
    return render_template('sample.html',
                           pred='{}.\nTarget failure is {}.\nFailure Type: {}'.format(msg, output, fail),
                           dtfail=tmp1[0], dttwf=tmp1[1], dthdf=tmp1[2],
                           dtpwf=tmp1[3], dtosf=tmp1[4], dtrnf=0, dangeralert=dangeralerts, warnalert=warnalerts,
                           infoalert=infoalerts)


@app.route('/predict_lr_manual', methods=['POST', 'GET'])
def predict_lr_manual():
    global lrModel
    svmIrisFile = open('LogisticsModel.pkl', 'rb')
    lrModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = float(request.form['air'])
    process = float(request.form['process'])
    rpm = float(request.form['rpm'])
    torque = float(request.form['torque'])
    tool = float(request.form['tool'])
    type = float(request.form['Type'])
    diff = float(air) - float(process)
    power = float(rpm) * float(torque)
    overstrain = float(torque) * float(tool)
    final = np.array([type, air, process, rpm, torque, tool, diff, power, overstrain]).reshape(1, 9)
    class_prediced = int(lrModel.predict(final)[0])
    output = str(class_prediced)
    tmp1 = [0, 0, 0, 0, 0, 0]
    fail = ""
    print(class_prediced)
    if class_prediced == 1:
        tmp1[0] = 1
        tmp1[3] = 1
        fail += "Power Failure"
    elif class_prediced == 2:
        tmp1[0] = 1
        tmp1[1] = 1
        fail += "Tool Wear Failure"
    elif class_prediced == 3:
        tmp1[0] = 1
        tmp1[4] = 1
        fail += "Overstrain Failure"
    elif class_prediced == 4:
        tmp1[0] = 1
        tmp1[5] = 1
        fail += "Random Failure"
    elif class_prediced == 5:
        tmp1[0] = 1
        tmp1[2] = 1
        fail += "Heat Dissipation Failure"

    RUL = 0
    tstatus = ''
    dangeralerts = ''
    warnalerts = ''
    infoalerts = ''
    if float(tool) < 190 and tmp1[1] == 0:
        RUL = abs(float(tool) - 210)
        tstatus = 'Safe Working Condition'
        infoalerts = "No Tool Failure Expected!"
    elif (float(tool) >= 190 and float(tool) <= 200) or tmp1[1] == 1:
        RUL = abs(tool - 210)
        tstatus = 'Near to Wearout'
        warnalerts = "Cutting Tool needs to be Replaced....Quality of Workpiece will decrease!"
    elif (float(tool) > 200 and float(tool) <= 220) or tmp1[1] == 1:
        RUL = abs(tool - 210)
        tstatus = 'Moderate Wearout'
        warnalerts = "Workpiece Quality has Reduced! Cutting Tool needs to be Replaced Immediately"
    elif float(tool) > 220 or tmp1[1] == 1:
        RUL = 0
        tstatus = 'Severe Wearout'
        dangeralerts = "Cutting Tool Needs Urgent Replacement! Workpiece Quality Damaging!"

    if overstrain > 11000 and int(type) == 0 or tmp1[4] == 1:
        warnalerts += "\nMachine Type is 'L'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 12000 and int(type) == 1 or tmp1[4] == 1:
        warnalerts += "\nMachine Type is 'M'. Torque is Exceeding. Workpiece will Damage."
    elif overstrain > 13000 and int(type) == 2 or tmp1[4] == 1:
        dangeralerts += "\nMachine Type is 'H'. Torque is Exceeding. Workpiece will Damage. Overstrain on Machine"

    if tmp1[3] == 1:
        dangeralerts += "\nPower is Fluctuating. Machine will Damage."
    if tmp1[2] == 1:
        dangeralerts += "\nThere is a Process Failure due to Heat dissipation failure!"
    if class_prediced == 1 or class_prediced == 2 or class_prediced == 3 or class_prediced == 4 or class_prediced == 5:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, fail),
                               knnfail=tmp1[0], knntwf=tmp1[1], knnhdf=tmp1[2],
                               knnpwf=tmp1[3], knnosf=tmp1[4], knnrnf=tmp1[5], dangeralert=dangeralerts,
                               warnalert=warnalerts,
                               infoalert=infoalerts)
    else:
        return render_template('sample.html',
                               pred='Machine machine is safe.\nTarget failure is {}.\nFailure Type: No Failure'.format(
                                   output), knnfail=tmp1[0], knntwf=tmp1[1], knnhdf=tmp1[2],
                               knnpwf=tmp1[3], knnosf=tmp1[4], knnrnf=tmp1[5])
    return output


@app.route('/load_model_manual', methods=['GET', 'POST'])
def load_model_manual():
    global svmIrisModel
    if 'rm' in request.form:
        # random()
        return render_template("sample.html")
    elif 'svm' in request.form:
        svm()
        return render_template("sample.html")
    elif 'dt' in request.form:
        dt()
        return render_template("sample.html")


def svm():
    global svmIrisModel
    svmIrisFile = open('SVMModel.pkl', 'rb')
    svmIrisModel = pickle.load(svmIrisFile)
    svmIrisFile.close()


def dt():
    global dtModel
    svmIrisFile = open('DT.pkl', 'rb')
    dtModel = pickle.load(svmIrisFile)
    svmIrisFile.close()


if __name__ == '__main__':
    # read_data()
    load_model()
    # thingspeak_post()
    # read_all_data()
    app.run(debug=True)
    # thingspeak_post()
