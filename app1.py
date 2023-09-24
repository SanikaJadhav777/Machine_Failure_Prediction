
from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io


l = ['No Failure', 'Power Failure', 'Tool Wear Failure', 'Overstrain Failure', 'Random Failures',
     'Heat Dissipation Failure']
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'machine_prediction_db'

mysql = MySQL(app)

fig, ax = plt.subplots(figsize=(6,6))
ax = sns.set_style(style="darkgrid")
df=pd.read_csv("predictive_maintenance.csv")
df = df.drop(["UDI", "Product ID"],axis=1)
df["Air temperature [K]"] = df["Air temperature [K]"] - 272.15
df["Process temperature [K]"] = df["Process temperature [K]"] - 272.15

df.rename(columns={"Air temperature [K]" : "Air temperature [°C]","Process temperature [K]" : "Process temperature [°C]"},inplace=True)
df["Temperature difference [°C]"] = df["Process temperature [°C]"] - df["Air temperature [°C]"]

@app.route('/')
@app.route('/startup')
def startup():
    return render_template("startup.html", msg="")

@app.route('/home')
def home():
    return render_template("sample.html", msg="")

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


#@app.route('/logout')
#def logout():
#    return redirect(url_for('login'))

@app.route('/close',  methods=['GET', 'POST'])
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


'''@app.route('/predict_svm', methods=['POST', 'GET'])
def predict_svm():
    global svmIrisModel
    svmIrisFile = open('SVMModel.pkl', 'rb')
    svmIrisModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = request.form['air']
    process = request.form['process']
    rpm = request.form['rpm']
    torque = request.form['torque']
    tool = request.form['tool']
    type = request.form['Type']
    final = np.array([air, process, rpm, torque, tool, type]).reshape(1, 6)
    class_prediced = int(svmIrisModel.predict(final)[0])
    output = str(class_prediced)
    if class_prediced == 1:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 2:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 3:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 4:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 5:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[class_prediced]))
    elif class_prediced == 0:
        return render_template('sample.html',
                               pred='Machine machine is safe.\nTarget failure is {}.\nFailure Type: {}'.format(output,l[class_prediced]))
    return output'''

@app.route('/predict_svm', methods=['POST', 'GET'])
def predict_svm():
    global svmIrisModel
    svmIrisFile = open('SVMModel.pkl', 'rb')
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
    final = np.array([air, process, rpm, torque, tool, type]).reshape(1, 6)
    class_prediced = int(svmIrisModel.predict(final)[0])
    output = str(class_prediced)
    tmp1 = [0, 0, 0, 0, 0]
    fail = "No Failure"
    if class_prediced == 1:
        tmp1[0] = 1
        if float(tool) >= 200 or float(tool) <= 240:
            fail += "Tool Wear Failure"
            tmp1[1] = 1
        elif diff >= 8.66 and float(rpm) >= 1380:
            tmp1[2] = 1
            fail += "Heat Dissipation Failure"
        elif power >= 3500 or power <= 9000:
            tmp1[3] = 1
            fail += "Power Failure"
        elif overstrain >= 11000 or overstrain <= 13000:
            tmp1[4] = 1
            fail += "Overstrain Failure"

        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, fail),
                               svmfail=tmp1[0], svmtwf=tmp1[1], svmhdf=tmp1[2],
                               svmpwf=tmp1[3], svmosf=tmp1[4], svmrnf=0)
    else:
        return render_template('sample.html',
                               pred='Machine machine is safe.\nTarget failure is {}.\nFailure Type: {}'.format(output,
                                                                                                               fail),
                               svmfail=tmp1[0], svmtwf=tmp1[1], svmhdf=tmp1[2],
                               svmpwf=tmp1[3], svmosf=tmp1[4], svmrnf=0)
    return output

@app.route('/data_plots')
def data_plots():
    return render_template('data_plots.html', msg='')

@app.route('/sample')
def sample():
    return render_template('sample.html', msg='')

'''@app.route('/predict_random', methods=['POST', 'GET'])
def predict_random():
    svmIrisFile = open('RandomForest.pkl', 'rb')
    RMModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = request.form['air']
    process = request.form['process']
    rpm = request.form['rpm']
    torque = request.form['torque']
    tool = request.form['tool']
    type = request.form['Type']
    final = np.array([air, process, rpm, torque, tool, type]).reshape(1, 6)
    class_prediced = int(RMModel.predict(final)[0])
    output = str(class_prediced)
    if class_prediced == 1:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 2:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 3:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 4:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 5:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[class_prediced]))
    elif class_prediced == 0:
        return render_template('sample.html',
                               pred='Machine machine is safe.\nTarget failure is {}.\nFailure Type: {}'.format(output,l[class_prediced]))
    return output'''
@app.route('/predict_random', methods=['POST', 'GET'])
def predict_random():
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

    return render_template('sample.html',
                           pred='{}.\nTarget failure is {}.\nFailure Type: {}'.format(msg, output, fail),
                           rffail=tmp1[0], rftwf=tmp1[1], rfhdf=tmp1[2],
                           rfpwf=tmp1[3], rfosf=tmp1[4], rfrnf=0)
    # return 0

'''
@app.route('/predict_decision', methods=['POST', 'GET'])
def predict_decision():
    global dtModel
    svmIrisFile = open('DT.pkl', 'rb')
    dtModel = pickle.load(svmIrisFile)
    svmIrisFile.close()
    air = request.form['air']
    process = request.form['process']
    rpm = request.form['rpm']
    torque = request.form['torque']
    tool = request.form['tool']
    type = request.form['Type']
    final = np.array([air, process, rpm, torque, tool, type]).reshape(1, 6)
    class_prediced = int(dtModel.predict(final)[0])
    output = str(class_prediced)
    if class_prediced == 1:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 2:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 3:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 4:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[
                                   class_prediced]))
    elif class_prediced == 5:
        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, l[class_prediced]))
    elif class_prediced == 0:
        return render_template('sample.html',
                               pred='Machine machine is safe.\nTarget failure is {}.\nFailure Type: {}'.format(output,l[class_prediced]))
    return output'''

@app.route('/predict_decision', methods=['POST', 'GET'])
def predict_decision():
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

    return render_template('sample.html',
                           pred='{}.\nTarget failure is {}.\nFailure Type: {}'.format(msg, output, fail),
                           dtfail=tmp1[0], dttwf=tmp1[1], dthdf=tmp1[2],
                           dtpwf=tmp1[3], dtosf=tmp1[4], dtrnf=0)
    # return 0

@app.route('/predict_lr', methods=['POST', 'GET'])
def predict_lr():
    global lrModel
    svmIrisFile = open('LRModel.pkl', 'rb')
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
    final = np.array([air, process, rpm, torque, tool, type]).reshape(1, 6)
    class_prediced = int(lrModel.predict(final)[0])
    output = str(class_prediced)
    tmp1=[0,0,0,0,0]
    fail = "No Failure"
    if class_prediced == 1:
        tmp1[0]=1
        if float(tool) >= 200 or float(tool) <= 240:
            tmp1[1]=1
            fail += "Tool Wear Failure"
        elif diff >= 8.66 and float(rpm) >= 1380:
            tmp1[2]=1
            fail += "Heat Dissipation Failure"
        elif power >= 3500 or power <= 9000:
            tmp1[3]=1
            fail += "Power Failure"
        elif overstrain >= 11000 or overstrain <= 13000:
            tmp1[4]=1
            fail += "Overstrain Failure"

        return render_template('sample.html',
                               pred='Machine may fail.\nTarget failure is {}.\nFailure Type: {}'.format(output, fail),knnfail = tmp1[0], knntwf = tmp1[1],knnhdf = tmp1[2],
                           knnpwf = tmp1[3],knnosf = tmp1[4],knnrnf = 0)
    else:
        return render_template('sample.html',
                               pred='Machine machine is safe.\nTarget failure is {}.\nFailure Type: {}'.format(output,fail),knnfail = tmp1[0], knntwf = tmp1[1],knnhdf = tmp1[2],
                           knnpwf = tmp1[3],knnosf = tmp1[4],knnrnf = 0)
    return output

@app.route('/load_model', methods=['GET', 'POST'])
def load_model():
    global svmIrisModel
    if 'rm' in request.form:
        #random()
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

@app.route('/plot_demo')
def plot_demo():
    return render_template('plot_charts.html', msg="")

if __name__ == '__main__':
    app.run(debug=True)
