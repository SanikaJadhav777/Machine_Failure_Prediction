from sklearn.ensemble import RandomForestClassifier
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import warnings
import category_encoders as ce

warnings.filterwarnings("ignore")

df = pd.read_csv("ai4i2020.csv")
df['Type'].replace(['L', 'M', 'H'], [0, 1, 2], inplace=True)
for column in df.columns:
    try:
        df[column] = df[column].astype(float)
    except:
        pass

df['td'] = df['Process temperature [K]'] - df['Air temperature [K]']
df["Process_temp_normalized"] = (df["Process temperature [K]"] - min(df['Air temperature [K]'])) / (
            max(df['Process temperature [K]']) - min(df['Air temperature [K]']))
df["Air_temp_normalized"] = (df["Air temperature [K]"] - min(df['Air temperature [K]'])) / (
            max(df['Process temperature [K]']) - min(df['Air temperature [K]']))
df['Temp_diff'] = df['Process_temp_normalized'] = df['Air_temp_normalized']
'''df_pass = df[df['Target'] == 0]
df_pass = df_pass[df_pass['Failure Type'] == 'No Failure']
df_fail = df[df['Target'] == 1]
df_fail = df_fail[df_fail['Failure Type'] != 'No Failure']'''

'''df1 = df[df['Failure Type'] == 'Heat Dissipation Failure']
df2 = df[df['Failure Type'] == 'Power Failure']
df3 = df[df['Failure Type'] == 'Overstrain Failure']
df4 = df[df['Failure Type'] == 'Tool Wear Failure']
df5 = df[df['Failure Type'] == 'Random Failures ']'''

'''df_fail['rpm_norm'] = (df_fail['Rotational speed [rpm]'] - min(df_fail['Rotational speed [rpm]'])) / (
            max(df_fail['Rotational speed [rpm]']) - min(df_fail['Rotational speed [rpm]']))
df_fail['Torque_norm'] = (df_fail['Torque [Nm]'] - min(df_fail['Torque [Nm]'])) / (
            max(df_fail['Torque [Nm]']) - min(df_fail['Torque [Nm]']))
df_fail['Tool_norm'] = (df_fail['Tool wear [min]'] - min(df_fail['Tool wear [min]'])) / (
            max(df_fail['Tool wear [min]']) - min(df_fail['Tool wear [min]']))
df_fail = df_fail.drop(
    columns=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
             'Tool wear [min]', 'Product ID', 'UDI', 'Temp_diff', 'td', 'Target'])'''
#df_fail = pd.get_dummies(data=df_fail, columns=['Type'], drop_first=True)
encoder = ce.OrdinalEncoder(cols=['Type', 'Failure Type'])
df = encoder.fit_transform(df)
scaler = LabelEncoder()
df_fail['Failure Type'] = scaler.fit_transform(df_fail['Failure Type'])
df_fail_targ = df_fail_train = df_fail.iloc[:, 0]
df_fail_train = df_fail.iloc[:, 1:]

df['rpm_norm'] = (df['Rotational speed [rpm]'] - min(df['Rotational speed [rpm]'])) / (
            max(df['Rotational speed [rpm]']) - min(df['Rotational speed [rpm]']))
df['Torque_norm'] = (df['Torque [Nm]'] - min(df['Torque [Nm]'])) / (max(df['Torque [Nm]']) - min(df['Torque [Nm]']))
df['Tool_norm'] = (df['Tool wear [min]'] - min(df['Tool wear [min]'])) / (
            max(df['Tool wear [min]']) - min(df['Tool wear [min]']))

df2 = df.drop(columns=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                       'Tool wear [min]', 'Product ID', 'UDI', 'Temp_diff', 'td', 'Failure Type'])

train = df2.drop(columns=['Target'])
tar = df2['Target']

from sklearn.tree import DecisionTreeClassifier

X_train2, X_test2, y_train2, y_test2 = train_test_split(train, tar, test_size=0.2, random_state=21)
decision = DecisionTreeClassifier()
decision.fit(X_train2, y_train2)
y_pred_dec = decision.predict(X_test2)

decision_train = round(decision.score(X_train2, y_train2) * 100, 2)
decision_accuracy = round(accuracy_score(y_pred_dec, y_test2) * 100, 2)

svmFile = open('DT.pkl', 'wb')
pickle.dump(decision, svmFile)
svmFile.close()
# model=pickle.load(open('model.pkl','rb'))
