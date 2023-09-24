import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
import seaborn as sns;
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import pickle
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('predictive_maintenance.csv')
df = pd.read_csv('predictive_maintenance.csv')
df['Type'].replace(['L', 'M', 'H'], [0, 1, 2], inplace=True)

for column in df.columns:
    try:
        df[column] = df[column].astype(float)
    except:
        pass

# show the numeric characters
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all').T

# copy the data
df_z_normalize = df.copy()
df_z_normalize = df_z_normalize.iloc[:, 2:8]
# Extracting features
df_z_normalize['air_process_diff'] = df_z_normalize['Process temperature [K]'] - df_z_normalize['Air temperature [K]']
df_z_normalize['power'] = df_z_normalize['Torque [Nm]'] * df_z_normalize['Rotational speed [rpm]']
df_z_normalize['overstrain'] = df_z_normalize['Tool wear [min]'] * df_z_normalize['Torque [Nm]']
encoder = ce.OrdinalEncoder(cols=['Failure Type'])
df = encoder.fit_transform(df)

scaler = LabelEncoder()
df_z_normalize['Failure Type'] = scaler.fit_transform(df['Failure Type'])
print(df_z_normalize['Failure Type'].value_counts())
'''
# apply normalization techniques
for column in df_z_normalize.columns:
	df_z_normalize[column] = (df_z_normalize[column] - df_z_normalize[column].mean()) / df_z_normalize[column].std()

# view normalized data
#display(df_z_normalize)
'''
x = df_z_normalize.copy()
Y = df.iloc[:, -6:-1]

X_cols = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
          'Tool wear [min]', 'air_process_diff', 'power', 'overstrain']

y_cols = ['Failure Type']

X = df_z_normalize[X_cols]
y = df[y_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=42)


def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
from sklearn.preprocessing import StandardScaler
#svc = make_pipeline(StandardScaler(), SVC(C=1.0,gamma='auto', class_weight='balanced'))
svc = SVC()
svc.fit(X_train, y_train)

# predict and evaluate
y_hat_train1 = svc.predict(X_train)
evaluate(y_train, y_hat_train1, 'train')

y_hat_test1 = svc.predict(X_test)
evaluate(y_test, y_hat_test1)

print('F1 Score is :', "{:.3f}".format(f1_score(y_hat_test1, y_test, average='micro')))
print('Model Accuracy Score on totally unseen data(Xtest) is:', accuracy_score(y_test, y_hat_test1) * 100, '%')

train_acc2 = round(svc.score(X_train, y_train) * 100, 1)
val_acc2 = round(accuracy_score(y_hat_test1, y_test) * 100, 2)
print('F1 Score is :', "{:.3f}".format(f1_score(y_hat_test1, y_test, average='micro')))
print('Model Accuracy Score on totally unseen data(Xtest) is:', accuracy_score(y_test, y_hat_test1) * 100, '%')
print('Model Precision Score on totally unseen data(Xtest) is:', precision_score(y_test, y_hat_test1,average='micro') * 100, '%')
print('Model Recall Score on totally unseen data(Xtest) is:', recall_score(y_test, y_hat_test1,average='micro') * 100, '%')
print('Training Accuracy: ',train_acc2)
print('Validation Accuracy:',val_acc2)

# inputt =[0.744376,-0.952342,-0.947313,0.068182,0.282186,-1.695899,0.498824,0.629412,-1.526468]
# inputt = [1.0,298.1,308.6,1551.0,42.8,0.0,10.5,66382.8,0.0]
# inputt = [0.744376,-0.902348,-1.014710,6.354294,-2.937818,-0.344848,0.299043,-3.071937,-1.200914]

# final=[np.array(inputt)]

# b = rfc.predict(final)
# print(b)

# =[float(x) for x in "0.151351 0.151351 0.222934 0.535714 0.000000 0".split(' ')]
# final=[np.array(inputt)]
# b = svc.predict(final)
rfcFile = open('SupportVectorModel.pkl', 'wb')
pickle.dump(svc, rfcFile)
rfcFile.close()
# model=pickle.load(open('model.pkl','rb'))
