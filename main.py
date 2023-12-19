import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

# Defining Feature Names 
index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
col_names = index_names + setting_names + sensor_names

# Importing Data 
df_train = pd.read_csv('COMPSCI 715/Final_Project/CMaps/TRAIN/train_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
df_test = pd.read_csv('COMPSCI 715/Final_Project/CMaps/TEST/test_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
y_valid = pd.read_csv('COMPSCI 715/Final_Project/CMaps/RUL/RUL_FD001.txt',sep='\s+',header=None,index_col=False,names=['RUL'])

print(y_valid)

print(df_train.shape)
print(df_test.shape)

train = df_train.copy()
test = df_test.copy()

# Data Inspection 
print(train)

print('Shape of the train dataset :' , train.shape)
print('Shape of the test dataset :' , test.shape)
print('The percentage of the validation dataset between test and train :' , len(test)/(len(test) + len(train)))

# Checking for NaN values in train dataset 
print('Total of not available values in the train dataset :' , train.isna().sum())

print(train.loc[:, ['unit_number', 'time_cycles']].describe())

print(train.loc[:,'s_1':].describe().transpose())

# Visual Representation of Turbofan Engines LifeSpan
max_time_cycles=train[index_names].groupby('unit_number').max()
plt.figure(figsize=(20,50))
ax=max_time_cycles['time_cycles'].plot(kind='barh',width=0.8, stacked=True,align='center')
plt.title('Turbofan Engines LifeSpan',fontweight='bold',size=30)
plt.xlabel('Time cycle',fontweight='bold',size=20)
plt.xticks(size=15)
plt.ylabel('unit',fontweight='bold',size=20)
plt.yticks(size=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visual Representation of distribution of maximum time cycles
sns.displot(max_time_cycles['time_cycles'],kde=True,bins=20,height=6,aspect=2)
plt.xlabel('max time cycle')

# Add RUL (Remaining Useful Life) to the dataset 
def add_RUL_column(df):
    train_grouped_by_unit = df.groupby(by='unit_number') 
    max_time_cycles = train_grouped_by_unit['time_cycles'].max() 
    merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number', right_index=True)
    merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
    merged = merged.drop("max_time_cycle", axis=1) 
    return merged

train = add_RUL_column(train)

print(train[['unit_number','RUL']])

# RUL (Remaining Useful Life) Analysis 
max_rul_u = train.groupby('unit_number').max().reset_index()
max_rul_u.head()

# Finding which features are good at detecting overheating for predictive maintenance
sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn')
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()

Sensor_dictionary={}
dict_list=[ "(Fan inlet temperature) (◦R)",
"(LPC outlet temperature) (◦R)",
"(HPC outlet temperature) (◦R)",
"(LPT outlet temperature) (◦R)",
"(Fan inlet Pressure) (psia)",
"(bypass-duct pressure) (psia)",
"(HPC outlet pressure) (psia)",
"(Physical fan speed) (rpm)",
"(Physical core speed) (rpm)",
"(Engine pressure ratio(P50/P2)",
"(HPC outlet Static pressure) (psia)",
"(Ratio of fuel flow to Ps30) (pps/psia)",
"(Corrected fan speed) (rpm)",
"(Corrected core speed) (rpm)",
"(Bypass Ratio) ",
"(Burner fuel-air ratio)",
"(Bleed Enthalpy)",
"(Required fan speed)",
"(Required fan conversion speed)",
"(High-pressure turbines Cool air flow)",
"(Low-pressure turbines Cool air flow)" ]
i=1
for x in dict_list :
    Sensor_dictionary['s_'+str(i)]=x
    i+=1
Sensor_dictionary

# Plotting sensor features along with RUL(Remaining Useful Life)
def plot_signal(df, Sensor_dic, signal_name):
    plt.figure(figsize=(13,5))
    for i in df['unit_number'].unique():
        if (i % 10 == 0):   #For a better visualisation, we plot the sensors signals of 20 units only
            plt.plot('RUL', signal_name, data=df[df['unit_number']==i].rolling(10).mean())

    plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 300, 25))
    plt.ylabel(Sensor_dic[signal_name])
    plt.xlabel('Remaining Useful Life')
    plt.show()

for i in range(1,22):
    try:
        plot_signal(train, Sensor_dictionary,'s_'+str(i))
    except:
        pass
    
for x in sensor_names:
    plt.figure(figsize=(13,7))
    plt.boxplot(train[x])
    plt.title(x)
    plt.show()

drop_labels = index_names + setting_names
X_train=train.drop(columns=drop_labels).copy()
X_train, X_test, y_train, y_test=train_test_split(X_train,X_train['RUL'], test_size=0.3, random_state=42)

print(train.loc[:,'s_1':].describe().transpose())

scaler = MinMaxScaler()
#Droping the target variable
X_train.drop(columns=['RUL'], inplace=True)
X_test.drop(columns=['RUL'], inplace=True)
#Scaling X_train and X_test
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.fit_transform(X_test)
#Conserve only the last occurence of each unit to match the length of y_valid
y_valid_new = df_test.groupby('unit_number').last().reset_index().drop(columns=drop_labels)
#scaling X_rul
X_valid_s=scaler.fit_transform(y_valid_new)

print(y_valid_new)

print(X_valid_s.shape)
print(y_valid.shape)

sensor_names=['s_{}'.format(i) for i in range(1,22) if i not in [1,5,6,10,16,18,19]]
pd.DataFrame(X_train_s,columns=['s_{}'.format(i) for i in range(1,22)])[sensor_names].hist(bins=100, figsize=(18,16))

# Models Implmenation 

#R2 score & RMSE & MAER
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
    
# Linear Regression
print("Linear Regression Performance:")
lr = LinearRegression()
lr.fit(X_train_s, y_train)  

y_lr_train = lr.predict(X_train_s) #Prediction on train data
evaluate(y_train,y_lr_train, label='train')

y_lr_test = lr.predict(X_test_s)  #Prediction on test data
evaluate(y_test, y_lr_test, label='test')

y_lr_valid= lr.predict(X_valid_s) #Prediction on validation data
evaluate(y_valid, y_lr_valid, label='valid')

# Random Forest 
print("Random Forest Regression Performance:")
rf = RandomForestRegressor(max_features="sqrt", random_state=42)
rf.fit(X_train_s, y_train)

y_rf_train = rf.predict(X_train_s)
evaluate(y_train,y_rf_train, label='train')

y_rf_test = rf.predict(X_test_s)
evaluate(y_test, y_rf_test, label='test')

y_rf_valid = rf.predict(X_valid_s)
evaluate(y_valid, y_rf_valid, label='valid')

# Regression Neural Network
def create_regression_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Train the regression neural network
X_train_nn = X_train_s  # Use the scaled training data
y_train_nn = y_train.values  # Convert pandas Series to numpy array

X_valid_nn = X_valid_s  # Use the scaled validation data

# Create and train the model
regression_model = create_regression_model(X_train_nn.shape[1])
regression_model.fit(X_train_nn, y_train_nn, epochs=50, batch_size=32, validation_data=(X_valid_nn, y_valid.values))

# Predictions on train, test, and validation sets
y_nn_train = regression_model.predict(X_train_nn).flatten()
y_nn_test = regression_model.predict(X_test_s).flatten()
y_nn_valid = regression_model.predict(X_valid_s).flatten()

# Evaluate the performance of the regression neural network
print("Neural Network Regression Performance:")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train_nn, y_nn_train)))
print("Valid RMSE:", np.sqrt(mean_squared_error(y_valid, y_nn_valid)))
print("Train R2:", r2_score(y_train_nn, y_nn_train))
print("Valid R2:", r2_score(y_valid, y_nn_valid))

# Classification Task

# Assuming you have a classification target 'classification_target' in your dataset
some_threshold = 100

df_train = add_RUL_column(df_train)
df_train_cls = df_train.copy()
df_train_cls['classification_target'] = (df_train_cls['RUL'] <= some_threshold).astype(int)

df_test = add_RUL_column(df_train)
df_test_cls = df_test.copy()
df_test_cls['classification_target'] = (df_test_cls['RUL'] <= some_threshold).astype(int)

X_train_cls = df_train_cls.drop(['RUL', 'classification_target'], axis=1)
y_train_cls = df_train_cls['classification_target']

X_test_cls = df_test_cls.drop(['RUL', 'classification_target'], axis=1)
y_test_cls = df_test_cls['classification_target']

scaler_cls = MinMaxScaler()
X_train_cls_s = scaler_cls.fit_transform(X_train_cls)
X_test_cls_s = scaler_cls.transform(X_test_cls)

# Logistic Regression for Classification
lr_cls = LogisticRegression(random_state=42)
lr_cls.fit(X_train_cls_s, y_train_cls)

# Predictions on train and test sets
y_lr_cls_train = lr_cls.predict(X_train_cls_s)
y_lr_cls_test = lr_cls.predict(X_test_cls_s)

# Evaluate the performance of logistic regression for classification
print("Logistic Regression Classification Performance:")
print("Train Accuracy:", accuracy_score(y_train_cls, y_lr_cls_train))
print("Test Accuracy:", accuracy_score(y_test_cls, y_lr_cls_test))
print("Train Precision:", precision_score(y_train_cls, y_lr_cls_train))
print("Test Precision:", precision_score(y_test_cls, y_lr_cls_test))
print("Train Recall:", recall_score(y_train_cls, y_lr_cls_train))
print("Test Recall:", recall_score(y_test_cls, y_lr_cls_test))
print("Train F1 Score:", f1_score(y_train_cls, y_lr_cls_train))
print("Test F1 Score:", f1_score(y_test_cls, y_lr_cls_test))
print("Train ROC AUC Score:", roc_auc_score(y_train_cls, lr_cls.predict_proba(X_train_cls_s)[:, 1]))
print("Test ROC AUC Score:", roc_auc_score(y_test_cls, lr_cls.predict_proba(X_test_cls_s)[:, 1]))

# Random Forest for Classification
rf_cls = RandomForestClassifier(random_state=42)
rf_cls.fit(X_train_cls_s, y_train_cls)

# Predictions on train and test sets
y_rf_cls_train = rf_cls.predict(X_train_cls_s)
y_rf_cls_test = rf_cls.predict(X_test_cls_s)

# Evaluate the performance of random forest for classification
print("Random Forest Classification Performance:")
print("Train Accuracy:", accuracy_score(y_train_cls, y_rf_cls_train))
print("Test Accuracy:", accuracy_score(y_test_cls, y_rf_cls_test))
print("Train Precision:", precision_score(y_train_cls, y_rf_cls_train))
print("Test Precision:", precision_score(y_test_cls, y_rf_cls_test))
print("Train Recall:", recall_score(y_train_cls, y_rf_cls_train))
print("Test Recall:", recall_score(y_test_cls, y_rf_cls_test))
print("Train F1 Score:", f1_score(y_train_cls, y_rf_cls_train))
print("Test F1 Score:", f1_score(y_test_cls, y_rf_cls_test))
print("Train ROC AUC Score:", roc_auc_score(y_train_cls, rf_cls.predict_proba(X_train_cls_s)[:, 1]))
print("Test ROC AUC Score:", roc_auc_score(y_test_cls, rf_cls.predict_proba(X_test_cls_s)[:, 1]))

# Classification Neural Network
def create_classification_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the classification neural network
X_train_cls_nn = X_train_cls_s  # Use the scaled training data
y_train_cls_nn = y_train_cls.values  # Convert pandas Series to numpy array

# Create and train the model
classification_model = create_classification_model(X_train_cls_nn.shape[1])
classification_model.fit(X_train_cls_nn, y_train_cls_nn, epochs=50, batch_size=32)

# Predictions on train and test sets
y_nn_cls_train = (classification_model.predict(X_train_cls_nn) > 0.5).astype(int).flatten()
y_nn_cls_test = (classification_model.predict(X_test_cls_s) > 0.5).astype(int).flatten()

# Evaluate the performance of the classification neural network
print("Neural Network Classification Performance:")
print("Train Accuracy:", accuracy_score(y_train_cls_nn, y_nn_cls_train))
print("Test Accuracy:", accuracy_score(y_test_cls, y_nn_cls_test))
print("Train Precision:", precision_score(y_train_cls_nn, y_nn_cls_train))
print("Test Precision:", precision_score(y_test_cls, y_nn_cls_test))
print("Train Recall:", recall_score(y_train_cls_nn, y_nn_cls_train))
print("Test Recall:", recall_score(y_test_cls, y_nn_cls_test))
print("Train F1 Score:", f1_score(y_train_cls_nn, y_nn_cls_train))
print("Test F1 Score:", f1_score(y_test_cls, y_nn_cls_test))
print("Train ROC AUC Score:", roc_auc_score(y_train_cls_nn, classification_model.predict(X_train_cls_nn)))
print("Test ROC AUC Score:", roc_auc_score(y_test_cls, classification_model.predict(X_test_cls_s)))
