# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier


# %%
path_dir = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/'

# %%
tdscfog_train = pd.DataFrame()
for files in os.listdir(path_dir):
    Dataframe = pd.read_csv(path_dir+files)
    tdscfog_train = pd.concat([tdscfog_train,Dataframe])

# %%
tdscfog_train

# %%
tdscfog_train.info()

# %%
tdscfog_train.describe()

# %%
def reduce_memory_usage(df):
    
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024 ** 2 
    print("Memory usage became: ",mem_usg," MB")
    
    return df

# %%
tdscfog_train = reduce_memory_usage(tdscfog_train)


# %%
X = tdscfog_train.iloc[:, 1:4]
y1 = tdscfog_train['StartHesitation']
y2 = tdscfog_train['Turn']
y3 = tdscfog_train['Walking']

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.2, random_state = 42)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.2, random_state = 42)
X_train, X_test, y3_train, y3_test = train_test_split(X, y3, test_size = 0.2, random_state = 42)

# %%
model1 = GaussianNB()
model2 = GaussianNB()
model3 = GaussianNB()

model1.fit(X_train, y1_train)
model2.fit(X_train, y2_train)
model3.fit(X_train, y3_train)

print('Accuracy for StartHesitation:', model1.score(X_test, y1_test))
print('Accuracy for Turn:', model2.score(X_test, y2_test))
print('Accuracy for Walking:', model3.score(X_test, y3_test))

# %%
from sklearn.metrics import classification_report, confusion_matrix

# Get the predictions for the three models on the test data.
y1_pred = model1.predict(X_test)
y2_pred = model2.predict(X_test)
y3_pred = model3.predict(X_test)

# Create a classification report for each model.
print('Classification Report for StartHesitation:')
print(classification_report(y1_test, y1_pred))

print('Classification Report for Turn:')
print(classification_report(y2_test, y2_pred))

print('Classification Report for Walking:')
print(classification_report(y3_test, y3_pred))

# Create a confusion matrix for each model.
print('Confusion Matrix for StartHesitation:')
print(confusion_matrix(y1_test, y1_pred))

print('Confusion Matrix for Turn:')
print(confusion_matrix(y2_test, y2_pred))

print('Confusion Matrix for Walking:')
print(confusion_matrix(y3_test, y3_pred))

# %%
tdcsfog_test_path = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog'

tdcsfog_test_list = []

for file_name in os.listdir(tdcsfog_test_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(tdcsfog_test_path, file_name)
        file = pd.read_csv(file_path)
        file['Id'] = file_name[:-4] + '_' + file['Time'].apply(str)
        tdcsfog_test_list.append(file)

tdcsfog_test = pd.concat(tdcsfog_test_list, axis = 0)

tdcsfog_test

# %%
tdcsfog_test = reduce_memory_usage(tdcsfog_test)

# %%
defog_test_path = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/defog'

defog_test_list = []

for file_name in os.listdir(defog_test_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(defog_test_path, file_name)
        file = pd.read_csv(file_path)
        file['Id'] = file_name[:-4] + '_' + file['Time'].apply(str)
        defog_test_list.append(file)

defog_test = pd.concat(defog_test_list, axis = 0)

defog_test

# %%
defog_test = reduce_memory_usage(defog_test)

# %%
test = pd.concat([tdcsfog_test, defog_test], axis = 0).reset_index(drop = True)
test

# %%
test_X = test.iloc[:, 1:4]

pred_y1 = model1.predict(test_X)
pred_y2 = model2.predict(test_X)
pred_y3 = model3.predict(test_X)

test['StartHesitation'] = pred_y1
test['Turn'] = pred_y2
test['Walking'] = pred_y3

test

# %%
test.describe()

# %%
submission = test.iloc[:, 4:].fillna(0.0)
submission

# %%
submission.to_csv("submission.csv", index = False)


