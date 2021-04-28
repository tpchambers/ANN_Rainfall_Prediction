####Theo Chambers ANNRainPrediction####

###############ImportingLibraries#########################

import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import numpy as np

#################Data Preprocessing#################
data = pd.read_csv("weatherAUS.csv")
data.head()
#look at data info
data.info()
#set categorical variables
categoricals = data.select_dtypes(include=['object']).columns.values.tolist()
#replace categoricals 
for i in categoricals:
    data[i].fillna(data[i].mode()[0],inplace=True) #inplace = True to replace

target=data.values[:,-1]
target = LabelEncoder().fit_transform(target)

#Preserve cyclicality 
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data.Date.dt.month
data['Day'] = data.Date.dt.day
data['Year'] = data.Date.dt.year
data['Cyclic_Month_Sine'] = np.sin(2*np.pi*data['Month']/12) #Divide my number of months
data['Cyclic_Day_Sine'] = np.sin(2*np.pi*data['Day']/31) # Days
data['Cyclic_Month_Cos'] = np.cos(2*np.pi*data['Month']/12)
data['Cyclic_Day_Cos'] = np.cos(2*np.pi*data['Day']/31)
#Replace Nas for numerical data
numerics = data.select_dtypes(include=['float64']).columns.values.tolist()
for i in numerics:
    data[i].fillna(data[i].mode()[0],inplace=True)
#drop date since we don't need it anymore
data.info()
data.drop(['Date'],axis=1)

#############DesigningInputArray#######################
import category_encoders as ce
#one-hot-encoding of features
categoricals = data.select_dtypes(include=['object']).columns.values.tolist()
cat_df = data.select_dtypes(include=['object']).copy()
encoder = ce.BinaryEncoder(cols=categoricals)
df_binary = encoder.fit_transform(cat_df)
data = data.drop(categoricals,axis=1)
result = pd.concat([data,df_binary],axis=1)
result.head()
features = result.drop(['RainTomorrow_0','RainTomorrow_1','Date','Month','Day'],axis=1)
features_array = features.values
features_array =  preprocessing.MinMaxScaler().fit_transform(features_array)
x = features_array
y = target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

########################testmodel#####################
from keras import callbacks
import tensorflow as tf
model = Sequential()
model.add(Dense(units = 16, activation = 'relu', input_dim = features_array.shape[1]))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(units = 32,  activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


early_stop = callbacks.EarlyStopping(
    min_delta=0.01, # minimium amount of change
    patience=5, # epochs before stopping
    restore_best_weights=True,
)


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
history1 = model.fit(X_train, y_train, batch_size = 32, epochs = 10,callbacks=[early_stop], validation_data=(X_test,y_test))


acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#######################HyperParameterOptimization##################

def model(activation):
    model = Sequential()
    model.add(Dense(units = 16,kernel_initializer = 'uniform', activation = activation, input_dim = features_array.shape[1]))
    model.add(Dense(units = 16, kernel_initializer = 'uniform',activation = activation))
    model.add(Dense(units = 32,  kernel_initializer = 'uniform',activation = activation))
    model.add(Dropout(0.25))
    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = activation))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

#Activation Functions
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def initial_model(x_train,y_train):
    classifier=KerasClassifier(build_fn=model,epochs=5,batch_size=32)
    parameters = {'activation':['sigmoid','relu','softmax']}
    grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=5)
    grid_search=grid_search.fit(X_train,y_train)
    parameters=grid_search.best_params_
    best_accuracy=grid_search.best_score_
    return classifier,parameters,best_accuracy,grid_search

classifier,best_parameters,best_accuracy,grid_search=initial_model(X_train,y_train)
print(best_parameters,best_accuracy)


#models run in Colab, must include imports as it is on the cloud
#Gaussian
from tensorflow.keras.layers import GaussianNoise
def model_with_noise(nodes=100,noise=.001):
    model=Sequential()
    model.add(GaussianNoise(noise,input_shape=(45,)))
    model.add(Dense(nodes, input_shape=(45,), activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

grid = {'noise':[.001,.01,.1,.2],'nodes':[100,150,300]}
model = KerasClassifier(build_fn=model_with_noise,epochs=3)
model_noise = GridSearchCV(model,param_grid=grid)
model_noise.fit(X_train,y_train)
(model_noise.best_params_,model_noise.best_score_)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
#Dropout test
def model_2(neurons=16,dropout_rate=.25):
    model = Sequential()
    model.add(Dense(neurons,kernel_initializer = 'uniform', activation = 'relu', input_dim = features_array.shape[1]))
    model.add(Dense(neurons, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dense(neurons,  kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def model_2_train(x_train,y_train):
    classifier=KerasClassifier(build_fn=model_2,epochs=5,batch_size=32)
    neurons = [16,32,64]
    dropout_rate = [.25,.3,.35,.4]
    param_grid = dict(neurons=neurons,dropout_rate=dropout_rate)
    grid_search = GridSearchCV(estimator=classifier,param_grid=param_grid,scoring='accuracy',cv=5)
    grid_search=grid_search.fit(X_train,y_train)
    parameters=grid_search.best_params_
    best_accuracy=grid_search.best_score_
    return classifier,parameters,best_accuracy,grid_search
classifier2,best_parameters2,best_accuracy2,grid_search2=model_2_train(X_train,y_train)

#Learning rate and momentum
from keras.optimizers import SGD
def model_lr_m(learn_rate=.001,momentum=0.6):
    model = Sequential()
    model.add(Dense(units = 16,kernel_initializer = 'uniform', activation = 'relu', input_dim = features_array.shape[1]))
    model.add(Dense(units = 16, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dense(units = 32,  kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    optimizer = SGD(lr=learn_rate,momentum=momentum)
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def initial_model(x_train,y_train):
    classifier=KerasClassifier(build_fn=model_lr_m,epochs=3,batch_size=32)
    parameters = {'learn_rate':[.001,.01,.01],'momentum':[.6,.8,.99]}
    grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=3)
    grid_search=grid_search.fit(X_train,y_train)
    parameters=grid_search.best_params_
    best_accuracy=grid_search.best_score_
    return classifier,parameters,best_accuracy,grid_search


#Inspecting graphs

#########################
initial_model = initial_model(X_train,y_train)
from keras import callbacks
import tensorflow as tf
model3 = Sequential()
model3.add(Dense(units = 64, activation = 'relu', input_dim = features_array.shape[1]))
model3.add(Dense(units = 64, activation = 'relu'))
model3.add(Dense(units = 64,  activation = 'relu'))
model3.add(Dropout(0.3))
model3.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
model3.add(Dropout(0.5))
model3.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

opt = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=.6)
model3.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
history3 = model3.fit(X_train, y_train, batch_size = 32, epochs = 10,callbacks=[early_stop], validation_data=(X_test,y_test))


import matplotlib.pyplot as plt
acc = history3.history['accuracy']
val_acc = history3.history['val_accuracy']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


model4 = Sequential()
model4.add(Dense(units = 64, activation = 'relu', input_dim = features_array.shape[1]))
model4.add(Dense(units = 64, activation = 'relu'))
model4.add(Dense(units = 64,  activation = 'relu'))
model4.add(Dropout(0.3))
model4.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
model4.add(Dropout(0.5))
model4.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model4.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history4 = model4.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test,y_test))



import matplotlib.pyplot as plt
acc = history4.history['accuracy']
val_acc = history4.history['val_accuracy']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



model5 = Sequential()
model5.add(Dense(units = 32, activation = 'relu', input_dim = features_array.shape[1]))
model5.add(Dense(units = 32, activation = 'relu'))
model5.add(Dense(units = 64,  activation = 'relu'))
model5.add(Dropout(0.3))
model5.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
model5.add(Dropout(0.3))
model5.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model5.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history5 = model5.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test,y_test))



import matplotlib.pyplot as plt
acc = history5.history['accuracy']
val_acc = history5.history['val_accuracy']
loss = history5.history['loss']
val_loss = history5.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


model_6=Sequential()
model_6.add(GaussianNoise(.001,input_shape=(45,)))
model_6.add(Dense(300, input_shape=(45,), activation='relu'))
model_6.add(Dense(1,activation='sigmoid'))
model_6.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history6 = model_6.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test,y_test))


acc = history6.history['accuracy']
val_acc = history6.history['val_accuracy']
loss = history6.history['loss']
val_loss = history6.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


def model_weight(init='uniform'):
    model = Sequential()
    model.add(Dense(units = 32,kernel_initializer = init, activation = 'relu', input_dim = features_array.shape[1]))
    model.add(Dense(units = 32, kernel_initializer = init,activation = 'relu'))
    model.add(Dense(units = 64,  kernel_initializer = init,activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 64, kernel_initializer = init, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 1, kernel_initializer = init, activation = 'sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def model_weight_train(x_train,y_train):
    classifier=KerasClassifier(build_fn=model_weight,epochs=3,batch_size=32)
    init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform']
    param_grid = dict(init=init)
    grid_search = GridSearchCV(estimator=classifier,param_grid=param_grid,scoring='accuracy',cv=3)
    grid_search=grid_search.fit(X_train,y_train)
    parameters=grid_search.best_params_
    best_accuracy=grid_search.best_score_
    return classifier,parameters,best_accuracy,grid_search
model_weight = model_weight_train(X_train,y_train)
(model_weight)


model7 = Sequential()
model7.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = features_array.shape[1]))
model7.add(Dense(units = 64, kernel_initializer = 'uniform',activation = 'relu'))
model7.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
model7.add(Dropout(0.3))
model7.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
model7.add(Dropout(0.3))
model7.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model7.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
history7 = model7.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test,y_test))
model7.summary()


acc = history7.history['accuracy']
val_acc = history7.history['val_accuracy']
loss = history7.history['loss']
val_loss = history7.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


model8 = Sequential()
model8.add(Dense(units = 32, activation = 'relu', input_dim = features_array.shape[1]))
model8.add(Dense(units = 32, activation = 'relu'))
model8.add(Dense(units = 64,  activation = 'relu'))
model8.add(Dropout(0.3))
model8.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
model8.add(Dropout(0.35))
model8.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
opt = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=.6)
model8.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
history8 = model3.fit(X_train, y_train, batch_size = 32, epochs = 10,callbacks=[early_stop], validation_data=(X_test,y_test))
model8.summary()



acc = history8.history['accuracy']
val_acc = history8.history['val_accuracy']
loss = history8.history['loss']
val_loss = history8.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




#model 3 final - best model
from keras import callbacks
import tensorflow as tf
final_model1 = Sequential()
final_model1.add(Dense(units = 64, activation = 'relu', input_dim = features_array.shape[1]))
final_model1.add(Dense(units = 64, activation = 'relu'))
final_model1.add(Dense(units = 64,  activation = 'relu'))
final_model1.add(Dropout(0.3))
final_model1.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
final_model1.add(Dropout(0.5))
final_model1.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

early_stop = callbacks.EarlyStopping(
    min_delta=0.01, # minimium amount of change
    patience=15, # epochs before stopping
    restore_best_weights=True,
)

opt = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=.6)
final_model1.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
history_final_1 = final_model1.fit(X_train, y_train, batch_size = 32, epochs = 150,callbacks=[early_stop], validation_data=(X_test,y_test))



acc = history_final_1.history['accuracy']
val_acc = history_final_1.history['val_accuracy']
loss = history_final_1.history['loss']
val_loss = history_final_1.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#####################ConfusionMatrix#########################

predictions = final_model2.predict(X_test)
predictions


from sklearn.metrics import confusion_matrix
import seaborn as sns
predictions = final_model2.predict(X_test)
y_pred = (predictions > .5)
matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(matrix, annot=True)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['0','1']))

#change fmt for decimal placees
sns.heatmap(matrix/np.sum(matrix), annot=True, 
            fmt='.2%', cmap='Blues')
y_pred = (predictions > .5)
matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(matrix, annot=True)
names = ['True Neg','False Pos','False Neg','True Pos']
names = np.asarray(labels).reshape(2,2)
sns.heatmap(matrix,annot=names, fmt="", cmap='Blues')



#model 1 
final_model2 = Sequential()
final_model2.add(Dense(units = 16, activation = 'relu', input_dim = features_array.shape[1]))
final_model2.add(Dense(units = 16, activation = 'relu'))
final_model2.add(Dense(units = 32,  activation = 'relu'))
final_model2.add(Dropout(0.25))
final_model2.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
final_model2.add(Dropout(0.5))
final_model2.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


early_stop = callbacks.EarlyStopping(
    min_delta=0.01, # minimium amount of change
    patience=15, # epochs before stopping
    restore_best_weights=True,
)


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
final_model2.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

history_final_2 = final_model2.fit(X_train, y_train, batch_size = 32, epochs = 150,callbacks=[early_stop], validation_data=(X_test,y_test))
final_model2.summary()

acc = history_final_2.history['accuracy']
val_acc = history_final_2.history['val_accuracy']
loss = history_final_2.history['loss']
val_loss = history_final_2.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

