# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keral.layers import Dense

#Read the training data
train_df = pd.read_csv('train.csv', index_col='ID')
train_df.head()

target = 'medv'

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_df)

# Print out the adjustment that the scaler applied to the total_earnings column of data
print("Note: median values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[13], scaler.min_[13]))
multiplied_by = scaler.scale_[13]
added = scaler.min_[13]

scaled_train_df = pd.DataFrame(scaled_train, columns=train_df.columns.values)




#Build the model
model = Sequential()

model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

X = scaled_train_df.drop(target, axis=1).values
Y = scaled_train_df[[target]].values

# Train the model
model.fit(
    X[10:],
    Y[10:],
    epochs=50,
    shuffle=True,
    verbose=2
)


#Making the predictions
prediction = model.predict(X[:1])
y_0 = prediction[0][0]
print('Prediction with scaling - {}',format(y_0))
y_0 -= added
y_0 /= multiplied_by
print("Housing Price Prediction  - ${}".format(y_0))


