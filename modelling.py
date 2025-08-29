import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('dataset.csv')
X = df.drop('kemiskinan',axis=1)
y = df['kemiskinan']

preprocessor = joblib.load('preprocessor.pkl')
X_processed = preprocessor.transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),   
    keras.layers.Dense(1, activation='sigmoid'), 
    ])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0
)

training = model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=50,
    verbose='auto',
    callbacks=[early_stop],
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1
)

y_pred = (model.predict(X_val) > 0.5).astype("int32")

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print("Accuaracy:", acc) 
print("Precision:", prec) 
print("Recall:", rec) 
print("Confusion Matrix", cm) 

model.save('model.h5')
print("Model Berhasil di save")



