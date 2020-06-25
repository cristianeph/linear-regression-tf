import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Descargando el dataset
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

# Configurando las columnas del dataset
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# Quitando las columnas con valores deconocidos
dataset.isna().sum()

# En caso que se detecten columnas con valores desconocidos, estas se eliminan
dataset = dataset.dropna()

# Estandarizando columnas categoricas a tipo numericas
# En este caso la columna 'Origin' contenia una columna con valor 1, 2 o 3
# Con esta linea se cambia la distribucion de estos valores a columnas con un flag
# El cual indicara si el registro pertenece o no al 'Pais de origen'

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()

# Se divide el dataset en 2
# 1. dataset de entrenamiento del modelo
# 2. dataset de pruebas para el modelo

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Se lista el dataset de entrenamiento

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# Se plotean las estadisticas del dataset de entrenamiento

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

# Se quita la columna a predecir que en este caso seria 'MPG'
# MPG: Millas por galon / rendimiento de combustible

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# Se normaliza el dataset:
# se aplica una división de la media sobre la desviación estándar para cumplir con este objetivo.
# Este proceso es necesario de realizar con el objetivo de escalar los valores de las variables
# entre 0 y 1 para hacer el proceso de aprendizaje más óptimo
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# Construcción del modelo:
# 3 capas densas, las 2 primeras tienen activación Relu y la primera
# recibe como parametro la estructura del dataset.
# La última no tiene activacion ya que es por defecto linear,
# teniendo en cuenta que emplearemos regresión lineal esto es correcto.
# Relu es una funcion de validacion optimizada para procesos computacionales de aprendizaje
# el cual ignora valores negativos y solo trata los mayores a 0.

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    # La función de pérdida será error medio cuadrático o “mse” y para las métricas
    # empleadas, para ver el rendimiento del modelo, se emplean funciones “mse” y
    # error medio absoluto “mae” de sus siglas en inglés.

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()

# Se inspecciona el modelo
model.summary()

# Se entrena el modelo para que recorra 1000 veces la data de entrenamiento

EPOCHS = 1000

# Se extrae 20% de la data como set de validación para ver posteriormente como está rindiendo.

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()])

# El proceso de aprendizaje se guarda en un histórico para ser analizado.

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# En este caso vemos que los valores de pérdida y su validación están creciendo considerablemente,
# por lo que posteriormente procederemos a optimizar el proceso de aprendizaje.

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# Se plotean los valores de validacion

plotter.plot({'Basic': history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

# Se plotean los valores de perdida

plotter.plot({'Basic': history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')

# Para optimizar el modelo se aplica el mecanismo llamado "Early Stopping",
# El cual analiza el proceso de aprendizaje en tiempo real (mientras se ejecutan las rondas)
# y se detiene cuando ve que la calidad de los resultados comienza a bajar considerablemente.
# Esto con el objetivo de mantener el nivel de confiabilidad.

model = build_model()

# El parametro mas importante de este mecanismo es la cantidad de rondas a verificar ('patience')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels,
                          epochs=EPOCHS, validation_split=0.2, verbose=0,
                          callbacks=[early_stop, tfdocs.modeling.EpochDots()])
plotter.plot({'Early Stopping': early_history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

# Se analiza el valor de error promedio del modelo para nuestra variable 'MGP' o Millas por galón

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# Se realizan las predicciones utilizando el set que separamos en pasos anteriores
# y comprobaremos la predicción contra el valor esperado.

test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)