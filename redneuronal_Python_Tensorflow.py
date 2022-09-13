#tensorflow librería de inteligencia artificial de google
import tensorflow as tf
import numpy as np

#entras y resultados --> celsius/fahrenheit
#capa de entrada y capa de salida
#las neuronas se conectan con conexiones que tienen un peso
#cada neurona menos la de entrada tiene un sesgo
# celsius x peso + sesgo
#todo se inicia de manera aleatoria y va aprendiendo
#aprende en la medida que ajusta los pesos

celsius = np.array([-40,10,0, 8, 15, 22, 18], dtype=float)
fahrenheit = np.array([-40,14,32, 46, 59, 72, 100], dtype=float)

#keras
#capas densas
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo= tf.keras.Sequential ([capa])

#compila prepara el modelo:
#optimizador Adam, para ajustar pesos y sesgos
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#fit entrana
#epochs: vueltas 1000
print("entrenando!")
historial=modelo.fit(celsius, fahrenheit,epochs=1000, verbose=False)
print("entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# epoca")
plt.ylabel("magnitud pérdida")
plt.plot(historial.history["loss"])

print("prediccion")
resultado = modelo.predict([90.0])
print("resultado" + str(resultado) + "fahrenheit")

print("variables internas del modelo")
print(capa.get_weights())
