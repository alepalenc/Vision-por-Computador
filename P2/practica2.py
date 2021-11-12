# -*- coding: utf-8 -*-








#########################################################################
#                                                                       #
#                              APARTADO 1                               #
#                                                                       #
#########################################################################


#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

# Importar librerías necesarias
import numpy as np
import math
import keras
import matplotlib.pyplot as plt
import keras.utils as np_utils

# Importar modelos y capas que se van a usar
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.layers import BatchNormalization, Dropout, UpSampling2D, Activation, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping

# Importar el optimizador a usar
from keras.optimizers import SGD, Adam

# Importar el conjunto de datos
from keras.datasets import cifar100

# Importar clases de preprocesamiento
from keras.preprocessing.image import ImageDataGenerator

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# A esta función solo se la llama una vez. Devuelve 4 
# vectores conteniendo, por este orden, las imágenes
# de entrenamiento, las clases de las imágenes de
# entrenamiento, las imágenes del conjunto de test y
# las clases del conjunto de test.
def cargarImagenes():
    # Cargamos Cifar100. Cada imagen tiene tamaño
    # (32, 32, 3). Nos vamos a quedar con las
    # imágenes de 25 de las clases.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data (label_mode ='fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape (train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]
    
    # Transformamos los vectores de clases en matrices.
    # Cada componente se convierte en un vector de ceros
    # con un uno en la componente correspondiente a la
    # clase a la que pertenece la imagen. Este paso es
    # necesario para la clasificación multiclase en keras.
    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)
    
    return x_train , y_train , x_test , y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve la accuracy de un modelo, 
# definida como el porcentaje de etiquetas bien predichas
# frente al total de etiquetas. Como parámetros es
# necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de
# keras (matrices donde cada etiqueta ocupa una fila,
# con un 1 en la posición de la clase a la que pertenece y un 0 en las demás).
def calcularAccuracy(labels, preds):
    labels = np.argmax(labels, axis = 1)
    preds = np.argmax(preds, axis = 1)
    accuracy = sum(labels == preds)/len(labels)
    return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución
# de la función de pérdida en el conjunto de train y
# en el de validación, y otra con la evolución de la
# accuracy en el conjunto de train y el de validación.
# Es necesario pasarle como parámetro el historial del
# entrenamiento del modelo (lo que devuelven las
# funciones fit() y fit_generator()).
def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()
    
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy','Validation accuracy'])
    plt.show()





#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

model = Sequential()
model.add(keras.Input(shape=(32,32,3)))
model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='softmax'))

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

opt_sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
opt_adam = Adam()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_sgd,
              metrics=['accuracy'])

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

x_train, y_train, x_test, y_test = cargarImagenes()
batch_size = 128
epochs = 30
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(validation_split=validation_split)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

pred = model.predict(x_test, batch_size=batch_size)
print("Accuracy:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)








#########################################################################
#                                                                       #
#                              APARTADO 2                               #
#                                                                       #
#########################################################################


#########################################################################
####################### MEJORA 1: NORMALIZACIÓN #########################
#########################################################################

##### DEFINICIÓN #####

model = Sequential()
model.add(keras.Input(shape=(32,32,3)))
model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='softmax'))


##### COMPILACIÓN #####

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_sgd,
              metrics=['accuracy'])


##### ENTRENAMIENTO #####

x_train, y_train, x_test, y_test = cargarImagenes()
batch_size = 128
epochs = 30
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   validation_split=validation_split)
datagen_train.fit(x_train)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)


##### PREDICCIÓN #####

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)
datagen_test.fit(x_test)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)





#########################################################################
####################### MEJORA 2: EARLY STOPPING ########################
#########################################################################

##### DEFINICIÓN #####

model = Sequential()
model.add(keras.Input(shape=(32,32,3)))
model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='softmax'))


##### COMPILACIÓN #####

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_sgd,
              metrics=['accuracy'])


##### ENTRENAMIENTO #####

x_train, y_train, x_test, y_test = cargarImagenes()
batch_size = 128
epochs = 150
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   validation_split=validation_split)
datagen_train.fit(x_train)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

early_stopping = EarlyStopping(monitor='val_loss', patience=6)

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 callbacks=early_stopping, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)


##### PREDICCIÓN #####

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)
datagen_test.fit(x_test)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)





#########################################################################
################### MEJORA 3: AUMENTO DE PROFUNDIDAD ####################
#########################################################################

##### DEFINICIÓN #####

"""
# modelo con UpSampling
model = Sequential()
model.add(keras.Input(shape=(32,32,3)))

model.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(3,3), activation='relu'))
model.add(UpSampling2D(size=(2,2), interpolation="bilinear"))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=72, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=96, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=112, kernel_size=(3,3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='softmax'))


# Modelo con kernels 1x1 y 3x3
model = Sequential()
model.add(keras.Input(shape=(32,32,3)))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(1,1), activation='relu'))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(1,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(1,1), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(1,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(Dense(25, activation='softmax'))
"""

# modelo que sustituye kernels 5x5 por 3x3 y añade filtros
model = Sequential()
model.add(keras.Input(shape=(32,32,3)))

model.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dense(25, activation='softmax'))


##### COMPILACIÓN #####

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_adam,
              metrics=['accuracy'])


##### ENTRENAMIENTO #####

x_train, y_train, x_test, y_test = cargarImagenes()
batch_size = 128
epochs = 150
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   validation_split=validation_split)
datagen_train.fit(x_train)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

early_stopping = EarlyStopping(monitor='val_loss', patience=4)

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 callbacks=early_stopping, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)


##### PREDICCIÓN #####

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)
datagen_test.fit(x_test)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)





#########################################################################
#################### MEJORA 4: BATCH NORMALIZATION ######################
#########################################################################

##### DEFINICIÓN #####

model = Sequential()
model.add(keras.Input(shape=(32,32,3)))

model.add(Conv2D(filters=8, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Conv2D(filters=16, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dense(25, activation='softmax'))


##### COMPILACIÓN #####

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_adam,
              metrics=['accuracy'])


##### ENTRENAMIENTO #####

x_train, y_train, x_test, y_test = cargarImagenes()
batch_size = 128
epochs = 150
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   validation_split=validation_split)
datagen_train.fit(x_train)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

early_stopping = EarlyStopping(monitor='val_loss', patience=4)

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 callbacks=early_stopping, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)


##### PREDICCIÓN #####

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)
datagen_test.fit(x_test)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)





#########################################################################
########################## MEJORA 5: DROPOUT ############################
#########################################################################

##### DEFINICIÓN #####

model = Sequential()
model.add(keras.Input(shape=(32,32,3)))

model.add(Conv2D(filters=8, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Conv2D(filters=16, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Flatten())
model.add(Dropout(rate=0.2))
model.add(Dense(400, activation='relu'))
model.add(Dense(25, activation='softmax'))


##### COMPILACIÓN #####

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_adam,
              metrics=['accuracy'])


##### ENTRENAMIENTO #####

x_train, y_train, x_test, y_test = cargarImagenes()
batch_size = 128
epochs = 150
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   validation_split=validation_split)
datagen_train.fit(x_train)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

early_stopping = EarlyStopping(monitor='val_loss', patience=6)

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 callbacks=early_stopping, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)


##### PREDICCIÓN #####

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)
datagen_test.fit(x_test)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)





#########################################################################
##################### MEJORA 6: DATA AUGMENTATION #######################
#########################################################################

##### DEFINICIÓN #####
"""
# Modelo con kernels 1x1 y 3x3
model = Sequential()
model.add(keras.Input(shape=(32,32,3)))

model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))
model.add(Conv2D(filters=16, kernel_size=(1,1), activation='relu'))

model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))
model.add(Conv2D(filters=16, kernel_size=(1,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))
model.add(Conv2D(filters=32, kernel_size=(1,1), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))
model.add(Conv2D(filters=32, kernel_size=(1,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Flatten())

model.add(Dropout(rate=0.2))
model.add(Dense(80, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='softmax'))
"""

model = Sequential()
model.add(keras.Input(shape=(32,32,3)))

model.add(Conv2D(filters=8, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Conv2D(filters=16, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))

model.add(Flatten())
model.add(Dropout(rate=0.2))
model.add(Dense(400, activation='relu'))
model.add(Dense(25, activation='softmax'))


##### COMPILACIÓN #####

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_sgd,
              metrics=['accuracy'])


##### ENTRENAMIENTO #####

x_train, y_train, x_test, y_test = cargarImagenes()
batch_size = 128
epochs = 150
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=20, 
                                   width_shift_range=[-4,4], 
                                   height_shift_range=[-4,4], 
                                   zoom_range=[0.9,1.1], 
                                   horizontal_flip=True,
                                   validation_split=validation_split)
datagen_train.fit(x_train)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)


##### PREDICCIÓN #####

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)
datagen_test.fit(x_test)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)








#########################################################################
#                                                                       #
#                              APARTADO 3                               #
#                                                                       #
#########################################################################


#########################################################################
################ CARGAR LAS LIBRERÍAS NECESARIAS ########################
#########################################################################

# Importar librerías necesarias
import math
import numpy as np
import keras
import keras.utils as np_utils
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Importar modelos y capas específicas que se van a usar
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.layers import BatchNormalization, Dropout, UpSampling2D, Activation, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping

# Importar el modelo ResNet50 y su respectiva función de preprocesamiento,
# que es necesario pasarle a las imágenes para usar este modelo
from keras.applications.resnet import ResNet50, preprocess_input

# Importar el optimizador a usar
from keras.optimizers import SGD
opt_sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

# Importar clases de preprocesamiento
from keras.preprocessing.image import ImageDataGenerator

#########################################################################
################## FUNCIÓN PARA LEER LAS IMÁGENES #######################
#########################################################################

# Dado un fichero train.txt o test.txt y el path donde se encuentran los
# ficheros y las imágenes, esta función lee las imágenes
# especificadas en ese fichero y devuelve las imágenes en un vector y 
# sus clases en otro.

def leerImagenes(vec_imagenes, path):
  clases = np.array([img.split('/')[0] for img in vec_imagenes])
  imagenes = np.array([img_to_array(load_img(path + "/" + img, 
                                             target_size = (224, 224))) 
                       for img in vec_imagenes])
  return imagenes, clases

#########################################################################
############# FUNCIÓN PARA CARGAR EL CONJUNTO DE DATOS ##################
#########################################################################

# Usando la función anterior, y dado el path donde se encuentran las
# imágenes y los archivos "train.txt" y "test.txt", devuelve las 
# imágenes y las clases de train y test para usarlas con keras
# directamente.

def cargarDatos(path):
  # Cargamos los ficheros
  train_images = np.loadtxt(path + "/train.txt", dtype = str)
  test_images = np.loadtxt(path + "/test.txt", dtype = str)
  
  # Leemos las imágenes con la función anterior
  train, train_clases = leerImagenes(train_images, path)
  test, test_clases = leerImagenes(test_images, path)
  
  # Pasamos los vectores de las clases a matrices 
  # Para ello, primero pasamos las clases a números enteros
  clases_posibles = np.unique(np.copy(train_clases))
  for i in range(len(clases_posibles)):
    train_clases[train_clases == clases_posibles[i]] = i
    test_clases[test_clases == clases_posibles[i]] = i

  # Después, usamos la función to_categorical()
  train_clases = np_utils.to_categorical(train_clases, 200)
  test_clases = np_utils.to_categorical(test_clases, 200)
  
  # Barajar los datos
  train_perm = np.random.permutation(len(train))
  train = train[train_perm]
  train_clases = train_clases[train_perm]

  test_perm = np.random.permutation(len(test))
  test = test[test_perm]
  test_clases = test_clases[test_perm]
  
  return train, train_clases, test, test_clases

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve el accuracy de un modelo, definido como el 
# porcentaje de etiquetas bien predichas frente al total de etiquetas.
# Como parámetros es necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de keras (matrices
# donde cada etiqueta ocupa una fila, con un 1 en la posición de la clase
# a la que pertenece y 0 en las demás).

def calcularAccuracy(labels, preds):
  labels = np.argmax(labels, axis = 1)
  preds = np.argmax(preds, axis = 1)
  
  accuracy = sum(labels == preds)/len(labels)
  
  return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución de la función
# de pérdida en el conjunto de train y en el de validación, y otra
# con la evolución del accuracy en el conjunto de train y en el de
# validación. Es necesario pasarle como parámetro el historial
# del entrenamiento del modelo (lo que devuelven las funciones
# fit() y fit_generator()).

def mostrarEvolucion(hist):

  loss = hist.history['loss']
  val_loss = hist.history['val_loss']
  plt.plot(loss)
  plt.plot(val_loss)
  plt.legend(['Training loss', 'Validation loss'])
  plt.show()
    
  acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']
  plt.plot(acc)
  plt.plot(val_acc)
  plt.legend(['Training accuracy','Validation accuracy'])
  plt.show()




######################### Cargar el conjunto de datos ##########################
x_train, y_train, x_test, y_test = cargarDatos("imagenes")




#######################################################################################
##### Usar ResNet50 preentrenada en ImageNet como un extractor de características #####
#######################################################################################

##### MODELO 0: Solo se reentrena una capa de salida #####

# Definición
resnet50 = ResNet50(include_top=False, weights='imagenet', pooling='avg')
for layer in resnet50.layers[:]:
  layer.trainable = False
model = Sequential()
model.add(resnet50)
model.add(Dense(200, activation='softmax'))

# Compilación
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_sgd,
              metrics=['accuracy'])

# Entrenamiento
batch_size = 64
epochs = 100
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                   validation_split=validation_split)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 callbacks=early_stopping, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)

# Predicción
datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy model0:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)





##### MODELO 1: Eliminar FC y salida; sustituir por nuevas FC y salida #####

# Definición
resnet50 = ResNet50(include_top=False, weights='imagenet', pooling='avg')
for layer in resnet50.layers[:]:
  layer.trainable = False
model = Sequential()
model.add(resnet50)
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(200, activation='softmax'))

# Compilación
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_sgd,
              metrics=['accuracy'])

# Entrenamiento
batch_size = 64
epochs = 100
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                   validation_split=validation_split)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 callbacks=early_stopping, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)

# Predicción
datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy model1:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)





##### MODELO 2: Eliminar AveragePooling, FC y salida; añadir nuevas capas #####

# Definición
resnet50 = ResNet50(include_top=False, weights='imagenet')
for layer in resnet50.layers[:]:
  layer.trainable = False
model = Sequential()
model.add(resnet50)
model.add(Dropout(rate=0.2))
model.add(Conv2D(filters=3072, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))
model.add(Dropout(rate=0.2))
model.add(Conv2D(filters=4096, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))
model.add(GlobalAveragePooling2D())
model.add(Dropout(rate=0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dense(200, activation='softmax'))

# Compilación
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_sgd,
              metrics=['accuracy'])

# Entrenamiento
batch_size = 64
epochs = 100
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                   validation_split=validation_split)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 callbacks=early_stopping, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)

# Predicción
datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy model2:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)





#######################################################################################
########################## Reentrenar ResNet50 (fine tunning) #########################
#######################################################################################

batch_size = 64
epochs = 50
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos
datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=validation_split)
train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)

# Añadir nuevas capas al final de ResNet50
resnet50 = ResNet50(include_top=False, weights='imagenet', pooling='avg')
x = resnet50.output
outputs = Dense(200, activation='softmax')(x)
model = Model(inputs=resnet50.input, outputs=outputs)

# Compilación
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_sgd,
              metrics=['accuracy'])

# Entrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 callbacks=early_stopping, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)

# Predicción
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy fine tuning:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)








#########################################################################
#                                                                       #
#                                BONUS                                  #
#                                                                       #
#########################################################################


#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

# Importar librerías necesarias
import numpy as np
import math
import keras
import matplotlib.pyplot as plt
import keras.utils as np_utils

# Importar modelos y capas que se van a usar
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Concatenate
from keras.layers import BatchNormalization, Dropout, Activation, AveragePooling2D
from keras.callbacks import EarlyStopping

# Importar el optimizador a usar
from keras.optimizers import SGD

# Importar el conjunto de datos
from keras.datasets import cifar100

# Importar clases de preprocesamiento
from keras.preprocessing.image import ImageDataGenerator

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# A esta función solo se la llama una vez. Devuelve 4 
# vectores conteniendo, por este orden, las imágenes
# de entrenamiento, las clases de las imágenes de
# entrenamiento, las imágenes del conjunto de test y
# las clases del conjunto de test.
def cargarImagenes():
    # Cargamos Cifar100. Cada imagen tiene tamaño
    # (32, 32, 3). Nos vamos a quedar con las
    # imágenes de 25 de las clases.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data (label_mode ='fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape (train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]
    
    # Transformamos los vectores de clases en matrices.
    # Cada componente se convierte en un vector de ceros
    # con un uno en la componente correspondiente a la
    # clase a la que pertenece la imagen. Este paso es
    # necesario para la clasificación multiclase en keras.
    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)
    
    return x_train , y_train , x_test , y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve la accuracy de un modelo, 
# definida como el porcentaje de etiquetas bien predichas
# frente al total de etiquetas. Como parámetros es
# necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de
# keras (matrices donde cada etiqueta ocupa una fila,
# con un 1 en la posición de la clase a la que pertenece y un 0 en las demás).
def calcularAccuracy(labels, preds):
    labels = np.argmax(labels, axis = 1)
    preds = np.argmax(preds, axis = 1)
    accuracy = sum(labels == preds)/len(labels)
    return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución
# de la función de pérdida en el conjunto de train y
# en el de validación, y otra con la evolución de la
# accuracy en el conjunto de train y el de validación.
# Es necesario pasarle como parámetro el historial del
# entrenamiento del modelo (lo que devuelven las
# funciones fit() y fit_generator()).
def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()
    
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy','Validation accuracy'])
    plt.show()





#########################################################################
########################### MÓDULO CONV #################################
#########################################################################
"""
Módulo que devuelve el output resultante de aplicar una convolución 
seguida de un Batch Normalization y una activación ReLU a un input dado.
x: input al que se le aplican las capas
filters: número de filtros de salida que devuelve la convolución
kernel_size: tamaño de la máscara de la convolución
strides: stride que se aplica en la convolución
padding: padding que se aplica en la convolución
"""
def conv_module(x, filters, kernel_size, strides=(1,1), padding='same'):
		x = Conv2D(filters, (kernel_size,kernel_size), strides=strides, padding=padding)(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)
		return x

#########################################################################
######################### MÓDULO INCEPTION ##############################
#########################################################################
"""
Módulo que devuelve el output resultante de concatenar las salidas de 
dos módulos de convolución, uno con máscara 1x1 y otro con máscara 3x3.
x: input al que se le aplican las capas
filters_1x1: número de filtros de salida que devuelve el módulo de convolución 1x1
filters_3x3: número de filtros de salida que devuelve el módulo de convolución 3x3
"""
def inception_module(x, filters_1x1, filters_3x3):
		conv_1x1 = conv_module(x, filters_1x1, 1)
		conv_3x3 = conv_module(x, filters_3x3, 3)
		x = Concatenate()([conv_1x1, conv_3x3])
		return x

#########################################################################
######################## MÓDULO DOWNSAMPLE ##############################
#########################################################################
"""
Módulo que devuelve el output resultante de concatenar las salidas de 
un módulo de convolución y un MaxPooling, ambos con tamaño de 
máscara/ventana 3x3 y stride 2x2.
x: input al que se le aplican las capas
filters: número de filtros de salida que usa el módulo de convolución
"""
def downsample_module(x, filters):
		conv_3x3 = conv_module(x, filters, 3, strides=(2,2), padding='valid')
		pool = MaxPooling2D((3,3), strides=(2,2))(x)
		x = Concatenate()([conv_3x3, pool])
		return x





#########################################################################
##################### DEFINICIÓN DEL MODELO  ############################
#########################################################################

# Definición del input del modelo y el primer módulo de convolución
inputs = keras.Input(shape=(32,32,3))
x = conv_module(inputs, 32, 3)

# Dos módulos inception seguidos de un módulo downsample
x = inception_module(x, 32, 32)
x = inception_module(x, 32, 48)
x = downsample_module(x, 80)

# Cuatro módulos inception seguidos de un módulo downsample
x = inception_module(x, 112, 48)
x = inception_module(x, 96, 64)
x = inception_module(x, 80, 80)
x = inception_module(x, 48, 96)
x = downsample_module(x, 96)

# Dos módulos inception seguidos de un AveragePooling
x = inception_module(x, 176, 160)
x = inception_module(x, 176, 160)
x = AveragePooling2D((3,3), strides=(2,2))(x)

# Dropout seguido de dos capas Fully Connected y el clasificador Softmax
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512)(x)
x = Activation("relu")(x)
x = Dense(25)(x)
outputs = Activation('softmax')(x)

# Creación del modelo
model = Model(inputs, outputs)




#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

opt_sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt_sgd,
              metrics=['accuracy'])




#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

x_train, y_train, x_test, y_test = cargarImagenes()
batch_size = 128
epochs = 100
validation_split = 0.1
steps_per_epoch = math.ceil((1-validation_split)*len(x_train)/batch_size)
validation_steps = math.ceil(validation_split*len(x_train)/batch_size)

datagen_train = ImageDataGenerator(featurewise_center=True, 
                                   featurewise_std_normalization=True, 
                                   rotation_range=20, 
                                   width_shift_range=[-4,4], 
                                   height_shift_range=[-4,4], 
                                   zoom_range=[0.9,1.1], 
                                   horizontal_flip=True, 
                                   validation_split=validation_split)
datagen_train.fit(x_train)

train_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen_train.flow(x_train, y_train, batch_size=batch_size, subset='validation')

hist = model.fit(train_generator, 
                 steps_per_epoch=steps_per_epoch, 
                 epochs=epochs, 
                 validation_data=validation_generator, 
                 validation_steps=validation_steps)




#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)
datagen_test.fit(x_test)
test_generator = datagen_test.flow(x_test, batch_size=1, shuffle=False)
pred = model.predict(test_generator, steps=len(test_generator))
print("Accuracy:", calcularAccuracy(y_test, pred))
mostrarEvolucion(hist)

