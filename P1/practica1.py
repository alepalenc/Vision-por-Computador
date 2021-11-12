# -*- coding: utf-8 -*-



import cv2
from matplotlib import pyplot as plt
import math
import numpy as np



"""
Devuelve el valor de la gaussiana 
centrada en el origen en el punto x

x: punto en el que se evalúa la gaussiana
sigma: desviación típica de la gaussiana
"""
def gaussiana(x, sigma):
  return math.exp((-x*x)/(2.0*sigma*sigma))


"""
Devuelve el valor de la primera derivada de la gaussiana 
centrada en el origen en el punto x

x: punto en el que se evalúa la gaussiana
sigma: desviación típica de la gaussiana
"""
def gaussiana1Deriv(x, sigma):
  return (-x/(sigma*sigma))*math.exp((-x*x)/(2.0*sigma*sigma))


"""
Devuelve el valor de la segunda derivada de la gaussiana 
centrada en el origen en el punto x

x: punto en el que se evalúa la gaussiana
sigma: desviación típica de la gaussiana
"""

def gaussiana2Deriv(x, sigma):
  return (-1.0/(sigma*sigma) + x*x/pow(sigma,4))*math.exp((-x*x)/(2.0*sigma*sigma))




"""
Devuelve la máscara de la gaussiana o de alguna de sus 
dos primeras derivadas discretizada

valor: parámetro para calcular la máscara (desv. típica o tamaño de la máscara)
flag_valor: valor = desviación típica si True, valor = tamaño máscara si False
derivada_mascara: derivada de la gaussiana usada para el cálculo de la máscara
  (0 si se usa la propia gaussiana)
"""
def mascaraGaussiana1D(valor, flag_valor, derivada_mascara):
  # Calcular sigma a partir de valor
  sigma = None
  if flag_valor == True:
    sigma = valor
  else:
    if valor%2 == 0:
      print("Error: la máscara no puede tener un tamaño par")
      exit(-1)
    # Máximo valor de sigma tal que, al truncar hacia arriba 3*sigma, 
    # obtenemos (tamaño_mascara-1)/2
    sigma = (valor-1.0)/6.0
  # t = (tamaño_mascara-1)/2
  t = int(math.ceil(3*sigma))
	
  # Inicializar máscara vacía
  mascara = np.zeros(2*t+1)
	
  # Calcular máscara en función de la derivada especificada
  if derivada_mascara == 0:
    mascara[t] = 1.0
    suma = mascara[t]
    for i in range(1, t+1):
      mascara[t+i] = mascara[t-i] = gaussiana(i, sigma)
      suma += 2*mascara[t+i]
    # En este caso, se divide por la suma para que la suma de los valores sea 1
    mascara /= suma
  elif derivada_mascara == 1:
    mascara[t] = 0.0
    for i in range(1, t+1):
      mascara[t+i] = gaussiana1Deriv(i, sigma)
      mascara[t-i] = -mascara[t+i]
  elif derivada_mascara == 2:
    mascara[t] = -1.0/(sigma*sigma)
    for i in range(1, t+1):
      mascara[t+i] = mascara[t-i] = gaussiana2Deriv(i, sigma)
  else:
    print("Error: el valor derivada_mascara debe ser 0, 1 ó 2")
    exit(-1)
  
  return mascara




"""
Devuelve la imagen orlada a izquierda y derecha por bordes de ceros

img: imagen a orlar
t: número de columnas que se añaden tanto a izquierda como a derecha
"""
def orlaCeros(img, t):
  orla = np.zeros((img.shape[0],t))
  return np.hstack((orla,img,orla))


"""
Devuelve la imagen orlada a izquierda y derecha por bordes reflejados (101)

img: imagen a orlar
t: número de columnas que se añaden tanto a izquierda como a derecha
"""
def orlaReflejo(img, t):
  reflejo_izq = np.flip(img[:,0:t], axis=1)
  reflejo_der = np.flip(img[:,img.shape[1]-t:img.shape[1]], axis=1)
  return np.hstack((reflejo_izq,img,reflejo_der))


"""
Devuelve la imagen orlada a izquierda y derecha por bordes replicados

img: imagen a orlar
t: número de columnas que se añaden tanto a izquierda como a derecha
"""
def orlaReplicados(img, t):
  reflejo_izq = np.outer(img[:,0], np.ones(t))
  reflejo_der = np.outer(img[:,img.shape[1]-1], np.ones(t))
  return np.hstack((reflejo_izq,img,reflejo_der))




"""
Devuelve una imagen sobre la que se ha aplicado una máscara replicada por filas

img: imagen sobre la que se aplica la máscara replicada
mascara: máscara a aplicar
flag_borde: tipo de borde con el que se orla la imagen
  0 --> borde con ceros
  1 --> borde reflejo 101
  2 --> borde replicados
"""
def aplicarMascaraReplicada(img, mascara, flag_borde):
  # Obtener una matriz con la máscara replicada por filas
  mascara_replicada = np.outer(np.ones(img.shape[0]), mascara)
	
  # Orlar la imagen según el borde especificado
  t = int((len(mascara)-1)/2)
  img_orlada = None
  if flag_borde == 0:
    img_orlada = orlaCeros(img, t)
  elif flag_borde == 1:
    img_orlada = orlaReflejo(img, t)
  elif flag_borde == 2:
    img_orlada = orlaReplicados(img, t)
  else:
    print('Error: valor no válido para flag_borde')
    exit(-1)
  
  # Aplicar la máscara replicada a la imagen
  img_final = np.zeros(img.shape, dtype=img.dtype)
  for i in range(t,img.shape[1]+t):
    img_final[:,i-t] = np.sum(np.multiply(img_orlada[:,i-t:i+t+1], mascara_replicada), axis=1)
  
  return img_final




"""
Devuelve la imagen que resulta de aplicar una correlación a partir de dos 
máscaras de dimensión 1 con la misma longitud, una por filas y otra por columnas

img: imagen sobre la que se aplica la máscara replicada
masc_x: máscara a aplicar por filas
masc_y: máscara a aplicar por columnas
flag_borde: tipo de borde con el que se orla la imagen
  0 --> borde con ceros
  1 --> borde reflejo 101
  2 --> borde replicados
"""
def correlacionMascaras1D(img, masc_x, masc_y, flag_borde):
  if len(masc_x) != len(masc_y):
    print("Error: las máscaras no tienen la misma longitud")
    exit(-1)
	
  if img.shape[0] < len(masc_x) or img.shape[1] < len(masc_x):
    print("Error: las máscaras tienen dimensiones superiores a las de la imagen")
    exit(-1)
  
  # Aplicar masc_x por filas
  img_corr = aplicarMascaraReplicada(img, masc_x, flag_borde)
  # Trasponer matriz, aplicar masc_y por filas y trasponer de nuevo
  # (Equivalente a aplicar masc_y por columnas)
  img_corr = img_corr.transpose()
  img_corr = aplicarMascaraReplicada(img_corr, masc_y, flag_borde)
  img_corr = img_corr.transpose()
	
  return img_corr




"""
Devuelve la imagen que resulta de aplicar el operador laplaciano

img: imagen sobre la que se calcula su laplaciana
valor: parámetro para calcular las máscaras (desv. típica o tamaño de la máscara)
flag_valor: valor = desviación típica si True, valor = tamaño máscara si False
flag_borde: tipo de borde con el que se orla la imagen
  0 --> borde con ceros
  1 --> borde reflejo 101
  2 --> borde replicados
"""
def laplaciana(img, valor, flag_valor, flag_borde):
  # Calcular las máscaras de la gaussiana y de su segunda derivada
  masc_gauss = mascaraGaussiana1D(valor, flag_valor, 0)
  masc_gauss_2deriv = mascaraGaussiana1D(valor, flag_valor, 2)
	
  # Obtener imágenes aplicando las máscaras de las derivadas parciales segundas
  img_deriv_xx = correlacionMascaras1D(img, masc_gauss, masc_gauss_2deriv, flag_borde)
  img_deriv_yy = correlacionMascaras1D(img, masc_gauss_2deriv, masc_gauss, flag_borde)
	
  # Calcular sigma según valor
  sigma = valor
  if flag_valor == False:
    sigma = (valor-1.0)/6.0
	
  # Sumar derivadas parciales y normalizar (multiplicar por sigma^2)
  img_laplaciana = (sigma*sigma)*(img_deriv_xx + img_deriv_yy)
	
  return img_laplaciana




"""
Devuelve una submuestra reducida compuesta por los píxeles que 
están en una fila y una columna impares en la imagen original

img: imagen a submuestrear
"""
def reducirSubmuestra(img):
  # Creo una submuestra con la mitad de filas y de columnas
  img_submuestra = np.zeros((int(img.shape[0]/2),int(img.shape[1]/2)))
  # Los píxeles usados para la submuestra son aquellos que en la 
  # imagen original se encuentran en una fila y una columna impares
  for i in range(img_submuestra.shape[0]):
    for j in range(img_submuestra.shape[1]):
      img_submuestra[i][j] = img[2*i+1][2*j+1]
  
  return img_submuestra




"""
Devuelve una submuestra ampliada en la que los píxeles de la imagen original 
se encuentran en una fila y una columna impares en la imagen ampliada. El resto 
de píxeles se calculan interpolando los píxeles más próximos obtenidos 
directamente de la imagen original.

ig: imagen a submuestrear
flag_fila: si True, se añade una fila adicional para que el número de filas sea impar
flag_columna: si True, se añade una columna adicional para que el número de columnas sea impar
"""
def ampliarSubmuestra(img, flag_fila=False, flag_columna=False):
  fila_adicional = columna_adicional = 0
  if flag_fila == True:
    fila_adicional = 1
  if flag_columna == True:
    columna_adicional = 1
	
  # Crear la submuestra ampliada
  img_submuestra = np.zeros((img.shape[0]*2+fila_adicional,img.shape[1]*2+columna_adicional))
	
  # Tratamiento especial para las dos primeras filas y columnas
  img_submuestra[0][0] = img_submuestra[0][1] = img_submuestra[1][0] = img_submuestra[1][1] = img[0][0]
  for i in range(1,img.shape[0]):
    img_submuestra[2*i][0] = img_submuestra[2*i+1][0] = img_submuestra[2*i+1][1] = img[i][0]
    img_submuestra[2*i][1] = (img[i][0]+img[i-1][0])/2
  for j in range(1,img.shape[1]):
    img_submuestra[0][2*j] = img_submuestra[0][2*j+1] = img_submuestra[1][2*j+1] = img[0][j]
    img_submuestra[1][2*j] = (img[0][j]+img[0][j-1])/2
  
  # Asignación de valores al resto de píxeles
  for i in range(1,img.shape[0]):
    for j in range(1,img.shape[1]):
      img_submuestra[2*i][2*j] = (img[i][j]+img[i-1][j]+img[i-1][j]+img[i-1][j-1])/4
      img_submuestra[2*i+1][2*j] = (img[i][j]+img[i][j-1])/2
      img_submuestra[2*i][2*j+1] = (img[i][j]+img[i-1][j])/2
      img_submuestra[2*i+1][2*j+1] = img[i][j]
  
  #Si es necesario, se añade una fila o una columna adicional según los flags
  if flag_fila == True:
    img_submuestra[-1,:] = img_submuestra[-2,:]
  if flag_columna == True:
    img_submuestra[:,-1] = img_submuestra[:,-2]
  
  return img_submuestra



"""
Devuelve la pirámide gaussiana de tamaño 4 de una imagen

img: imagen de la que se calcula su pirámide gaussiana
valor: parámetro para calcular la máscara (desv. típica o tamaño de la máscara)
flag_valor: valor = desviación típica si True, valor = tamaño máscara si False
"""
def piramideGaussiana(img, valor, flag_valor):
  # Crear la imagen de la pirámide gaussiana
  piramide = np.zeros((img.shape[0],img.shape[1]+int(img.shape[1]/2)))
  # Copiar la imagen original a la izquierda de la imagen pirámide
  piramide[0:img.shape[0],0:img.shape[1]] = img
	
  altura = 0
  img_subm = img.copy()
  mascara_gaussiana = mascaraGaussiana1D(valor, flag_valor, 0)
  for i in range(3):
    # Aplicar filtro gaussiano
    img_subm = correlacionMascaras1D(img_subm, mascara_gaussiana, mascara_gaussiana, 2)
    # Obtener submuestra reducida
    img_subm = reducirSubmuestra(img_subm)
    # Guardar submuestra justo abajo de la submuestra anterior 
    # y a la izquierda de la original
    piramide[altura:altura+img_subm.shape[0],img.shape[1]:img.shape[1]+img_subm.shape[1]] = img_subm
    altura += img_subm.shape[0]
  
  return piramide




"""
Devuelve la imagen normalizada en el intervalo [0,1]

img: imagen a normalizar
"""
def normalizarIntervalo01(img):
  # Obtener máximo y mínimo
  minimo, maximo = 0, 1
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if minimo > img[i][j]:
        minimo = img[i][j]
      elif maximo < img[i][j]:
        maximo = img[i][j]
  
  # Normalizar si es necesario
  img_normalizada = img.copy()
  if minimo < 0 or maximo > 1:
    img_normalizada = (1/(maximo-minimo))*(img-minimo*np.ones(img.shape))
	
  return img_normalizada




"""
Devuelve la lista de imágenes normalizada en el intervalo [0,1], usando como 
máximo y mínimo los valores máximo y mínimo globales de entre todas las 
imágenes, respectivamente.

lista_imgs: lista de imágenes a normalizar
"""
def normalizarListaImagenes01(lista_imgs):
  # Obtener máximo y mínimo globales
  minimo, maximo = 0, 1
  for img in lista_imgs:
    for i in range(img.shape[0]):
       for j in range(img.shape[1]):
        if minimo > img[i][j]:
          minimo = img[i][j]
        elif maximo < img[i][j]:
          maximo = img[i][j]
  
  # Normalizar si es necesario
  lista_norm = []
  if minimo < 0 or maximo > 1:
    for img in lista_imgs:
      lista_norm.append( (1/(maximo-minimo))*(img-minimo*np.ones(img.shape)) )
  else:
    lista_norm = lista_imgs.copy()
  
  return lista_norm




"""
Devuelve la pirámide laplaciana de tamaño 4 de una imagen

img: imagen de la que se calcula su pirámide laplaciana
valor: parámetro para calcular la máscara (desv. típica o tamaño de la máscara)
flag_valor: valor = desviación típica si True, valor = tamaño máscara si False
"""
def piramideLaplaciana(img, valor, flag_valor):
  # Crear la imagen de la pirámide laplaciana
  piramide = np.zeros((img.shape[0],img.shape[1]+int(img.shape[1]/2)))
	
  # Crear una lista con todas las submuestras a añadir a la pirámide
  lista_submuestras = []
	
  # Crear las tres primeras imágenes laplacianas y añadirlas a la lista
  img_subm = img.copy()
  mascara_gaussiana = mascaraGaussiana1D(valor, flag_valor, 0)
  for i in range(3):
    # Crear copia de submuestra reducida anterior con filtro gaussiano
    img_subm_auxiliar = correlacionMascaras1D(img_subm, mascara_gaussiana, mascara_gaussiana, 2)
    # Reducir submuestra anterior con filtro gaussiano
    img_subm = reducirSubmuestra(img_subm_auxiliar)
    # Actualizar flags de filas y columnas impares
    flag_fila = flag_columna = False
    if img_subm_auxiliar.shape[0]%2 == 1:
      flag_fila = True
    if img_subm_auxiliar.shape[1]%2 == 1:
      flag_columna = True
    # Añadir nuevaa laplaciana a la lista:
    # submuestra con filtro gaussiano menos ella misma reducida y luego ampliada
    lista_submuestras.append( 
        img_subm_auxiliar - ampliarSubmuestra(img_subm, flag_fila, flag_columna) )
	
  # Normalizar lista de submuestras laplacianas al intervalo [0,1]
  lista_submuestras = normalizarListaImagenes01(lista_submuestras)
	
  # Añadir la última imagen obtenida con filtro gaussiano, previamente normalizada
  lista_submuestras.append( normalizarIntervalo01(img_subm) )
  
  # Guardar la primera submuestra laplaciana a la izquierda de la pirámide
  piramide[0:lista_submuestras[0].shape[0],0:lista_submuestras[0].shape[1]] = (
      lista_submuestras[0] )
	
  # Guardar el resto de submuestras justo debajo de la submuestra anterior 
  # y a la izquierda de la primera
  h = 0
  for i in range(1,4):
    dims = lista_submuestras[i].shape
    piramide[h:h+dims[0],img.shape[1]:img.shape[1]+dims[1]] = (
        lista_submuestras[i] )
    h += lista_submuestras[i].shape[0]
  
  return piramide




"""
Devuelve la imagen híbrida que resulta de hacer la suma de una imagen con 
un filtro gaussiano y otra imagen con un filtro laplaciano.

img1: imagen a la que se le aplica el filtro gaussiana
img2: imagen a la que se le aplica el filtro laplaciano
sigma1: desviación típica de las máscaras del filtro gaussiano
sigma1: desviación típica de las máscaras del filtro laplaciano
peso1: valor por el que se multiplica la imagen 1 filtrada antes de sumarla
peso1: valor por el que se multiplica la imagen 2 filtrada antes de sumarla
flag_dibujar: True si se quieren dibujar las tres imágenes en una misma ventana
"""
def imagenHibrida(img1, img2, sigma1, sigma2, peso1=1, peso2=1, flag_dibujar=True):
  if img1.shape != img2.shape:
    print("Error: las imágenes no tienen las mismas dimensiones")
    exit(-1)
	
  #Aplicar filtro de paso bajo a imagen 1
  masc_gauss = mascaraGaussiana1D(sigma1, True, 0)
  img1_frec_bajas = correlacionMascaras1D(img1, masc_gauss, masc_gauss, 2)
	
  #Aplicar filtro de paso alto a imagen 2
  img2_frec_altas = laplaciana(img2, sigma2, True, 2)
	
  #Normalizar las dos imágenes al intervalo [0,1]
  img1_frec_bajas = normalizarIntervalo01(img1_frec_bajas)
  img2_frec_altas = normalizarIntervalo01(img2_frec_altas)
	
  #Crear la imagen híbrida y normalizarla al intervalo [0,1]
  img_hibrida = peso1*img1_frec_bajas + peso2*img2_frec_altas
  img_hibrida = normalizarIntervalo01(img_hibrida)
	
  #Mostrar las tres imágenes en una misma ventana
  if flag_dibujar == True:
    plt.imshow(np.hstack((img1_frec_bajas,img_hibrida,img2_frec_altas)), cmap="gray")
    plt.show()
	
  return img_hibrida




"""
Función que opera con imágenes a color.
Devuelve la imagen híbrida que resulta de hacer la suma de una imagen con 
un filtro gaussiano y otra imagen con un filtro laplaciano.

img1: imagen a color a la que se le aplica el filtro gaussiana
img2: imagen a color a la que se le aplica el filtro laplaciano
sigma1: desviación típica de las máscaras del filtro gaussiano
sigma1: desviación típica de las máscaras del filtro laplaciano
peso1: valor por el que se multiplica la imagen 1 filtrada antes de sumarla
peso1: valor por el que se multiplica la imagen 2 filtrada antes de sumarla
"""
def imagenHibridaColor(img1, img2, sigma1, sigma2, peso1=1, peso2=1):
  if img1.shape != img2.shape:
    print("Error: las imágenes no tienen las mismas dimensiones")
    exit(-1)
	
  img_hibrida = np.zeros(img1.shape)
  for i in range(3):
    componente_img1 = img1[:,:,i]
    componente_img2 = img2[:,:,i]
    img_hibrida[:,:,i] = imagenHibrida( 
      componente_img1, componente_img2, sigma1, sigma2, peso1, peso2, False )
	
  return img_hibrida










if __name__ == "__main__":
  print("///// APARTADO 1A /////")
  print()
	
  sigma = 3.0
  print("SIGMA =", sigma)
  print()
  cv2.waitKey(0)
	
  mascara_gauss = mascaraGaussiana1D(sigma, True, 0)
  print("Máscara de la gaussiana:\n", mascara_gauss)
  plt.figure(figsize=(5,5))
  plt.plot(np.arange(-(len(mascara_gauss)-1)/2, (len(mascara_gauss)-1)/2+1),mascara_gauss,'o',markersize=2)
  plt.show()
  print()
  cv2.waitKey(0)
	
  mascara_gauss_1deriv = mascaraGaussiana1D(sigma, True, 1)
  print("Máscara de la primera derivada de la gaussiana:\n", mascara_gauss_1deriv)
  plt.figure(figsize=(5,5))
  plt.plot(np.arange(-(len(mascara_gauss_1deriv)-1)/2, (len(mascara_gauss_1deriv)-1)/2+1),mascara_gauss_1deriv,'o',markersize=2)
  plt.show()
  print()
  cv2.waitKey(0)
	
  mascara_gauss_2deriv = mascaraGaussiana1D(sigma, True, 2)
  print("Máscara de la segunda derivada de la gaussiana:\n", mascara_gauss_2deriv)
  plt.figure(figsize=(5,5))
  plt.plot(np.arange(-(len(mascara_gauss_2deriv)-1)/2, (len(mascara_gauss_2deriv)-1)/2+1),mascara_gauss_2deriv,'o',markersize=2)
  plt.show()
  print()
  cv2.waitKey(0)
	
  tam = 15
  print("TAMAÑO MÁSCARA =", tam)
  print()
  cv2.waitKey(0)
	
  mascara_gauss = mascaraGaussiana1D(tam, False, 0)
  print("Máscara de la gaussiana:\n", mascara_gauss)
  plt.figure(figsize=(5,5))
  plt.plot(np.arange(-(len(mascara_gauss)-1)/2, (len(mascara_gauss)-1)/2+1),mascara_gauss,'o',markersize=2)
  plt.show()
  print()
  cv2.waitKey(0)
	
  mascara_gauss_1deriv = mascaraGaussiana1D(tam, False, 1)
  print("Máscara de la primera derivada de la gaussiana:\n", mascara_gauss_1deriv)
  plt.figure(figsize=(5,5))
  plt.plot(np.arange(-(len(mascara_gauss_1deriv)-1)/2, (len(mascara_gauss_1deriv)-1)/2+1),mascara_gauss_1deriv,'o',markersize=2)
  plt.show()
  print()
  cv2.waitKey(0)
	
  mascara_gauss_2deriv = mascaraGaussiana1D(tam, False, 2)
  print("Máscara de la segunda derivada de la gaussiana:\n", mascara_gauss_2deriv)
  plt.figure(figsize=(5,5))
  plt.plot(np.arange(-(len(mascara_gauss_2deriv)-1)/2, (len(mascara_gauss_2deriv)-1)/2+1),mascara_gauss_2deriv,'o',markersize=2)
  plt.show()
  print()
  cv2.waitKey(0)
	
  print("///// FIN APARTADO 1A /////")
  print()
  print()
  print()
  cv2.waitKey(0)
  
	
  
	
  print("///// APARTADO 1B /////")
  print()
	
  tam = 15
  print("TAMAÑO MÁSCARA =", tam)
  print()
  
  print("Imagen original: ")
  img = cv2.imread('./imagenes/bicycle.bmp', 0)
  plt.imshow(img, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
	
  print("Imagen convolucionada: ")
  mascara_gauss = mascaraGaussiana1D(tam, False, 0)
  img_conv = correlacionMascaras1D(img, mascara_gauss, mascara_gauss, 0)
  plt.imshow(img_conv, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
	
  print("Imagen convolucionada con GaussianBlur: ")
  img_conv_gaussianblur = cv2.GaussianBlur(img, (tam,tam), (tam-1)/6.0)
  plt.imshow(img_conv_gaussianblur, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
	
  print("///// FIN APARTADO 1B /////")
  print()
  print()
  print()
  cv2.waitKey(0)
  
  
	
	
  print("///// APARTADO 1C /////")
  print()
	
  for tam in range(3, 15, 4):
    print("TAMAÑO MÁSCARA =", tam)
    mascara_gauss = mascaraGaussiana1D(tam,False,0)
    mascara_gauss1deriv = mascaraGaussiana1D(tam,False,1)
    mascaras_cv2 = cv2.getDerivKernels(1, 0, tam)
    mascara_gauss1deriv_cv2 = mascaras_cv2[0].transpose()[0]
    mascara_gauss_cv2 = mascaras_cv2[1].transpose()[0]
    print("Gaussiana:")
    print("Máscara del apartado 1A:")
    print(mascara_gauss)
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(-(tam-1)/2, (tam-1)/2+1),mascara_gauss,'o',markersize=2)
    plt.show()
    print("Máscara de cv2.getDerivKernels:", )
    print(mascara_gauss_cv2)
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(-(tam-1)/2, (tam-1)/2+1),mascara_gauss_cv2,'o',markersize=2)
    plt.show()
		
    print()
    cv2.waitKey(0)
		
    print("Derivada de la gaussiana:")
    print("Máscara del apartado 1A:")
    print(mascara_gauss1deriv)
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(-(tam-1)/2, (tam-1)/2+1),mascara_gauss1deriv,'o',markersize=2)
    plt.show()
    print("Máscara de cv2.getDerivKernels:", )
    print(mascara_gauss1deriv_cv2)
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(-(tam-1)/2, (tam-1)/2+1),mascara_gauss1deriv_cv2,'o',markersize=2)
    plt.show()
		
    print()
    cv2.waitKey(0)
	
  for sigma in range(1, 6, 2):
    print("SIGMA =", sigma)
    tam = 2*math.ceil(3*sigma)+1
    mascara_gauss = mascaraGaussiana1D(sigma,True,0)
    mascara_gauss1deriv = mascaraGaussiana1D(sigma,True,1)
    mascaras_cv2 = cv2.getDerivKernels(1, 0, tam)
    mascara_gauss1deriv_cv2 = mascaras_cv2[0].transpose()[0]
    mascara_gauss_cv2 = mascaras_cv2[1].transpose()[0]
    print("Gaussiana:")
    print("Máscara del apartado 1A:")
    print(mascara_gauss)
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(-(tam-1)/2, (tam-1)/2+1),mascara_gauss,'o',markersize=2)
    plt.show()
    print("Máscara de cv2.getDerivKernels:", )
    print(mascara_gauss_cv2)
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(-(tam-1)/2, (tam-1)/2+1),mascara_gauss_cv2,'o',markersize=2)
    plt.show()
		
    print()
    cv2.waitKey(0)
		
    print("Derivada de la gaussiana:")
    print("Máscara del apartado 1A:")
    print(mascara_gauss1deriv)
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(-(tam-1)/2, (tam-1)/2+1),mascara_gauss1deriv,'o',markersize=2)
    plt.show()
    print("Máscara de cv2.getDerivKernels:", )
    print(mascara_gauss1deriv_cv2)
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(-(tam-1)/2, (tam-1)/2+1),mascara_gauss1deriv_cv2,'o',markersize=2)
    plt.show()
		
    print()
    cv2.waitKey(0)
	
  print("///// FIN APARTADO 1C /////")
  print()
  print()
  print()
  cv2.waitKey(0)
  
	
	
	
  print("///// APARTADO 1D /////")
  print()
	
  sigma = 3.0
	
  # Calcular la máscara de la gaussiana
  mascara_gauss = mascaraGaussiana1D(sigma, True, 0)
  print("Máscara de la gaussiana con sigma =", sigma, ":\n", mascara_gauss)
  plt.figure(figsize=(5,5))
  plt.plot(np.arange(-(len(mascara_gauss)-1)/2, (len(mascara_gauss)-1)/2+1),mascara_gauss,'o',markersize=2)
  plt.show()
	
  # Calcular la máscara normalizada de la segunda derivada de la gaussiana
  mascara_gauss_2deriv = (sigma*sigma)*mascaraGaussiana1D(sigma, True, 2)
  print("Máscara normalizada de la segunda derivada de la gaussiana con sigma =", sigma, ":\n", mascara_gauss_2deriv)
  plt.figure(figsize=(5,5))
  plt.plot(np.arange(-(len(mascara_gauss_2deriv)-1)/2, (len(mascara_gauss_2deriv)-1)/2+1),mascara_gauss_2deriv,'o',markersize=2)
  plt.show()
  print()
  cv2.waitKey(0)
	
  # Laplaciana sigma=1.0, bordes cero
  print("Laplaciana con sigma = 1.0 y bordes cero: ")
  plt.imshow(laplaciana(img, 1.0, True, 0), cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  # Laplaciana sigma=1.0, bordes replicados
  print("Laplaciana con sigma = 1.0 y bordes replicados: ")
  plt.imshow(laplaciana(img, 1.0, True, 2), cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  # Laplaciana sigma=3.0, bordes cero
  print("Laplaciana con sigma = 3.0 y bordes cero: ")
  plt.imshow(laplaciana(img, 3.0, True, 0), cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  # Laplaciana sigma=3.0, bordes replicados
  print("Laplaciana con sigma = 3.0 y bordes replicados: ")
  plt.imshow(laplaciana(img, 3.0, True, 2), cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
	
  print("///// FIN APARTADO 1D /////")
  print()
  print()
  print()
  cv2.waitKey(0)
	
	
	
	
  print("///// APARTADO 2A /////")
  print()
	
  print("Pirámide gaussiana con tamaño de máscara = 5:")
  img = cv2.imread('./imagenes/bicycle.bmp', 0)
  piramide = piramideGaussiana(img, 5, False)
  plt.imshow(piramide, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
	
  print("Pirámide gaussiana con tamaño de máscara = 11:")
  img = cv2.imread('./imagenes/bicycle.bmp', 0)
  piramide = piramideGaussiana(img, 11, False)
  plt.imshow(piramide, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
	
  print("Pirámide gaussiana con sigma = 1.0:")
  img = cv2.imread('./imagenes/bicycle.bmp', 0)
  piramide = piramideGaussiana(img, 1.0, True)
  plt.imshow(piramide, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
	
  print("Pirámide gaussiana con sigma = 3.0:")
  img = cv2.imread('./imagenes/bicycle.bmp', 0)
  piramide = piramideGaussiana(img, 3.0, True)
  plt.imshow(piramide, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
	
  print("///// FIN APARTADO 2A /////")
  print()
  print()
  print()
  cv2.waitKey(0)
	
	
	
	
  print("///// APARTADO 2B /////")
  print()
	
  print("Pirámide laplaciana con tamaño de máscara = 5:")
  img = cv2.imread('./imagenes/bicycle.bmp', 0)
  piramide = piramideLaplaciana(img, 5, False)
  plt.imshow(piramide, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
  print("Pirámide laplaciana con tamaño de máscara = 11:")
  img = cv2.imread('./imagenes/bicycle.bmp', 0)
  piramide = piramideLaplaciana(img, 11, False)
  plt.imshow(piramide, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
  print("Pirámide laplaciana con sigma = 1.0:")
  img = cv2.imread('./imagenes/bicycle.bmp', 0)
  piramide = piramideLaplaciana(img, 1.0, True)
  plt.imshow(piramide, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
  print("Pirámide laplaciana con sigma = 3.0:")
  img = cv2.imread('./imagenes/bicycle.bmp', 0)
  piramide = piramideLaplaciana(img, 3.0, True)
  plt.imshow(piramide, cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
  print("///// FIN APARTADO 2B /////")
  print()
  print()
  print()
  cv2.waitKey(0)
	
	
	
	
  print("///// APARTADO 3 /////")
  print()
	
  print("Imagen híbrida bicicleta-moto:")
  sigma1, sigma2 = 2.0, 33/20
  img1 = cv2.imread('./imagenes/motorcycle.bmp', 0)
  img2 = cv2.imread('./imagenes/bicycle.bmp', 0)
  img_hibrida = imagenHibrida(img1, img2, sigma1, sigma2, 4, 3)
  plt.imshow(piramideGaussiana(img_hibrida, 7, False), cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
	
  print("Imagen híbrida gato-perro:")
  sigma1, sigma2 = 3.0, 33/20
  img1 = cv2.imread('./imagenes/cat.bmp', 0)
  img2 = cv2.imread('./imagenes/dog.bmp', 0)
  img_hibrida = imagenHibrida(img1, img2, sigma1, sigma2, 4, 3)
  plt.imshow(piramideGaussiana(img_hibrida, 7, False), cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
	
  print("Imagen híbrida Marilyn-Einstein:")
  sigma1, sigma2 = 1.0, 3.0
  img1 = cv2.imread('./imagenes/marilyn.bmp', 0)
  img2 = cv2.imread('./imagenes/einstein.bmp', 0)
  img_hibrida = imagenHibrida(img1, img2, sigma1, sigma2, 4, 5)
  plt.imshow(piramideGaussiana(img_hibrida, 7, False), cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
	
  print("Imagen híbrida submarino-pez:")
  sigma1, sigma2 = 3.0, 11/20
  img1 = cv2.imread('./imagenes/submarine.bmp', 0)
  img2 = cv2.imread('./imagenes/fish.bmp', 0)
  img_hibrida = imagenHibrida(img1, img2, sigma1, sigma2, 1, 1)
  plt.imshow(piramideGaussiana(img_hibrida, 7, False), cmap='gray')
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
	
  print("///// FIN APARTADO 3 /////")
  print()
  print()
  print()
  cv2.waitKey(0)
	
	
	
	
  print("///// BONUS 1 /////")
  print()
	
  print("Imagen híbrida bicicleta-moto a color:")
  sigma1, sigma2 = 2.0, 1.0
  img1 = cv2.imread('./imagenes/motorcycle.bmp')
  img2 = cv2.imread('./imagenes/bicycle.bmp')
  img_hibrida = imagenHibridaColor(img1, img2, sigma1, sigma2, 4, 3)
  plt.imshow( cv2.cvtColor((255*img_hibrida).astype(np.uint8), cv2.COLOR_BGR2RGB) )
  plt.show()
  print()
  cv2.waitKey(0)
  print()
  
	
  print("Imagen híbrida gato-perro a color:")
  sigma1, sigma2 = 3.0, 33/20
  img1 = cv2.imread('./imagenes/cat.bmp')
  img2 = cv2.imread('./imagenes/dog.bmp')
  img_hibrida = imagenHibridaColor(img1, img2, sigma1, sigma2, 4, 3)
  plt.imshow( cv2.cvtColor((255*img_hibrida).astype(np.uint8), cv2.COLOR_BGR2RGB) )
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
	
  print("Imagen híbrida submarino-pez a color:")
  sigma1, sigma2 = 3.0, 11/20
  img1 = cv2.imread('./imagenes/submarine.bmp')
  img2 = cv2.imread('./imagenes/fish.bmp')
  img_hibrida = imagenHibridaColor(img1, img2, sigma1, sigma2, 1, 1)
  plt.imshow( cv2.cvtColor((255*img_hibrida).astype(np.uint8), cv2.COLOR_BGR2RGB) )
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
	
  print("///// FIN BONUS 1 /////")
  print()
  print()
  print()
  cv2.waitKey(0)
	
	
	
	
  print("///// BONUS 2 /////")
  print()
	
  
  sigma1, sigma2 = 3.0, 61/30
  peso1, peso2 = 4, 3
  
	
	
  img1 = cv2.imread('./imagenes/monte_cervino.jpg', 0)
  img2 = cv2.imread('./imagenes/torre_eiffel.jpg', 0)
	
  img1 = img1[20:583,0:800]
  img2 = img2[:,290:690]
	
  print("Imagen monte Cervino restringida: ")
  plt.imshow(img1, cmap="gray")
  plt.show()
  print()
  print("Imagen torre Eiffel restringida: ")
  plt.imshow(img2, cmap="gray")
  plt.show()
  print()
	
  masc_gauss = mascaraGaussiana1D(1.0, True, 0)
  img1_auxiliar = correlacionMascaras1D(img1, masc_gauss, masc_gauss, 2)
  img1 = np.zeros((img1_auxiliar.shape[0], int(img1_auxiliar.shape[1]/2)))
  for j in range(img1.shape[1]):
    img1[:,j] = img1_auxiliar[:,2*j]
	
  print("Imagen monte Cervino restringida y estrechada: ")
  plt.imshow(img1, cmap="gray")
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
  print("Imagen híbrida: ")
  img_hibrida = imagenHibrida(img1, img2, sigma1, sigma2, peso1, peso2, False)
  plt.imshow(img_hibrida, cmap="gray")
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
  print("Pirámide gaussiana: ")
  plt.imshow(piramideGaussiana(img_hibrida, 7, False), cmap='gray')
  plt.show()
  print()
  print()
  print()
  cv2.waitKey(0)
	
	
  
  img1 = cv2.imread('./imagenes/monte_cervino.jpg')
  img2 = cv2.imread('./imagenes/torre_eiffel.jpg')
	
  img1 = img1[20:583,0:800,:]
  img2 = img2[:,290:690,:]
	
  print("Imagen monte Cervino restringida: ")
  plt.imshow( cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) )
  plt.show()
  print()
  print("Imagen torre Eiffel restringida: ")
  plt.imshow( cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) )
  plt.show()
  print()
	
  masc_gauss = mascaraGaussiana1D(1.0, True, 0)
  img1_auxiliar = np.zeros(img1.shape)
  for i in range(3):
    img1_auxiliar[:,:,i] = correlacionMascaras1D(img1[:,:,i], masc_gauss, masc_gauss, 2)
  img1 = np.zeros((img1_auxiliar.shape[0], int(img1_auxiliar.shape[1]/2), 3))
  for j in range(img1.shape[1]):
    img1[:,j,:] = img1_auxiliar[:,2*j,:]
  
  print("Imagen monte Cervino restringida y estrechada: ")
  plt.imshow( cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2RGB) )
  plt.show()
  print()
  cv2.waitKey(0)
  print()
	
  print("Imagen híbrida: ")
  img_hibrida = imagenHibridaColor(img1, img2, sigma1, sigma2, peso1, peso2)
  plt.imshow( cv2.cvtColor((255*img_hibrida).astype(np.uint8), cv2.COLOR_BGR2RGB) )
  plt.show()
  print()
  cv2.waitKey(0)
  print()
  
	
  print("///// FIN BONUS 2 /////")
  print()
  print()
  print()
  cv2.waitKey(0)
	
