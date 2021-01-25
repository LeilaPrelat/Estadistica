# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:38:18 2019

@author: Usuario
"""

#1) Tomar un numero al azar (equiprobable) entre 0 y 1 (no sé si incluye al 1)

from random import random

print(random())

from random import uniform

#tomar un numero al azar (equiprobable) entre 5 y 8 (más general)

print(uniform(5, 8))

##########################################################################################################################

#2) Fijarse si el número random es menor que 0.75

from random import random

x=random()

lim=0.75

if x<lim:
    print(x, 'es menor que 0.75')
else:
    print(x)
    
##########################################################################################################################
    
#3) Repetir 1000 veces los dos pasos anteriores. Con while: la acción que se repite se termina cuando se cumple una condición. Con for: la acción se repite una cantidad determinada de veces. 

#from random import random
#
#for j in range(1000):    
#    i=random()
#    if i<0.75:
#        print(i, 'es menor que 0.75')
#    else:
#        print(i)
        
from random import random

menores=list([]) #es una lista
        
for j in range(1000):    
    i=random()
    if i<lim:
        print(i, 'es menor que 0.75')
        menores.append(i)
    else:
        print(i)
print(menores)

#########################################################################################################

#4) Ver cuántos números son menores que 0.75
#
#from random import random
#
#count=0
#
#menores=list([]) #es una lista
#        
#for j in range(1000):    
#    i=random()
#    if i<0.75:
#        print(i, 'es menor que 0.75')
#        count=count+1 
#        menores.append(i)
#    else:
#        print(i)
#print(count) #cuántos números son menores que 0.75
#print(menores)
#
##Versión más corta: 
#
#count=0
#        
#for j in range(1000):    
#    i=random()
#    if i<0.75:
#        count=count+1 
#print(count)

#4) Ver el largo de la lista menores: cuántos de la lista anterior son menores que 0.75

print(len(menores)) # = n 

#####################################################################################################

#5)A) Hacer un histograma de los números < 0.75: 

#acá hago trampa porque no estoy usando los mismos números que el ejercicio anterior sino unos nuevos, pero son todos randoms así que supongo que no importa.

#bins: son los rectángulos del histograma, que tienen que tener ancho 0.05=L/(cantidad de bins) entonces despejo el num_bins

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import hist
from random import uniform      
#
#s=np.random.uniform(0, 0.75, 1000) #para qué repetir 1000 veces si puedo obtener una tira random de 1000 números con este renglón
#
#L_s=max(s)-min(s)
#
#num_bins_s= int(L_s*20)
#
#print(num_bins_s)
#
#plt.hist(s, normed=True, bins=num_bins_s, facecolor='blue')
# 
#plt.show()

#Ocurre algo raro en este gráfico: el eje "y" toma valores mayores a 1, el eje "y" es probabilidad ? 

#k=np.random.uniform(-0.5, 2, 1000) #para qué repetir 1000 veces si puedo obtener una tira random de 1000 números con este renglón

#5)B) Hacer un histograma de los números < 0.75 guardados en la lista menores: 

L_k=max(menores)-min(menores)

print(L_k)

num_bins_k= int(L_k*20)

print(num_bins_k)

n, bins, patches = plt.hist(menores, normed=False, bins=num_bins_k, facecolor='red', edgecolor = 'black')

plt.xlim(-0.5,2)

plt.show()

#########################################################################################################

#6) Superponer el histograma con una función f=N*alfa. Acá no estoy segura de lo que es "n" y "N" (si ambas son variables o sólo una de ellas). 

#6)A) n es la cantidad de números menores que 0.75 entonces f es una recta horizontal

import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from matplotlib.pylab import hist, show
from random import uniform  

N=1000
#
#k=np.random.uniform(0, 1, N)

#count=0
#
#for j in range (N): 
#    if k[j]<0.75:
#        count=count+1
#print(count)  #count = n = cuántos números son menores que 0.75   

#la función es una recta horizontal

#n=len(menores) cantidad de numeros menores a 0.75

modoLeila=1

if modoLeila==0:

    L_k=max(menores)-min(menores)
    
    print(L_k)
    
    num_bins_k= int(L_k*20)
    
    print(num_bins_k)
    
    plt.figure()
    n, bins, patches = plt.hist(menores, normed=False, bins=num_bins_k, facecolor='red', edgecolor = 'black')
    plt.plot(menores, len(menores)*0.75*N*0.05*np.ones(len(menores)), 'b.')
    
    a=15
    
    plt.xlabel('Muestra', fontsize=a)
    plt.ylabel('Cantidad de veces', fontsize=a)
    plt.title('Ejercicio 6 guía cero', fontsize=a)
    
    
    plt.xlim(-0.5,2)
    
    plt.show()

#Los gráficos se superponen pero no se ve el histograma por la escala. 

#6)B n es la cantidad de números menores que 0.75 entonces f es una recta horizontal
    
else:

    L_k=max(menores)-min(menores)
    
    num_bins_k= int(L_k*20)
    
    print(num_bins_k)
    
    y=np.ones(len(menores))
    
    for k in range(len(menores)):
        y[k]=menores[k]*0.75*N*0.05
    print(y)
    
    len(y)==len(menores)
      
    plt.figure()
    n, bins, patches = plt.hist(menores, normed=False, bins=num_bins_k, facecolor='red', edgecolor = 'black')
    plt.plot(menores, y, 'b.')
    
    a=15
    
    plt.xlabel('Muestra', fontsize=a)
    plt.ylabel('Cantidad de veces', fontsize=a)
    plt.title('Ejercicio 6 guía cero', fontsize=a)
    
    
    plt.xlim(-0.5,2)
    
    plt.show()    

#########################################################################################################

#7) Para cada bien medir su distancia a f. 

#Hay 15 bins y 14 cajas. Bins son las rayitas, n son las alturas maximas de las cajitas len(n) = len(cajitas)=len(bins-1)  

if modoLeila==0:
    #distancia al centro de las cajitas
    f=lambda x: len(menores)*0.75*N*0.05
    dist=list([])
    for i in range(num_bins_k-1):
        dist.append(f((bins[i+1]+bins[i])/2)-n[i]) #f(promedio en x de cada cajita)-altura de cada cajita
    print(dist)
    print(len(dist))
    print(np.mean(dist))
    print(np.std(dist))
    #distancia a los bins
    dist_bins=list([])
    for i in range(num_bins_k):
        dist_bins.append(f(bins[i])-n[i]) #f(promedio en x de cada cajita)-altura de cada cajita
    print(dist_bins)
    print(len(dist_bins))
    print(np.mean(dist_bins))
    print(np.std(dist_bins))   

if modoLeila==1:
    f=lambda x: x*0.75*N*0.05
    dist=list([])
    for i in range(num_bins_k-1):
        dist.append(n[i]-f((bins[i+1]+bins[i])/2)) #f(promedio en x de cada cajita)-altura de cada cajita
    print(dist)
    print(len(dist))
    print(np.mean(dist))
    print(np.std(dist))
    #distancia a los bins
    dist_bins=list([])
    for i in range(num_bins_k):
        dist_bins.append(n[i]-f(bins[i])) #f(promedio en x de cada cajita)-altura de cada cajita
    print(dist_bins)
    print(len(dist_bins))
    print(np.mean(dist_bins))
    print(np.std(dist_bins)) 
    
       
    
    