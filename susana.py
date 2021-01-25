## -*- coding: utf-8 -*-

#"""
#Created on Sat Feb  2 01:23:52 2019
#
#@author: Leila

#Version aburrida

from random import uniform

N=10**6

cambio=0

nocambio=0

for i in range(N):

    x=int(uniform(1, 4)) #La llave está en la caja número x

    #print('La llave está en', x)

    y1=int(uniform(1,4)) #mi primera elección de caja 

    #print('Mi primera elección es la caja', y1)

    #Susana nos da info: dónde no está la llave

    s=[1,2,3]

    for j in s:
        if j!=x and j!=y1:
                #print ('Susana dice que la llave no está en la caja',s[j])
                susana=j
                break 
        
        #cambio de caja

    for k in s:
        if k!=susana and k!=y1:
            y2=k
            if y2==x:
                #print('Ganaste en la segunda elección')
                cambio=cambio+1
            else:
                #print('Perdiste, estaba bien tu primera opcion')
                nocambio=nocambio+1
                #print('La llave estaba en la caja', x)   
 
print('Cantidad de veces que gane cambiando mi primera eleccion:', cambio)

print('Cantidad de veces que gane sin cambiar de elección',  nocambio)