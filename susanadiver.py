# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 22:37:18 2019

@author: Usuario
"""



from random import uniform

#import numpy as np

#Version divertida

x=int(uniform(1, 4)) #La llave está en la caja número x

#print('la llave está en', x)

y1=int(input('¿En qué caja está la llave?')) #primera elección del jugador

#print(y1)

s=[1,2,3]

for j in s:
    if j!=x and j!=y1:
        print ('Susana dice que la llave no está en la caja',j)
        susana=j
        break 

rta=input('¿Querés cambiar de caja?')

if rta=='si' or rta=='Si' or rta=='Sí' or rta=='sí' or rta==str(1) or rta=='y':
    for k in s:
        if k!=susana and k!=y1:
            y2=k
            if y2==x:
                print('Ganaste! La llave estaba en la caja', x)
            else:
                print('Perdiste! La llave estaba en la caja', x)
else:
    if y1==x:
        print('Ganaste! La llave estaba en la caja', x)
    else:
        print('Perdiste! La llave estaba en la caja',x)

print(x)

#Cambio de caja

#cambio=0
#
#y2=input('¿En qué caja está la llave?') #segunda elección
#
#if int(y2)==x:
#    print('Ganaste en la segunda elección')
#    cambio=cambio+1
#else:
#    print('Perdiste')

#------------------------------------------------------------------------------