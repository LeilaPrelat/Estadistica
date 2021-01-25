# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 23:03:44 2019

@author: Usuario
"""

#Ejercicio 1: Tiro 10 dados al azar, cual es la probabilidad de que tres de dados sean 6?

from random import uniform   
import math

N=10**6

exito=0 

for l in range(N):
    x=[] #creo los 10 dados

    for j in range(10):
        xi=int(uniform(1, 7)) #tiro los diez dados: cada uno puede valer lo que sea de 1 a 6
        x.append(xi)
    
    #print(x)   

    #Cual es la probabilidad de que tres de dados sean 6?

    cantidad=[] #nueva lista de 0 y 1 (1 cuando salió un 6, 0 en el resto de los casos)

    for k in range(10):
        if x[k]==6:
            cantidad.append(1)
        else:
            cantidad.append(0)

    #print(cantidad)
    
    #sumo los elementos de la lista cantidad = cantidad de veces que salió el 6 en x
    total=0
    for m in range(10): 
        total=total+cantidad[m]

    #print(total)   

    #exito es que el 6 haya salido exactamente 3 veces

    if total==3:
        exito=exito+1
    
print('Hubo 3 de los 10 dados en los que salió 6:', exito, 'veces')

print('Cantidad de éxitos respecto del total de intentos:', (exito/N)) 

#Lo que nos importa son sólo dos opciones: sale 6 tres veces (éxito) o no sale 6 tres veces (fracaso).
#Además tengo sólo dos probabilidades: p(prob de que salga 6)=1/6 y q(prob de que no salga 6)=5/6.
#La distribución es binomial, distribución en la cual siempre hay sólo dos opciones: fracaso o éxito. 

probabilidad=5**7*math.factorial(10)*(math.factorial(7)*math.factorial(3)*6**10)**(-1)

print('La probabilidad del ejercicio 1 es:', probabilidad)

#%%
#Ejercicio 2: Tengo 100 resistencias, 20 malas y 80 buenas. Cuál es la probabilidad de sacar 5 y 5 ? (saco 10 resistencias al azar)

from random import uniform   
import math

#Al igual que en el ejer 1, Lo que nos importa son sólo dos opciones: sacar 5 resist buenas y 5 malas (éxito) o cualquier otro caso (fracaso).
#Además tengo sólo dos probabilidades: p(prob de sacar resist mala)=0.2 y q(prob de sacar resist buena)=0.8 
#Pero la distribución no es binomial, es hipergeométrica. distribución en la cual siempre hay sólo dos opciones: fracaso o éxito (como en la binomial)
#pero en la hipergeométrica la probabilidad va cambiando (saco una resistencia buena y para sacar una segunda buena tengo menos probabilidad, porque tengo menos casos favorables) 

N=10**6

exito=0 

#Hay dos formas de hacer esto: pongo de manera random las resistencias buenas y malas y entonces saco
#las primeras 10 de la lista (opcion 1) o las resistencias que saco, las saco de manera random (opcion 2). 

#opcion 1:

x=[] #lista de 100 resistencias
    
for b in range(80): #80 resistencias buenas (las buenas son los 1)
    x.append(1)
for m in range(20): #20 resistencias malas (las malas son los 0)
    x.append(0)

for l in range(N):
    
   #Saco 10 resistencias de manera random
     
    r=[] #PROBLEMA: cómo elegir números random distintos 
       
    nodisponibles=[]

    ri=int(uniform(0,100))      
    r.append(ri)
    nodisponibles.append(ri)
    
    while len(r)<10:
        rj=int(uniform(0,100)) 
        if rj not in nodisponibles: 
            r.append(rj)
            nodisponibles.append(rj)
            
    eleccion=[] #elección random de los elementos de x 
    
    for e in range(10):
        eleccion.append(x[r[e]])
        
    #sumo los elementos de la lista eleccion random = cantidad de veces que salió una resistencia buena en eleccion random
    total=0
    for m in range(10): 
        total=total+eleccion[m]
    
    if total==5:
        exito=exito+1

print('Cantidad de éxitos respecto del total de intentos:', exito/N)

#Probabilidad hipergeométrica 

import math
probabilidad=math.factorial(80)*math.factorial(20)*math.factorial(90)*math.factorial(10)*(math.factorial(75)*math.factorial(5)*math.factorial(15)*math.factorial(5)*math.factorial(100))**(-1)
#
print('La probabilidad del ejercicio 2 es:', probabilidad)
#%%
#Ejer 3: Tengo un par de dados (a,b) y gana quien acierte cuántas veces van a sumar 7

from random import uniform   
import collections
import numpy as np


modas=[]

for k in range(10**2):

    cantidad=[] #cantidad de veces que hubo que tirar los dados para que el acierto sea 10 (M=10)
    
    for i in range(10**3): #repito 10000 veces el experimento y me quedo con 4 vmp (valores mas probables)
        
        M=0

        N=0

        while M<10:                                     #queremos que M sea 10
            x=(int(uniform(1, 7)), int(uniform(1, 7))) #dos dados
            N=N+1                                      #contar cada vez que se tiran los dos dados
            if x[0]+x[1]==7:                            #éxito
                M=M+1
            
            #print(M, N)

        cantidad.append(N)

    print(len(cantidad))    

    counter=collections.Counter(cantidad)

    #print(mode(cantidad))    

    print(modas)

    modas.append(counter.most_common(1)[0][0])

np.savetxt('Ejer3valoresmasprobables.txt', modas)

counter2=collections.Counter(modas)

print(counter2.most_common(6))


#Primero hay que calcular la probabilidad de que dos dados sumen 7 "M" veces al tirarlos una cantidad "N",
#eso es una binomial:
 
#probabilidad=5**7*math.factorial(10)*(math.factorial(7)*math.factorial(3)*6**10)**(-1)

#Ejer 8:
import math
import matplotlib.pyplot as plt
import numpy as np
import operator as op
from functools import reduce
#
#def S(m): 
#    S=m*np.log(m)-m+0.5*np.log(2*np.pi*m)+(12*m)**(-1)
#    return S
#
#def ncr(a,b):
#    ncr=np.exp(S(a)-S(b)-S(a-b))
#    return ncr

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

N=5000

m=int(N*0.01) #gente que vota a A

n=500 #0=<n=<N

#p=N*0.01*(n**(-1))

#k=m/2    #0=<k=<m (falta ver casos límites)

Prob = lambda k: ncr(m,k)*ncr(N-m, 500-k)*(ncr(N,500))**(-1)

#Prob = lambda n, k: ncr(n,k)*math.factorial(N*0.01)*math.factorial(N*0.99)*(math.factorial(N*0.01-k)*math.factorial(N*0.99-(n-k))*ncr(N,n))**(-1)

k = range(1,m) 

y=[]

for i in k:
    y.append(Prob(i))

plt.plot(k, y, 'r.')

plt.grid(1)

#Ejer 8:

#a)
import matplotlib.pyplot as plt
import numpy as np

def fa(k):
    if k==0:
        return (1-0.01)**(500)
    else:
        return fa(k-1)*(500-(k-1))*0.01*(k*(1-0.01))**(-1)

suma=0

for i in range(10):
    suma=suma+fa(i)
print(suma)    

rta = 1-suma

print(rta)

x=range(500)

ya=[]

for j in range(500):
    ya.append(fa(j))
    
plt.plot(x, ya, 'b.', linewidth = 2, label = 'binomial')
plt.legend(loc = 'best')
plt.grid(1)

plt.show

#b)
import matplotlib.pyplot as plt
import numpy as np

prod=1

for i in range(50):
    prod=prod*(4500-i)*((5000-i)**(-1))
print(prod)

#prod=0.005014411795816377
    
def fb(k):
    if k==0:
        return 0.005014411795816377
    else:
        return fb(k-1)*(50-(k-1))*(500-(k-1))*(k*(4451+k-1))**(-1)

sumb=0

for i in range(10):
    sumb=sumb+fb(i)
print(sumb)    

rtb=1-sumb

print(rtb)

xb=range(50)
yb=[]

for j in range(50):
    yb.append(fb(j))
    
plt.plot(xb, yb, 'r.', linewidth = 2, label = 'hipergeometrica')
plt.legend(loc = 'best')
plt.grid(1)

plt.show


#c: Hago el b con otro valor de N y pruebo . 

#Tengo que lograr que (rta-rtb)/rta < 0.001, despejo rtb y obtengo 0.0310689 < rtb =< 0.0311

N=1.15*10**6

#m=N*0.01

prod=1

for i in range(int(N*0.01)):
    prod=prod*(N-500-i)*((N-i)**(-1))
print(prod)

  
def fb(k):
    if k==0:
        return prod
    else:
        return fb(k-1)*(N*0.01-(k-1))*(500-(k-1))*(k*(N-500-0.01*N+k))**(-1)

sumb=0

for i in range(10):
    sumb=sumb+fb(i)
print(sumb)

rtb=1-sumb

print(rtb)

#%%
#Ejer 10: Messi

import math
import matplotlib.pyplot as plt
import numpy as np


f = lambda n: 1-n*(0.182)*(1-0.182)**(n-1)-(1-0.182)**n #1-prob de hacer 1 gol - prob de hacer 0 goles

x=range(1000)

y=[]

for n in range(1000):
    y.append(f(n))
    
plt.plot(x, y, 'b.')
plt.grid(1)
plt.plot(x, 0.9*np.ones(len(x)), 'r-')

plt.show


def binomial(k,num,p):
    if k==0:
        return (1-p)**num
    else:
        return p*(1-p)**(-1)*(num-k+1)*k**(-1)*binomial(k-1,num,p)
 

#%%



