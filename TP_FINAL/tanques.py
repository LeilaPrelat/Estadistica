# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:19:35 2019

@author: Usuario
"""
from random import uniform  
import numpy as np
import matplotlib.pyplot as plt
#Adivinemos la cantidad total de tanques. 

#incógnita es n=25


def numerodetanque(n):
    return int(uniform(1,n+1))

def distribuciondem(m,k,n1): #distribución de probabilidad de m variando k 
    if k==1:
        return 1/n1
    else:
        return k*(m-k+1)*distribuciondem(m,k-1,n1)*((k-1)*(n1-k+1))**(-1)


def distdemax(n,k,x):
    return n**(-1)*k*(x/(n-1)-1/(n-1))**(k-1)

def experimento(N,k,a,b):
    maxi=[]
    for i in range(N):
        x=[]
        for i in range(k):
            x.append(int(uniform(a,b+1)))  #patentes observadas (de las más nuevas) #k=uniform(0,n) #puedo observar cero patentes nuevas y puedo observar todas las patentes nuevas que existen en circulacion (n)
        maxi.append(np.max(x))
    return maxi


n=3000
k=2000

x=[]
for i in range(k):
    x.append(int(uniform(1,n+1)))
y=[]
for i in range(k):    
    y.append(distdemax(n,k,x[i]))
    
plt.figure()
plt.plot(x,y,'r.')

#Teorema central del límite:
 
def gaussiana(x,mu,sigma):
    return sigma**(-1)*(2*np.pi)**(-1/2)*np.e**(-(x-mu)**2*(2*sigma)**(-2))    

x2=x

y2=[]
for i in range(k):    
    y2.append(gaussiana(x[i],np.max(x),(np.max(x)-k)/k))

plt.plot(x2,y2,'b.')
#m=np.max(x)
#print(distribuciondem(23,15,25))

#print(np.max(x)+int((np.max(x)-k)/k), distribuciondem(np.max(x),k,np.max(x)+int((np.max(x)-k)/k)))



















