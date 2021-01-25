# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:03:03 2019

@author: Usuario
"""
from random import uniform  
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def experimento(N,k,a,b):
    maxi=[]
    for i in range(N):
        x=[]
        for i in range(k):
            x.append(int(uniform(a,b+1)))  #patentes observadas (de las más nuevas) #k=uniform(0,n) #puedo observar cero patentes nuevas y puedo observar todas las patentes nuevas que existen en circulacion (n)
        maxi.append(np.max(x))
    return maxi

#Tomo desde la patente AD200AA en adelante: patentes(['A','D',2,0,0,'A','A'])=2163201. 
#La ultima patente que existe= 2428510 entonces 2428510-2163201=265309=n1

a=2163201
b=2428510

#Cambio el origen, números más chicos

a1=1
b1=265309

k=1000
N=1000

y=experimento(N,k,a1,b1)

plt.figure()

bins=np.linspace(np.min(y),np.max(y),40)
n, bins, patches = plt.hist(y, bins, normed=1, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i intentos' %(N))

def distribuciondem(m,k,n1): #distribución de probabilidad de m variando k 
    if k==1:
        return 1/n1
    else:
        return k*(m-k+1)*distribuciondem(m,k-1,n1)*((k-1)*(n1-k+1))**(-1)


def funciondem(m,k):
    return (m-1)*funciondem(m-1,k)*((m-k)**(-1))

#No se puede comparar con la teórica porque números grandes
n1=b1-a1
m=np.linspace(a1,b1,b1-a1+1) #m>=k, n1>=k, m<=n1
y2=[]
for i in m:
    y2.append(distribuciondem(i,k,n1))

plt.plot(m,y2,'r.')
plt.grid(1)
    
#Para el ítem b) uso los rtas del experimento (valores de m)
    
#%% 
    
#item b) 
    
def distribuciondem(m,k,n1): #distribución de probabilidad de m variando k 
    if k==1:
        return 1/n1
    else:
        return k*(m-k+1)*distribuciondem(m,k-1,n1)*((k-1)*(n1-k+1))**(-1)

a1=1
b1=265309
n1=b1-a1
k=5000 #observo 25 patentes de las más grandes y me quedo con el máximo 

#m=np.linspace(a1,b1,b1-a1+1) #m>=k, n1>=k, m<=n1. Deberia barrer n para cada m fijo. 

m=265109

rangoden=np.linspace(m,m+int((m-k)/k),m+int((m-k)/k)-m+1)
y2=0
for i in rangoden:
    y2=likelihood(m,k,i)+y2
    print(y2)

def likelihood(m,k,n):
    return np.log(k)+(n-k)*np.log(n-k)+(m-1)*np.log(m-1)-n*np.log(n)-(m-k)*np.log(m-k)

#Opcion de los histogramas

def experimento(N,k,a,b):
    maxi=[]
    for i in range(N):
        x=[]
        for i in range(k):
            x.append(int(uniform(a,b+1)))  #patentes observadas (de las más nuevas) #k=uniform(0,n) #puedo observar cero patentes nuevas y puedo observar todas las patentes nuevas que existen en circulacion (n)
        maxi.append(np.max(x))
    return maxi

a=2163201
b=2428510

y=experimento(N,k,a1,b1)

#if y[i]==b: #Hallar cuántas veces aparece y==b en todos los experimentos. Repertir más veces y hacer histograma
    
#%%

listatotaln=[]

m=265109
probmfijodadon=[]
ncotasup=265109+3000
listan=np.linspace(m,ncotasup,ncotasup-m+1)
for n in listan:
    a1=1
   
    k=277
    N=200
    
    y=experimento(N,k,a1,n)  
    
    cantidaddem=len(np.where(np.array(y)==m)[0])
    probmfijodadon.append(cantidaddem/N)
    print (n-listan[0])

    listatotaln=listatotaln+list(n*np.ones(cantidaddem))

plt.figure()

bins=np.linspace(m,ncotasup,40)
#n, bins, patches = plt.hist(probmfijodadon, bins, normed=1, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i intentos' %(N))
n, bins, patches = plt.hist(listatotaln, bins, normed=1, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i intentos' %(N))
plt.grid(1)


plt.figure()

plt.plot(listan,probmfijodadon,'r.')    
plt.grid(1)
#%%
def distdemax(n,k,x): #x son las patentes 
    return n**(-1)*k*(x/(n-1)-1/(n-1))**(k-1)
        
k=277
m=265109
probmfijodadon=[]
ncotasup=265109+264+800
listan=np.linspace(m,ncotasup,ncotasup-m+1)
    
probmdadon2=distdemax(listan,k,m)

normaliz=np.sum(probmdadon2)

probndadom=distdemax(listan,k,m)/normaliz

plt.figure()

plt.plot(listan,probndadom,'b.')    
plt.grid(1)

#%%

def distribuciondem(m,k,n1): #distribución de probabilidad de m variando k 
    if k==1:
        return 1/n1
    else:
        return k*(m-k+1)*distribuciondem(m,k-1,n1)*((k-1)*(n1-k+1))**(-1)
k=277
m=265109
probmfijodadon=[]
ncotasup=10**6
listan=np.linspace(m,ncotasup,ncotasup-m+1)
"""
for n in listan:
    a1=1
   
    k=277
    N=1500
    
    y=distribuciondem(m,k,n)  
    
    cantidaddem=len(np.where(np.array(y)==m)[0])
    probmfijodadon.append(cantidaddem/N)
    print (n-listan[0])

    listatotaln=listatotaln+list(n*np.ones(cantidaddem))
"""


y=distribuciondem(m,k,listan)
ynorm=y/np.sum(y)

esperanza=np.sum(listan*ynorm)
std=(np.sum((listan-esperanza)**2*ynorm))**0.5

esperteor=(m-1)*(k-1)/(k-2)

stdteor=((k-1)*(m-1)*(m-k+1)*(k-2)**-2/(k-3))**0.5

plt.figure()

plt.plot(listan,ynorm,'b.')    
plt.grid(1)
    

#%%








