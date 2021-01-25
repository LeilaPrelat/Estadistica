# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:21:29 2019

@author: Usuario
"""

#Ejer 6) Teorema central del límite: 

#a) Una distribución binomial de parámetros n y p es aproximadamente normal para grandes valores de n, y p no demasiado cercano a 0 o a 1

import matplotlib.pyplot as plt
import numpy as np
import statistics as stats

def binomial(k,num,p):
    if k==0:
        return (1-p)**num
    else:
        return p*(1-p)**(-1)*(num-k+1)*k**(-1)*binomial(k-1,num,p)
    
def gaussiana(x,mu,sigma):
    return sigma**(-1)*(2*np.pi)**(-1/2)*np.e**(-(x-mu)**2*(2*sigma)**(-2))
    
num1=5
p1=0.2
   
y1=[] #Binomial 1
 
for k in range(num1+1):
    y1.append(binomial(k,num1,p1))
    
mu1=num1*p1
sigma1=(num1*p1*(1-p1))**(1/2)
    
g1=[] #Gaussiana 1

for k in range(num1+1):
    g1.append(gaussiana(k,mu1,sigma1))

num2=30
p2=0.4
   
y2=[] #Binomial 2
 
for k in range(num2+1):
    y2.append(binomial(k,num2,p2))
    
mu2=num2*p2
sigma2=(num2*p2*(1-p2))**(1/2)
    
g2=[] #Gausseana 2

for x in range(num2+1):
    g2.append(gaussiana(x,mu2,sigma2))

x1=np.linspace(0,num1,num1+1)
x2=np.linspace(0,num2,num2+1)

plt.figure()
#
plt.plot(x1, y1, 'r.', label='Distribución Binomial con n=%i y probabilidad=%.1f' %(num1,p1))
plt.plot(x2, y2, 'b.', label='Distribución Binomial con n=%i y probabilidad=%.1f' %(num2,p2))
plt.plot(x1, g1, 'g.', label=r'Distribución Gaussiana con $\mu$=%i y sigma=%.2f' %(mu1,sigma1))
plt.plot(x2, g2, 'k.', label=r'Distribución Gaussiana con $\mu$=%i y sigma=%.2f' %(mu2,sigma2))

plt.title('Ejercicio 6.a')
plt.legend(loc='best')
plt.grid(1)

#%%

#Ejer 6) Teorema central del límite:

#b) Una distribución de Poisson con parámetro λ es aproximadamente normal para grandes valores de λ.

import matplotlib.pyplot as plt
import numpy as np

def poisson(k,nu):
    if k==0:
        return (np.e)**(-nu)
    else: 
        return nu*(k**(-1))*poisson(k-1,nu)
    
def gaussiana(x,mu,sigma):
    return sigma**(-1)*(2*np.pi)**(-1/2)*np.e**(-(x-mu)**2*(2*sigma)**(-2))

N=80

poi1=[] #Poisson 1

nu1=4

for k in range(N):
    poi1.append(poisson(k,nu1))
    
mu1=nu1
sigma1=nu1**(1/2)
    
g1=[] #Gaussiana 1

for l in range(N):
    g1.append(gaussiana(l,mu1,sigma1))
    
poi2=[] #Poisson 2

nu2=10

for k in range(N):
    poi2.append(poisson(k,nu2))

mu2=nu2
sigma2=nu2**(1/2)
    
g2=[] #Gaussiana 2

for l in range(N):
    g2.append(gaussiana(l,mu2,sigma2))
    
poi3=[] #Poisson 3

nu3=40

for k in range(N):
    poi3.append(poisson(k,nu3))   
    
mu3=nu3
sigma3=nu3**(1/2)
    
g3=[] #Gaussiana 3

for l in range(N):
    g3.append(gaussiana(l,mu3,sigma3))

x=np.linspace(0,N-1,N)

plt.plot(x, poi1, 'r.', label=r'Distribución de Poisson con $\nu$=%i' %(nu1))
plt.plot(x, poi2, 'b.', label=r'Distribución de Poisson con $\nu$=%i' %(nu2))
plt.plot(x, poi3, 'g.', label=r'Distribución de Poisson con $\nu$=%i' %(nu3))

plt.plot(x, g1, 'm.', label=r'Distribución Gaussiana con $\mu$=%i y $\sigma$=%.2f' %(mu1,sigma1))
plt.plot(x, g2, 'y.', label=r'Distribución Gaussiana con $\mu$=%i y $\sigma$=%.2f' %(mu2,sigma2))
plt.plot(x, g3, 'k.', label=r'Distribución Gaussiana con $\mu$=%i y $\sigma$=%.2f' %(mu3,sigma3))

plt.title('Ejercicio 6.b')
plt.legend(loc='best')
plt.grid(1)

#%%

#Ejer 8

#Variables independientes y randoms Xi: 

from random import uniform  
    
y=[]  

N=100

for k in range(100):
    x=[]
    for i in range(N):
        x.append(uniform(0,1))   
    z=0   
    for j in range(len(x)):
        z=z+x[j]
    y.append(z)
    
mu=0.5
sigma=1/12

resultados=[]

for l in range(len(y)):
    resultados.append(gaussiana(y[l],mu,sigma))


x2=np.linspace(0,N-1,N)
   
plt.plot(resultados, y, 'k.', label='Distribución Gaussiana con mu=%i' %(mu))


plt.plot(y, resultados, 'b.', label='Distribución Gaussiana con mu=%i' %(mu))

#%%

from random import uniform  
import numpy as np
import statistics as stats

def randoms(n):
    y=[]
    for i in range(int(n)):
        y.append(uniform(0,1))
    return y

def z(n):
    return (np.sum(randoms(n))-n*0.5)*(n*1)**(-1/2)    

x=np.linspace(500,2000,1501)

resultados=[]

for j in x:
    resultados.append(z(int(j)))
    print (j)
    
plt.plot(x, resultados, 'k.')

#%%

#Ejer 3

tita=np.linspace(0.6,1.6,100)
y=[]
for i in range(len(tita)):
    y.append(np.tan(tita[i]))

plt.plot(tita, y, 'k.')
plt.grid(1)

#%%

def Bmax(Fmax,C,epsilon):
    return (Fmax-C)*((epsilon*10*60)**(-1))

Fmax=2384
C=69
epsilon=0.1

def Bmin(Fmin,C,epsilon):
    return (Fmin-C)*((epsilon*10*60)**(-1))

Fmin=992

def Delta(b,i,*argv):
    inc=[]
    for arg in argv:
        inc.append(arg)
    if b==1:
        inc[i]=inc[i]+argv[i]*10**(-14)
    if b==0:
        inc[i]=inc[i]-argv[i]*10**(-14)
    return inc     
        
def gradiente(f,*argv):
    derivada=[]
    for i in range(len(argv)):
        derivada.append((f(*Delta(1,i,*argv))-f(*Delta(0,i,*argv)))/(2*argv[i]*10**(-14)))
    return derivada

def Error(a,f,*argv): #poner primero las variables y despues los errores: x1,x2,x3,e1,e2,e3
    if len(argv)%2==0:
        variables=[]
        for i in range(int(len(argv)/2)):
            variables.append(argv[i])
        errores=[]
        for j in range(int(len(argv)/2),int(len(argv))):
            errores.append(argv[j])
        if a==1:   #x1,x2,x3 no son independientes 
            cov=0
            for i in range(int(len(argv)/2)):
                for j in range(int(len(argv)/2)):
                    cov=cov+gradiente(f,*variables)[i]*gradiente(f,*variables)[j]*errores[i]*errores[j]  
        if a==0:  #x1,x2,x3 son independientes
            cov=0
            for i in range(int(len(argv)/2)):
                cov=cov+gradiente(f,*variables)[i]*gradiente(f,*variables)[i]*errores[i]*errores[i]
        return cov 
    else: 
        print('Las variables y los errores de f no tienen la misma dimensión')
        
#Armamos dos funciones con los datos: f(x1,x2,x3) y g(y1,y2,y3) 
def Covarianza(a,f,g,*argv): #En argv poner primero las variables y despues los errores: x1,x2,x3,ex1,ex2,ex3,y1,y2,y3,ey1,ey2,ey3
    if len(argv)%4==0: 
        variablesdef=[]
        for i in range(int(len(argv)/4)):
            variablesdef.append(argv[i])
        erroresdex=[]
        for k in range(int(len(argv)/4),int(len(argv)/2)):
            erroresdex.append(argv[k])
        variablesdeg=[]
        for j in range(int(len(argv)/2),int(len(argv)*3/4)):
            variablesdeg.append(argv[j])
        erroresdey=[]
        for l in range(int(len(argv)*3/4),int(len(argv))):
            erroresdey.append(argv[l])
        print (variablesdef,erroresdex,variablesdeg,erroresdey)
        if a==1:   #x1,x2,x3 no son independientes  
            cov=0
            for i in range(int(len(argv)/4)):
                for j in range(int(len(argv)/4)):
                    cov=cov+gradiente(f,*variablesdef)[i]*gradiente(g,*variablesdeg)[j]*erroresdex[i]*erroresdey[j]
        if a==0:
            cov=0
            for i in range(int(len(argv)/4)):
                cov=cov+gradiente(f,*variablesdef)[i]*gradiente(g,*variablesdeg)[i]*erroresdex[i]*erroresdey[i] 
        return cov

#%%

#Ejer 5
    
#tc=tf=10*60
def Bmax(Fmax,C,epsilon):
    return (Fmax-C)*((epsilon*10*60)**(-1))

Fmax=2384
C=69
epsilon=0.1

def Bmin(Fmin,C,epsilon):
    return (Fmin-C)*((epsilon*10*60)**(-1))

Fmin=992


ErrorBmax=Error3(Bmax,Fmax,Fmax**(1/2),C,C**(1/2),epsilon,epsilon*0.06,0)

ErrorBmin=Error3(Bmin,Fmin,Fmin**(1/2),C,C**(1/2),epsilon,epsilon*0.06,0)

#Cov2=covarianza(Bmax,Fmax,Fmax**(1/2),C,C**(1/2),epsilon,epsilon*0.06,Bmin,Fmin,Fmin**(1/2),C,C**(1/2),epsilon,epsilon*0.06)
#
#Cov1=covarianza(Bmax,Fmax,Fmax**(1/2),C,C**(1/2),epsilon,epsilon*0.06,Bmax,Fmax,Fmax**(1/2),C,C**(1/2),epsilon,epsilon*0.06)
#
#Cov3=covarianza(Bmin,Fmin,Fmin**(1/2),C,C**(1/2),epsilon,epsilon*0.06,Bmin,Fmin,Fmin**(1/2),C,C**(1/2),epsilon,epsilon*0.06)


#%%

#Ejer 8: Cuadrados mínimos. Los valores en el eje x no tienen error mientras que los 
#valor en el eje y tienen todos el mismo error (sigma**2)

#Ajuste lineal: a1+a2*x. a1 y a2 se obtienen de minimizar Sn

#from sympy import Derivative, diff, simplify

def S(x,y,a1,a2):
    if len(x)==len(y):
        s=0
        for i in range(len(x)):
            s=s+(y[i]-(a1+a2*x[i]))**2
        return s
    else: 
        print('x e y no tienen las mismas dimensiones')

#def S_a1(x,y,a1,a2):
#    h1=0.005
#    return (S(x,y,a1+1,a2)-S(x,y,a1,a2))*(h1**(-1))

Datos=np.transpose(np.loadtxt('datosejer8guia5.txt', delimiter='\t'))

x=Datos[0]
y=Datos[1]

x2=[]
for i in range(len(x)):
    x2.append(x[i]**2)
#Obs: np.sum(x2)=np.sum(x**2)
    
xy=[]

for i in range(len(x)):
    xy.append(x[i]*y[i])
#Obs: np.sum(xy)=np.sum(x*y)    
    
Delta=len(x)*np.sum(x2)-(np.sum(x))**2

a1=(np.sum(x2)*np.sum(y)-np.sum(x)*np.sum(xy))*(Delta**(-1))
a2=(len(x)*np.sum(xy)-np.sum(x)*np.sum(y))*(Delta**(-1))

sigma=0.3

sigmaa1=((sigma**2)*(Delta**(-1))*np.sum(x2))**(1/2)
sigmaa2=((sigma**2)*(Delta**(-1))*len(x))**(1/2)

plt.figure()
plt.plot(x, y, 'r.', label='Datos')
plt.errorbar(x,y,sigma,linestyle='None',label='Error')
plt.title('Ejercicio 8.d guía 5')
plt.legend(loc='best')
plt.grid(1)


#FALTA EL AJUSTE LINEAL

#%%

#Ejer 9

Datos=np.transpose(np.loadtxt('Datos ejer 8 guía 5.txt', delimiter='\t'))

x=Datos[0]
y=Datos[1]

def gaussiana(x,mu,sigma):
    return sigma**(-1)*(2*np.pi)**(-1/2)*np.e**(-(x-mu)**2*(2*sigma)**(-2))


sigma=0.3
y2=[]
for i in range(len(x)):
    a1=uniform(0.731,2.173)
    a2=uniform(0.513,1.085)
    y2.append(gaussiana(x[i],a1+a2*x[i],sigma))


plt.figure()
plt.plot(x, y2, 'r.', label='Datos')
plt.plot(x, y, 'b.', label='Datos')








