# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:21:29 2019

@author: Usuario
"""

#Ejer 6) Teorema central del límite: 

#a) Una distribución binomial de parámetros n y p es aproximadamente normal para grandes valores de n, y p no demasiado cercano a 0 o a 1

import matplotlib.pyplot as plt
import numpy as np

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
plt.plot(x1, g1, 'g.', label=r'Distribución Gaussiana con $\mu$=%i y $\sigma$=%.2f' %(mu1,sigma1))
plt.plot(x2, g2, 'k.', label=r'Distribución Gaussiana con $\mu$=%i y $\sigma=%.2f' %(mu2,sigma2))

plt.title('Ejercicio 6.a guía 4')
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

plt.title('Ejercicio 6.b guía 4')
plt.legend(loc='best')
plt.grid(1)

#%%

#Ejer 7: Aproximación de la Binomial con la función acumulativa normal canónica

import numpy as np

def binomial(k,num,p):
    if k==0:
        return (1-p)**num
    else:
        return p*(1-p)**(-1)*(num-k+1)*k**(-1)*binomial(k-1,num,p)
    
p=1/3 #respuesta correcta

num=100 #cien preguntas 

#k va desde 40 hasta 100 (cantidad de éxitos)

rta=0
for i in range(40):
    rta=rta+binomial(i,num,p)

print('Probabilidad de aprobar respondiendo al azar:', 1-rta)

#
#def funcioncaracteristica(t,mu,sigma):
#    return [np.cos(mu*t)*np.e**(-0.5*(sigma*t)**2),np.sin(mu*t)*np.e**(-0.5*(sigma*t)**2)]

from scipy import special

#mu=num*p
#sigma=(num*p*(1-p))**(1/2)

a=40
b=100

t2=(b-num*p+0.5)*(num*p*(1-p))**(-1/2)

t1=(a-num*p-0.5)*(num*p*(1-p))**(-1/2)

#print(funcioncaracteristica(t2,mu,sigma)[0] - funcioncaracteristica(t1,mu,sigma)[0])
#
#print(funcioncaracteristica(t2,mu,sigma)[1] - funcioncaracteristica(t1,mu,sigma)[1])

mu2=0
sigma2=1

def phi(t):
    return 0.5*(1+ special.erf((t-mu2)*(sigma2*np.sqrt(2))**(-1)))
    
print ('Probabilidad de aprobar respondiendo al azar con aproximación:',phi(t2)-phi(t1))
    
#%%

#Ejer 8

from random import uniform  
import numpy as np

def randoms(n):
    y=[]
    for i in range(int(n)):
        y.append(uniform(0,1))
    return y

#El histograma de las medias (para muestras grandes) tiene distribución gaussiana: teorema central
#El promedio de las medias (la media de las medias) es la media de la muestra. 
    
#Primero hallamos el promedio de muuuchas muestras=la media de una muestra grande.
#np.mean(randoms(1000)) es 0.505819 y np.mean(resultados) con x=np.linspace(500,1000,501) es 0.500466
#entonces el mu que ponemos es 0.5
    
#Ahora OJO no es lo mismo para el sigma: 
#(stats.pvariance(randoms(1000)))**(1/2) es 0.290099 y (stats.pvariance(resultados))**(1/2) con x=np.linspace(500,1000,501) es 0.010444
#El sigma de los promedios es más chico 
    
mu=0.5
sigma=0.29

def f(n): #función generadora de una distribución N(0,1)
    return (np.mean(randoms(n))-mu)*n*(sigma*n**(1/2))**(-1)

resultados=[]

#x=np.linspace(500,1000,501)

x=np.linspace(1,500,500)

for j in x:
    resultados.append(f(j))
  
num_bins=50
bins1=np.linspace(-0,1,20)
bins2=np.linspace(-10,10,21)

plt.figure()
n, bins, patches = plt.hist(randoms(len(x)), bins1, normed=1, facecolor='red', alpha=1,  edgecolor='black', linewidth=1.2, label='Histograma de la distribución uniforme con %i datos'%(len(x)))
plt.title('Ejercicio 8 guía 4')
plt.legend(loc='best')
plt.grid(1)

plt.figure()
n, bins, patches = plt.hist(resultados, bins2, normed=1, facecolor='green', alpha=1,  edgecolor='black', linewidth=1.2, label='Histograma de f(z) con %i datos'%(len(x)))

resultados2=[]
for l in bins:
    resultados2.append(gaussiana(l,0,1))

plt.plot(bins, resultados2, 'm.', label=r'Distribución Gaussiana con $\mu$=%i y $\sigma$=%.2f' %(mu,sigma))

plt.title('Ejercicio 8 guía 4')
plt.legend(loc='best')
plt.grid(1)

#%%

#Ejer 8 ítem d) No se puede usar el teorema central del límite para la distribución de Cauchy.

def cauchy(x):
    return (np.pi*(1+x**2))**(-1)

def muchoscauchy(n):
    cauchies=[]
    for i in range(n):
        cauchies.append(cauchy(x))
    return cauchies


x=np.linspace(-500,500,1001)
y=[]
for i in x:
    y.append(cauchy(i))
       
mu=np.mean(y)
sigma=np.std(y)

n=500
muestra=[]
for i in range(n):
    muestra.append(int(uniform(0,len(x))))

y2=[]

for i in range(n):
    y2.append(y[muestra[i]])
    
plt.figure()
plt.plot(muestra, y2, 'b.')
plt.title('Ejercicio 8 guía 4') 

#%%

#Ejer 9: Primero vamos a aproximar la hipergeométrica por una binomial (el N -total de la población- es mucho mayor que el n -gente que vamos a encuestar- como en la guía 2 ejer 8)

#Se desea conocer la intención de voto (esperanza) con un nivel de confianza del 95% 
   
p=0.45 #ítem a)
num=9900

a=0
b=9900*0.95

t2=(b-num*p+0.5)*(num*p*(1-p))**(-1/2)
t1=(a-num*p-0.5)*(num*p*(1-p))**(-1/2)

mu2=0
sigma2=1

def phi(t):
    return 0.5*(1+ special.erf((t-mu2)*(sigma2*np.sqrt(2))**(-1)))
    
print (phi(t2)-phi(t1))


#A partir del nivel de confianza calculo la cantidad de sigmas que necesito 

mu=num*p

sigma=(num*p*(1-p))**(1/2)

#%%

#Ejer 10: 

import numpy as np
import statistics as stats

def covarianza(x,y):
    if len(x)==len(y):
        cov=0
        for i in range(len(x)):
            cov=cov+(x[i]-np.mean(x))*(y[i]-np.mean(y))/len(x)
        return cov
    else:
        print('x e y no tienen la misma dimensión')
    
def correlacion(x,y):
    if len(x)==len(y):
        return covarianza(x,y)*(stats.pstdev(x)*stats.pstdev(y))**(-1)
    else:
        print('x e y no tienen la misma dimensión')
    

promediosx=[]
promediosy=[]

varx=[]
vary=[]

correlaciondetodos=[]

j=np.linspace(1,4,4) #Tengo 4 datos

for i in j:
    
    Datos=np.transpose(np.loadtxt('Datos%i.txt'%(int(i)), delimiter='\t'))

    xi=Datos[0]
    yi=Datos[1]
    
    plt.figure()
    plt.plot(xi, yi, 'b.', label='Datos%i con promedios %.2f y %.2f, varianza %.2f y %.2f y correlacion  %.2f'%(int(i),np.mean(xi),np.mean(yi),np.var(xi),np.var(yi),correlacion(xi,yi)))
    plt.title('Ejercicio 10 guía 4')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y') 
    plt.legend(loc='best')
    plt.grid(1)
    
    promediosx.append(np.mean(xi))
    promediosy.append(np.mean(yi))
    
    varx.append(np.var(xi))
    vary.append(np.var(yi))
    
    correlaciondetodos.append(correlacion(xi,yi))

print(promediosx)
print(promediosy)

print(varx)
print(vary)

print(correlaciondetodos)

#%%

#Ejer 11: Distribución multinormal 

#Creamos la matriz de covarianza para dos variables, una matriz de 2x2:
    
def V(x,y):
    matrizdecovarianza=0
    matrizdecovarianza.append([stats.pvariance(x),covarianza(x,y)])
    matrizdecovarianza.append([covarianza(x,y),stats.pvariance(y)])
    return matrizdecovarianza


#%%
#ítem e: elipses de covarianza
    
def Q(x1,mu1,sigma1,x2,mu2,sigma2,ro):
    if -1<=ro<=1:
        return (((x1-mu1)/sigma1)**2+((x2-mu2)/sigma2)**2-2*ro*((x1-mu1)/sigma1)*((x2-mu2)/sigma2))*((1-ro**2)**(-1))
    else:
        print('la correlación $ro(x1,x2)$ no está entre -1 y 1')

sigma1=1
sigma2=2
mu1=0
mu2=0
ro=-0.1


n=500
x1=np.arange(-n*sigma1,n*sigma1)
x2=np.arange(-n*sigma1,n*sigma1)
#
x1t=[]
x2t=[]
for i in x1:
    for j in x2:
            if Q(i,mu1,sigma1,j,mu2,sigma2,ro)==1:
                x1t.append(i)
                x2t.append(j)

X1,X2=np.meshgrid(x1,x2)
base_contour=1
    
cs = plt.contour(X1,X2,Q(X1,mu1,sigma1,X2,mu2,sigma2,ro),3)
plt.clabel(cs, inline=1, fontsize=10)

#plt.plot(x1t,x2t,'g-')
plt.plot(x1,np.ones(len(x1))*mu2,'r.')
plt.plot(np.ones(len(x2))*mu1,x2,'b.')

plt.title('Ejercicio 11 guía 4, elipses de covarianza')
#plt.xlim([-n*mu1*2,n*mu1*2])
#plt.ylim([-n*mu2*2,n*mu2*2])
plt.legend(loc='best')
plt.grid(1)
plt.show()


#%%



sigma1=1
sigma2=1
mu1=0
mu2=0
ro=0.8





























