# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:50:23 2019

@author: Usuario

"""


#%%

from random import uniform   
import matplotlib.pyplot as plt
import numpy as np

x=[]

N=10**3

def Normal(mu,sigma,x):
    return sigma**(-1)*(2*np.pi)**(-0.5)*np.e**((-0.5*((x-mu)/sigma)**2))

mu=3

sigma=8

y=[]

x=np.linspace(-N/2,N/2,N)

for j in x:
    y.append(Normal(mu,sigma,j)) 
    
plt.plot(x, y, 'r.')

plt.legend(loc='best')
plt.grid(1)
plt.figure()

#%%

#Ejer 7)a)

from random import uniform   
import matplotlib.pyplot as plt
import numpy as np

x1=[]

N=10**3

for i in range(N):
    
    x1.append(uniform(0,1))


y=[]

for j in range(N):
    y.append(np.e**(x1[j]))

#x2=np.linspace(1,1/np.e,N)

x2=[]

for k in range(N):
    x2.append((y[k])**(-1))
    

plt.plot(x1, 1*np.ones(len(x1)), 'b.', label='Distribución uniforme')
plt.plot(y, x2, 'r.', label='Cambio de variable $y=e^x$')

plt.title('Ejercicio 7 con N=%i' %(N))

plt.legend(loc='best')
plt.grid(1)
plt.figure()

#%%

#Ejer 7)b)

from random import uniform   
import matplotlib.pyplot as plt
import numpy as np


x1=[]

N=10**3

for i in range(N):
    
    x1.append(uniform(0,1))


y=[]

for j in range(N):
    y.append(np.log((x1[j])))

#x2=np.linspace(1,1/np.e,N)

x2=[]

for k in range(N):
    x2.append(np.e**(y[k]))
    

plt.plot(x1, 1*np.ones(len(x1)), 'b.', label='Distribución uniforme')
plt.plot(y, x2, 'r.', label='Cambio de variable $y=e^x$')

plt.title('Ejercicio 7 con N=%i' %(N))

plt.legend(loc='best')
plt.grid(1)
plt.figure()

#%%

#Ejer 8)


N=10**3

M=10**3

u=[]

for k in range(M):

    x=[]
    
    for i in range(N):
    
        x.append(uniform(0,1))

    u.append(np.max(x))

y=[]
for j in range(N):
    y.append((x[j])**(N-1))

print(len(u))

plt.plot(u, y, 'r.', label='Cambio de variable $y=e^x$')

plt.title('Ejercicio 7 con N=%i' %(N))

plt.legend(loc='best')
plt.grid(1)
plt.figure()

#%%

#Problemas computacionales

#Ejer 9

import numpy as np

def cauchy(x):
    return (np.pi*(1+x**2))**(-1)
    
def gaussiana(x,mu,sigma):
    return sigma**(-1)*(2*np.pi)**(-1/2)*np.e**(-(x-mu)**2*(2*sigma)**(-2))


#ítem a) 
    
x=np.linspace(-10,10,200)

yc=[]
for k in x:
    yc.append(cauchy(k))
    
mu=0
sigma=0.75

yg=[]
for k in x:
    yg.append(gaussiana(k,mu,sigma)*cauchy(0)*(gaussiana(0,mu,sigma))**(-1))

plt.figure()

plt.plot(x, yc, 'b.', label='Distribución de Cauchy')
plt.plot(x, yg, 'g.', label=r'Distribución Gaussiana normalizada con $\mu$=%i y $\sigma$=%.2f' %(mu,sigma))

plt.title('Ejercicio 9.a guía 3')
plt.legend(loc='best')
plt.grid(1)

#%%
#ítem b) Se pueden construir las colas de Cauchy con suma de dos gaussianas

def sumadegaussianas(a,x,mu1,sigma1,mu2,sigma2):
    return a*gaussiana(x,mu1,sigma1)+(1-a)*gaussiana(x,mu2,sigma2)

a=0.5
mu1=0
sigma1=0.75
mu2=0
sigma2=3

x=np.linspace(-20,20,400)

ys=[]
for k in x:
    ys.append(sumadegaussianas(a,k,mu1,sigma1,mu2,sigma2)*cauchy(0)*(sumadegaussianas(a,0,mu1,sigma1,mu2,sigma2))**(-1))
    
yc=[]
for k in x:
    yc.append(cauchy(k))

plt.figure()
    
plt.plot(x, ys, 'r.', label=r'Suma de gaussianas normalizada con $\mu1$=%i, $\sigma1$=%.2f, $\mu2$=%i y $\sigma2$=%.2f' %(mu1,sigma1,mu2,sigma2))
plt.plot(x, yc, 'b.', label='Distribución de Cauchy')

plt.title('Ejercicio 9.b guía 3')
plt.legend(loc='best')
plt.grid(1)

#%%

#Ejer10: Montecarlo. Generar números pseudoaleatorios con distribución f(t)-desconocida analíticamente- en un dominio [a,b] y valor máximo fm (f acotada en [a,b]) 

def cauchy(x):
    return (np.pi*(1+x**2))**(-1)

def u(a,b,y):
    return a+(b-a)*y

def v(fm,z):
    return fm*z

#Dominio:
a=5
b=10

#máximo de la función: 
fm=cauchy(0) 

muestra=[]


N=10**5

for i in range(N):
    y=uniform(0,1)
    z=uniform(0,1)

    if v(fm,z)<=cauchy(u(a,b,y)):
        muestra.append(u(a,b,y))
 

ejey=[]    
for j in range(len(muestra)):
    ejey.append(v(fm,muestra[j]))
    
plt.figure()    
plt.plot(muestra, ejey, 'b.', label='Distribución de Cauchy')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

