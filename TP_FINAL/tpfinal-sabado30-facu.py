# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:54:07 2019

@author: leila
"""
from random import uniform  
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


#%%

#Ejer 2

nummuestras=7 #cantidad de txt 

ABC=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
abc=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def patentes(lista):
    patente=[]
    for l in range(7):
        patente.append(lista[l])
    a,b,c,d,e,f,g=patente
    rta=[]
    for m in range(len(ABC)):
        if a==ABC[m]:
            rta0=m+1
        if b==ABC[m]:
            rta1=m+1
    rta.append(rta0)
    rta.append(rta1)
    rta.append(int(c))
    rta.append(int(d))
    rta.append(int(e))
    for m in range(len(ABC)):
        if f==ABC[m]:
            rta5=m+1 
        if g==ABC[m]:
            rta6=m+1 
    rta.append(rta5)
    rta.append(rta6)
    num=rta[6]*26**0+(rta[5]-1)*26**1+rta[4]*26**2+rta[3]*10*26**2+rta[2]*10*10*26**2+(rta[1]-1)*10**3*(26**2)+(rta[0]-1)*10**3*(26**3)
    return (num,rta)

num, rta=patentes('ZZ999ZZ')
print(num)

#Separado por barrio (tengo 7 muestras):
muestrascrudas=[] #no les aplique la funcion patente todavia
muestrasnum=[] #me quede con el num cuando le aplique la funcion patente
muestrasrta=[] #me quede con el rta cuando le aplique la funcion patente

largomuestras=[]

#Junto todos los barrios: 
muestratotcrudo=[]
muestratotnum=[]
muestratotrta=[]
nummuestra=[]
barrio=[]

for i in range(nummuestras):
    muestracrudai =[]
    muestranumi=[]
    
    muestrartai=[]

    with open('muestra%i.txt' %(i+1), "r") as f:
        for line in f:
            patente=line
            muestracrudai.append(patente)
            numi,rtai=patentes(patente)
            muestranumi.append(numi)
            muestrartai.append(rtai)
            muestratotcrudo.append(patente)
            muestratotnum.append(numi)
            muestratotrta.append(rtai)
            nummuestra.append(i+1)
            barrio.append()
    muestrascrudas.append(muestracrudai)
    muestrasnum.append(muestranumi)
    muestrasrta.append(muestrartai)
    largomuestras.append(len(muestranumi))
    
#%%
    
#Ejer 3.a:version discreta

#Uniforme escalonada: (teórica)
    
xt=np.linspace(1,2428510,2428510)
yt=np.linspace(0,1,2428510)

plt.figure()

plt.step(xt,yt,'r')

muestratotnum_sort=[1]+list(np.sort(muestratotnum))+[2428510]

pk =[0]+list(np.linspace(1/len(muestratotnum),1,len(muestratotnum)))+[1]

plt.step(muestratotnum_sort,pk,'b',where='post')
plt.grid(1)

#Para calcular el máximo de la resta hay que rellenar la Kolmogorov:
  
pktot=[]     
for i in range(len(muestratotnum_sort)-1):
    pktot=pktot+list(np.ones(muestratotnum_sort[i+1]-muestratotnum_sort[i])*pk[i])    
pktot=pktot+[1]
plt.plot(xt,pktot,'k.')
    
#Calcular la resta:
listadn=np.abs(np.array(pktot)-yt)
print(np.max(listadn))
    
listapos=np.where(listadn==np.amax(listadn))[0]
    
tablakolm=1.36*len(muestratotnum)**-0.5

#Agregar el estadístico al gráfico


#%%
    
#Ejer 3a

patmax=patentes(['A','D',5,9,2,'M','F'])[0] 
lendata=len(muestratotnum)

#Uniforme escalonada: (teórica)
    

def unifcdf(x):  #Se asume que los x a los que se les aplica la función estan en el intervalo [1,patmax]
    return ((x-1)/(patmax-1))

ks,pv=st.kstest(muestratotnum,unifcdf)

tablakolm=1.36*lendata**-0.5

if ks<tablakolm:
    print ('No puedo rechazar la hipótesis nula')
else:
    print ('Rechazo la hipótesis nula')
    
    
plt.figure()

plt.plot([-115000,1,patmax,2600000],[0,0,1,1],'r-',label='CDF teórica uniforme')

muestratotnum_sort=[-115000]+list(np.sort(muestratotnum))+[2600000]

pk =[0]+list(np.linspace(1/len(muestratotnum),1,len(muestratotnum)))+[1]
plt.xlim([-110000,2500000])
plt.step(muestratotnum_sort,pk,'b',where='post',label='EDF')
plt.legend(loc=2)
plt.grid(1)


#%%

#Ejer 3.b: H1: exponencial = kolmogorov

#Exponencial: (teórica)

def exponencial(x,lambbda):
    return 1-np.e**(-lambbda*x)
    



#%%
    
#Ejer 4.a: La patente más nueva

#binomial negativa
#m = el natural correspondiente a la patente más nueva observada
#k = número total de patentes nuevas observadas
#n = número de autos con patentes nuevas en circulación (es decir, hay n patentes disponibles para observar)

def distribuciondem(m,k,n1): #distribución de probabilidad de m 
    if k==1:
        return 1/n1
    else:
        return k*(m-k+1)*distribuciondem(m,k-1,n1)*((k-1)*(n1-k+1))**(-1)

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

k=25 #observo 25 patentes de las más grandes y me quedo con el máximo 
n1=265309 #Me quedo con los últimos 265309 valores posibles de las patentes (los 265309 valores más grandes)
N=10 #repito lo de observar 25 patentes y quedarme con el máximo de aquéllas unas 1000 veces

y=experimento(N,k,a,b)#máximos de cada muestra 

m=np.linspace(a,b,b-a+1) #m>=k, n1>=k, m<=n1
y2=[]
for i in m:
    y2.append(distribuciondem(i,k,n1))

plt.figure()

bins=np.linspace(np.max(y)-200,np.max(y),50)
n, bins, patches = plt.hist(y, bins, normed=1, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i intentos' %(N))
   
plt.plot(m,y2,'r.')
plt.grid(1)

#El gráfico tiende asintóticamente al valor de n 

#%%    

#Ejer 4.b: La patente más nueva


#Ahora queremos estimar el n1 (número total de patentes que existen) utilizando distribucion de m

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
k=25 #observo 25 patentes de las más grandes y me quedo con el máximo 
N=10

simu=[]
M=30
for i in range(M):
    simu.append(np.max(experimento(N,k,a,b+i)))

#Un prior no informativo es una uniforme. 
        
#%%


