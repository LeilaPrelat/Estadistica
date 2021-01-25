# -*- coding: utf-8 -*-
"""
Created on Monday April 1 12:54:41 2019

@author: Leila
"""
from random import uniform  
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.stats as st
import time
import os
import csv
from scipy.special import gamma

tamletra=12
tamlegend=10

directorio=input('¿En qué carpeta guardaste los txt?')

#%% Estadística en la calle

print ('Ejer 2: Convertir patentes a números')
os.chdir(directorio)
with open('Barriosmuestras - Hoja1.CSV', newline='') as csvfile:#importa csv pares
    barrioscsv = list(csv.reader(csvfile))

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
        if a==ABC[m] or a==abc[m]:
            rta0=m+1
        if b==ABC[m] or b==abc[m]:
            rta1=m+1
    rta.append(rta0)
    rta.append(rta1)
    rta.append(int(c))
    rta.append(int(d))
    rta.append(int(e))
    for m in range(len(ABC)):
        if f==ABC[m] or f==abc[m]:
            rta5=m+1 
        if g==ABC[m] or g==abc[m]:
            rta6=m+1 
    rta.append(rta5)
    rta.append(rta6)
    num=rta[6]*26**0+(rta[5]-1)*26**1+rta[4]*26**2+rta[3]*10*26**2+rta[2]*10*10*26**2+(rta[1]-1)*10**3*(26**2)+(rta[0]-1)*10**3*(26**3)
    return (num,rta)

num, rta=patentes(['Z','Z',9,9,9,'Z','Z'])
print(num==26**4*10**3)

#Separado por barrio (tengo 7 muestras):
muestrascrudas=[] #no les aplique la funcion patente todavia
muestrasnum=[] #me quede con el num cuando le aplique la funcion patente
muestrasrta=[] #me quede con el rta cuando le aplique la funcion patente

largomuestras=[]

#Junto todos los barrios: 
muestratotcrudo=[]
muestratotnum=[]
muestratotrta=[]

#Poner la carpeta donde se guardan las muestras de las patentes: 
os.chdir(directorio)

for i in range(nummuestras):
    muestracrudai =[]
    muestranumi=[]
    muestrartai=[]

    with open('muestra%i.txt' %(i+1), "r") as f:
        for line in f:
            patente=[]
            for l in range(7):
                patente.append(line[l])
            muestracrudai.append(patente)
            numi,rtai=patentes(patente)
            muestranumi.append(numi)
            muestrartai.append(rtai)
            muestratotcrudo.append(patente)
            muestratotnum.append(numi)
            muestratotrta.append(rtai)
    muestrascrudas.append(muestracrudai)
    muestrasnum.append(muestranumi)
    muestrasrta.append(muestrartai)
    largomuestras.append(len(muestranumi))

#%% ¿Uniformemente distribuidas?
    
print ('Ejer 3.0: Comparación entre la distribución teórica discreta y su aproximación a continua') 

patmax=patentes('AD592MF')[0] #última patente existente
valores_patentes=np.linspace(1,patmax,patmax)

def unif_cdf(x):  
    return ((x-1)/(patmax-1))

ytc=unif_cdf(valores_patentes)
    
puntosy=valores_patentes*1/patmax
puntosxmax=valores_patentes+1

#Sin zoom:
plt.figure()
plt.plot(valores_patentes,puntosy,'k.', label='Distribución teórica $F_o(x)$ discreta')
plt.hlines(puntosy,valores_patentes,puntosxmax,'b')
plt.plot(valores_patentes,ytc,'g-', label='Distribución teórica $F_o(x)$ aprox continua')
plt.title('Ejer 3.0: Comparación entre las distribuciones teóricas')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Patentes',fontsize=tamletra) 
plt.grid(1)

#Con zoom: 
plt.figure()
plt.plot(valores_patentes,puntosy,'k.', label='Distribución teórica $F_o(x)$ discreta')
plt.hlines(puntosy,valores_patentes,puntosxmax,'b')
plt.plot(valores_patentes,ytc,'g-', label='Distribución teórica $F_o(x)$ aprox continua')
plt.title('Ejer 3.0: Comparación entre las distribuciones teóricas')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Patentes',fontsize=tamletra) 
plt.xlim([0,6])
plt.ylim([0,2*10**-6])
plt.grid(1)

#%%
"""
def Kolmogorov(lista_data,Fo):
    lista_data_sort=np.sort(np.array(lista_data))
    n=len(lista_data)
    lista_dn=[]
    for i in range(1,n+1):
        lista_dn.append(Fo(lista_data_sort[i-1])-(i-1)/n) #pares y 0
        lista_dn.append(i/n-Fo(lista_data_sort[i-1])) #impares
    Dn_obs=np.max(lista_dn)
    Patente_posicion=np.where(lista_dn==Dn_obs)[0]
    
    for j in Patente_posicion:
        if j%2==0:
            i=int(j/2)+1
            Patente_Dn_obs=lista_data_sort[i-1]
            ecdf_obs=(i-1)/n
            cdf_obs=Fo(lista_data_sort[i-1])
        elif j%2==1:
            i=int((j+1)/2)
            Patente_Dn_obs=lista_data_sort[i-1]
            ecdf_obs=i/n
            cdf_obs=Fo(lista_data_sort[i-1])
    return (Dn_obs,Patente_Dn_obs,ecdf_obs,cdf_obs)
"""
#%% ¿Uniformemente distribuidas?
print ('Ejer 3.1: Sn(x) y Fo(x)') 

patmax=patentes('AD592MF')[0]
#Construir Sn(x):
muestratotnum_sort=[1]+list(np.sort(muestratotnum))+[patmax]
pk =[0]+list(np.linspace(1/len(muestratotnum),1,len(muestratotnum)))+[1]

def Kolmogorov(lista_data,Fo):
    lista_data_sort=np.sort(np.array(lista_data))
    n=len(lista_data)
    lista_dn1=[]
    lista_dn2=[]
    for i in range(1,n+1):
        lista_dn1.append(Fo(lista_data_sort[i-1])-(i-1)/n) #pares y 0
        lista_dn2.append(i/n-Fo(lista_data_sort[i-1])) #impares
    lista_dn=lista_dn1+lista_dn2
    Dn_obs=np.max(lista_dn)
    Patente_posicion=np.where(lista_dn==Dn_obs)[0]
    Patente_Dn_obs=[]
    ecdf_obs=[]
    cdf_obs=[]
    for j in Patente_posicion:
        if j<len(lista_dn1):
            Patente_Dn_obs.append(lista_data_sort[j])
            ecdf_obs.append(j/n)
            cdf_obs.append(Fo(lista_data_sort[j]))
        else:
            Patente_Dn_obs.append(lista_data_sort[j-len(lista_dn1)])
            ecdf_obs.append((j-len(lista_dn1))/n)
            cdf_obs.append(Fo(lista_data_sort[j-len(lista_dn1)]))
    return (Dn_obs,Patente_Dn_obs,ecdf_obs,cdf_obs)

Dn_obs,Patente_Dn_obs,ecdf_obs,cdf_obs=Kolmogorov(muestratotnum, unif_cdf)

#Comparamos con la tabla de Kolmogorov
tablakolm=1.36*(len(muestratotnum)**(-1/2))

plt.figure()
plt.plot(valores_patentes,ytc,'g-', label='Distribución teórica $F_o(x)$')
plt.step(muestratotnum_sort,pk,'b',where='post', label='Distribución de Kolmogorov') 

for j in range(len(Patente_Dn_obs)):
    plt.plot([Patente_Dn_obs[j],Patente_Dn_obs[j]],[cdf_obs[j],ecdf_obs[j]],'k-',label='Patente del $D_n$ observado: %i' %(Patente_Dn_obs[j]))

plt.ylabel('Probabilidad',fontsize=tamletra)  
plt.xlabel('Patentes',fontsize=tamletra)  
plt.legend(loc='best',fontsize=tamlegend)

if Dn_obs<=tablakolm:
    print('Se acepta Ho con una significancia de 0,05')
else:
    print('Se rechaza Ho con una significancia de 0,05')

#Para agregar el p value, defino la distribución del estadístico:    

#Distribución del estadístico (fórmula del Frodensen con la sumatoria desde 1 hasta cotsupder (cota superior de r))
def D(z,cotsupder):
    sumder=0
    for j in range(1,cotsupder):
        sumder=sumder+(-1)**(j-1)*np.e**(-2*j**2*z**2)
    return 1-2*sumder

cotsupder=15**3

#Este es un one side test (es modulo de la distancia)
pvalue=1-D(Dn_obs*len(muestratotnum)**(0.5),int(cotsupder))
print('El p value (con la acumulada) es:', pvalue) #formula que vale para la distribucion acumulada del estadistico

plt.title('Ejer 3.1: Sn(x) y Fo(x), el $D_n$ observado es %.4f y el pvalue es %.4f' %(Dn_obs,pvalue))
plt.grid(1)

#%% Grafico la distribución acumulada del estadístico:

z=np.linspace(0.005,2,400)
y=D(z,cotsupder)
alpha=0.05

x=z*len(muestratotnum)**(-1/2)
plt.figure()
plt.plot(x,y,'r.',label='Distribución acumulada de Dn')
plt.title('Ejer 3.1: Distribución acumulada de Dn')
plt.ylabel('Probabilidad de $D_n \leq z/\sqrt{n}$',fontsize=tamletra) 
plt.xlabel('Valores de $z/\sqrt{n}$',fontsize=tamletra) 
plt.axvline(x=Dn_obs,label='Dn observado')
plt.axhline(y=1-pvalue,color='g',label='1-pvalue')
plt.axvline(x=tablakolm,color='y',label='Valor crítico del Dn (tabla de Kolmogorov)')
plt.axhline(y=1-alpha,color='k',label='1-alfa')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

#%% Grafico la densidad de distribución del estadístico (simulación, tarda bastante)

def unifcdf(x):  
    return ((x-1)/(patmax-1))

def D(n,patmax): #Fo puede ser la uniforme (item a) o la exponencial (item b)
    x=[]
    for i in range(n):
        x.append(int(uniform(1,patmax+1)))
    xtot_sort=[1]+list(np.sort(x))+[patmax]
    pk=[0]+list(np.linspace(1/n,1,n))+[1]
    pktot=[]     #Rellenar la kolmogorov
    for i in range(n-1):
        pktot=pktot+list(np.ones(xtot_sort[i+1]-xtot_sort[i])*pk[i])    
    pktot=pktot+[1]
    listadn=np.abs(np.array(pk)-np.array(unifcdf(np.linspace(1,patmax,len(pk)))))
    return (np.where(listadn==np.amax(listadn))[0],np.max(listadn))

N=50
print(D(len(muestratotnum),patmax)[0])
Dn=[]
for i in range(N):
    Dn.append(D(len(muestratotnum),patmax)[1])
    
plt.figure()
bins=np.linspace(0,0.1,30)
plt.title('Ejer 3.1: Densidad de distribución de $D_n$ con simulaciones')
plt.xlabel('Valores de $D_n$',fontsize=tamletra) 
nbins, bins, patches = plt.hist(Dn, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=1.2)

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*(1-nbins[j]))**(1/2)/N)

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.axvline(x=Dn_obs,label='Estadístico $D_n$ observado')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

pvalue_Dobs=0
i=len(xerror)-1
while xerror[i]>=Dn_obs: 
    pvalue_Dobs=(bins[i+1]-bins[i])*nbins[i]+pvalue_Dobs   
    i=i-1
    
print('El pvalue del D_obs es:',pvalue_Dobs) 

#%% Distribucion exponencial

lambbda=4*10**(-7)
lendata=277
tablakolm=1.36*(lendata**(-1/2))

N=10**3
exito=0
lista_dn=[]
for i in range(N):
    lista_data=np.random.exponential(lambbda**(-1),lendata)
    Dn=Kolmogorov(lista_data,unifcdf)
    lista_dn.append(Dn)
    if Dn>tablakolm:
        exito=exito+1
print(exito/N)

plt.figure()
bins=np.linspace(np.min(lista_dn),np.max(lista_dn),30)
plt.title('Ejer 3.2: Distribución acumulada de $D_n$ con simulaciones cuando H1 es cierta')
plt.xlabel('Valores de $D_n$',fontsize=tamletra) 
nbins, bins, patches = plt.hist(lista_dn, bins, density=True, cumulative=True,facecolor='green', alpha=1,  edgecolor='black', linewidth=1.2)

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*(1-nbins[j]))**(1/2)/N)

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)
##
plt.figure()
bins=np.linspace(np.min(lista_dn),np.max(lista_dn),30)
plt.title('Ejer 3.2: Densidad de distribucion de $D_n$ con simulaciones cuando H1 es cierta')
plt.xlabel('Valores de $D_n$',fontsize=tamletra) 
nbins, bins, patches = plt.hist(lista_dn, bins, density=True,facecolor='green', alpha=1,  edgecolor='black', linewidth=1.2)

plt.axvline(x=tablakolm,label='Valor crítico para un alfa de 0.05')
xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*(1-nbins[j]))**(1/2)/N)

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

#%% 

#La patente del auto más nuevo. Primero acotemos la longitud de las patentes (cambiamos el origen). 

start_time1 = time.time()

#Distribución teórica de m, dados k y n:
def P_m(m,k,n): 
    if k==1:
        return 1/n
    else:
        return k*(m-k+1)*P_m(m,k-1,n)*((k-1)*(n-k+1))**(-1)    
 
#Simulamos la distribución de m
def experimento(N,k,a,b):
    maxi=[]
    for i in range(N):
        x=[]
        for i in range(k):
            x.append(int(uniform(a,b+1)))  #patentes observadas (de las más nuevas) #k=uniform(0,n) #puedo observar cero patentes nuevas y puedo observar todas las patentes nuevas que existen en circulacion (n)
        maxi.append(np.max(x))
    return maxi

a1=2163201
b1=2428510
m1=2428310

#Corremos el origen:    
a2=1
b2=265309 #sería el n 
m2=265109 

k=277
x_P1=np.arange(m1-5000,b1) 
y_P1=P_m(x_P1-a1,k,b1-a1)

N=10**3
y_exp1=experimento(N,k,a1,b1)

plt.figure()
plt.plot(x_P1,y_P1,'r.', label='Distribución P(m;k,n) acortando el rango de las patentes')
bins=np.linspace(np.min(y_exp1),np.max(y_exp1),50)
nbins, bins, patches = plt.hist(y_exp1, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*(1-nbins[j]))**(1/2)/N)

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.title('Ejer 4.1: Distribución de m')
plt.xlabel('Patentes',fontsize=tamletra) 
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

plt.figure()

x_P2=np.arange(m2-5000,b2) 
y_P2=P_m(x_P2,k,b2)

N=10**3
y_exp2=experimento(N,k,a2,b2)
plt.plot(x_P2,y_P2,'b.', label='Distribución P(m;k,n) acortando el rango de las patentes')
bins=np.linspace(np.min(y_exp2),np.max(y_exp2),50)
nbins, bins, patches = plt.hist(y_exp2, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*k*(1-nbins[j]))**(1/2)/(N*k))

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.title('Ejer 4.1: Distribución de m')
plt.xlabel('Patentes',fontsize=tamletra) 
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

print(time.time()-start_time1)

#%% 
#La patente del auto más nuevo. Ahora usamos stirling para poder usar todo el rango de las patentes. 

start_time2 = time.time()
#Logaritmo de likelihood usando stirling (aproximación de la distribución de m)
def likelihood_stirling(m,k,n):
    return np.log(k)+(n-k)*np.log(n-k)+(m-1)*np.log(m-1)-n*np.log(n)-(m-k)*np.log(m-k)

a1=2163201
b1=2428510
m1=2428310

k=277
x=np.arange(m1-6000,b1)
y_stirling=np.e**(likelihood_stirling(x-a1,k,b1-a1)+1) #Hay que sumarle 1 porque la aprox de stirling achica un poco los valores

y_exp=experimento(N,k,a1,b1)

plt.figure()
plt.plot(x,y_stirling,'b.', label='Distribución del log(P(m;k,n)) con stirling')
bins=np.linspace(np.min(y_exp),np.max(y_exp),50)
nbins, bins, patches = plt.hist(y_exp, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*k*(1-nbins[j]))**(1/2)/(N*k))

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.title('Ejer 4.1: Distribución de m con stirling')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Patentes',fontsize=tamletra)
plt.grid(1)  

print(time.time()-start_time2)   

#%% 
#La patente del auto más nuevo. Ahora usamos la distribucion del maximo 
 
start_time3 = time.time()

def distdemax(n,k,x):
    return n**(-1)*k*(x/(n-1)-1/(n-1))**(k-1)

a1=2163201
b1=2428510
m1=2428310

y_exp=experimento(N,k,a1,b1)

k=277
x=np.arange(m1-6000,b1)
y_max=distdemax(b1-a1,k,x-a1)

plt.figure()
plt.plot(x,y_max,'k.', label='Distribución del m con G3E8')
bins=np.linspace(np.min(y_exp),np.max(y_exp),50)
nbins, bins, patches = plt.hist(y_exp, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*k*(1-nbins[j]))**(1/2)/(N*k))

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.title('Ejer 4.1: Distribución de m con G3E8')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Patentes',fontsize=tamletra)
plt.grid(1)  

print(time.time()-start_time3)   

#%% La patente del auto más nuevo. Item 2: P(n;k,m) a partir de P(m;k,n)

print ('Ejercicio 4.2: P(n;k,m) a partir de P(m;k,n)')    

b1=patentes('AD592MF')[0]
m1=2428310
k=1500
ncotasup=b1+5*int((m1-k)/k)

def P_n(n,k,m,ncotasup):
    rango_n=np.linspace(m,ncotasup,ncotasup-m+1) #Rango en el que variamos n
    y=P_m(m1,k,rango_n)
    ynorm=y/np.sum(y)
    return (rango_n,ynorm)

rango_n=P_n(b1,k,m1,ncotasup)[0]
ynorm=P_n(b1,k,m1,ncotasup)[1]

esperanza_obtenida=np.sum(rango_n*ynorm)
std_obtenida=(np.sum(ynorm*(rango_n-esperanza_obtenida)**2))**0.5

#Valores teoricos
esperanza_teorica=(m1-1)*(k-1)/(k-2)
std_teorica=((k-1)*(m1-1)*(m1-k+1)*(k-2)**-2/(k-3))**0.5

plt.figure()
plt.plot(rango_n,ynorm,'b.',label='P(n;k,m) con k=%i y m=%i' %(k,m1))    
plt.title('Ejer 4.2: P(n;k,m) a partir de P(m;k,n)')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Rango de n', fontsize=tamletra)
plt.grid(1)

#%% La patente del auto más nuevo. Item 3: Evaluar en k y m obtenidos. 

k=277
m1=2428310

listan=P_n(b1,k,m1,ncotasup)[0]
ynorm=P_n(b1,k,m1,ncotasup)[1]

esperanza_obtenida=np.sum(listan*ynorm)
std_obtenida=(np.sum(ynorm*(listan-esperanza_obtenida)**2))**0.5

#Valores teoricos
esperanza_teorica=(m1-1)*(k-1)/(k-2)
std_teorica=((k-1)*(m1-1)*(m1-k+1)*(k-2)**-2/(k-3))**0.5

plt.figure()
plt.plot(listan,ynorm,'b.',label='P(n;k,m) con k=%i y m=%i' %(k,m1))    
plt.title('Ejer 4.3: P(n;k,m) a partir de P(m;k,n)')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Rango de n', fontsize=tamletra)
plt.grid(1)

#Al aumentar el k, la estimación es mejor. 

#%% La patente del auto más nuevo. Item 4: 

def Estimacion_n(n,k,m,ncotasup):
    rango_n=np.linspace(m,ncotasup,ncotasup-m+1) #Rango en el que variamos n
    y=P_m(m1,k,rango_n)
    ynorm=y/np.sum(y)
    Estimacion_n=np.sum(rango_n*ynorm)
    #std=(np.sum(ynorm*(rango_n-Estimacion_n)**2))**0.5
    return Estimacion_n

#Generar N valores para m con el experimento 

k=277
m1=2428310
a1=2163201
b1=2428510
N=10**3
ncotasup=b1+5*int((m1-k)/k)
rango_m=experimento(N,k,1,b1)
n1=Estimacion_n(b1,k,m1,ncotasup)

estpeor=0
estimaciones=[]
for i in range(N):
    est=Estimacion_n(b1,k,rango_m[i],ncotasup)
    estimaciones.append(est)
    if np.abs(est-b1)>=np.abs(n1-b1): #two-side test
        estpeor=estpeor+1
    print (i)
print (estpeor/N)

plt.figure()   
bins=np.linspace(np.min(estimaciones),np.max(estimaciones),50)
nbins, bins, patches = plt.hist(estimaciones, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))
plt.plot([b1,b1],[0,0.00010],'b-')
plt.plot([n1,n1],[0,0.00010],'r-')
plt.plot([2*b1-n1,2*b1-n1],[0,0.00010],'r-')

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*k*(1-nbins[j]))**(1/2)/(N*k))

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.legend(loc='best',fontsize=tamlegend)
plt.title('Ejer 4.4: Estimaciones bayesianas de n')
plt.grid(1)

#%% 

patentesanmartin=[] #San Martin es el Barrio 0
os.chdir(directorio)
with open('patentes san martin.txt', "r") as f:
    for line in f:
        patentesanmartin.append(patentes(line)[0])
               
patentesrecoleta=muestrasnum[0]+muestrasnum[1]+muestrasnum[2]+muestrasnum[4]+muestrasnum[5]+muestrasnum[6]  #Recoleta es el Barrio 1

patentesanmartin=list(dict.fromkeys(patentesanmartin))   #Elimino las patentes repetidas
patentesrecoleta=list(dict.fromkeys(patentesrecoleta))   #Elimino las patentes repetidas 

#Junto todo en una misma muestra y la ordeno:
patentestot=np.array(patentesanmartin+patentesrecoleta)
patentestot_sort=np.sort(patentestot)

barriosort=[]

#rank=0

def Wilcoxon(lista1,lista2):
    patentestot=np.array(list(lista1)+list(lista2))
    patentestot_sort=np.sort(patentestot)
    rank1=0
    rank2=0
    for i in range(len(patentestot)):
        if patentestot_sort[i]!=patentestot_sort[i-1]:
            posiciones=np.where(patentestot==patentestot_sort[i])[0]
            posicionessort=np.where(patentestot_sort==patentestot_sort[i])[0]
            sumandorank=np.mean(posicionessort)
            for j in range(len(posicionessort)):
                posicion=posiciones[j]
                if posicion>=len(lista1):
                    rank2=rank2+sumandorank
                else:
                    rank1=rank1+sumandorank
    if len(lista1)<=len(lista2):
        w=rank1
        n=len(lista1)
        m=len(lista2)
    else:
        w=rank2
        m=len(lista1)
        n=len(lista2)        
    Ew=(n+m+1)*n/2 #Esperanza
    Vw=n*m*(n+m+1)/12 #Varianza
    sw=Vw**0.5
    if w>Ew:
        z=(w-Ew-1/2)/sw
        pv=2*np.min([1/2 + erf(z*2**-0.5)/2,1-(1/2 + erf(z*2**-0.5)/2)])
        return (z,pv)
    elif w<Ew:
        z=(w-Ew+1/2)/sw
        pv=2*np.min([1/2 + erf(z*2**-0.5)/2,1-(1/2 + erf(z*2**-0.5)/2)])
        return (z,pv)
    else: 
        z=(w-Ew)/sw
        pv=2*np.min([1/2 + erf(z*2**-0.5)/2,1-(1/2 + erf(z*2**-0.5)/2)])
        return (z,pv)
    
#def Qsn(p):  #Defino la quantile function de la standard normal distribution N(0,1)
#    return(2**0.5*erfinv(2*p-1))

z_obs=Wilcoxon(patentesanmartin,patentesrecoleta)[0]
pw_obs=Wilcoxon(patentesanmartin,patentesrecoleta)[1]   #Calculo el pvalor en un two sided test #Evaluo la CDF de N(0,1) en z 

print ('El estadístico y el pvalor del test de Wilcoxon son, respectivamente:',z_obs,pw_obs)

zp,pvp=st.ranksums(patentesrecoleta,patentesanmartin)  #Calculo esto para comparar (paquete de python)

#%% Test de hipotesis del G8E4. Estadistico U 

def U(lista1,lista2):
    m=len(lista1)
    n=len(lista2)
    s1=0
    for i in range(m):
        s1=s1+(lista1[i]-np.mean(lista1))**2
    s2=0
    for j in range(n):
        s2=s2+(lista2[j]-np.mean(lista2))**2
    u=(np.mean(lista1)-np.mean(lista2))*((m+n-2)**(0.5))*((m**(-1)+n**(-1))*(s1+s2))**(-0.5)
    return u

b1=patentes('AD592MF')[0]
m=500
n=600

def unif(a,b,lenlista):
    x=[]
    for j in range(lenlista):
        x.append(uniform(a,b))
    return x

N=10**4
valores_de_U=[]
for i in range(N):
    lista1=unif(1,b1,m)
    lista2=unif(1,b1,n)
    valores_de_U.append(U(lista1,lista2))

#Aplicar la distribución de U a los datos 
Uobs=np.round(U(patentesanmartin,patentesrecoleta),3)
print('El estadístico observado es', Uobs)
    
plt.figure()   
bins=np.linspace(Uobs*3-0.25,-Uobs*3+0.25,int(-Uobs*6)*5)
#bins=np.linspace(-4*np.abs(Uobs),4*np.abs(Uobs),41)
nbins, bins, patches = plt.hist(valores_de_U, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma de %i valores de U' %(N))

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*(1-nbins[j]))**(1/2)/N)

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.title('Ejer 5.2: Distribución de densidad de U')

plt.axvline(x=Uobs,label='Estadístico observado')
plt.axvline(x=-Uobs,label='Estadístico observado')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

#El p value en un two side test cuando la distribucion no es simetrica no esta bien definido
a=np.where(np.round(np.array(bins),3)==np.round(Uobs,3))[0][0]
rta1=0
i=0
while xerror[i]<=Uobs: 
    rta1=(bins[i+1]-bins[i])*nbins[i]+rta1   
    i=i+1 
    
b=np.where(np.round(np.array(bins),3)==np.round(-Uobs,3))[0][0]     
rta2=0
i=len(xerror)-1
while xerror[i]>=-Uobs:
    rta2=(bins[i+1]-bins[i])*nbins[i]+rta2
    i=i-1
    
#VALE SI UOBS ESTA TIRADO A LA IZQUIERDA. PENSAR MEJOR QUE ONDA  
Uobs=U(patentesanmartin,patentesrecoleta)
pt_obs=2*np.min([len(np.where(valores_de_U<=Uobs)[0])/len(valores_de_U),len(np.where(valores_de_U>=Uobs)[0])/len(valores_de_U)])

#%% Combinando los tests

#Los p value tienen distribucion uniforme, se puede hallar la dist de T analiticamente y comparar

def T(pw,pv):
    return -2*np.log(pw*pv)

N=10**4
pw=0.2
pt=0.24

#Test independendientes: (el valor de pw no influye en el de pt y viceversa)
lista_pw=unif(0,1,N)
lista_pv=unif(0,1,N)

#Calculo correlación entre pvtot y pwtot
lista_pw=np.array(lista_pw)
lista_pt=np.array(lista_pv)
corr=np.mean((lista_pw-np.mean(lista_pw))*(lista_pv-np.mean(lista_pv)))*(np.std(lista_pv))**-1*(np.std(lista_pw))**-1
print (corr)

lista_T=[]
for i in range(N):
    lista_T.append(T(lista_pw[i],lista_pv[i]))
plt.figure()   
#bins=np.linspace(np.min(lista_pw),np.max(lista_pw),30)
#nbins, bins, patches = plt.hist(lista_pw, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma de %i valores de T(pw,pt)' %(N))
#plt.figure()   
#bins=np.linspace(np.min(lista_pt),np.max(lista_pt),30)
#nbins, bins, patches = plt.hist(lista_pt, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma de %i valores de T(pw,pt)' %(N))

plt.figure()   
bins=np.linspace(np.min(lista_T),np.max(lista_T),30)
nbins, bins, patches = plt.hist(lista_T, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma de %i valores de T(pw,pt)' %(N))

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*(1-nbins[j]))**(1/2)/N)

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.title('Ejer 6.1: Distribución de densidad de T(pw,pt)')

def gamma_pdf(x,alfa,beta): #cuando alfa es entero
    return x**(alfa-1)*np.e**(-x/beta)*beta**(-alfa)*gamma(alfa)**(-1)

alfa=2 #sume dos exponenciales
beta=2 #lambda es 1/2 y beta es 1/lambda

lista_x=np.linspace(0,20,1000)
lista_y=gamma_pdf(lista_x,alfa,beta)

plt.plot(lista_x,lista_y,'r.',label='Distribución de densidad teórica de gamma(x,2,2)')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

#Esperanza del histograma
mean=0
for i in range(len(bins)-1):
    mean=mean+(bins[i+1]-bins[i])*(nbins[i])
    
#Esperanza de la funcion gamma=alfa/beta
    
#%%
#Generar dos muestras con la misma esperanza (a+b)/2 y distribucion uniforme (para usar lo del ejer 5)
#Se tiene que cumplir (a1+b1)/2=(a2+b2)/2 entonces a1+b1=a2+b2
    
N=10**3

a1=1
b1=patentes('AD592MF')[0]
lenlista1=500

a2=a1
b2=b1
lenlista2=600

pvtot=[]
pwtot=[]
for i in range(N):
    lista1=unif(a1,b1,lenlista1)
    lista2=unif(a2,b2,lenlista2)

    Uobs=U(lista1,lista2)
    pvtot.append(2*np.min([len(np.where(valores_de_U<=Uobs)[0])/len(valores_de_U),len(np.where(valores_de_U>=Uobs)[0])/len(valores_de_U)]))
    pwtot.append(Wilcoxon(lista1,lista2)[1])

#Calculo correlación entre pvtot y pwtot (estan correlacionados porque usamos las mismas listas)
pvtot=np.array(pvtot)
pwtot=np.array(pwtot)
corr=np.mean((pvtot-np.mean(pvtot))*(pwtot-np.mean(pwtot)))*(np.std(pvtot))**-1*(np.std(pwtot))**-1
print ('La correlacion entre pv y pw es:',corr)

T_Ho_cierta=[]
for i in range(N):
    Ti=T(pwtot[i],pvtot[i])
    if np.isfinite(Ti): #quiero solo numeros finitos 
        T_Ho_cierta.append(Ti)

plt.figure()   
bins=np.linspace(np.min(T_Ho_cierta),np.max(T_Ho_cierta),100)
nbins, bins, patches = plt.hist(T_Ho_cierta, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma de %i valores de T(pw,pt)' %(N))

xerror=[] 
for i in range(len(bins)-1):
    xerror.append((bins[i+1]+bins[i])*0.5)
error=[]
for j in range(len(nbins)):
    error.append((nbins[j]*N*(1-nbins[j]))**(1/2)/N)

plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
plt.title('Ejer 6.1: Distribución de densidad de T(pw,pt) cuando Ho es cierta')

#Comparamos con la dist de T anterior
alfa=2 #sume dos exponenciales
beta=2 #lambda es 1/2 y beta es 1/lambda

lista_x=np.linspace(0,20,1000)
lista_y=gamma_pdf(lista_x,alfa,beta)

plt.plot(lista_x,lista_y,'r.',label='Distribución de densidad teórica de gamma(x,2,2)')


pwobs=Wilcoxon(patentesanmartin,patentesrecoleta)[1]
#Uobs=np.round(U(patentesanmartin,patentesrecoleta),3)
Uobs=U(patentesanmartin,patentesrecoleta)
pvobs=2*np.min([len(np.where(valores_de_U<=Uobs)[0])/len(valores_de_U),len(np.where(valores_de_U>=Uobs)[0])/len(valores_de_U)])

T_obs=T(pwobs,pvobs)
pvalue_Tobs=0
i=len(xerror)-1
while xerror[i]>=T_obs: 
    pvalue_Tobs=(bins[i+1]-bins[i])*nbins[i]+pvalue_Tobs   
    i=i-1 
print('El pvalue del T_obs es:',pvalue_Tobs) 
  
plt.axvline(x=T_obs,label='Estadístico observado')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)
#El problema no surge de las distribuciones de pw y pv, que siguen siendo uniformes:

#plt.figure()  
#plt.title('Histograma de %i valores de pw' %(N)) 
#bins=np.linspace(np.min(pwtot),np.max(pwtot),30)
#nbins, bins, patches = plt.hist(pwtot, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5)
#plt.legend(loc='best',fontsize=tamlegend)
#plt.xlabel('Valores de pw')
#plt.grid(1)
#
#plt.figure()   
#plt.title('Histograma de %i valores de pv' %(N)) 
#bins=np.linspace(np.min(pvtot),np.max(pvtot),30)
#nbins, bins, patches = plt.hist(pvtot, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5)
#plt.legend(loc='best',fontsize=tamlegend)
#plt.xlabel('Valores de pv')
#plt.grid(1)

#%% Se tu propia verduga. Estadístico del test = m y su densidad de distribución es P_m(m,k,b1)

m1=2428310
a1=2163201
b1=2428510
k=277

rango_m=np.arange(m1+1,b1+1)
#y=P_m(m,k,b1)

#rta=0
#for i in range(len(m)):
#    if m[i]<=m1:
#        rta=m[i]*P_m(m[i],k,b1)+rta
        
#Es mas facil calcular 1-pvalue y despejar: tomo los m[i]>m1:
rta=0
for i in rango_m:
    rta=1*P_m(i,k,b1)+rta #la distancia entre cada bin es 1 en este caso       
        
print('El p-value es', 1-rta)

#Para la significancia voy a elegir 0.05 porque, dentro de todos los alfa posibles para aceptar Ho con un pvalue de 0.98, es el típico valor
   
#%%

 
    
    
    
    
    
    