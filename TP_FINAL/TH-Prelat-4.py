# -*- coding: utf-8 -*-
"""
Created on Monday April 1 12:54:41 2019

@author: Leila
"""
from random import uniform  
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import time
import os
import csv
from scipy.special import gamma

tamletra=12
tamlegend=10

directorio=input('¿En qué carpeta guardaste los txt?')

comparacion_discreta_continua=0  
distribucion_acumulada_Dn=0 
distdemax=0

#%% Estadística en la calle

print ('Ejer 2: Convertir patentes a números')
os.chdir(directorio)
#Ejer 2

with open('Barriosmuestras - Hoja1.CSV', newline='') as csvfile:#importa csv pares
    barrioscsv = list(csv.reader(csvfile))

nummuestras=14 #cantidad de txt 

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
    return (num)

def inversapatentes(num):
    if num<1 or num>26**4*10**3:
        return ('Fuera de rango')  
    num=int(num)-1  
    rta0=int(num*(26**3*10**3)**-1) 
    rta1=int((num-rta0*26**3*10**3)*(26**2*10**3)**-1) 
    rta2=int((num-rta0*26**3*10**3-rta1*26**2*10**3)*(26**2*10**2)**-1) 
    rta3=int((num-rta0*26**3*10**3-rta1*26**2*10**3-rta2*26**2*10**2)*(26**2*10**1)**-1) 
    rta4=int((num-rta0*26**3*10**3-rta1*26**2*10**3-rta2*26**2*10**2-rta3*26**2*10**1)*(26**2)**-1) 
    rta5=int((num-rta0*26**3*10**3-rta1*26**2*10**3-rta2*26**2*10**2-rta3*26**2*10**1-rta4*26**2)*(26**1)**-1) 
    rta6=int((num-rta0*26**3*10**3-rta1*26**2*10**3-rta2*26**2*10**2-rta3*26**2*10**1-rta4*26**2-rta5*26)) 
    pat0=abc[rta0]
    pat1=abc[rta1]
    pat2='%i' %(rta2)
    pat3='%i' %(rta3)
    pat4='%i' %(rta4)
    pat5=abc[rta5]
    pat6=abc[rta6]
    pat=pat0+pat1+pat2+pat3+pat4+pat5+pat6
    return (pat)


#Separado por barrio (tengo nummuestras muestras):
muestrasnum=[] #me quede con el num cuando le aplique la funcion patente


#Junto todos los barrios: 
muestratotnum=[]

#Veo el largo de cada muestra y a que barrio corresponde
largomuestras=[]
barrio=[]

for i in range(nummuestras):
    muestranumi=[]
    with open('muestra%i.txt' %(i+1), "r") as f:
        for line in f:
            if line!='\n': #Para sacarse de encima ese caso
                patente=line
                numi=patentes(patente)
                muestranumi.append(numi)
                muestratotnum.append(numi)
    muestrasnum.append(muestranumi)
    largomuestras.append(len(muestranumi))
    barrio.append(barrioscsv[i+1][1])

#%% ¿Uniformemente distribuidas?
    
print ('Ejer 3.0: Comparación entre la distribución teórica discreta y su aproximación a continua') 

patmax=patentes('AD592MF') #última patente existente

def unif_cdf(x):  
    return ((x-1)/(patmax-1))

valores_patentes=np.linspace(1,patmax,patmax)
ytc=unif_cdf(valores_patentes)   

if comparacion_discreta_continua==1:
    puntosy=valores_patentes*1/patmax
    puntosxmax=valores_patentes+1
    #Sin zoom: (Tarda bastante)
    plt.figure()
    plt.plot(valores_patentes,puntosy,'k.', label='Distribución teórica $F_o(x)$ discreta')
    plt.hlines(puntosy,valores_patentes,puntosxmax,'b')
    plt.plot(valores_patentes,ytc,'g-', label='Distribución teórica $F_o(x)$ aprox continua')
    plt.title('Ejer 3.0: Comparación entre las distribuciones teóricas')
    plt.legend(loc='best',fontsize=tamlegend)
    plt.xlabel('Patentes',fontsize=tamletra) 
    plt.grid(1)
    
    #Con zoom: (Tarda poco)
    plt.figure()
    plt.plot(valores_patentes[0:15],puntosy[0:15],'k.', label='Distribución teórica $F_o(x)$ discreta')
    plt.hlines(puntosy[0:15],valores_patentes[0:15],puntosxmax[0:15],'b')
    plt.plot(valores_patentes[0:15],ytc[0:15],'g-', label='Distribución teórica $F_o(x)$ aprox continua')
    plt.title('Ejer 3.0: Comparación entre las distribuciones teóricas')
    plt.legend(loc='best',fontsize=tamlegend)
    plt.xlabel('Patentes',fontsize=tamletra) 
    plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
    plt.xlim([0,6])
    plt.ylim([0,2*10**-6])
    plt.grid(1)

#%% ¿Uniformemente distribuidas?
print ('Ejer 3.1: Sn(x) y Fo(x)') 


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
            ecdf_obs.append((j-len(lista_dn1)+1)/n)
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
pvalue_F=1-D(Dn_obs*len(muestratotnum)**(0.5),int(cotsupder))
print('El p value con la cdf es:', pvalue_F) #formula que vale para la distribucion acumulada del estadistico

plt.title('Ejer 3.1: $S_n(x)$ y $F_o(x)$, el $D_n$ observado es %.4f y el P value es %.4f' %(Dn_obs,pvalue_F))
plt.grid(1)

#%% Distribución acumulada del estadístico del Frodensen:

if distribucion_acumulada_Dn==1:
    z=np.linspace(0.005,2,400)
    y=D(z,cotsupder)
    alpha=0.05
    x=z*len(muestratotnum)**(-1/2)
    
    plt.figure()
    plt.plot(x,y,'r.',label='Distribución acumulada de Dn')
    plt.title('Distribución acumulada de Dn')
    plt.ylabel('Probabilidad de $D_n \leq z/\sqrt{n}$',fontsize=tamletra) 
    plt.xlabel('Valores de $z/\sqrt{n}$',fontsize=tamletra) 
    plt.axvline(x=Dn_obs,label='Dn observado')
    plt.axhline(y=1-pvalue_F,color='g',label='1- P value')
    plt.axvline(x=tablakolm,color='y',label='Valor crítico del Dn (tabla de Kolmogorov)')
    plt.axhline(y=1-alpha,color='k',label='1-alfa')
    plt.legend(loc='best',fontsize=tamlegend)
    plt.grid(1)

#%% Simulación del estadístico Dn:

N=10**3
lendata=len(muestratotnum)
Dn=[]
for i in range(N):
    lista_data=[]
    for i in range(lendata):
        lista_data.append(int(uniform(1,patmax+1)))
    Dn.append(Kolmogorov(lista_data,unif_cdf)[0])

pvalue_s=len(np.where(Dn>=Dn_obs)[0])/len(Dn)
print('El pvalue del D_obs con %i simulaciones:' %(N),pvalue_s) 

distribucion='pdf'

def Error_bines(nbins,bins,N):
    xerror=[] 
    for i in range(len(bins)-1):
        xerror.append((bins[i+1]+bins[i])*0.5)
    error=[]
    for j in range(len(nbins)): 
        error.append(((nbins[j]/N)*(1-nbins[j]/N))**(1/2))
    return (xerror,error)

if distribucion=='pdf':
    plt.figure()
    bins=np.linspace(np.min(Dn),np.max(Dn),50)
    plt.title('Densidad de distribución de $D_n$ con simulaciones, el P value es %.2f' %(pvalue_s))
    plt.xlabel('Valores de $D_n$',fontsize=tamletra) 
    nbins, bins, patches = plt.hist(Dn, bins, density=1,facecolor='green', alpha=1,  edgecolor='black', linewidth=1.2, label='Histograma con %i simulaciones'%(N))    
    plt.errorbar(Error_bines(nbins,bins,N)[0],nbins,Error_bines(nbins,bins,N)[1],linestyle='None',label='Error')
    plt.axvline(x=Dn_obs,label='Estadístico $D_n$ observado')
    plt.legend(loc='best',fontsize=tamlegend)
    plt.grid(1)

#if distribucion=='cdf':
#    plt.figure()
#    bins=np.linspace(np.min(Dn),np.max(Dn),50)
#    plt.title('Distribución acumulada de $D_n$ con simulaciones')
#    plt.xlabel('Valores de $D_n$',fontsize=tamletra) 
#    nbins, bins, patches = plt.hist(Dn, bins, density=True,cumulative=True,facecolor='green', alpha=1,  edgecolor='black', linewidth=1.2, label='Histograma con %i simulaciones'%(N))    
#    xerror=[] 
#    for i in range(len(bins)-1):
#        xerror.append((bins[i+1]+bins[i])*0.5)
#    error=[]
#    for j in range(len(nbins)):
#        error.append((nbins[j]*N*lendata*(1-nbins[j]))**(1/2)/(N*lendata))
#    
#    plt.errorbar(xerror,nbins,error,linestyle='None',label='Error')
#    plt.axvline(x=Dn_obs,label='Estadístico $D_n$ observado')
#    plt.legend(loc='best',fontsize=tamlegend)
#    plt.grid(1)

#%% Distribucion exponencial
    
print ('Ejer 3.2: Distribución exponencial') 

lambbda=4*10**(-7)
tablakolm=1.36*(len(muestratotnum)**(-1/2))

N=10**3
exito=0
lista_dn=[]
for i in range(N):
    lista_data=np.random.exponential(lambbda**(-1),len(muestratotnum))
    Dn=Kolmogorov(lista_data,unif_cdf)[0]
    lista_dn.append(Dn)
    if Dn>tablakolm:
        exito=exito+1
print('El poder del test es', exito/N)
print(np.min(Dn)>tablakolm) #Como todos los valores generados de los Dn son mayores que el tablakolm, el poder del test es 1

plt.figure()
bins=np.linspace(np.min(lista_dn),np.max(lista_dn),30)
plt.title('Ejer 3.2: Densidad de distribucion de $D_n$ con simulaciones cuando H1 es cierta')
plt.xlabel('Valores de $D_n$',fontsize=tamletra) 
nbins, bins, patches = plt.hist(lista_dn, bins, density=True,facecolor='green', alpha=1,  edgecolor='black', linewidth=1.2)
plt.axvline(x=tablakolm,label='Valor crítico para un alfa de 0.05')
plt.errorbar(Error_bines(nbins,bins,N)[0],nbins,Error_bines(nbins,bins,N)[1],linestyle='None',label='Error')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

#%% 
print ('Ejer 4.1: Distribución de m') 

start_time1 = time.time()

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

a=1
b=patentes('AD592MF')
m=np.max(muestratotnum)
inf=1*10**4
k=len(muestratotnum)
x_P=np.arange(m-inf,b) 
y_P=P_m(x_P-a,k,b-a)

N=10**3
y_exp=experimento(N,k,a,b)

plt.figure()
plt.xlim([(m-inf),b*1.0005])
plt.plot(x_P,y_P,'r.', label='Distribución P(m;k,n)')
bins=np.linspace(np.min(y_exp),np.max(y_exp),50)
nbins, bins, patches = plt.hist(y_exp, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))
plt.errorbar(Error_bines(nbins,bins,N*k)[0],nbins,Error_bines(nbins,bins,N*k)[1],linestyle='None',label='Error')
plt.title('Ejer 4.1: Distribución de m')
plt.xlabel('Patentes',fontsize=tamletra) 
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

#Haciendo lo mismo con la aprox stirling, el tiempo computacional es casi el mismo
print(time.time()-start_time1)

#Podemos utilizar como distribución de m la del máximo de la guía 3: (aproximamos a una continua)
if distdemax==1:
    def distdemax(n,k,x):
        return n**(-1)*k*(x/(n-1)-1/(n-1))**(k-1)
    
    x=np.arange(m-inf,b) 
    y_max=distdemax(patmax,k,x)
    
    plt.figure()
    plt.xlim([(m-inf),b*1.0005])
    plt.plot(x,y_max,'k.', label='Distribución del m con G3E8')
    bins=np.linspace(np.min(y_exp),np.max(y_exp),50)
    nbins, bins, patches = plt.hist(y_exp, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))
    plt.errorbar(Error_bines(nbins,bins,N*k)[0],nbins,Error_bines(nbins,bins,N*k)[1],linestyle='None',label='Error')
    plt.title('Ejer 4.1: Distribución de m con G3E8')
    plt.legend(loc='best',fontsize=tamlegend)
    plt.xlabel('Patentes',fontsize=tamletra)
    plt.grid(1)  

#%% La patente del auto más nuevo. Item 2: P(n;k,m) a partir de P(m;k,n)

print ('Ejercicio 4.2: P(n;k,m) a partir de P(m;k,n)')    


def P_n(n,k,m,ncotasup):
    rango_n=np.linspace(m,ncotasup,ncotasup-m+1) #Rango en el que variamos n
    y=P_m(m1,k,rango_n)
    ynorm=y/np.sum(y)
    return (rango_n,ynorm)

b=patmax
m1=np.max(muestratotnum)
k=len(muestratotnum)
ncotasup=b+5*int((m1-k)/k)

rango_n=P_n(b,k,m1,ncotasup)[0]
ynorm=P_n(b,k,m1,ncotasup)[1]

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

print ('Ejercicio 4.3: P(n;k,m) con k y m de los datos')   

k=len(muestratotnum)
m1=np.max(muestratotnum)

listan=P_n(b,k,m1,ncotasup)[0]
ynorm=P_n(b,k,m1,ncotasup)[1]

esperanza_obtenida=np.sum(listan*ynorm)
std_obtenida=(np.sum(ynorm*(listan-esperanza_obtenida)**2))**0.5

print ('Estimación bayesiana para la patente máxima existente (en número y patente):',esperanza_obtenida,inversapatentes(esperanza_obtenida))
print ('Desviación estándar de la patente máxima existente (en número):',std_obtenida)
print ('Intervalo de la patente máxima existente (mínima y máxima patente):',inversapatentes(esperanza_obtenida-std_obtenida),inversapatentes(esperanza_obtenida+std_obtenida))

#Valores teoricos
esperanza_teorica=(m1-1)*(k-1)/(k-2)
std_teorica=((k-1)*(m1-1)*(m1-k+1)*(k-2)**-2/(k-3))**0.5

plt.figure()
plt.plot(listan,ynorm,'g.',label='P(n;k,m) con k=%i y m=%i' %(k,m1))    
plt.title('Ejer 4.3: P(n;k,m) a partir de P(m;k,n)')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Rango de n', fontsize=tamletra)
plt.grid(1)

#Al aumentar el k, la estimación es mejor. 

#%% La patente del auto más nuevo. Item 4: 

print ('Ejercicio 4.4: Estimaciones bayesianas para n')   

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

print ('Ejercicio 5.1: Wilcoxon')   

patentesanmartin=[] 
os.chdir(directorio)
with open('patentes san martin.txt', "r") as f:
    for line in f:
        patentesanmartin.append(patentes(line))
               
patentesrecoleta=[]
for i in range(len(barrio)):
    if barrio[i]=='Recoleta':       
        patentesrecoleta=patentesrecoleta+muestrasnum[i]
        
patentescaballito=[]
for i in range(len(barrio)):
    if barrio[i]=='Caballito':
        patentescaballito=patentescaballito+muestrasnum[i]

patentesparquepatricios=[]
for i in range(len(barrio)):
    if barrio[i]=='Parque Patricios/ConstituciÃ³n':
        patentesparquepatricios=patentesparquepatricios+muestrasnum[i]

patentesanmartin=list(dict.fromkeys(patentesanmartin))   #Elimino las patentes repetidas
patentesrecoleta=list(dict.fromkeys(patentesrecoleta))   #Elimino las patentes repetidas 
patentescaballito=list(dict.fromkeys(patentescaballito))   #Elimino las patentes repetidas
patentesparquepatricios=list(dict.fromkeys(patentesparquepatricios))   #Elimino las patentes repetidas 


def Wilcoxon(lista1,lista2):
    listatot=np.array(list(lista1)+list(lista2))
    listatot_sort=np.sort(listatot)
    rank1=0
    rank2=0
    for i in range(len(listatot)):
        if listatot_sort[i]!=listatot_sort[i-1]:
            posiciones=np.where(listatot==listatot_sort[i])[0]
            posicionessort=np.where(listatot_sort==listatot_sort[i])[0]
            sumandorank=np.mean(posicionessort+1)
            for j in range(len(posiciones)):
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
    z=(w-Ew)/sw #para crear una normal(0,1)
    pv=2*np.min([1/2 + erf(z*2**-0.5)/2,1-(1/2 + erf(z*2**-0.5)/2)])  #Calculo el pvalor en un two sided test #Evaluo la CDF de N(0,1) en z 
    return (z,pv)
    

print ('Se obtuvieron los siguientes estadísticos y el pvalor del test de Wilcoxon para los barrios')
z_obs,pw_obs=Wilcoxon(patentesanmartin,patentesrecoleta)
print ('Recoleta y San Martín:',z_obs,pw_obs)

z_obs,pw_obs=Wilcoxon(patentesanmartin,patentesparquepatricios)
print ('Parque Patricios y San Martín:',z_obs,pw_obs)

z_obs,pw_obs=Wilcoxon(patentesrecoleta,patentescaballito)
print ('Caballito y Recoleta:',z_obs,pw_obs)

z_obs,pw_obs=Wilcoxon(patentesparquepatricios,patentesrecoleta)
print ('Recoleta y Parque Patricios:',z_obs,pw_obs)


#zp,pvp=st.ranksums(patentesrecoleta,patentesanmartin)  #Calculo esto para comparar (paquete de python)

#%% Test de hipotesis del G8E4. Estadistico U 

print ('Ejercicio 5.2: Test de hipotesis del G8E4')   #Advertencia!: Tarda 20 segundos aprox

start_time=time.time()

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

def unif(a,b,lenlista):
    x=[]
    for j in range(lenlista):
        x.append(uniform(a,b))
    return x


b1=patmax
m=len(patentesanmartin)
n=len(patentesrecoleta)

N=10**3
valores_de_U=[]
for i in range(N):
    lista1=unif(1,b1,m)
    lista2=unif(1,b1,n)
    valores_de_U.append(U(lista1,lista2))

#Aplicar la distribución de U a los datos 
Uobs=U(patentesanmartin,patentesrecoleta)
pt_obs=2*np.min([len(np.where(valores_de_U<=Uobs)[0])/len(valores_de_U),len(np.where(valores_de_U>=Uobs)[0])/len(valores_de_U)])

bins=np.linspace(-4*np.abs(Uobs),4*np.abs(Uobs),41)

#Uobs=np.round(U(patentesanmartin,patentesrecoleta),3)
print('El estadístico observado y el pvalor son:', Uobs,pt_obs)
    
plt.figure()   

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

print (time.time()-start_time)

#%% Combinando los tests

print ('Ejercicio 6.1: Combinando los tests') 

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
print ('La correlacion entre pv y pw simulados con dos distribuciones uniformes independientes es:',corr)

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

print ('Ejercicio 6.2: P values correlacionados')   #Advertencia!: Tarda 90 segundos aprox

starttime=time.time()

N=10**3

a1=1
b1=patmax
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
pvalue_Tobs=len(np.where(T_Ho_cierta>=T_obs)[0])/len(T_Ho_cierta)
print('El pvalue del T_obs es:',pvalue_Tobs) 
  
plt.axvline(x=T_obs,label='Estadístico observado')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

print (time.time()-starttime)

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

print ('Ejercicio 7') 

def P_m_cdf(m,k,n):
    return P_m(m,k,n)*m/k

m1=patentes('ad571uc')
b1=patmax
k=len(muestratotnum)-500
#
#rango_m=np.arange(m1+1,b1+1) #se calcula la probabilidad de hallar algo mejor y para el p value es 1-eso
#
#rta=0
#for i in rango_m:
#    rta=1*P_m(i,k,b1)+rta #la distancia entre cada bin es 1 en este caso       
#        
#print('El p-value es', 1-rta)

print('El p-value es', P_m_cdf(m1,k,b1)) #Test a cola izquierda (algo peor es un m mas chico)

#Para la significancia voy a elegir 0.05
   
#%%

 
    
    
    
    
    
    