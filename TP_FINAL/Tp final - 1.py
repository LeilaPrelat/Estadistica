# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:54:07 2019

@author: leila
"""
from random import uniform  
from scipy.special import erf
from scipy.special import erfinv
from scipy.special import kolmogorov
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import csv


#%%

#Ejer 2
print ('Ejercicio 2')
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
            barrio.append(barrioscsv[i+1][1])
    muestrascrudas.append(muestracrudai)
    muestrasnum.append(muestranumi)
    muestrasrta.append(muestrartai)
    largomuestras.append(len(muestranumi))
    
#%%
"""    
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
"""

#%%
    
#Ejer 3a
print ('Ejercicio 3.1')
patmax=patentes('AD592MF')[0] 
lendata=len(muestratotnum)

#Uniforme escalonada: (teórica)
  
  

def unifcdf(x):  #Se asume que los x a los que se les aplica la función estan en el intervalo [1,patmax]
    return ((x-1)/(patmax-1))

ks,pvp=st.kstest(muestratotnum,unifcdf)


tablakolm=1.36*lendata**-0.5
"""
if ks<tablakolm:
    print ('No puedo rechazar la hipótesis nula')
else:
    print ('Rechazo la hipótesis nula')
"""    
pv=kolmogorov(ks*lendata**0.5)
alpha=0.05
if pv>alpha:
    print ('No puedo rechazar la hipótesis nula')
else:
    print ('Rechazo la hipótesis nula')

plt.figure()

x=np.linspace(0,0.15,2000)
y=kolmogorov(x*lendata**0.5)

plt.plot(x,y,'r.')
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
print ('Ejercicio 3.2')
#Exponencial: (teórica)
#Ejer 3.b: Calculo la potencia del test de Kolmogorov del item anterior (es decir, considerando que H0 es la distribución uniforme): Debo calcular la probabilidad de rechazar H0 (ksexp>tablakolm) dado que los datos poseen una distribución exponencial con parámetro lambdaa=4*10**-7 (o sea, fueron generados con dicha distribución).

lambdaa=4*10**-7

totintentos=10**4
rech=0
for i in range(totintentos):
    muestraexp=np.random.exponential(lambdaa**-1,lendata)
    ksexp,pvexp=st.kstest(muestraexp,unifcdf)
    if ksexp>tablakolm:
        rech=rech+1
Potencia=rech/totintentos
print (Potencia)



#%%
    
#Ejer 4.a: La patente más nueva
print ('Ejercicio 4.1')
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

a=patentes('AD200AA')[0]
b=patentes('AD592MF')[0]
a1=1
b1=b-a


"""
k=25 #observo 25 patentes de las más grandes y me quedo con el máximo 
n1=265309 #Me quedo con los últimos 265309 valores posibles de las patentes (los 265309 valores más grandes)
N=10 #repito lo de observar 25 patentes y quedarme con el máximo de aquéllas unas 1000 veces
"""

n1=b-a
k=1000
N=1000

y=experimento(N,k,a,b)#máximos de cada muestra 



plt.figure()

bins=np.linspace(np.min(y),np.max(y),50)
n, bins, patches = plt.hist(y, bins, normed=1, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i intentos' %(N))


#m=np.linspace(a,b,b-a+1) #m>=k, n1>=k, m<=n1
npuntos=4000  #Para cubrir lo necesario del gráfico
m=np.linspace(n1-npuntos,n1,npuntos+1) #m>=k, n1>=k, m<=n1
y2=distribuciondem(m,k,n1)
m=m+a
#y2=[]
#for i in m:
#    y2.append(distribuciondem(i,k,n1))
#    print (i)
"""
#m=np.linspace(a,b,b-a+1) #m>=k, n1>=k, m<=n1
m=np.linspace(k,n1,n1-k+1) #m>=k, n1>=k, m<=n1
y2=[]
for i in m:
    y2.append(distribuciondem(i,k,n1))
    print (i)
"""
plt.plot(m,y2,'r.')
plt.grid(1)

#El gráfico tiende asintóticamente al valor de n 

#%%    

#Ejer 4.b: La patente más nueva

print ('Ejercicio 4.2')

def distribuciondem(m,k,n1): #distribución de probabilidad de m variando k 
    if k==1:
        return 1/n1
    else:
        return k*(m-k+1)*distribuciondem(m,k-1,n1)*((k-1)*(n1-k+1))**(-1)
k=277 
m=265109
probmfijodadon=[]
ncotasup=265109+264+8000
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
std=(np.sum(ynorm*(listan-esperanza)**2))**0.5

esperteor=(m-1)*(k-1)/(k-2)

stdteor=((k-1)*(m-1)*(m-k+1)*(k-2)**-2/(k-3))**0.5

plt.figure()

plt.plot(listan,ynorm,'b.')    
plt.grid(1)

#%%

#Intento del 4b con experimento
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


#Ejer 5

print ('Ejercicio 5.1')    

patentesanmartin=[] #San Martin es el Barrio 0

with open('patentes san martin.txt', "r") as f:
    for line in f:
        patentesanmartin.append(patentes(line)[0])
        
        
patentesrecoleta=muestrasnum[0]+muestrasnum[1]+muestrasnum[2]+muestrasnum[4]+muestrasnum[5]+muestrasnum[6]  #Recoleta es el Barrio 1

patentesanmartin=list(dict.fromkeys(patentesanmartin))   #Elimino las patentes repetidas
patentesrecoleta=list(dict.fromkeys(patentesrecoleta))   #Elimino las patentes repetidas
  

patentestot=np.array(patentesanmartin+patentesrecoleta)
patentestot_sort=np.sort(patentestot)

barriosort=[]

#rank=0

rankrecoleta=0
ranksanmartin=0
for i in range(len(patentestot)):
    posiciones=np.where(patentestot==patentestot_sort[i])[0]
    if len(posiciones)==1:
        posicion=posiciones[0]
        if posicion>=len(patentesanmartin):
            barriosort.append(1)
            rankrecoleta=rankrecoleta+i
        else:
            barriosort.append(0)
            ranksanmartin=ranksanmartin+i
    else:
        print ('Hay un problema. Hay repetidos')
    
#    if barriosort[i]!=barriosort[i-1]:
#        rank=rank+1

if len(patentesanmartin)<=len(patentesrecoleta):
    w=ranksanmartin
    n=len(patentesanmartin)
    m=len(patentesrecoleta)
else:
    w=rankrecoleta
    m=len(patentesanmartin)
    n=len(patentesrecoleta)    


print (w)

Ew=(n+m+1)*n/2
Vw=n*m*(n+m+1)/12
sw=Vw**0.5

def Qsn(p):  #Defino la quantile function de la standard normal distribution N(0,1)
    return(2**0.5*erfinv(2*p-1))


if w>Ew:
    z=(w-Ew-1/2)/sw

if w<Ew:
    z=(w-Ew+1/2)/sw


pv=2*np.min([1/2 + erf(z*2**-0.5)/2,1-(1/2 + erf(z*2**-0.5)/2) ])   #Calculo el pvalor en un two sided test #Evaluo la CDF de N(0,1) en z 

print ('El estadístico y el pvalor del test de Wilcoxon son, respectivamente:')
print (z)
print (pv)


zp,pvp=st.ranksums(patentesrecoleta,patentesanmartin)  #Calculo esto para comparar

