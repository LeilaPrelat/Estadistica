# -*- coding: utf-8 -*-
"""
Created on Monday April 1 17:54:41 2019

@author: Leila
"""
from random import uniform  
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import time
import os
from scipy.special import gamma

tamletra=12
tamlegend=10

directorio=input('¿En qué carpeta guardaste los txt?')

comparacion_discreta_continua=0  
frodensen_cdf_Dn=0 
simulacion_pdf_Dn=0
dist_de_max=0
dist_stirling=0

#%% 2.Estadística en la calle

print ('Ejer 2: Convertir patentes a números y viceversa')

os.chdir(directorio)
start_time_tot=time.time()

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

patentesrecoleta=[] 
with open('patentesrecoleta.txt', "r") as f:
    for line in f:
        patentesrecoleta.append(patentes(line))
        
patentesrecoleta=list(dict.fromkeys(patentesrecoleta))   #Elimino patentes repetidas (por si hay) 
lendata=len(patentesrecoleta)

#%% 3.1¿Uniformemente distribuidas?
    
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

print ('Ejer 3.1: Sn(x) y Fo(x)') 

#Construir Sn(x):
patentesrecoleta_sort=[1]+list(np.sort(patentesrecoleta))+[patmax]
pk =[0]+list(np.linspace(1/len(patentesrecoleta),1,len(patentesrecoleta)))+[1]

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

Dn_obs,Patente_Dn_obs,ecdf_obs,cdf_obs=Kolmogorov(patentesrecoleta, unif_cdf)

#Comparamos con la tabla de Kolmogorov
tablakolm=1.36*(lendata**(-1/2))

plt.figure()
plt.plot(valores_patentes,ytc,'g-', label='Distribución teórica $F_o(x)$')
plt.step(patentesrecoleta_sort,pk,'b',where='post', label='Distribución de Kolmogorov') 

for j in range(len(Patente_Dn_obs)):
    plt.plot([Patente_Dn_obs[j],Patente_Dn_obs[j]],[cdf_obs[j],ecdf_obs[j]],'k-',label='Patente del $D_n$ observado: (%s)' %(inversapatentes(Patente_Dn_obs[j])))

plt.ylabel('Probabilidad',fontsize=tamletra)  
plt.xlabel('Patentes',fontsize=tamletra)  
plt.legend(loc='best',fontsize=tamlegend)

if Dn_obs<=tablakolm:
    print('Se acepta Ho con una significancia de 0,05')
else:
    print('Se rechaza Ho con una significancia de 0,05')

#Para calcular el P value, uso la distribución acumulada del estadístico:    

#Distribución del estadístico (fórmula (3) del informe)
def D(z,cotsupder):
    sumder=0
    for j in range(1,cotsupder):
        sumder=sumder+(-1)**(j-1)*np.e**(-2*j**2*z**2)
    return 1-2*sumder
cotsupder=15**3

#Fórmula (4) del informe: 
pvalue_F=1-D(Dn_obs*len(patentesrecoleta)**(0.5),int(cotsupder))
print('El p value con la cdf del estadístico es:', pvalue_F) 
plt.title('Ejer 3.1: $S_n(x)$ y $F_o(x)$, el $D_n$ observado es %.4f y el P value es %.4f' %(Dn_obs,pvalue_F))
plt.grid(1)

# Distribución acumulada del estadístico del Frodensen (fórmula (3) del informe):

if frodensen_cdf_Dn==1:
    z=np.linspace(0.005,2,400)
    y=D(z,cotsupder)
    alpha=0.05
    x=z*len(patentesrecoleta)**(-1/2)
    
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

#%% Simulación de la densidad de distribución del estadístico Dn:

def Error_bines(nbins,bins,N):
    xerror=[] 
    for i in range(len(bins)-1):
        xerror.append((bins[i+1]+bins[i])*0.5)
    error=[]
    for j in range(len(nbins)): 
        error.append(((nbins[j]/N)*(1-nbins[j]/N))**(1/2))
    return (xerror,error)

N=10**3
Dn=[]
for i in range(N):
    lista_data=[]
    for i in range(lendata):
        lista_data.append(int(uniform(1,patmax+1)))
    Dn.append(Kolmogorov(lista_data,unif_cdf)[0])

pvalue_s=len(np.where(Dn>=Dn_obs)[0])/len(Dn)
print('El pvalue del D_obs con %i simulaciones de la pdf del estadístico:' %(N),pvalue_s) 

if simulacion_pdf_Dn==1:
    plt.figure()
    bins=np.linspace(np.min(Dn),np.max(Dn),50)
    plt.title('Densidad de distribución de $D_n$ con simulaciones, el P value es %.2f' %(pvalue_s))
    plt.xlabel('Valores de $D_n$',fontsize=tamletra) 
    nbins, bins, patches = plt.hist(Dn, bins, density=1,facecolor='green', alpha=1,  edgecolor='black', linewidth=1.2, label='Histograma con %i simulaciones'%(N))    
    plt.errorbar(Error_bines(nbins,bins,N)[0],nbins,Error_bines(nbins,bins,N)[1],linestyle='None',label='Error')
    plt.axvline(x=Dn_obs,label='Estadístico $D_n$ observado')
    plt.legend(loc='best',fontsize=tamlegend)
    plt.grid(1)

#%% 3.2.Distribución exponencial
    
print ('Ejer 3.2: Distribución exponencial') 

lambbda=4*10**(-7)

N=10**3
exito=0
lista_dn=[]
for i in range(N):
    lista_data=np.random.exponential(lambbda**(-1),lendata)
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

#%% 4.1.La patente del auto más nuevo: Distribución de m

print ('Ejer 4.1: Distribución de m') 

def P_m(m,k,n): 
    if k==1:
        return 1/n
    else:
        return k*(m-k+1)*P_m(m,k-1,n)*((k-1)*(n-k+1))**(-1)    
 
#Simulamos la distribución de m y la comparamos con su teórica P_m(m,k,n)
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
m=np.max(patentesrecoleta)
inf=1*13**4
k=lendata
x_P=np.arange(m-inf,b) 
y_P=P_m(x_P-a,k,b-a)

N=15**3
y_exp=experimento(N,k,a,b)

plt.figure()
plt.xlim([(m-inf),b*1.0005])
plt.plot(x_P,y_P,'r.', label='Distribución P(m;k,n)')
bins=np.linspace(np.min(y_exp),np.max(y_exp),80)
nbins, bins, patches = plt.hist(y_exp, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))
plt.errorbar(Error_bines(nbins,bins,N*k)[0],nbins,Error_bines(nbins,bins,N*k)[1],linestyle='None',label='Error') #Como se usa N veces el experimento, los intentos son N*k
plt.title('Ejer 4.1: Distribución de m')
plt.xlabel('Patentes',fontsize=tamletra) 
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

#Podemos utilizar como distribución de m la del máximo de la guía 3: (aproximamos a una continua)
if dist_de_max==1:
    def distdemax(n,k,x):
        return n**(-1)*k*(x/(n-1)-1/(n-1))**(k-1)
    
    x=x_P
    y_max=distdemax(patmax,k,x)
    
    plt.figure()
    plt.xlim([(m-inf),b*1.0005])
    plt.plot(x,y_max,'k.', label='Distribución del m con G3E8')
    bins=np.linspace(np.min(y_exp),np.max(y_exp),80)
    nbins, bins, patches = plt.hist(y_exp, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))
    plt.errorbar(Error_bines(nbins,bins,N*k)[0],nbins,Error_bines(nbins,bins,N*k)[1],linestyle='None',label='Error')
    plt.title('Ejer 4.1: Distribución de m con G3E8')
    plt.legend(loc='best',fontsize=tamlegend)
    plt.xlabel('Patentes',fontsize=tamletra)
    plt.grid(1)  
    
# Haciendo lo mismo con la aprox stirling, el tiempo computacional es casi el mismo y se nota la diferencia entre la distribucion de m con stirling y con la fórmula
if dist_stirling==1:    
    def likelihood_stirling(m,k,n):
        return np.log(k)+(n-k)*np.log(n-k)+(m-1)*np.log(m-1)-n*np.log(n)-(m-k)*np.log(m-k)
    
    y_P2=y_P
    y_stirling=np.e**(likelihood_stirling(x_P-a,k,b-a))
    
    plt.figure()
    plt.xlim([(m-inf),b*1.0005])
    plt.plot(x_P,y_P,'r.', label='Distribución P(m;k,n)')
    plt.plot(x_P,y_stirling,'b.', label='Distribución P(m;k,n) con stirling')
    plt.plot(x,y_max,'k.', label='Distribución del m con G3E8')
    plt.title('Comparación de distintas formas para obtener P(m;k,n)')
    plt.legend(loc='best',fontsize=tamlegend)
    plt.xlabel('Patentes',fontsize=tamletra)
    plt.grid(1)  

#%% La patente del auto más nuevo. Item 2: P(n;k,m) a partir de P(m;k,n)

print ('Ejercicio 4.2: P(n;k,m) a partir de P(m;k,n)')    

def P_n(k,m,ncotasup):
    rango_n=np.linspace(m,ncotasup,ncotasup-m+1) #Rango en el que variamos n
    y=P_m(m,k,rango_n)
    ynorm=y/np.sum(y)
    esperanza_n=np.sum(rango_n*ynorm)
    std_n=(np.sum(ynorm*(rango_n-esperanza_n)**2))**0.5
    return (rango_n,ynorm,esperanza_n,std_n)

b=patmax
m=np.max(patentesrecoleta) + 7000 #pongo un m distinto al mio
k=50 #pongo un k distinto al mío 
ncotasup=b+5*int((m-k)/k)
rango_n,ynorm,esperanza_n,std_n=P_n(k,m,ncotasup)

plt.figure()
plt.plot(rango_n,ynorm,'b.',label='P(n;k,m) con k=%i y m=%i(%s)' %(k,m,inversapatentes(m)))    
plt.title('Ejer 4.2: P(n;k,m) a partir de P(m;k,n)')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Rango de n', fontsize=tamletra)
plt.grid(1)

esperanza_obtenida=esperanza_n
std_obtenida=std_n

esperanza_teorica=(m-1)*(k-1)/(k-2)
std_teorica=((k-1)*(m-1)*(k-3)**(-1)*(k-2)**(-2)*(m-k+1))**0.5

print('Valor real de n:', patmax)
print('Intervalo de n obtenido', np.round(esperanza_obtenida-std_obtenida,3),np.round(esperanza_obtenida+std_obtenida,3))
print('Error porcentual de la estimación obtenida',np.round((std_obtenida/esperanza_obtenida)*10**2,3))
print('Intervalo de n teórico', np.round(esperanza_teorica-std_teorica,3),np.round(esperanza_teorica+std_teorica,3))

if esperanza_obtenida-std_obtenida<=patmax<=esperanza_obtenida+std_obtenida: 
    print('El valor real de n está dentro de la estimación bayesiana obtenida :)')
else:
    print('El valor real de n no está dentro de la estimación bayesiana obtenida')

#%% La patente del auto más nuevo. Item 3: Evaluar en k y m obtenidos. 

print ('Ejercicio 4.3: P(n;k,m) con k y m de los datos') 
  
#Uso mis datos:
k=lendata
m=np.max(patentesrecoleta)

print('Valor real de n:', patmax)
print('Intervalo de n obtenido', np.round(esperanza_obtenida-std_obtenida,3),np.round(esperanza_obtenida+std_obtenida,3))
print('Error porcentual de la estimación obtenida',np.round((std_obtenida/esperanza_obtenida)*10**2,3))
print('Intervalo de n teórico', np.round(esperanza_teorica-std_teorica,3),np.round(esperanza_teorica+std_teorica,3))

if esperanza_obtenida-std_obtenida<=patmax<=esperanza_obtenida+std_obtenida: 
    print('El valor real de n está dentro de la estimación bayesiana obtenida :)')

plt.figure()
plt.plot(rango_n,ynorm,'g.',label='P(n;k,m) con k=%i y m=%i(%s)' %(k,m,inversapatentes(m)))    
plt.title('Ejer 4.3: P(n;k,m) a partir de P(m;k,n)')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Rango de n', fontsize=tamletra)
plt.grid(1)

#%% La patente del auto más nuevo. Item 4: (sección que más tarda en ejecutarse)

print ('Ejercicio 4.4: Estimaciones bayesianas para n')   

#Generar N valores para m con el experimento 
N=15**3
rango_m=experimento(N,k,1,b)
n1=P_n(k,m,ncotasup)[2] #estimación de n con mi m 

estpeor=0
estimaciones=[]
for i in range(N):
    est=P_n(k,rango_m[i],ncotasup)[2]
    estimaciones.append(est)
    if np.abs(est-b)>=np.abs(n1-b): 
        estpeor=estpeor+1

P_estpeor=estpeor/N
print ('La probabilidad de obtener una peor estimación es:', P_estpeor)

plt.figure()   
bins=np.linspace(np.min(estimaciones),np.max(estimaciones),50)
nbins, bins, patches = plt.hist(estimaciones, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma con %i experimentos' %(N))
plt.plot([b,b],[0,0.00010],'b-', label='Última patente existente')
plt.plot([n1,n1],[0,0.00010],'r-',label='Estimación obtenida con m=%i'%(m))
plt.plot([2*b-n1,2*b-n1],[0,0.00010],'k-',label='Patente equidistante')
plt.errorbar(Error_bines(nbins,bins,N*k)[0],nbins,Error_bines(nbins,bins,N*k)[1],linestyle='None',label='Error')
plt.legend(loc='best',fontsize=tamlegend)
plt.title('Ejer 4.4: Estimaciones bayesianas de n. P value= %.3f'%(P_estpeor))
plt.xlabel('Estimaciones bayesianas de n', fontsize=tamletra)
plt.grid(1)

#%% ¿Independiente del barrio?

print ('Ejercicio 5.1: Wilcoxon')   

patentesanmartin=[] 
os.chdir(directorio)
with open('patentes san martin.txt', "r") as f:
    for line in f:
        patentesanmartin.append(patentes(line))
                      
patentesanmartin=list(dict.fromkeys(patentesanmartin))   #Elimino las patentes repetidas

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

z_obs,pw_obs=Wilcoxon(patentesanmartin,patentesrecoleta)
print('El estadístico de Wilcoxon y el P value para Recoleta y San Martín:',z_obs,pw_obs)

#%% ¿Independiente del barrio? (Tarda 20 segundos aprox)

print ('Ejercicio 5.2: Test de hipotesis del G8E4')   

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

m=len(patentesanmartin)
n=lendata
N=10**3
valores_de_U=[]
for i in range(N):
    lista1=unif(1,b,m)
    lista2=unif(1,b,n)
    valores_de_U.append(U(lista1,lista2))

Uobs=U(patentesanmartin,patentesrecoleta)
pt_obs=2*np.min([len(np.where(valores_de_U<=Uobs)[0])/len(valores_de_U),len(np.where(valores_de_U>=Uobs)[0])/len(valores_de_U)])

print('El estadístico observado y el P value del test G8E4 son:', Uobs,pt_obs)
    
plt.figure()
bins=np.linspace(np.min(valores_de_U),np.max(valores_de_U),80)   
nbins, bins, patches = plt.hist(valores_de_U, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma de %i valores de U' %(N))
plt.errorbar(Error_bines(nbins,bins,N)[0],nbins,Error_bines(nbins,bins,N)[1],linestyle='None',label='Error')
plt.title('Ejer 5.2: Distribución de densidad de U, con pt=%.3f'%(pt_obs))
plt.axvline(x=Uobs,label='Estadístico observado')
plt.axvline(x=-Uobs,label='-Estadístico observado (asumiendo simetría)')
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Valores del estadístico U', fontsize=tamletra)
plt.grid(1)

#%% Combinando los tests

print ('Ejercicio 6.1: Combinando los tests') 

def T(pw,pv):
    return -2*np.log(pw*pv)

def gamma_pdf(x,alfa,beta): #cuando alfa es entero
    return x**(alfa-1)*np.e**(-x/beta)*beta**(-alfa)*gamma(alfa)**(-1)

N=10**4
pw=0.2
pt=0.24

alfa=2 #sume dos exponenciales
beta=2 #lambda es 1/2 y beta es 1/lambda

#Test independendientes: (el valor de pw no influye en el de pt y viceversa)
lista_pw=unif(0,1,N)
lista_pt=unif(0,1,N)

#Calculo correlación entre pvtot y pwtot (no prueba que sean independientes, lo calculo de chusma)
lista_pw=np.array(lista_pw)
lista_pt=np.array(lista_pt)
corr=np.mean((lista_pw-np.mean(lista_pw))*(lista_pt-np.mean(lista_pt)))*(np.std(lista_pt))**-1*(np.std(lista_pw))**-1
print ('La correlacion entre las listas simuladas de manera independiente es:',corr)

lista_T=[]
for i in range(N):
    lista_T.append(T(lista_pw[i],lista_pt[i]))

lista_x=np.linspace(0,20,1000)
lista_y=gamma_pdf(lista_x,alfa,beta)

#Esperanza del histograma
mean=0
for i in range(len(bins)-1):
    mean=mean+(bins[i+1]-bins[i])*(nbins[i])
    
plt.figure()  
plt.plot(lista_x,lista_y,'r.',label='Pdf de gamma(x,2,2) (su esperanza es 1)') 
bins=np.linspace(np.min(lista_T),np.max(lista_T),30)
nbins, bins, patches = plt.hist(lista_T, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma de %i valores de T(pw,pt), esperanza=%.3f y corr=%.3f' %(N,mean,corr))
plt.errorbar(Error_bines(nbins,bins,N)[0],nbins,Error_bines(nbins,bins,N)[1],linestyle='None',label='Error')    
plt.title('Ejer 6.1: Distribución de densidad de T(pw,pt)')    
plt.legend(loc='best',fontsize=tamlegend)
plt.xlabel('Valores del estadístico T', fontsize=tamletra)
plt.grid(1)
    
#%%
#Generar dos muestras con la misma esperanza (a+b)/2 y distribucion uniforme (para usar lo del ejer 5)

print ('Ejercicio 6.2: P values correlacionados')   #Advertencia!: Tarda 90 segundos aprox

N=10**3

a=1
b=patmax
lenlista1=500

a2=a
b2=b
lenlista2=600

pvtot=[]
pwtot=[]
for i in range(N):
    lista1=unif(a,b,lenlista1)
    lista2=unif(a,b,lenlista2)

    Uobs=U(lista1,lista2)
    pvtot.append(2*np.min([len(np.where(valores_de_U<=Uobs)[0])/len(valores_de_U),len(np.where(valores_de_U>=Uobs)[0])/len(valores_de_U)]))
    pwtot.append(Wilcoxon(lista1,lista2)[1])
    
#Calculo correlación entre pvtot y pwtot (están correlacionados porque usamos las mismas listas)
pvtot=np.array(pvtot)
pwtot=np.array(pwtot)
corr=np.mean((pvtot-np.mean(pvtot))*(pwtot-np.mean(pwtot)))*(np.std(pvtot))**-1*(np.std(pwtot))**-1
print ('La correlacion entre pv y pw es:',corr)

#Esperanza del histograma
mean=0
for i in range(len(bins)-1):
    mean=mean+(bins[i+1]-bins[i])*(nbins[i])

T_Ho_cierta=[]
for i in range(N):
    Ti=T(pwtot[i],pvtot[i])
    if np.isfinite(Ti): #quiero sólo numeros finitos (algunos valores son -inf cuando los p values tomaron valores muy chicos)
        T_Ho_cierta.append(Ti)

plt.figure()   
bins=np.linspace(np.min(T_Ho_cierta),np.max(T_Ho_cierta),60)
nbins, bins, patches = plt.hist(T_Ho_cierta, bins, density=True, facecolor='green', alpha=1,  edgecolor='black', linewidth=0.5, label='Histograma de %i valores de T(pw,pt), esperanza=%.3f y corr=%.3f' %(N,mean,corr))
plt.errorbar(Error_bines(nbins,bins,N)[0],nbins,Error_bines(nbins,bins,N)[1],linestyle='None',label='Error')

#Comparamos con la dist de T anterior
alfa=2 #sume dos exponenciales
beta=2 #lambda es 1/2 y beta es 1/lambda

lista_x=np.linspace(0,20,1000)
lista_y=gamma_pdf(lista_x,alfa,beta)

plt.plot(lista_x,lista_y,'r.',label='Distribución de densidad teórica de gamma(x,2,2)')

pwobs=Wilcoxon(patentesanmartin,patentesrecoleta)[1]
Uobs=U(patentesanmartin,patentesrecoleta)
pvobs=2*np.min([len(np.where(valores_de_U<=Uobs)[0])/len(valores_de_U),len(np.where(valores_de_U>=Uobs)[0])/len(valores_de_U)])

T_obs=T(pwobs,pvobs)
pvalue_Tobs=len(np.where(T_Ho_cierta>=T_obs)[0])/len(T_Ho_cierta)
print('El pvalue del T_obs es:',pvalue_Tobs) 

plt.title('Ejer 6.2: Pdf de T(pw,pt) cuando Ho es cierta, con P value=%.3f'%(pvalue_Tobs))
plt.xlabel('Valores del estadístico T', fontsize=tamletra)  
plt.axvline(x=T_obs,label='Estadístico T observado')
plt.legend(loc='best',fontsize=tamlegend)
plt.grid(1)

#%% Sé tu propia verduga

print ('Ejer 7: Sé tu propia verduga') 

def P_m_cdf(m,k,n):
    return P_m(m,k,n)*m/k

m=np.max(patentesrecoleta)
b=patmax
k=lendata

print('El P value para la patente máxima como estadistico es', np.round(P_m_cdf(m,k,b),5)) #Test a cola izquierda (algo peor es un m mas chico)

patentesrecoletaalreves=patmax-np.array(patentesrecoleta)+1
minima=np.max(patentesrecoletaalreves) #Es la patente mínima!

print('El P value para la patente mínima como estadistico es', np.round(P_m_cdf(minima,k,b),5)) #Test a cola izquierda (algo peor es un m mas chico)

#%%

print ('Tiempo total de ejecución en minutos:', (time.time()-start_time_tot)/60)
    
    
    
    
    
    