# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:24:53 2019

@author: Usuario
"""
import numpy as np
import matplotlib.pyplot as plt

#%%

#Ejer 2

nummuestras=7 #cantidad de txt 

ABC=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
abc=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def patentes(lista):
    a,b,c,d,e,f,g=lista
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

num, rta=patentes(['Z','Z',9,9,9,'Z','Z'])
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

    
#%%

#Ejer 3. Discreto:

def uniformeacumulativa(a,b):
    m=1/(b-a)
    x=np.linspace(a,b,b-a+1)
    y=[]
    for i in range(len(x)):
        y.append(x[i]*m-m)
    for j in range(len(x)):
        xt=[]
        yt=[]
        deltax=np.linspace(x[j],x[j+1]-1,x[j+1]-x[j])
        for l in deltax:
            xt.append(l)
            yt.append(y[j])
    return (x,y)

x1=uniformeacumulativa(1,2428510)[0]
y1=uniformeacumulativa(1,2428510)[1]

muestratotnum_sort=np.sort(muestratotnum)

#k1=np.linspace(1,muestratotnum_sort[0]-1, muestratotnum_sort[0]-1-1+1) #asociarles el cero
#ky1=[]
#for j in range(muestratotnum_sort[0]-1):
#    ky1.append(0)
#
#k2=muestratotnum_sort
#ky2=[]
#kky2=np.linspace(1,len(muestratotnum_sort),len(muestratotnum_sort))
#for j in kky2:
#    ky2.append(j/len(muestratotnum_sort))
#
#k3=(muestratotnum_sort[-1],2428510, 2428510-muestratotnum_sort[-1]+1) 
#ky3=np.ones(2428510-muestratotnum_sort[-1]+1)  
#
plt.figure()
plt.plot(x1, y1, 'b.', label='Distribución acumulativa de la uniforme (teórica)')

#kxtot=[]
#for j in range(len(k1)):
#    kxtot.append(k1[j])
#for l in range(len(k2)):
#    kxtot.append(k2[l])
#for m in range(len(k3)):
#    kxtot.append(k3[m])
#
#kytot=[]
#for j in range(len(ky1)):
#    kytot.append(ky1[j])
#for l in range(len(ky2)):
#    kytot.append(ky2[l])
#for m in range(len(ky3)):
#    kytot.append(ky3[m])
#
#
#plt.plot(k1, ky1, 'r.')
#plt.plot(k2, ky2, 'r.')
#plt.plot(k3, ky3, 'r.')

a=1
b=2428510
x=np.linspace(a,b,b-a+1)

y=[]
for i in range(muestratotnum_sort[0]-1):
    y.append(0)
for j in range(len(muestratotnum_sort)+1):
    y.append(j/len(muestratotnum_sort))
for l in range(2428510-muestratotnum_sort[-1]-1):
    y.append(1)
    
plt.plot(kxtot, kytot, 'g.')

#%%
#Ejer 3. Continuo:
    
#Uniforme

#patentes(['A','D',5,9,2,'M','F'])=(2428510, [1, 4, 5, 9, 2, 13, 6])
        
def uniformeacumulativa(a,b):
    m=1/(b-a)
    x=np.linspace(a,b,b-a+1)
    y=[]
    for i in range(len(x)):
        y.append(x[i]*m-m)
    return (x,y)

x1=uniformeacumulativa(1,2428510)[0]
y1=uniformeacumulativa(1,2428510)[1]

plt.figure()
plt.plot(x1, y1, 'b.', label='Distribución acumulativa de la uniforme (teórica)')
muestratotnum_sort=np.sort(muestratotnum)
p = 1. * np.arange(len(muestratotnum)) / (len(muestratotnum) - 1)
plt.step(muestratotnum_sort,p)
plt.grid(1)

#Kolmogorov

#muestratotnum_sort=np.sort(muestratotnum) #ordenado de menor a mayor 
#
#def kolmogorov(a,b):
#    x=np.linspace(a,b,b-a+1)
#    y=[]
#    for j in range(len(x)):
#        if x[j]<muestratotnum_sort[0]:
#            y.append(0)
#    for j in range(len(x)):
#        if muestratotnum_sort[0]<=x[j]<=muestratotnum_sort[len(muestratotnum_sort)-1]:
#            y.append(j/len(muestratotnum_sort))
#    for j in range(len(x)):
#        if muestratotnum_sort[len(muestratotnum_sort)-1]<x[j]:
#            y.append(1)
#    return (x,y)
#    
#x2=kolmogorov(1,2428510)[0]
#y2=kolmogorov(1,2428510)[1]    
#plt.figure()
#plt.plot(x2, y2, 'r-', label='Distribución acumulativa de la uniforme (teórica)')
#plt.grid(1)


#%%

#Ejer 3: Discreto 

xt=np.linspace(1,2428510,2428510)
yt=np.linspace(0,1,2428510)

plt.figure()

plt.step(xt,yt,'r')

muestratotnum_sort=[1]+list(np.sort(muestratotnum))+[2428510]

pk =[0]+list(np.linspace(1/len(muestratotnum),1,len(muestratotnum)))+[1]

plt.step(muestratotnum_sort,pk,'b',where='post')
plt.grid(1)

#Para calcular la resta hay que rellenar la Kolmogorov:
  
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
    

    
    
    
    
    
    
    
    
    




