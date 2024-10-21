import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import sparse

#Queremos programar una funcion que resuleva la ecuacion con el metodo explicito Euler forward

def Euler_forward(x0,v0,h,w,alpha,tsim):
    #Construyo A
    Id = np.identity(2)
    B = np.array([[0,1],[-w**2,-alpha]])
    A = Id + h*B

    #Creo los x y v de longitud el tiempo de simulacion 
    t = np.arange(0,tsim,h)
    xa = np.zeros(len(t))
    v = np.zeros(len(t))
    #Pongo valor inicial
    xa[0] = x0
    v[0] = v0
    #Lleno las listas
    for i in range(1,len(t)):
        xa[i],v[i] = np.dot(A,np.array([xa[i-1],v[i-1]]))
    return t,xa,v 

#Queremos programar una funcion que resuleva la ecuacion con el metodo implicito Euler backward

def Euler_backward(x0,v0,h,w,alpha,tsim):
    #Construyo A
    Id = np.identity(2)
    B = np.array([[0,1],[-w**2,-alpha]])
    A = Id - h*B
    A = np.linalg.inv(A)
    #A=np.array([[1,h],[-h*w**2,-h*alpha+1]])
    #Creo los x y v de longitud el tiempo de simulacion 
    t = np.arange(0,tsim,h)
    xb = np.zeros(len(t))
    v = np.zeros(len(t))
    #Pongo valor inicial
    xb[0] = x0
    v[0] = v0
    #Lleno las listas
    for i in range(1,len(t)):
        xb[i],v[i] = np.dot(A,np.array([xb[i-1],v[i-1]]))
        
    return t,xb,v 

#Queremos programar una funcion que resuleva la ecuacion con el metodo semi-implicito Crank-Nicolson

def Cronk(x0,v0,h,w,alpha,tsim):
    #Creamos las matrices
    Id = np.identity(2)
    B=np.array([[0,1],[-w**2,-alpha]])
    A1=Id+h*B
    A=Id-h*B
    A2=np.linalg.inv(A)
    Acn=0.5*(A1+A2)
    #Creo los x y v de longitud el tiempo de simulacion 
    t=np.arange(0,tsim,h)
    xc=np.zeros(len(t))
    v=np.zeros(len(t))
    #Pongo valor inicial    
    xc[0]=x0
    v[0]=v0
    #Lleno las listas
    for i in range(1,len(t)):
        xc[i],v[i]=np.dot(Acn,np.array([xc[i-1],v[i-1]]))
    return t,xc,v


#Falta comparar los valores de cada metodo con el exacto

def Exacto(alpha,w,x_0,v_0,t_final):
    t=np.arange(0,t_final,h)
    Amp=np.sqrt(x_0**2+((v_0+0.5*alpha*x_0)/(w))**2)
    ang=np.arctan((x_0*w)/(v_0+0.5*alpha*x_0))
    return Amp*np.exp(-0.5*alpha*t)*np.sin(w*t+ang)

#Doy valores iniciales
x0=1
v0=0
h=0.1    #variacion de tiempo, probar para varios y comprobar la estabilidad del metodo
w=1.2        #frecuencia angular
alpha=0.2    #coeficiente de amortiguamiento
tsim=30      #tiempo de simulacion
t = np.arange(0,tsim,h)

#Falta crear y añadir las diferencias de cada metodo con la exacta
dife1 = np.zeros(len(t))
dife2 = np.zeros(len(t))
dife3 = np.zeros(len(t))

Sol1 = Euler_forward(x0,v0,h,w,alpha,tsim)
Sol2 = Euler_backward(x0, v0, h, w, alpha, tsim)
Sol3 = Cronk(x0, v0, h, w, alpha, tsim)
Sol4 = Exacto(alpha, w, x0, v0, tsim)

xe = Sol4
xa = Sol1[1]
xb = Sol2[1]
xc = Sol3[1]

dife1 = abs(xa - xe)
dife2 = abs(xb - xe)
dife3 = abs(xc - xe)

#Graficamos
plt.figure()
plt.plot(Sol1[0],Sol1[1],label='Euler forward')
plt.plot(Sol2[0],Sol2[1],label='Euler backward')
plt.plot(Sol3[0],Sol3[1],label='Cronk')
plt.plot(Sol4[0],Sol4[1],label='Exacta')
plt.xlabel('tiempo (s)')
plt.ylabel('posición x (m)')
plt.title(label='OSCILADOR ARMÓNICO AMORTIGUADO h  = 1e-01s')
#plt.savefig('Osc.arm.amor. h=0.1.png')
plt.legend(loc='best')

plt.figure()
plt.plot(t, dife1, label='Dif Euler forward')
plt.plot(t, dife2, label='Dif Euler backwards')
plt.plot(t, dife3, label='Dif Euler cronk')
plt.legend(loc='best')
plt.xlabel('tiempo (s)')
plt.ylabel('error')
plt.title(label='ERROR DE LOS MÉTODOS')
#plt.savefig('Errores h=0.00001.png')
plt.show()






