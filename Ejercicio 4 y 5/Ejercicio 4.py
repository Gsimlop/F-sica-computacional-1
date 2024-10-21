#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:34:24 2023

@author: gabriel
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from scipy.sparse import diags



###############################Parte 1###############################

'''
La ecuación de difusión es una EDP que describe fluctuaciones de densidad en un 
material que se difunde. También se usa para describir procesos que exhiben 
un comportamiento de difusión.
Resolveremos por tanto una EDP parabólica.
'''
########Solucion con matrices

#Datos del problema
D = 1e-2 #Coeficiente de difusión
L = 1 #Define el dominio espacial
dx = 5e-2 #Variación espacial
nx = int(L/dx) #Define el número de puntos en los que se discretiza el espacio
t = 100 #Define el dominio temporal
dt = (0.5*dx**2)/(2*D) #Define la variación de tiempo
nt = int(t/dt) #Define el número de puntos en los que se discretiza el tiempo

if 2*D > dx**2/dt: print('No se cumple la condición de Estabilidad') #Es facil comprobar como si D es grande el metodo explota


def difusion_ftcs():
    
    #Creamos la matriz
    diag = -2*np.ones(nx) 
    off = 1*np.ones(nx-1) 
    A = scipy.sparse.diags([diag, off, off], [0,-1,1],shape=(nx,nx)).toarray() #Creamos matriz tridiagonal
    Id = scipy.sparse.eye(nx).toarray() 
    B = Id + (D*dt/dx**2)*A
    
    #Modificamos la matriz para aplicar las condiciones de contorno
    B[0,:] = 0 ; B[0, 0] = 1 ; B[nx-1,:] = 0 ; B[nx-1, nx-1] = 1
    
    #Define el vector de solución inicial, lo usaremos en la animacion
    
    T0 = np.zeros(nx)
    for i in range(1, nx-1):
        T0[i] = 100 #imponemos condicion inicial
    
    T = [] #Definimos una lista para añadir los valores de la solución
    T.append(T0)
    for i in range(nt):
        T_Temp = np.dot(B, T0) 
        T.append(T_Temp) 
        T0 = T_Temp 
    return T


#Animación
def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(x, difusion_ftcs()[i])
    #Actualizar el valor del tiempo
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*i)) 
    return line,time_text



x = np.arange(0, L, dx) #Define el espacio en el que se dibuja   
fig = plt.figure()
ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 120)) #Define el límite de los ejes para que no se muevan
line, = ax.plot([], [], lw=2)
plt.title('Evolución de T de la barra a lo largo del tiempo')
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nt, interval=10, blit=True)
plt.show()
     
x1 = np.linspace(0,L,nx)
t1 = np.linspace(0,t,nt+1)
#La figura al final
plt.figure()
plt.suptitle('Evolución temperatura barra',fontsize=12)

plt.subplot(1, 2, 1)
plt.plot(x1,difusion_ftcs()[0])
plt.title('Barra inicial')
plt.xlabel('Posicion de la T')
plt.ylabel('T barra')
#plt.show()

#La figura al final

plt.subplot(1, 2, 2)
plt.plot(x1,difusion_ftcs()[-1])
plt.ylim(0, 100)
plt.title('Barra final')
plt.xlabel('Posicion de la T')
plt.ylabel('T barra')
plt.show()


plt.figure()

malla1 = np.array(difusion_ftcs())

X,Y = np.meshgrid(x1,t1)

plt.contourf(X, Y, malla1, nx, cmap=plt.cm.hot)

plt.xticks(())
plt.yticks(())
plt.show()

###############################Parte 2###############################

'''
En esta parte tenemos condiciones iniciales distintas y resolveremos 
el problema con el metodo de Crank-Nicolson
'''

Ti = 100*np.ones(nx)  # Condición inicial: temperatura constante de 100°C en todo el dominio 
Ti[0] =  0.0  # Condiciones de contorno: temperatura constante de 0°C en los extremos
Ti[-1] = 50

#######Matrices tridiagonales para el método Crank-Nicolson
#Creamos una matriz A tridiagonal con la ecuación de Cranck-Nicolson 
diag = (1 + D * dt / dx**2)*np.ones(nx)
down = (-D * dt / (2 * dx**2))*np.ones(nx-1)
up = (-D * dt / (2 * dx**2))*np.ones(nx-1)
A = diags([diag, up, down], [0,1,-1], shape = (nx,nx)).toarray()

A[0,:] = 0 ; A[-1,:] = 0 ; A[0, 0] = A[-1, -1] = 1.0

diag = (1 - D * dt / dx**2)*np.ones(nx)
down2 = (D * dt / (2 * dx**2))*np.ones(nx-1)
up2 = (D * dt / (2 * dx**2))*np.ones(nx-1)
B = diags([diag, up2, down2], [0,1,-1], shape = (nx,nx)).toarray()

B[0,:] = 0 ; B[-1,:] = 0 ; B[0, 0] = B[-1, -1] = 1.0

#Creamos una lista a la cual le añadiremos la evolucion temporal de la barra 
#tendrá por tanto longitud del numero de puntos del que discretizamos el tiempo

lista2 = [[] for i in range(nt)]

# Evolución temporal utilizando el método Crank-Nicolson
#Rellenamos lista creada con las soluciones
for n in range(nt):
    Ti = np.linalg.solve(A, np.dot(B, Ti))
    lista2[n] = Ti

#Creamos los linspace para graficar
x2 = np.linspace(0,L,nx)
t2 = np.linspace(0,len(lista2),nt)

#Graficamos

plt.figure()
plt.plot(x2, Ti, label='Temperatura')
plt.xlabel('Posición')
plt.ylabel('Temperatura (°C)')
plt.show()   

#Lo vemos en una malla
plt.figure()

malla2 = np.array(lista2)

X2,Y2 = np.meshgrid(x2,t2)

plt.contourf(X2, Y2, malla2, nx, cmap=plt.cm.hot)

plt.xticks(())
plt.yticks(())
plt.show()

#Animación

fig, ax = plt.subplots()
line, = ax.plot(x2,lista2[0])
plt.title('Evolución de T de la barra a lo largo del tiempo')
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')

def update(i):
    new_data = lista2[i]
    line.set_ydata(new_data)
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*i)) 
    return line,time_text

ani = FuncAnimation(fig,update,frames = len(lista2),interval = 50)
plt.show()



# =============================================================================
# Ejercicio 3
# =============================================================================

'''
Disminuye ahora el coeficiente a D = 10−3 entre x = [0,4, 0,6] lo que puede representar,
por ejemplo, una zona dentro del conductor con un material con un calor especifico mas 
alto y repetiremos el caso anterior. 
En este caso crearemos una matriz que ejerza de derivada y luego otra que aplique las condiciones
Construiremos entonces una matriz con cond. de contorno y calc. la solucion
'''

#Definimos las condiciones a la difusividad pedidas en el problema
D=(1e-2)*np.ones(nx+1)  #La barra tiene un coef. de difusividad de 1e-2 menos 0,4 a 0,6
D[40:61]= 1e-3          #Es un 61 pooque lo queremos hacer hasta el 60
a=D*dt/dx**2        


diag3=(-2)*np.ones(nx+1)
down3=np.ones(nx)
up3 = np.ones(nx)

#Matriz de la derivada
A3=diags([diag3, down3, up3], [0,-1,1],shape=(nx+1,nx+1)).toarray()

#Condiciones del método

B3=np.linalg.inv(np.identity(nx+1)-np.transpose(a/2*A3)).dot(np.identity(nx+1)+np.transpose(a/2*A3))
B3[0,]=B3[nx,]=0 ; B3[0,0]=B3[nx,nx]=1

x3=np.linspace(0,L,nx+1)
t3=np.linspace(0,dt*nt,nt)


#Mostramos esta vez la solución en forma de matriz
Ti3=np.zeros((nt,nx+1))
Ti3[0,1:nx]=100
Ti3[0,nx]=50

for i in range (1,nt):
    Ti3[i] = np.dot(B3,Ti3[i-1])
    

#####Gráficamos

plt.figure()
plt.plot(x3,Ti3[-1,:])
plt.show()

X3,Y3=np.meshgrid(x3,t3)

plt.figure()
plt.contourf(X3,Y3,Ti3,nx,cmap = 'hot')
plt.show()

lista3=Ti3.tolist()

fig, ax = plt.subplots()
line, = ax.plot(x3,lista3[0])
plt.title('Evolución de T de la barra a lo largo del tiempo')
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')

def update(i):
    new_data = lista3[i]
    line.set_ydata(new_data)
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*i)) 
    return line,time_text

ani = FuncAnimation(fig,update,frames = len(lista3),interval = 50)
plt.show()



