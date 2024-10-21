import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation

'''
Se nos pide resolver la ecuación de Schrödinger dependiente del tiempo en 1D con el método de Crank-Nicholson
para una particula libre en un pozo infinito, o para un potencial cuadratico (oscilador armonico)
'''

#Definimos los parametros
L = 10      #Longitud del dominio espacial
Nx = 600
dx = L/Nx    #Paso espacial
t = 0.5
Nt = 2500
dt = t/Nt    #Paso temporal
k = 15*np.pi 
sigma = 0.5

cte = (1j*dt)/(2*dx**2) #Constante que acompaña a la matriz

#Definimos el intervalo espacial

x = np.arange(-L/2, L/2+dx, dx) 

#Lo haremos para una paquete de ondas, por lo tanto sabemos que la solucion correspondiente a valores definidos de la energía 
#y del momento viene dada por una onda plana:
def part_libre(): 
    return np.exp((-1/2)*(x/sigma)**2)*np.exp(1j*k*x)

#Vamos a crear las matrices que vamos a usar en el método de Crank-Nicholson

#Creamos la matriz tridiagonal del k+1

diag1 = (1 + 2*cte)*np.ones(Nx+1) 
sup1 = inf1 = -cte*np.ones(Nx) 
A1 = scipy.sparse.diags([diag1, sup1, inf1], [0,-1,1],shape=(Nx+1,Nx+1), dtype = complex).toarray() 

#Creamos la matriz tridiagonal del k

diag2 = (1 - 2*cte)*np.ones(Nx+1) 
sup2 = inf2 = cte*np.ones(Nx) 
A2 = scipy.sparse.diags([diag2, sup2, inf2], [0,-1,1],shape=(Nx+1,Nx+1), dtype = complex).toarray() 

#Resolvemos para obtener la matriz

B = np.dot(np.linalg.inv(A1), A2) #Genera la matriz final del sistema de ecuaciones

#Modifico B para aplicar las condiciones de contorno

B[:,0] = 0 ; B[:,-1] = 0

#Creamos un array para añadir la solucion e imponemos la condicion de una part. libre

u = np.zeros([Nt+1, Nx+1], dtype = complex) 
u[0] = part_libre() 

#Aplico el método de Crank-Nicholson

for i in range(Nt):
    u[i+1] = np.dot(B, u[i])
    
    
#Animaciones
'''
Animamos para el módulo al cuadrado de la función de onda de la partícula libre. La multiplicación de Psil por su conjugado produce el módulo al cuadrado, 
que en el contexto de la mecánica cuántica corresponde a la probabilidad de encontrar la partícula en una posición determinada.
'''
listal = u * u.conjugate().tolist()
fig, ax = plt.subplots()
line, = ax.plot(x, listal[0])
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')
def update_module(i):
    new_data = listal[200*i]
    line.set_ydata(new_data)
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*i)) 
    return line,time_text

ani_module = animation.FuncAnimation(fig, update_module, frames=Nt, interval=100, blit=True)
plt.show()
#Vemos que esta normalizada
'''
Animamos la parte real de la funcion de onda
'''
lista2 = u.real.tolist()
fig, ax = plt.subplots()
line, = ax.plot(x, lista2[0])
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')
def update_real(i):
    new_data = lista2[200*i]
    line.set_ydata(new_data)
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*i)) 
    return line,time_text

ani_real =  animation.FuncAnimation(fig, update_real, frames=Nt, interval=100, blit=True)
plt.show()

'''
Animamos la parte imaginaria de la funcion de onda
'''

lista3 = u.imag.tolist()
fig, ax = plt.subplots()
line, = ax.plot(x, lista3[0])
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')
def update_imaginary(i):
    new_data = lista3[200*i]
    line.set_ydata(new_data)
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*i)) 
    return line,time_text

ani_imaginary = animation.FuncAnimation(fig, update_imaginary, frames=Nt, interval=100, blit=True)
plt.show()



