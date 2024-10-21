import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
'''
Tenemos que resolver la ecuacion de adveccion y el metodo elegido es el metodo 
explicito upwind, downwind y diferencias centrales 
'''
L = 10
Nx = 100
dx = L/Nx
t = 10
Nt = 100
dt = t/Nt
c = 1     #Definimos la velocidad de la onda(Importante si positivo o negativo)
beta = (c**2*dt**2)/(dx**2)  #Definimos beta que usaremos en el extra
Estabilidad = (c*dt)/(dx)
#Estudiamos el limite de estabilidad, esta vendrá determinada por el numero de Courant
if Estabilidad > 1: print('Se rompe el criterio de estabilidad para c = 1')
else: print('Se cumple el criterio de estabilidad para c=1') 
 
alpha = 1               #Definimos alpha que es limite de estabilidad en nuestro caso, 
#si lo ponemos positivo funciona el upwind y explota el downwind, y la direccion de la onda es positiva
#si lo ponemos negativo funciona el downwind y explota de upwind, y la direccion es la contraria
#esto es debido a que en el negativo al metodo que llamamos downwind en el primer caso cambia a ser el
#upwind y viceversa.
#Lo que define que sea upwind o downwind es si va a la misma direccion que la onda, por eso al cambiar el sigo de esta cambian los metodos

x = np.linspace(0, 10, Nx)

#Definimos la funcion inicial

u_up = np.zeros((Nt, Nx))
u_down = np.zeros((Nt, Nx))
u_cen = np.zeros((Nt, Nx))

u_up[0] = u_down[0] = u_cen[0] = [np.exp(-10 * (i - 1) ** 2) for i in x] 

#Implementamos los metodos
for k in range(0, Nt - 1):
    # Método Upwind velocidad y onda misma direccion
    for i in range(1, Nx-1):
        u_up[k + 1, i] = u_up[k, i] - alpha * (u_up[k, i] - u_up[k, i - 1])
   
    # Método Downwind velocidad y onda distinta direccion
    for i in range(Nx - 1):
        u_down[k + 1, i] = u_down[k, i] - alpha * (u_down[k, i+1] - u_down[k, i])
    
    # Método de Diferencias Centradas
    for i in range(1, Nx - 1):
        u_cen[k + 1,i] = u_cen[k, i] - 0.5 * alpha * (u_cen[k, i + 1] - u_cen[k, i - 1])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def animate(k):
    axes[0].cla()
    axes[0].plot(x, u_up[k, :])
    axes[0].set_title('Upwind')
    axes[0].set(xlim=(0,L),ylim=(-1.5,1.5))
    
    axes[1].cla()
    axes[1].plot(x, u_down[k, :])
    axes[1].set_title('Downwind')
    axes[1].set(xlim=(0,L),ylim=(-1.5,1.5))
    
    axes[2].cla()
    axes[2].plot(x, u_cen[k, :])
    axes[2].set_title('Diferencias centrales')
    axes[2].set(xlim=(0,L),ylim=(-1.5,1.5))

ani= FuncAnimation(fig, animate, frames=Nt + 1, interval=1)
plt.show()
#Observamos cuando el metodo explota 

####################################Extra####################################
'''
Dado que el metodo de diferencias centrales explota con las condiciones puestas
vamos a programar el método de Lax-Wendroff es una extensión que mejora la precisión 
en la captura de ondas más afiladas y rápidas.
'''
#Seguimos con la misma dinamica que en los 3 metodos anteriores
u = np.zeros((Nt,Nx))
u[0] = [np.exp(-10*(i-1)**2) for i in x]
#Introducimos el metodo de Lax-wendroff
for k in range(0,Nt-1):
    u[k+1,1:-1] = u[k,1:-1]- 0.5 *alpha*(u[k,2:]-u[k,:-2]) + 0.5 * beta*(u[k,2:]-2*u[k,1:-1]+u[k,:-2])
#Animamos
fig = plt.figure()

ax = plt.axes(xlim=(0,L),ylim=(-1.5,1.5))
ax.set_title('Lax-Wendroff')
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')
line, = ax.plot([],[])  #la coma es para que plot no devuelva una tupla

x = np.linspace(0,10,Nx)
def animatet(k):
    line.set_data(x,u[k,:])
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*k)) 
    return line,time_text
ani = FuncAnimation(fig,animatet,frames=Nt+1, interval=10)
plt.show()

#Vemos como con los mismos parámetros iniciales Lax-Wendroff no explota y grafica 
#satisfactoriamente la solución de la ecuación, como cabría esperar. 
#Mejora la precisión en la captura de ondas más afiladas y rápidas
