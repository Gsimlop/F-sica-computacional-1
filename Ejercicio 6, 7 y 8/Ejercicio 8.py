import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
Tenemos que resolver la ecuacion de burgers no viscosa, ya que su termino de difusion es nulo
Este ejercicio sigue la misma estructura que el anterior.
'''

# Parámetros
L = 10
Nx = 100
dx = L/Nx
t = 10
Nt = 100
dt = t/Nt
c = 1
beta = (c*dt)/(dx)
beta1 = (c*dt)/(dx**2*100)
#Estudiamos el limite de estabilidad, esta vendrá determinada por el numero de Courant
if beta > 1: print('Se rompe el criterio de estabilidad para c = 1')
else: print('Se cumple el criterio de estabilidad para c=1') 



x = np.linspace(0, L, Nx)

# Condiciones iniciales
u_up = np.zeros((Nt, Nx))
u_cen = np.zeros((Nt, Nx))
u_up[0] = u_cen[0] = [3 * np.sin((2 * np.pi * i) / L) for i in x]

# Implementación de los métodos
for k in range(0, Nt - 1):
    # Método Upwind
    for i in range(Nx-1):
        u_up[k + 1, i] = u_up[k, i] - beta * (u_up[k, i] - u_up[k, (i -1 )])
        u_up[k][0]=u_up[k][-2]; u_up[k][-1]=u_up[k][1] #Condiciones periódicas
        
    # Método de Diferencias Centradas
    for i in range(Nx-1):
       u_cen[k + 1, i] = u_cen[k, i] -   beta1 * (u_cen[k, (i + 1) ] - u_cen[k, (i - 1)])
       u_cen[k][0]=u_cen[k][-2]; u_cen[k][-1]=u_cen[k][1] #Condiciones periódicas
       
     
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
def animate(k):
    axes[0].cla()
    axes[0].plot(x, u_up[k, :])
    axes[0].set_title('Upwind')
    axes[0].set(xlim=(0,L),ylim=(-5,5))

    axes[1].cla()
    axes[1].plot(x, u_cen[k, :])
    axes[1].set_title('Diferencias centrales')
    axes[1].set(xlim=(0,L),ylim=(-5,5))
# Animación
ani = FuncAnimation(fig, animate, frames=Nt+1 , interval=10)
plt.show()

####################################Extra####################################
'''
Como extra voy a programar el ejemplo de downwind que hemos aprendido en el 
ejercicio anterior, de tal manera que como cabría esperar obtenemos el mismo 
resultado.
'''
c = -1 #Cambiando el signo obtendremos las mismas conclusiones que en el ejercicio 7
beta = (c*dt)/(dx)

u_down = np.zeros((Nt, Nx))

u_down[0] = [3 * np.sin((2 * np.pi * i) / L) for i in x]

for k in range(0, Nt - 1):
    # Método Downwind
    for i in range(Nx-1):
        u_down[k + 1, i] = u_down[k, i] - beta * (u_down[k, i+1] - u_down[k, (i )])
        u_down[k][0]=u_down[k][-2]; u_down[k][-1]=u_down[k][1] #Condiciones periódicas
        
#Animamos
fig = plt.figure()

ax = plt.axes(xlim=(0,L),ylim=(-5,5))
ax.set_title('Downwind')
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')
line, = ax.plot([],[])  #la coma es para que plot no devuelva una tupla
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')

def animatet(k):
    line.set_data(x,u_down[k,:])
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*k)) 
    return line, time_text
ani = FuncAnimation(fig,animatet,frames=Nt+1, interval=10)
plt.show()