
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse





###############################EXTRA###############################

'''
A lo largo de la practica he puesto un pulso de onda triangular ejemplo
 a parte del seno, en este extra pondremos tambien un pulso cuadrado y 
 una combinacion de ambos




#######Para un pulso triangular
for i in range(0,Nx):
    if x[i]<=0.8*L:
        S[0,i] = 1.25*x[i]/L
    else:
        S[0,i] = 5.-5.*x[i]/L
S[1,:]=S[0,:]

#######Para un pulso cuadrado
for i in range(0, Nx):
    if x[i] <= 0.5 * L:
        S[0, i] = 0.25  # Asignar valor de 0.25 para la parte inferior de la onda cuadrada
    else:
        S[0, i] = -0.25  # Asignar valor de -0.25 para la parte superior de la onda cuadrada

S[1,:]=S[0,:]

#########Para un pulso triangular en el primer 40 por ciento de la cuerda
for i in range(0, Nx):
    if x[i] <= 0.4 * L:
        S[0,1] = 1.25*x[i]/L
    elif 0.4 * L < x[i] <= 0.8 * L:
        S[0, i] = 0.5  # Parte baja de la onda cuadrada en el siguiente 40 por ciento
    else:
        S[0, i] = -0.5  # Parte alta de la onda cuadrada en el último 20 por ciento

S[1,:]=S[0,:]
'''


############################Parte 1############################


'''
Resolveremos la ecuacion en derivadas parciales hiperbolicas a la que le hemos añadido 
un termino de una fuerza de rozamiento producida por la viscosidad del fluido.
Este termino depende de dt a primer orden y en la funcion se llamara tro.
'''

Nx=100    #Pasos de posicion
Nt=10000   #Pasos de tiempo
L = 1     #Longitud de la cuerda 
T = 40    
kappa= 1e-3
rho= 1e-2
c = 0.5
dx = L/Nx   
dt = 5e-3
tfin=dt*Nt 
n = 1         #nodos del seno, probar para varios y ver como cambian estos nodos


x = np.linspace(0,L,Nx+1)

#Creamos la matriz S, la cual usaremos para introducir la forma de la onda
S = np.zeros((Nt+1,Nx+1))
#Tenemos en cuenta que la cuerda parte del reposo(la ''soltamos'' con la funcion deseada)
# Y que tenemos que fijar la cuerda

S[0,:] = np.sin(n*np.pi*x)
S[0,2:] = S[0,1:-1] ; S[1,:] = S[0,:] 
'''
#Tambien lo podemos hacer para un pulso triangular en vez de un seno, por ejemplo

for i in range(0,Nx):
    if x[i]<=0.8*L:
        S[0,i] = 1.25*x[i]/L
    else:
        S[0,i] = 5.-5.*x[i]/L
S[1,:]=S[0,:]

#Para un pulso cuadrado
for i in range(0, Nx):
    if x[i] <= 0.5 * L:
        S[0, i] = 0.25  # Asignar valor de 0.25 para la parte inferior de la onda cuadrada
    else:
        S[0, i] = -0.25  # Asignar valor de -0.25 para la parte superior de la onda cuadrada
'''
#Creamos la funcion para resolver la ecuacion con el etodo explicito
#Al aplicar metodo FDTD a la ecuacion de la onda obtendremos:
def Explicita(u):
    for k in range(1,Nt): 
        cte = 1/((1/dt**2 + (2*kappa)/(dt*rho)))
        temporal = -u[k-1,1:-1]+2*u[k,1:-1]     #parte temporal segundo orden   
        espacial = u[k,2:]+u[k,:-2]-2*u[k,1:-1] #parte espacial segundo orden
        tro = u[k,1:-1]*2*kappa/(dt*rho)        #parte temporal primer orden, añadida con la fuerza de rozamiento de la viscosidad
        u[k+1,1:-1]= cte * (espacial/dx**2-tro+temporal/dt**2)
    return u

S = Explicita(S)

#Graficamos el resultado

fig = plt.figure()
ax = plt.axes(xlim=(0,L), ylim=(-np.max(S),np.max(S)))
line, = ax.plot([], [], lw=2)
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')
def animate(k):
    line.set_data(x,S[k,:])
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*k)) 
    return line,time_text

ani = FuncAnimation(fig,animate, blit=False,
                              interval=1)
plt.show()

############################Parte 2############################

x1 = np.linspace(0,L,Nx)
#Constantes que se quedan multiplicando a los terminos al aplicar el metodo implicito
#a la ecuacion problema
a=(c**2)/(dx**2)         #cte. multiplica a la parte espacial de orden 2
b=1/(dt**2)              #cte. multiplica a la parte temporal de orden 2
c1= (2*kappa)/(rho*dt)    #cte. multiplica a la parte temporal de orden 1

##########Definimos la tridiagonal

d=np.ones(Nx)*(-2)
o=np.ones(Nx-1)
u=np.ones(Nx-1)

I=np.eye(Nx)

A=sparse.diags([d,o,u],[0,1,-1],shape=(Nx,Nx)).toarray() #matriz de -2 y 1

#########Matriz inversa por metodo implicito
M = np.linalg.inv(-a * A + (b+c1) * I)
 
M1  = M * (c1 + 2*b) * I     #matriz del término del paso actual mult. por la inversa de M
M2 = M * b * I            #matriz con el término paso anterior mult. por la inversa de M

def Implicito(n,M1,M2):

   Sol = np.zeros((Nt,Nx))   #Rellenamos con las soluciones

   Sol[0]=np.sin(n*np.pi*x1)
   Sol[1]=np.sin(n*np.pi*x1) #condición para partir del reposo
   #Sol[0,0]=0
   #Sol[0,-1]=0
   #Sol[1,0]=0
   #Sol[1,-1]=0 
   #esas son las condiciones de contorno
   for i in range(1,Nt-1): 
       bloque1 = np.dot(M1,Sol[i])        #Son dos terminos debido a los dos pasos que necesitamos al ser una ecuacion de orden 2
       bloque2 = np.dot(M2,Sol[i-1])
       Sol[i+1] = bloque1 - bloque2 
        
   return Sol

s = Implicito(n,M1,M2)

#Las 2 condiciones iniciales vienen por los dos pasos anteriores que necesitamos

x = np.arange(0, L, dx) 
 
#Graficamos

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x, s[frame])
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*frame)) 
    return line,time_text

fig, ax = plt.subplots()
line, = ax.plot([], [],color='blue')
plt.title('MÉTODO IMPLÍCITO')
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')
# Configurar los límites de los ejes x e y
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)

# Crear la animación
ani = FuncAnimation(fig, update, frames=Nt, interval=100,init_func=init, blit=True)

# Mostrar la animación
plt.show()

############################Parte 3############################
'''
Si metemos un termino fuente adicional la ecuacion resultante tambien 
se conoce como ecuacion del telegrafo. Aplicaremos FDTD a la ecuacion 
planteada, siguiendo con la metodologia de la parte 1 obtendremos:
'''

x = np.linspace(0,L,Nx+1)

'''
#Tambien lo podemos hacer para un pulso triangular en vez de un seno, por ejemplo

for i in range(0,Nx):
    if x[i]<=0.8*L:
        S[0,i] = 1.25*x[i]/L
    else:
        S[0,i] = 5.-5.*x[i]/L
S[1,:]=S[0,:]
'''

# inicializo la onda con condiciones iniciales
S = np.zeros((Nt+1,Nx+1))
S[0,:] = np.sin(n*np.pi*x)              #Probar para varios n
S[0,2:] = S[0,1:-1] ; S[1,:] = S[0,:]


#Metodo FDTD para la ecuación del telégrafo
#Misma idea que Parte 1, solo que con distinta ecuacion

def telegrafo(u):
    cte = (1/dt**2+1/dt)
    for k in range(1,Nt-1):
        bloque1 = S[k,2:]+S[k,:-2]-2*S[k,1:-1]
        bloque2 = S[k,1:-1]*(1/dt-2)
        bloque3 = -S[k-1,1:-1]+2*S[k,1:-1]
        S[k+1,1:-1] = 1/cte*(bloque1/dx**2+bloque2+bloque3/dt**2)
    return S

u = telegrafo(u)
#graficamos la animación
fig = plt.figure()
ax = plt.axes(xlim=(0,L), ylim=(-1,1))
line, = ax.plot([], [], lw=2)
time_text=ax.text(0.75,0.95,'',fontsize=10,transform=ax.transAxes,color='black')
def animate(k):
    line.set_data(x,u[k,:])
    time_text.set_text('Tiempo: {:.4f} s'.format(dt*k)) 
    return line,time_text

ani = FuncAnimation(fig,animate, blit=False,
                              interval=1)

plt.show()

#En la animación, además de la vibración principal, se obtienen unas pequeñas 
#ondas en la propia cuerda que corresponden a una vibración de alta frecuencia
#debida a que la primera derivada no está bien ajustada



