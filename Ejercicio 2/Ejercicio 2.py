import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy as sp
import time

################################Parte 1################################
'''
Resolveremos la Ecuacion de Laplace para un problema dado con Condiciones de Dirichlet, 
en donde dado el valor de φ en la frontera de la region de interes,
encontraremos su valor en cualquier otro punto.
'''
#Definimos las dimensiones de la matriz deseada
M = 30
N = 30
x,y = np.arange(N), np.arange(M)

#Creamos las submatrices las cuales usaremos para construir la matriz que necesitamos

A = np.eye(N,M)
Ia = np.zeros((N,M))
Ia[0,0]=1 ; Ia[-1,-1]=1

B = -4*np.eye(N,M) + np.eye(N,M,k=1) + np.eye(N,M,k=-1)
B[0,0]=1 ;B[-1,-1]=1; B[0,1]=0; B[-1,-2]=0

Ib = np.eye(N,M)
Ib[0,0]=0 ; Ib[-1,-1]=0

C = Ib
Ic = np.eye(N,M,k=1) + np.eye(N,M,k=-1)
Ic[0,1]= 0; Ic[-1,-2]=0 

#Ahora obtenemos la matriz deseada con un producto tensorial, usaremos sparse.kron
#El orden en sparse.kron es muy importante.

D = sparse.kron(Ia,A).toarray() + sparse.kron(Ib,B).toarray() + sparse.kron(Ic,C).toarray() #ojo con el orden

####Aplicamos las condiciones de contorno

#Condiciones de contorno 1
R = np.zeros((N*M,1))
R[:M] = 100

#Resolvemos el sistema

sol1 = np.linalg.solve(D,R)
sol = sol1.reshape((M,N))

#Graficamos
X,Y = np.meshgrid(x,y)
plt.contourf(X,Y,sol,alpha=.75,cmap=plt.cm.coolwarm)
plt.colorbar(label='Potencial(V)')
plt.title('ECUACIÓN DE LAPLACE METODO DIRECTO')
plt.xlabel('Posición (m)')
plt.ylabel('Potencial (V)')
#plt.savefig('Ecuacion Laplace 1.png')
plt.show()


################################Parte 2################################

'''
Ahora tenemos que resolver el mismo problema con un metodo iterativo, 
que en este caso será el de Jacobi. 
'''

N=30
#La malla de incógnitas tiene N*N puntos
error=1e-2

r=np.zeros(N*N)
r[0:N]=100

def Jacobi(A): #metemos la matriz con condiciones
    D=np.diag(np.diag(A)) #separamos su diagonal
    #Se ha introducido la matriz invD manualmente porque es costoso obtenerla
    invD=sparse.diags([-0.25*np.ones(N*N)],[0]).toarray()
    B=A-D #le quitamos la diagonal a la matriz introducida
    #Asigno un valor inicial al potencial (aleatorio) en los puntos interiores
    sol=np.random.rand(N**2)
    iteraciones=0
    while True:
        sol_nueva=np.dot(invD,r-(np.dot(B,sol)))
        iteraciones+=1
        if abs(np.linalg.norm(sol_nueva)-np.linalg.norm(sol))<error:
            matriz_sol=sol_nueva.reshape(N,N)
            return matriz_sol, iteraciones
        sol=sol_nueva

Sol,Iteraciones=Jacobi(D)

X,Y = np.meshgrid(x,y)
plt.contourf(X,Y,Sol,alpha=.75,cmap=plt.cm.coolwarm)
plt.colorbar(label='Potencial(V)')
plt.title('EC. LAPLACE JACOBI')
plt.xlabel('Posición (m)')
plt.ylabel('Potencial (V)')
plt.show()
plt.show(block=True)



################################Parte 3################################

#En Neumann dado el valor de ∇φ en la frontera de la region de interes, queremos encontrar 
#su valor en cualquier otro punto. En este caso la diferencia principal se encuentra en las matrices 
#A y B, las cuales deberán llegar las condiciones de contorno de Neumann

M = 30
N = 30
x,y = np.arange(N), np.arange(M)


I=np.eye(M) #matriz identidad

#Condiciones de contorno 1
R = np.zeros((N*M,1))
R[:M] = 100

#Creamos las matrices con las condiciones de Neumann


A = -4*np.eye(N,M) + np.eye(N,M,k=1) + np.eye(N,M,k=-1)
A[0][1]=2 ; A[M-1][N-2]=2 #matriz tridiagonal con las condiciones de Neumann

B = np.eye(N,M,k=1) + np.eye(N,M,k=-1)


mat1=sparse.kron(I,A).toarray()
mat2=sparse.kron(B,I).toarray()

MATN= mat1+mat2 #matriz con las condiciones de Neumann


sol=sp.linalg.solve(MATN,R).reshape((M,N)) #Aplicamos la solucion del metodo directo a la matriz con condiciones de Neumann

#Graficamos

X,Y = np.meshgrid(x,y)
plt.contourf(X,Y,sol,alpha=.75,cmap=plt.cm.coolwarm)
plt.colorbar(label='Potencial(V)')
plt.title('EC. LAPLACE COND. NEUMANN')
plt.xlabel('Posición (m)')
plt.ylabel('Potencial (V)')
#plt.savefig('Ecuacion Laplace Neumann.png')
plt.show()

#Definimos Jacobi con Neumann

PhiJN = Jacobi(MATN) #Aplicamos el metodo iterativo de Jacobi a las matrices con condiciones de Neumann

#Graficamos

X,Y = np.meshgrid(x,y)
plt.contourf(X,Y,sol,alpha=.75,cmap=plt.cm.coolwarm)
plt.colorbar(label='Potencial(V)')
plt.title('EC. LAPLACE JACOBI Y COND. DE NEUMANN')
plt.xlabel('Posición (m)')
plt.ylabel('Potencial (V)')
#plt.savefig('Ecuacion Laplace con Jacobi y Neumann.png')
plt.show()

################################Extra################################

'''
Como extra crearemos de nuevo la funcion Jacobi de otra forma enseñada en métodos numéricos de primero
de carrera e implementaremos sobrejacobi. Comparando así los tiempos de resolucion y viendo como 
el metodo de sobrerelajacion es mas rapido que el de relajacion.
'''

Nx = 50
Ny = 50
N = Nx*Ny
V = 100
w = 0.8
Delta = 1
Phi = np.zeros([Nx,Ny])
x,y = Delta*np.arange(Nx) , Delta*np.arange(Ny)


Phi[0,:]= V

def Jacobi(Phi):
    Phip=np.copy(Phi)
    err = 1e-6       #Error que queremos tolerar
    continua= True     #Para meternos en el bucle
    while continua:
        continua=False
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                Phi_a = Phi[i,j]
                Phip[i,j] = 1/4*(Phi[i+1,j]+Phi[i-1,j]+Phi[i,j+1]+Phi[i,j-1]) 
                Phi=Phip
                dPhi = np.abs(Phi[i,j]-Phi_a) 
                if dPhi>err:       #Mientras la diferencia entre phi's sea mayor que el error se seguirá metiendo en el bucle
                    continua=True
    return Phi 

def SobreJacobi(Phi):
    Phip=np.copy(Phi)
    w = 0.8
    err = 1e-6
    sigue= True
    while sigue:
        sigue=False
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                Phi_a = Phi[i,j]
                Phip[i,j] = (1+w)/4*(Phi[i+1,j]+Phi[i-1,j]+Phi[i,j+1]+Phi[i,j-1])-w*Phi[i,j]
                Phi=Phip
                dPhi = np.abs(Phi[i,j]-Phi_a)
                if dPhi>err:
                    sigue=True
    return Phi

#COMPARAMOS TIEMPOS CON JACOBI
t1J = time.time()
PhiJ = Jacobi(Phi)
t2J = time.time()
tJ = t2J - t1J
#COMPARAMOS TIEMPOS CON SOBREJACOBI
t1SJ = time.time()
PhiSJ = SobreJacobi(Phi)
t2SJ = time.time()
tSJ = t2SJ - t1SJ

#IMPRIMIMOS POR PANTALLA LA COMPARACION
print('El tiempo que tarda en resolver Jacobi con Dirichlet es: ' , round(tJ,3),'segundos')
print('El tiempo que tarda en resolver Jacobi con Dirichlet es: ' , round(tSJ,3),'segundos')

X,Y = np.meshgrid(x,y)
plt.contourf(X,Y,PhiJ, alpha=.75,cmap=plt.cm.coolwarm)
plt.colorbar(label='Potencial(V)')
plt.title('EC.LAPLACE JACOBI2 COND. DIRICHLET')
plt.xlabel('Posición (m)')
plt.ylabel('Potencial (V)')
#plt.savefig('Ecuacion Laplace con Jacobi2 y Dirichlet.png')
plt.show()


X,Y = np.meshgrid(x,y)
plt.contourf(X,Y,PhiSJ, alpha=.75,cmap=plt.cm.coolwarm)
plt.colorbar(label='Potencial(V)')
plt.title('EC.LAPLACE SOBREJACOBI COND. DIRICHLET')
plt.xlabel('Posición (m)')
plt.ylabel('Potencial (V)')
#plt.savefig('Ecuacion Laplace con SobreJacobi y Dirichlet.png')
plt.show()


