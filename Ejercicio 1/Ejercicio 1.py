import numpy as np
import time
import random
import matplotlib.pyplot as plt

#Implementamos una funcion a la que le pasaremos 4 arrays d,u,o y r
#Y nos devuelve un array con las soluciones del sistema
def TridiagonalSolver(d,o,u,r):
    #Creamos un array de ceros de longitud deseada
    h=np.zeros(len(o))
    p=np.zeros(len(d)) 
    #Definimos el inicial
    h[0]=o[0]/d[0] 
    p[0]=r[0]/d[0]
    #Creamos la recursividad
    for i in range(1,len(p)):
        p[i]=(r[i]-u[i-1]*p[i-1])/(d[i]-u[i-1]*h[i-1])
        if not i==len(p)-1:
            h[i]=o[i]/(d[i]-u[i-1]*h[i-1])

    x=np.zeros(len(d))

    x[-1]=p[-1]

    for i in range(len(p)-2,-1,-1):
        x[i]=p[i]-h[i]*x[i+1]

    return x

#Definimos una funcion que cree la matriz trigonal
def MatrizA(d,o,u):
    A=np.zeros((len(d),len(d)))
    for i in range(0,len(d)):
        A[i][i]=d[i]
        if not i==len(d)-1:
            A[i][i+1]=o[i]
            A[i+1][i]=u[i]
    return A

#Ahora definimos el tiempo de simulacion y creamos las listas de los 3 metodos que vamos a comparar

N=200  #tamaño del sistema, cuidado el tiempo de solucion es exponencial

#Creamos las listas a las que vamos a añadir los resultados a graficar con cada tipo de calculo

listaN=[]
t_TridiagonalSolver=[]
t_solve=[]
t_inv=[]

#Creamos un for en el que solucionaremos una matriz random y contaremos el tiempo que 
#necesita cada metodo

for n in range(2,N+1):    
    
    listaN.append(n)
    d=[random.random() for i in range(n)]
    o=[random.random() for i in range(n-1)]
    u=[random.random() for i in range(n-1)]
    r=[random.random() for i in range(n)]
    A=MatrizA(d,o,u) #Creamos la matriz
    
    #empezamos a contar para el metodo de la funcion
    
    ini_TridiagonalSolver=time.time()
    
    x_TridiagonalSolver=TridiagonalSolver(d,o,u,r) 
    
    fin_TridiagonalSolver=time.time()
    
    t_TridiagonalSolver.append(fin_TridiagonalSolver-ini_TridiagonalSolver) 

    #empezamos a contar para el metodo del linalg
    
    ini_solve=time.time()
    
    x_solve=np.linalg.solve(A,r)
    
    fin_solve=time.time()
    
    t_solve.append(fin_solve-ini_solve) 
    
    #empezamos a contar para el metodo de la inversa
    
    ini_inv=time.time()
    
    invA=np.linalg.inv(A)
    
    x_inv=np.dot(invA,r)
    
    fin_inv=time.time()
    
    t_inv.append(fin_inv-ini_inv) #añadimos a la grafica
    
print('El método de la inversa tarda:',  round(np.sum(t_inv),3),sep=' ')
print('El método de la tridiagonal tarda:', round(np.sum(t_TridiagonalSolver),3), sep=' ')
print('El método con el solve tarda:' , round(np.sum(t_solve),3), sep= ' ')


##########################Extra#############################################
#Si ponemos el time.time() fuera del for y repetimos el proceso podemos ver como el tiempo que tarda en resolverse es exponencial


listaN2=[]
t_TridiagonalSolver2=[]

ini_TridiagonalSolver=time.time()

for n in range(2,N+1):    
    
    listaN2.append(n)
    d=[random.random() for i in range(n)]
    o=[random.random() for i in range(n-1)]
    u=[random.random() for i in range(n-1)]
    r=[random.random() for i in range(n)]
    A=MatrizA(d,o,u) #Creamos la matriz
    
    #empezamos a contar para el metodo de la funcion
        
    x_TridiagonalSolver=TridiagonalSolver(d,o,u,r) 
    
    fin_TridiagonalSolver=time.time()
    
    t_TridiagonalSolver2.append(fin_TridiagonalSolver-ini_TridiagonalSolver) 

print('El método de la tridiagonal tarda:', round(np.sum(t_TridiagonalSolver),3), sep=' ')

#Creamos el grafico

plt.figure('Resolución Ec.Lineal')
plt.title('Tiempos de resolución')
plt.plot(listaN,t_TridiagonalSolver,label='TridiagonalSolver')
plt.plot(listaN,t_solve,label='np.linalg.solve')
plt.plot(listaN,t_inv,label='np.linalg.inv')
plt.xlabel('N')
plt.ylabel('t (s)')
plt.legend(loc='best')
#plt.savefig('Texp3.png')
plt.show()

plt.figure('Resolución Ec.Lineal')
plt.title('Crecimiento exponencial de resolución')
plt.plot(listaN2,t_TridiagonalSolver2,label='TridiagonalSolver')
plt.xlabel('N')
plt.ylabel('t (s)')
plt.legend(loc='best')