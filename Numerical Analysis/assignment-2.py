# Solution of 1D Poisson’s equation with FDM
# Wei Wei (c) 2024
import numpy as np
import matplotlib.pyplot as plt

def func1(x):
    return 3*x-2

def func2(x):
    return x**2+3*x-2

def uex1(x):
    return -1/2*x**3 + x**2 + 3/2*x + 1

def uex2(x):
    return -1/12*x**4 - 1/2*x**3 + x**2 + 15/4*x + 1

n = 5 #the number of subinterval
xgrid = np.linspace(0, 3, n+1) #grid nodes including boundaries

h = (3-0)/(n) #step size

f1 = func1(xgrid)
f2 = func2(xgrid)

fig1 = plt.figure()
plt.plot(xgrid, f1, color='blue', marker='o', label='$f_1$')
plt.plot(xgrid, f2, color='red', marker='o', label='$f_2$')
plt.legend()
plt.title('Right-hand-side functions $f_1(x)$ and $f_2(x)$ with $n=5$')
plt.xlabel('$x$')
plt.ylabel('$f$')
plt.show()

u1ex = uex1(xgrid)
u2ex = uex2(xgrid)

lower_diag = np.ones(n-2)
diag = -2*np.ones(n-1)
upper_diag = np.ones(n-2)
A = -1/h**2*(np.diag(diag, 0) + np.diag(lower_diag, -1) + np.diag(upper_diag, 1))

fig3 = plt.figure()
plt.spy(A, marker='o', color='green')
plt.title("Structure of matrix $A$")
plt.show()

eigenvalues_computed, eigenvectors_computed = np.linalg.eig(A)
idx = np.argsort(eigenvalues_computed)
eigenvalues_sorted = eigenvalues_computed[idx]


add_term = np.zeros(n-3)
add_term = np.insert(add_term, 0, 1/h**2)
add_term = np.append(add_term, 1/h**2)
f1rhs = f1[1:-1] + add_term
f2rhs = f2[1:-1] + add_term
    
u1_solve = np.linalg.solve(A, f1rhs)
u2_solve = np.linalg.solve(A, f2rhs)

u1 = np.insert(u1_solve, 0, 1)
u1 = np.append(u1, 1)

u2 = np.insert(u2_solve, 0, 1)
u2 = np.append(u2, 1)

fig2 = plt.figure()
plt.plot(xgrid, u1ex, color='blue', marker='o', label='$u^{ex}_1$')
plt.plot(xgrid, u1, color='blue', linestyle =  "dashed", marker='o', label='$u_1$')
plt.plot(xgrid, u2ex, color='red', marker='o', label='$u^{ex}_2$')
plt.plot(xgrid, u2, color='red', linestyle =  "dashed", marker='o', label='$u_2$')
plt.legend(fontsize=19)
plt.title('Comparing the exact solutions $u^{ex}_1$ $u^{ex}_2$ and the Finite Difference Method approximations $u_1(x)$ and $u_2(x)$ for $n=5$', fontsize=19)
plt.xlabel('$x$', fontsize=15)
plt.ylabel('$u$', fontsize=15)
plt.show()

L2_u1 = np.linalg.norm(u1ex-u1)
L2_u2 = np.linalg.norm(u2ex-u2)

RMSE_u1 = L2_u1/np.sqrt(n-1)
RMSE_u2 = L2_u2/np.sqrt(n-1)


err_n_u1 = []
err_n_u2 = []
n_iterate = np.arange(2, 12, 1)
for i in n_iterate:
    n = i
    xgrid = np.linspace(0, 3, n+1) 
    
    h = (3-0)/n
    
    f1 = func1(xgrid) 
    f2 = func2(xgrid)

    u1ex = uex1(xgrid)
    u2ex = uex2(xgrid)

    lower_diag = np.ones(n-2)
    diag = -2*np.ones(n-1)
    upper_diag = np.ones(n-2)
    A = -1/h**2*(np.diag(diag, 0) + np.diag(lower_diag, -1) + np.diag(upper_diag, 1))

    if n >= 3:
        add_term = np.zeros(n-3)
        add_term = np.insert(add_term, 0, 1/h**2)
        add_term = np.append(add_term, 1/h**2)
        f1rhs = f1[1:-1] + add_term
        f2rhs = f2[1:-1] + add_term
    else:
        f1rhs = f1[1:-1] + 2*np.ones(n-1)*1/h**2
        f2rhs = f1[1:-1] + 2*np.ones(n-1)*1/h**2
    

    u1_solve = np.linalg.solve(A, f1rhs)
    u2_solve = np.linalg.solve(A, f2rhs)

    u1 = np.insert(u1_solve, 0, 1)
    u1 = np.append(u1, 1)

    u2 = np.insert(u2_solve, 0, 1)
    u2 = np.append(u2, 1)

    L2_u1 = np.linalg.norm(u1ex-u1)
    L2_u2 = np.linalg.norm(u2ex-u2)

    RMSE_u1 = L2_u1/np.sqrt(n-1)
    RMSE_u2 = L2_u2/np.sqrt(n-1)
    
    err_n_u1.append(RMSE_u1)
    err_n_u2.append(RMSE_u2)

fig4, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(n_iterate, err_n_u1, marker='o', color = 'blue')
ax1.set_xlabel('$n$', fontsize=15)
ax1.set_ylabel('$RMSE(u^{ex},u)$', fontsize=15)
ax1.set_title('Rate of convergence for $n = 2$ to $12$', fontsize=17)
ax2.plot(n_iterate[3:], np.log10(err_n_u1[3:]), marker='o', color = 'blue')
ax2.set_xlabel('$n$', fontsize=15)
ax2.set_ylabel('$log_{10}RMSE(u^{ex},u)$', fontsize=15)
ax2.set_title('The logarithm of the global error for $n = 5$ to $12$', fontsize=17)
fig4.suptitle("Convergence rate of using Finite Difference Method approximation for the source function $f_1(x)=3x-2$", fontsize=19)

fig5 = plt.figure()
plt.title('$f_1(x)=3x-2$', fontsize=19)
plt.plot(n_iterate, np.log10(err_n_u2), marker='o', color = 'red')
plt.ylabel('$log_{10}RMSE(u^{ex},u)$', fontsize=15)
plt.xlabel('$n$', fontsize=15)
plt.title('Convergence rate of using Finite Difference Method approximation for the source function $f_2(x)=x^2+3x-2$', fontsize=19)
