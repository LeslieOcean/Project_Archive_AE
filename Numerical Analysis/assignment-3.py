# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:00:19 2024

@author: weiwe
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

def FDLaplacian2D(LeftX, RightX, LeftY, RightY, Nx, Ny, dx, dy):
    
    Dx = 1/dx*sp.diags([-1,1],[-1,0],shape=(Nx, Nx-1), format='csc')
    Dy = 1/dy*sp.diags([-1,1],[-1,0],shape=(Ny, Ny-1), format='csc')

    Lxx = Dx.transpose().dot(Dx)
    Lyy = Dy.transpose().dot(Dy)

    Ix = sp.eye(Nx-1)
    Iy = sp.eye(Ny-1)

    A = sp.kron(Iy, Lxx) + sp.kron(Lyy, Ix)
    return A

def sourcefunc(x,y):
    f = 0
    alpha = 40
    for i in range(1, 10):
        for j in range(1, 5):
            f += np.exp(-alpha*(x-i)**2-alpha*(y-j)**2)
    return f

def coeffK1(x,y):
    K = 1.0
    return K

def coeffK2(x,y):
    K = 1 + 0.1*(x+y+x*y)
    return K

def create2DLFVM(Nx,Ny,x,y,coeffFun):
    N = (Nx-1)*(Ny-1)
    
    diag_uu = np.zeros(N - (Nx - 1))  
    diag_u = np.zeros(N - 1)          
    diag = np.zeros(N)                
    diag_l = np.zeros(N - 1)          
    diag_ll = np.zeros(N - (Nx - 1))
    
    for j in range(Ny-1):
        for i in range(Nx-1): 
            idx = j*(Nx-1)+i
            diag[idx]=((coeffFun(x[i,j],y[i,j]+1/2*dy)/dy**2)+(coeffFun(x[i,j]+1/2*dx,y[i,j])/dx**2)+(coeffFun(x[i,j],y[i,j]-1/2*dy)/dy**2)+(coeffFun(x[i,j]-1/2*dx,y[i,j])/dx**2))
            if j < (Ny-2):
                diag_uu[idx]=-(coeffFun(x[i,j],y[i,j]+1/2*dy)/dy**2)
                
            if i < (Nx-2):
                diag_u[idx]=-(coeffFun(x[i,j]+1/2*dx,y[i,j])/dx**2)

            if i > 0:
                diag_l[idx-1]=-(coeffFun(x[i,j]-1/2*dx,y[i,j])/dx**2)

            if j > 0:
                diag_ll[idx-(Nx-1)]=-(coeffFun(x[i,j],y[i,j]-1/2*dy)/dy**2)

    A = sp.diags([diag_uu, diag_u, diag, diag_l, diag_ll],[Nx-1,1,0,-1,-(Nx-1)],format='csc')
    return A

LeftX = 0.0
RightX = 10.0
LeftY = 0.0
RightY = 5.0
Nx = 4 # number of intervals in x-direction
Ny = 4 # number of intervals in y-direction
dx = (RightX-LeftX)/Nx # grid step in x-direction
dy = (RightY-LeftY)/Ny # grid step in y-direction

x,y = np.mgrid[LeftX+dx:RightX-dx: complex(0, Nx-1), LeftY+dy:RightY-dy: complex(0, Ny - 1)]
f = sourcefunc(x, y)

# visualizing the source function
plt.ion()
plt.figure(1)
plt.clf()
plt.imshow(f.T, origin='lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet') # use the f array here
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
# lexicographic source vector
fLX = np.reshape(f, -1, order='F') # reshape 2D f array into 1D fLX array

A = FDLaplacian2D(LeftX, RightX, LeftY, RightY, Nx, Ny, dx, dy)
# plt.spy(A)
# plt.show()

u = la.spsolve(A, fLX)

# reshaping the solution vector into 2D array
uArr = np.reshape(u, (Nx-1, Ny-1), order='F')

# visualizing the solution
plt.figure(2)
plt.clf()
plt.imshow(uArr.T, origin='lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")

# visualizing two coefficient functions
xk = np.linspace(LeftX, RightX, Nx)
yk = np.linspace(LeftY, RightY, Ny)
K1 = np.zeros((Nx,Ny))
K2 = np.zeros((Nx,Ny))
for j in range(Ny):
    for i in range(Nx):      
     K1[i,j] = coeffK1(xk[i], yk[j])
     K2[i,j] = coeffK2(xk[i], yk[j])
     
plt.ion()
plt.figure(3)
plt.clf()

ax1 = plt.subplot(211)
im1 = ax1.imshow(K1, origin="lower", extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
plt.ylabel("y")
plt.colorbar(im1, ax=ax1, label="K1 Value")

ax2 = plt.subplot(212)
im2 = ax2.imshow(K2, origin="lower", extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(im2, ax=ax2, label="K2 Value")
plt.show()

A_FV_K1 = create2DLFVM(Nx, Ny, x, y, coeffK1)
A_FV_K2 = create2DLFVM(Nx, Ny, x, y, coeffK2)

u_K1 = la.spsolve(A_FV_K1,fLX)
u_K2 = la.spsolve(A_FV_K2,fLX)

uArr_K1 = np.reshape(u_K1, (Nx-1, Ny-1), order='F')
uArr_K2 = np.reshape(u_K2, (Nx-1, Ny-1), order='F')

# visualizing the solution
plt.figure(4)
plt.imshow(uArr_K1.T, origin='lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.figure(5)
plt.imshow(uArr_K2.T, origin='lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet') # use the f array here
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.show()