
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import math
import time
import matplotlib.animation as animation
import matplotlib.ticker as tkr

np.random.seed(10) #comment out after trials

def FDLaplacian2D(LeftX, RightX, LeftY, RightY, Nx, Ny, dx, dy):
    
    Dx = 1/dx*sp.diags([-1,1],[0,1],shape=(Nx, Nx+1), format='csc')
    Dy = 1/dy*sp.diags([-1,1],[0,1],shape=(Ny, Ny+1), format='csc')

    Lxx = Dx.transpose().dot(Dx)
    Lyy = Dy.transpose().dot(Dy)

    Ix = sp.eye(Nx+1)
    Iy = sp.eye(Ny+1)

    A = sp.kron(Iy, Lxx) + sp.kron(Lyy, Ix)
    return A

def perturbation(rxy):
    for i in range(rxy.shape[0]):
        for j in range(rxy.shape[1]):
            r = 0.01*(a+b)*np.random.rand()
            rxy[i,j] = r
    return rxy

def sourcef_u(u_n, v_n, A, Du, a, Kappa):
    f_u = -Du*A.dot(u_n) + Kappa*(a-u_n+(u_n**2) * v_n)
    return f_u
    
def sourcef_v(u_n, v_n, A, Dv, b, Kappa):
    f_v = -Dv*A.dot(v_n) + Kappa*(b-(u_n**2) * v_n)
    return f_v

#FE Method function
def FE_Method(kappa, Nt, A, u_0, v_0):  
    ht = T/Nt
    
    u_n = u_0.reshape(-1) #initialize the iteration
    v_n = v_0.reshape(-1)
    
    
    vt = []
    ut = []
    stable = True
    for i in range(Nt):
        u_n1 = u_n + ht*sourcef_u(u_n, v_n, A, Du, a, kappa)
        v_n1 = v_n + ht*sourcef_v(u_n, v_n, A, Dv, b, kappa)
        u_n1 = np.clip(u_n1, -1e6, 1e6)
        v_n1 = np.clip(v_n1, -1e6, 1e6)
        u_n = u_n1
        v_n = v_n1
        ut.append(u_n1)
        vt.append(v_n1)
        
        upper_thresh = v_0.max()+2
        lower_thresh = v_0.min()-2
        
        if np.any(v_n1 > upper_thresh) or np.any(v_n1 < lower_thresh):
            stable = False
            break

    return stable, vt, ut

   
def empirical_stability(h, kappa_val, A, u_0, v_0):
    results ={}
    for kappa in kappa_val:
        Nt_0 = math.ceil(T*max(Du*8/h**2+kappa, Dv*8/h**2)/2)
        Nt_emp = Nt_0 + int(kappa*20) #assume a simple linear function
        print(f"N_t for FE Method: {Nt_emp}")
        stable, vt, ut = FE_Method(kappa, Nt_emp, A, u_0, v_0)
        results[kappa] = (Nt_emp, stable, vt, ut)
    return results

def animate_vt(results, kappa, Nx, Ny, ani_vt, initial=0):
    Nt_emp, stable, vt, ut = results[kappa]
    
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    #image for the initial conditions
    im = ax.imshow(vt[initial].reshape((Nx+1,Ny+1)), origin = 'lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet', animated=ani_vt)
    plt.colorbar(im, ax=ax)
    
    if ani_vt == True:
        def update(frame):
            #print(f"Frame {frame}: {vt[frame].shape}")
            im.set_array(vt[frame].reshape(Nx+1, Ny+1))
            return (im,)
    
        ani = animation.FuncAnimation(fig, update, frames=range(initial, len(vt)), interval=50, blit=False)
        
        ani.save("v_t_animation.gif", writer="pillow")


def FE_visual(results, kappa, Nx, Ny):
    Nt_emp, stable, vt, ut = results[kappa]
    
    fig3 = plt.figure("FE_T_u")
    ax3 = fig3.add_subplot(111)
    im3 = ax3.imshow(ut[-1].reshape((Nx+1, Ny+1)), origin = 'lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
    plt.colorbar(im3, ax=ax3)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    
    fig4 = plt.figure("FE_T_v")
    ax4 = fig4.add_subplot(111)
    im4 = ax4.imshow(vt[-1].reshape((Nx+1, Ny+1)), origin = 'lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
    plt.colorbar(im4, ax=ax4)
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    
def J11(Du, A, kappa, ui, vi):
    I = sp.eye(A.shape[0])
    dfudu = -Du*A + kappa*(-I + 2*sp.diags(ui*vi))
    return dfudu

def J12(kappa, ui):
    dfudv = kappa*sp.diags(ui**2)
    return dfudv

def J21(kappa, ui, vi):
    dfvdu = -2*kappa*sp.diags(ui*vi)
    return dfvdu

def J22(Dv, A, kappa, ui):
    dfvdv = -Dv*A - kappa*sp.diags(ui**2)
    return dfvdv

def Jacobian(Du, Dv, A, kappa, ui, vi):
    J011 = J11(Du, A, kappa, ui, vi)
    J012 = J12(kappa, ui)
    J021 = J21(kappa, ui, vi)
    J022 = J22(Dv, A, kappa, ui)
    
    J = sp.bmat([[J011,J012],[J021,J022]], format='csr')
    return J

#BE-NR Method Function - failed trial
def BE_NR_Method(kappa, Nt, A, u_0, v_0):
    epsilon= 1e-3 #tolerance
    
    ht = T/Nt
    u_k = u_0.reshape(-1)
    v_k = v_0.reshape(-1)
    
    vt = []
    ut = []
    renorm_list = [] #norm of the residual at each inner iteration
    iterate_count = False
    start_time = time.time()
    
    for nt in range(Nt): #outer iteration
        i = 0 #counter the inner iteration
        
        u_k_copy = u_k.copy()
        v_k_copy = v_k.copy()
        
        u_k1 = u_k.copy() #u^{k+1}_0
        v_k1 = v_k.copy()
        
        # residual_uk1 = u_k+ht*sourcef_u(u_k1, v_k1, A, Du, a, kappa)-u_k1 
        # residual_vk1 = v_k+ht*sourcef_v(u_k1, v_k1, A, Du, b, kappa)-v_k1
        
        # residual = np.concatenate((residual_uk1,residual_vk1))
        # re_norm = np.linalg.norm(residual)
        
        while np.linalg.norm(u_k_copy+ht*sourcef_u(u_k1, v_k1, A, Du, a, kappa)-u_k1)> epsilon: #inner iteration
            residual_uk1 = u_k_copy+ht*sourcef_u(u_k1, v_k1, A, Du, a, kappa)-u_k1
            residual_vk1 = v_k_copy+ht*sourcef_v(u_k1, v_k1, A, Du, b, kappa)-v_k1
            residual = np.concatenate((residual_uk1,residual_vk1))
            re_norm = np.linalg.norm(residual)
            renorm_list.append(re_norm)
            
            J = Jacobian(Du, Dv, A, kappa, u_k1, v_k1)
            I = sp.eye(J.shape[0])
            #delta = sp.linalg.inv(I-ht*J).dot(residual) #SparseEfficiencyWarning: spsolve is more efficient when sparse b is in the CSC matrix format
            delta = sp.linalg.spsolve(I - ht * J, residual) 
            i += 1
            
            u_k1 += delta[:A.shape[0]]
            v_k1 += delta[A.shape[0]:]
            
            if i == 3:
                iterate_count = True #require 3 iterations
                
        u_k = u_k1
        v_k = v_k1
        ut.append(u_k)
        vt.append(v_k)
        
    end_time = time.time()
    cpu_time = end_time - start_time
    return iterate_count, cpu_time, vt, ut, renorm_list
#failed trial
def empirical_Nt(kappa_val, A, u_0, v_0):
    results = {}
    for kappa in kappa_val:
        Nt_emp = 120 #assume a function
        print(f"The automatically chosen Nt: {Nt_emp} ")
        iterate_count, time1, vt, ut, renorm_list = BE_NR_Method(kappa, Nt_emp, A, u_0, v_0)
        results[kappa] = (Nt_emp, iterate_count, time1, vt, ut, renorm_list)
        print(f"the norm of the residual of each inner iteration: {renorm_list}")
        print(f"CPU time for BENR Method: {time1:.6f} seconds")
    return results


def BE_visual(uT, vT, kappa, Nx, Ny):
    
    fig5=plt.figure("BE_u_T")
    ax5 = fig5.add_subplot(111)
    im5 = ax5.imshow(uT.reshape((Nx+1, Ny+1)), origin = 'lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
    plt.colorbar(im5, ax=ax5)
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")

    fig6=plt.figure("BE_v_T")
    ax6 = fig6.add_subplot(111)
    im6 = ax6.imshow(vT.reshape((Nx+1, Ny+1)), origin = 'lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
    plt.colorbar(im6, ax=ax6)
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")

#final trial
def BE_NR_Method_new(kappa, Nt, A, u0, v0, tol=1e-3, max_inner_iter=20):
    ht = T / Nt
    N_nodes = u0.size
    w_n = np.concatenate([u0.flatten(), v0.flatten()])
    u_hist = [u0.copy()]
    v_hist = [v0.copy()]
    
    start_time = time.time()
    for step in range(Nt):
        w_guess = w_n.copy()
        inner_iter = 0
        
        for inner in range(max_inner_iter):
            u = w_guess[:N_nodes]
            v = w_guess[N_nodes:]
            
            f_u = sourcef_u(u, v, A, Du, a, kappa)
            f_v = sourcef_v(u, v, A, Dv, b, kappa)
            f = np.concatenate([f_u, f_v])
            
            F = w_guess - w_n - ht * f
            residual_norm = np.linalg.norm(F)
            print(f"time step={step}, i={inner_iter}, residual norm={residual_norm}")
            if residual_norm < tol:
                break
            
            J_f = Jacobian(Du, Dv, A, kappa, u, v)
            J_F = sp.eye(2 * N_nodes) - ht * J_f  # I - ht * J_f
            
            delta_w = sp.linalg.spsolve(J_F, -F)
            w_guess += delta_w
            inner_iter += 1
        
        w_n = w_guess.copy()
        u_hist.append(w_n[:N_nodes].reshape(u0.shape))
        v_hist.append(w_n[N_nodes:].reshape(v0.shape))
    
    end_time = time.time()
    cpu_time = end_time - start_time
    
    return u_hist[-1], v_hist[-1], cpu_time

def find_optimal_Nt(kappa):
    Nt = int(78.983*np.exp(0.1531*kappa))
    print(f"The automatically chosen Nt: {Nt} ")
    return Nt
    
#problem parameters
Du = 0.05 #slow diffusion
Dv = 1 #fast diffusion
kappa = 5 #reaction constants
a = 0.1305
b = 0.7695
T = 20
kappa_values = [2, 5, 10] #for empirical formula

LeftX = 0.0
RightX = 4.0
LeftY = 0.0
RightY = 4.0

######################### modest grid ###########################
# Nx = 25
# Ny = 25
# dx = (RightX-LeftX)/Nx
# dy = (RightY-LeftY)/Ny

# #small perturbation
# rxy = np.zeros((Nx+1, Ny+1))
# rxy = perturbation(rxy)
# #inital conditions
# u_0 = a + b + rxy
# v_0 = b/(a+b)**2*np.ones((Nx+1, Ny+1))
# A = FDLaplacian2D(LeftX, RightX, LeftY, RightY, Nx, Ny, dx, dy)

# results = empirical_stability(dx, kappa_values, A, u_0, v_0) #finding f(kappa)

#animate_vt(results, kappa_values[0], Nx, Ny, True, initial=2500) #animation of vt since some time step
#################################################################


#verifying with fine mesh case
Nx = 100
Ny = 100
dx = (RightX - LeftX) / Nx
dy = (RightY - LeftY) / Ny

rxy = np.zeros((Nx + 1, Ny + 1))
rxy = perturbation(rxy)
u_0 = a + b + rxy
v_0 = (b / (a + b)**2) * np.ones((Nx + 1, Ny + 1))

A = FDLaplacian2D(LeftX, RightX, LeftY, RightY, Nx, Ny, dx, dy)

#only select one method a time, so the visualization won't pop up error
FE = False
BE = True
initial_condition_visualization = False

if FE:
    start_time = time.time()
    results1 = empirical_stability(dx, [kappa_values[1]], A, u_0, v_0)
    end_time = time.time()
    
    cpu_time = end_time - start_time
    print(f"CPU time for FE Method: {cpu_time:.6f} seconds")
    
    #visualize the final time step
    FE_visual(results1, kappa_values[1], Nx, Ny)


if BE:
    # BE_results = empirical_Nt([5], A, u_0, v_0)  failed trial
    # BE_visual(BE_results, 5, Nx, Ny)

    Nt_be = find_optimal_Nt(kappa)
    u_be_T, v_be_T, time1 = BE_NR_Method_new(kappa_values[1], Nt_be, A, u_0, v_0)
    print(f"CPU time for BE-NR Method: {time1:.6f} seconds")
    
    BE_visual(u_be_T, v_be_T, kappa, Nx, Ny)

if initial_condition_visualization:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    im1 = ax1.imshow(u_0, origin = 'lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    im2 = ax2.imshow(v_0, origin = 'lower', extent=[LeftX, RightX, LeftY, RightY], cmap='jet')
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

plt.show()
