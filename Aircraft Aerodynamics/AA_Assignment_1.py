import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#camber line distribution
def camber(x, m, p):
    if x >=0 and x <= p:
        z = m/p**2*(2*p*x - x**2)
    else:
        z = m/(1-p)**2*(1-2*p+2*p*x-x**2)
    return z

#camber line slope - to compute the unit vector
def grad(x, m, p):
    if x >=0 and x <= p:
        dzdx = m/p**2*(2*p - 2*x)
    else:
        dzdx = m/(1-p)**2*(2*p - 2*x)
    return dzdx

#the velocity induced by a vortex at a given point
def VOR2D(gamma, x1, z1, x0, z0):
    r2 = (x1 - x0)**2 + (z1 - z0)**2
    I = np.array([[0, 1], [-1, 0]])
    vector = np.array([[x1 - x0], [z1 - z0]])
    V = gamma / (2 * np.pi * r2) * np.dot(I, vector)
    return V


def circulation(aoa, npanel, m, p, c, imposed=False):
    x = np.linspace(0, c, npanel, endpoint=False)
    l = c/npanel #panel length
    x0 = x+0.25*l #vortex point locations
    x1 = x+0.75*l #collocation points
    grad1 = np.ones(shape = (2, x.shape[0])) #unit normal vectors at collocation points
    z = []
    z0 = []
    z1 = []
    for i in range(npanel):
        z.append(camber(x[i], m, p))
        z0.append(camber(x0[i], m, p))
        z1.append(camber(x1[i], m, p))  
        grad1[1, i] = grad(x1[i], m, p) #gradient at collocation points
    
    grad2 = grad1 / np.linalg.norm(grad1, axis=0) #unit tangential vectors
    n_unit = np.array([-grad2[1], grad2[0]]) #unit normal vectors
    
    #influence coefficient matrix
    a = np.zeros(shape=(npanel, npanel))
    for i in range(npanel): #collocation points//rows
        for j in range(npanel): #downwash due to circualtion at different vortex points//column
            a[i, j] = np.dot(n_unit[:,i], VOR2D(1.0, x1[i], z1[i], x0[j], z0[j])) #
    
    #finding circulations
    U = np.array([[Uinf*np.cos(aoa*2*np.pi/180)],[Uinf*np.sin(aoa*2*np.pi/180)]])
    RHS = np.zeros(npanel)
    for i in range(npanel):
        RHS[i] = -np.dot(n_unit[:,i],U)
    cir = np.linalg.solve(a, RHS)
    
    if imposed == True: #impose the circulation to the trailing edge due to Kutta condition
        cir=np.append(cir,0)
        x=np.append(x,1.0)
    
    return cir, x, l
    
def lift_coeffi(cir, rho, Uinf, c):
    deltaL = rho*Uinf*cir
    L = np.sum(deltaL)
    cl = L/(1/2*rho*Uinf**2*c)
    return cl
    
def moment_coeffi(x, cir, rho, Uinf, c, l):
    deltaL = rho*Uinf*cir
    x_ac = c/4
    deltax = x+0.25*l-x_ac
    M = -np.dot(deltax, deltaL.T)
    cm = M/(1/2*rho*Uinf**2*c**2)
    return cm

def dcldalpha_npanel(results, npanel_val):
    cl_alpha_linear = 2*np.pi
    plt.hlines(cl_alpha_linear, 0, npanel_val[-1], label='Theoretical value')
    for npanel in npanel_val:
        plt.scatter(npanel, results[npanel][0]*180/2/np.pi)
    plt.xlabel('Number of Panels', fontsize=18)
    plt.ylabel('$Cl_{alpha}$[1/rad]', fontsize=18)
    plt.title("Convergence study of $Cl_{alpha}$", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.show()

def interpolate_func(dataset, x):
    inter_func = interp1d(x, dataset)
    return inter_func

NACA = '1412'
m = float(NACA[0])/100 #maximum camber (nondimentional)
p = float(NACA[1])/10 #the location of maximum camber (nondimensional)
npanel_val = [2,6,10,20,30,50,70,90,120,200]
c = 1 #chord length
Uinf = 1
aoa_val = np.arange(-5, 16, 1) #deg
rho = 1.225

convergence = True
pressure_difference = False
moment_lift_coeff_alpha = False
camber_modified = True

if convergence:
    #aoa_val = np.arange(-3, 5, 1) 
    results = {}
    for npanel in npanel_val:
        cl_list = np.zeros(aoa_val.shape[0])
        cm_list = np.zeros(aoa_val.shape[0])
        for i in range(aoa_val.shape[0]):
            cir_strength, x, l = circulation(aoa_val[i], npanel, m, p, c)
            cl_list[i] = lift_coeffi(cir_strength, rho, Uinf, c)
            cm_list[i] = moment_coeffi(x, cir_strength, rho, Uinf, c, l)
        dcldalpha = (cl_list[-1]-cl_list[0])/(aoa_val[-1]-aoa_val[0])
        results[npanel] = dcldalpha, cl_list, cm_list
    
    #dcldalpha_npanel(results, npanel_val)

if pressure_difference:
    aoa_0 = 0
    cir_0, x, l = circulation(aoa_0, 50, m, p, c, imposed=True)
    deltap = rho*Uinf*cir_0/l
    cp = deltap/(1/2*rho*Uinf**2)
    data = np.loadtxt('C:\Python\cp_data_1412.txt',skiprows=3)
    x1, cp1 = data[:, 0], data[:, 2]
    pane = int((len(x1)/2))
    x_up = x1[:pane+1]
    cp_low_func = interpolate_func(cp1[pane-1:], x1[pane-1:])
    cp_low = cp_low_func(x_up)
    deltacp1 = np.zeros(pane+1)
    for i in range(pane+1):
        deltacp1[i] = cp_low[i]-cp1[i]
    
    plt.plot(x,cp,label='Thin airfoil')
    plt.plot(x_up,deltacp1,label='XFOIL')
    plt.xlabel('x/c[-]')
    plt.ylabel(r'$\Delta C_p$[-]')
    plt.legend()
    plt.show()

if moment_lift_coeff_alpha:
    #xfoil data
    data = np.loadtxt('C:\Python\cm_cl_alpha.txt', skiprows=12) 
    aoa, cm, cl = data[:,0], data[:,4], data[:,1]
    #experiment data
    aoa_ex = np.append(aoa, [16, 17, 18, 19, 20])
    cm_ex = np.append(np.ones(len(aoa))*(-0.1),[-0.11, -0.23, -0.3, -0.36])
    cl_ex = np.append(np.linspace(-0.3,1.58,len(aoa)),[1.6, 1.2, 1.1, 1.05, 1])
    #comparison
    plt.plot(aoa_val, results[50][1], label='Thin airfoil $C_l$')
    plt.plot(aoa_val, results[50][2], label='Thin airfoil $C_m$')
    plt.plot(aoa, cm, label='XFOIL $C_m$')
    plt.plot(aoa, cl, label='XFOIL $C_l$')
    plt.plot(aoa_ex, cl_ex, label='Experiment $C_l$')
    plt.plot(aoa_ex[:-1], cm_ex, label='Experiment $C_m$')
    plt.xlabel('$alpha$[deg]')
    plt.ylabel('$C_m$ and $C_l$[-]')
    plt.legend()
    plt.show()
    
if camber_modified:
    m_mod = m*2 #increasing the camber thickness to get NACA 2412
    p_mod = p-0.2 #forward maximum camber to get NACA 1212
    
    cl_2412 = np.zeros(len(aoa_val))
    cm_2412 = np.zeros(len(aoa_val))
    cl_1212 = np.zeros(len(aoa_val))
    cm_1212 = np.zeros(len(aoa_val))
    
    for i in range(len(aoa_val)):
        cir_2412, x, l = circulation(aoa_val[i], 50, m_mod, p, c)
        cir_1212, x, l = circulation(aoa_val[i], 50, m, p_mod, c)
        cl_2412[i] = lift_coeffi(cir_2412, rho, Uinf, c)
        cm_2412[i] = moment_coeffi(x, cir_2412, rho, Uinf, c, l)
        cl_1212[i] = lift_coeffi(cir_1212, rho, Uinf, c)
        cm_1212[i] = moment_coeffi(x, cir_1212, rho, Uinf, c, l)
    
    plt.plot(aoa_val, results[50][1], label='NACA1412 $C_l$')
    plt.plot(aoa_val, results[50][2], label='NACA1412 $C_m$')
    plt.plot(aoa_val, cl_2412, label='NACA2412 $C_l$')
    plt.plot(aoa_val, cm_2412, label='NACA2412 $C_m$')
    plt.plot(aoa_val, cl_1212, label='NACA1212 $C_l$')
    plt.plot(aoa_val, cm_1212, label='NACA1212 $C_m$')
    plt.xlabel('$alpha$[deg]')
    plt.ylabel('$C_m$ and $C_l$[-]')
    plt.legend()
    plt.show()