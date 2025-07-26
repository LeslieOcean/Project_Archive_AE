
import numpy as np
from scipy.optimize import fsolve
import matplotlib as mlp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class UnderexpandedJet:
    def __init__(self, M_e, P_a, P_e, x_A, y_A, gamma):
        self.M_e = M_e
        self.P_a = P_a
        self.P_e = P_e
        self.x_A = x_A
        self.y_A = y_A
        self.gamma = gamma
        
        self.P_t = self.total_pressure(P_e, M_e, gamma)

    #some important calculations, angles are in [rad]
    def prandtl_meyer_angle(self, M):
        if M <= 1:
            raise ValueError(f"Mach number must be greater than 1 for Prandtl-Meyer function, but got M={M}")
        nu = np.sqrt((self.gamma+1)/(self.gamma-1))*np.arctan(np.sqrt((self.gamma-1)/(self.gamma+1)*(M**2-1)))-np.arctan(np.sqrt(M**2-1))
        return float(nu)

    def mach_angle(self, M):
        mu = np.arcsin(1/M)
        return float(mu)
        
    def V_plus(self, nu, phi, uni=True):
        if uni == True:
            return float(nu-phi)
        else:
            return (nu-phi).astype(float).tolist()
        
    def Gamma_plus(self, mu, phi, uni=True):
        if uni == True:
            return float(np.tan(phi+mu))
        else:
            return (np.tan(phi+mu)).astype(float).tolist()

    def V_minus(self, nu, phi, uni=True):
        if uni == True:
            return float(nu+phi)
        else:
            return (nu+phi).astype(float).tolist()

    def Gamma_minus(self, mu, phi, uni=True):
        if uni == True:
            return float(np.tan(phi-mu))
        else:
            return (np.tan(phi-mu)).astype(float).tolist()

    def total_pressure(self, P_e, M_e, gamma):
        P_t = P_e*(1+(gamma-1)/2*M_e**2)**(gamma/(gamma-1))
        return P_t
    
    def static_pressure(self, M, P_t, gamma):
        return P_t * (1 + (gamma - 1) / 2 * M**2)**(-gamma / (gamma - 1))
        
        
    #mach number calculated from pressure
    def mach_number_pressure(self, P_t, P_a):
        M = np.sqrt((2/(self.gamma-1))*((P_t/P_a)**((self.gamma-1)/self.gamma)-1))
        return float(M)
    
    #calculates mach number from prandtl-meyer angle
    def mach_number_nu(self, nu, uni=True):
        if uni == True:
            M = fsolve(lambda M: self.prandtl_meyer_angle(M)-nu, 1.5).item()
            return float(M)
        else:
            M = np.zeros(len(nu))
            for i in range(len(nu)):
                M[i] = fsolve(lambda M: self.prandtl_meyer_angle(M)-nu[i], 1.5).item()
            return M
    
    #region properties
    def nozzle_exit_0(self):
        nu_0 = self.prandtl_meyer_angle(self.M_e)
        mu_0 = self.mach_angle(self.M_e)
        phi_0 = 0  #flow direction at nozzle exit
        return {
            "region": 0,
            "M": self.M_e,
            "nu": nu_0,
            "phi": phi_0,
            "mu": mu_0,
            "v_plus": self.V_plus(nu_0, phi_0),
            "G_plus": self.Gamma_plus(mu_0, phi_0),
            "v_minus": self.V_minus(nu_0, phi_0),
            "G_minus": self.Gamma_minus(mu_0, phi_0),
            "P_t": self.P_t,
        }

    def uni_region_1(self, v_plus, region):
        M_1 = self.mach_number_pressure(self.P_t, self.P_a)
        nu_1 = self.prandtl_meyer_angle(M_1)
        phi_1 = nu_1 - v_plus
        mu_1 = self.mach_angle(M_1)
        return {
            "region": region,	
            "M": M_1,
            "nu": nu_1,
            "phi": phi_1,
            "mu": mu_1,
            "v_plus": self.V_plus(nu_1, phi_1),
            "G_plus": self.Gamma_plus(mu_1, phi_1),
            "v_minus": self.V_minus(nu_1, phi_1),
            "G_minus": self.Gamma_minus(mu_1, phi_1),
        }

    def uni_region_2(self, v_minus, region, phi_2=0):
        nu_2 = v_minus - phi_2
        M_2 = self.mach_number_nu(nu_2)
        mu_2 = self.mach_angle(M_2)
        return {
            "region": region,
            "M": M_2,
            "nu": nu_2,
            "phi": phi_2,
            "mu": mu_2,
            "v_plus": self.V_plus(nu_2, phi_2),
            "G_plus": self.Gamma_plus(mu_2, phi_2),
            "v_minus": self.V_minus(nu_2, phi_2),
            "G_minus": self.Gamma_minus(mu_2, phi_2),
        }
    
    def simple_wave_region_1(self, v_plus, phi, region, uni=False):
        nu_4 = v_plus + phi
        M_4 = np.array([self.mach_number_nu(nu) for nu in nu_4])
        mu_4 = np.array([self.mach_angle(M) for M in M_4])            
        return {
            "region": region,
            "M": M_4,
            "nu": nu_4,
            "phi": phi,
            "mu": mu_4,
            "v_plus": self.V_plus(nu_4, phi, uni),
            "G_plus": self.Gamma_plus(mu_4, phi, uni),
            "v_minus": self.V_minus(nu_4, phi, uni),
            "G_minus": self.Gamma_minus(mu_4, phi, uni),
        }
    
    def non_simple_region_1(self, v_minus_BC, v_plus_BE, nu_BE, phi_BE, region):
        nu_5, phi_5, M_5, mu_5, G_minus_5, G_plus_5 = [np.zeros((n, n)) for _ in range(6)]
        v_minus_5 = v_minus_BC
        v_plus_5 = v_plus_BE
        for i in range(n): #along gamma plus
            for j in range(n): #along gamma minus
                if j == i:
                    nu_5[i, j] = nu_BE[i]
                    phi_5[i, j] = phi_BE[i]
                else:
                    nu_5[i, j] = 1/2*(v_minus_5[j]+v_plus_5[i])
                    phi_5[i, j] = 1/2*(v_minus_5[j]-v_plus_5[i])
                M_5[i,j] = self.mach_number_nu(nu_5[i, j])
                mu_5[i,j] = self.mach_angle(M_5[i, j])
                G_plus_5[i,j] = np.tan((phi_5[i,j]+mu_5[i,j]+phi_5[i-1, j]+mu_5[i-1,j])/2)
                G_minus_5[i,j] = np.tan((phi_5[i,j]-mu_5[i,j]+phi_5[i,j-1]-mu_5[i,j-1])/2)
        return {
            "region": region,
            "M": M_5,
            "nu": nu_5,
            "phi": phi_5,
            "mu": mu_5,
            "v_plus": v_plus_5,
            "v_minus": v_minus_5,
            "G_plus": G_plus_5,
            "G_minus": G_minus_5,
        }
    
    def non_simple_region_2(self, v_minus_DG, v_plus_DF, nu_DG, phi_DG, region):
        nu_7, phi_7, M_7, mu_7, G_minus, G_plus = [np.zeros((n, n)) for _ in range(6)]
        v_minus_7 = v_minus_DG
        v_plus_7 = v_plus_DF
        for i in range(n): #along gamma minus
            for j in range(n): #along gamma plus
                if j == i:
                    nu_7[i, j] = nu_DG[i]
                    phi_7[i, j] = phi_DG[i]
                else:
                    nu_7[i, j] = 1/2*(v_minus_7[i]+v_plus_7[j])
                    phi_7[i, j] = 1/2*(v_minus_7[i]-v_plus_7[j])
                M_7[i,j] = self.mach_number_nu(nu_7[i, j])
                mu_7[i,j] = self.mach_angle(M_7[i, j])
                G_minus[i,j] = np.tan((phi_7[i, j-1]-mu_7[i, j-1]+phi_7[i,j]-mu_7[i,j])/2)
                G_plus[i,j] = np.tan((phi_7[i-1,j]+mu_7[i-1,j]+phi_7[i,j]+mu_7[i,j])/2)
        return {
            "region": region,
            "M": M_7,
            "nu": nu_7,
            "phi": phi_7,
            "mu": mu_7,
            "v_plus": v_plus_7,
            "v_minus": v_minus_7,
            "G_minus": G_minus,
            "G_plus": G_plus,
        }
    
    def line(self, nu_1_plus, phi_1, nu_2_minus, phi_2):
        nu_a = (nu_1_plus+nu_2_minus)/2 + (phi_2-phi_1)/2
        phi_a = (phi_1+phi_2)/2 + (nu_2_minus-nu_1_plus)/2
        M_a = np.array([self.mach_number_nu(nu) for nu in nu_a])
        mu_a = np.array([flow.mach_angle(M) for M in M_a])
        return {
            "nu": nu_a,
            "phi": phi_a,
            "M": M_a,
            "mu": mu_a,
            "v_plus": self.V_plus(nu_a, phi_a, uni=False),
            "G_plus": self.Gamma_plus(mu_a, phi_a, uni=False),
            "v_minus": self.V_minus(nu_a, phi_a, uni=False),
            "G_minus": self.Gamma_minus(mu_a, phi_a, uni=False),
        }
                              
    
    
    def upper_filter(self, M_matrix):
        region_upper = np.where(np.triu(np.ones_like(M_matrix)), M_matrix, np.nan)
        return region_upper


#define some constants and properties
flow = UnderexpandedJet(M_e=2, P_a=101325, P_e=2*101325, x_A=0, y_A=1, gamma=1.4) #y_A can be modifies according to the real height of nozzle exist

#number of characteristics
n = 21
plot = True

#get uniform region properties
region_0 = flow.nozzle_exit_0()
region_1 = flow.uni_region_1(v_plus=region_0["v_plus"], region=1) #for the first uniform region 1 outside the nozzle
region_2 = flow.uni_region_2(v_minus=region_1["v_minus"], region=2) #for the second uniform region 2 outside the nozzle
region_3 = flow.uni_region_1(v_plus=region_2["v_plus"], region=3) #for the third uniform region 3 outside the nozzle
y_B = 0
x_B = (y_B - flow.y_A)/region_0["G_minus"] + flow.x_A

#get simple wave region 4 mach distribution
phi_4 = np.linspace(region_0["phi"], region_1["phi"], n)
region_4 = flow.simple_wave_region_1(v_plus=region_0["v_plus"], phi=phi_4, region=4) #for the simple wave region 4

nu_B = region_0["nu"]
phi_B = region_0["phi"]
mu_B = region_0["mu"]
line_BC = flow.line(nu_B, phi_B, region_4["nu"], region_4["phi"])
slope_BC = np.tan((phi_B+mu_B+line_BC["phi"]+line_BC["mu"])/2)
x_BC = np.zeros(n)
y_BC = np.zeros(n)
for i in range(n):
    x_BC[i]=fsolve(lambda x: region_4["G_minus"][i]*(x - flow.x_A) + flow.y_A - slope_BC[i]*(x - x_B) - y_B, 0)[0]
    y_BC[i]=region_4["G_minus"][i]*(x_BC[i] - flow.x_A) + flow.y_A
x_C = x_BC[-1]
y_C = y_BC[-1]

x_D = fsolve(lambda x: region_1["G_plus"]*(x - x_C) + y_C - np.tan(region_1["phi"])*(x - flow.x_A) - flow.y_A, 0)[0]
y_D = np.tan(region_1["phi"])*(x_D - flow.x_A) + flow.y_A
x_AC = np.linspace(flow.x_A, x_C, n)
y_AC = flow.y_A + region_1["G_minus"]*(x_AC - flow.x_A)

#get non-simple wave region 5 properties
nu_BE = line_BC["v_minus"]
phi_BE = np.zeros(n)
line_BE = flow.line(nu_BE, phi_BE, region_4["nu"], region_4["phi"])
locate_a = np.tan((line_BC["phi"]-line_BC["mu"]+line_BE["phi"]-line_BE["mu"])/2)
y_E = 0
y_BE = np.zeros(n)
x_BE = (y_BE - y_BC)/locate_a + x_BC  
x_E = x_BE[-1]

region_5 = flow.non_simple_region_1(v_minus_BC=line_BC["v_minus"], v_plus_BE=line_BE["v_plus"], nu_BE=line_BE["nu"], 
                                    phi_BE=line_BE["phi"], region=5)


#get properties of region 6 and mach distribution

nu_CE = region_5["nu"][:, -1]
phi_CE = region_5["phi"][:, -1]
M_CE = flow.mach_number_nu(nu_CE, uni=False)
mu_CE = np.array([flow.mach_angle(M) for M in M_CE])
region_6 = {"nu": nu_CE, "phi": phi_CE, "M": M_CE, "mu": mu_CE, "v_plus": flow.V_plus(nu_CE, phi_CE, uni=False), "v_minus": region_1["v_minus"], 
            "G_plus": flow.Gamma_plus(mu_CE, phi_CE, uni=False), "G_minus": region_1["G_minus"]}

nu_D = region_1["nu"]
phi_D = region_1["phi"]
mu_D = region_1["mu"]
line_DF = flow.line(region_6["nu"], region_6["phi"], nu_D, phi_D)

#get region 7 properties
M_DG = flow.mach_number_pressure(flow.P_t, flow.P_a)
nu_DG = flow.prandtl_meyer_angle(M_DG)*np.ones(n)
mu_DG = flow.mach_angle(M_DG)*np.ones(n)
phi_DG = nu_DG*np.ones(n) - region_6["v_plus"]
v_minus_DG = flow.V_minus(nu_DG, phi_DG, uni=False)
v_plus_DG = flow.V_plus(nu_DG, phi_DG, uni=False)
G_minus_DG = flow.Gamma_minus(mu_DG, phi_DG, uni=False)
G_plus_DG = flow.Gamma_plus(mu_DG, phi_DG, uni=False)
line_DG = {"nu": nu_DG, "phi": phi_DG, "M": M_DG, "mu": mu_DG, "v_minus": v_minus_DG, "v_plus": v_plus_DG, "G_minus": G_minus_DG, "G_plus": G_plus_DG}  

region_7 = flow.non_simple_region_2(line_DG["v_minus"], line_DF["v_plus"], line_DG["nu"], line_DG["phi"], region=7)

nu_FG = region_7["nu"][:, -1]
phi_FG = region_7["phi"][:, -1]
M_FG = flow.mach_number_nu(nu_FG, uni=False)
mu_FG = np.array([flow.mach_angle(M) for M in M_FG])
region_8 = {"nu": nu_FG, "phi": phi_FG, "M": M_FG, "mu": mu_FG, "v_plus": region_2["v_plus"], "v_minus": region_7["v_minus"], 
            "G_plus": region_2["G_plus"], "G_minus": flow.Gamma_minus(mu_FG, phi_FG, uni=False)} #or slope_DF[j]
nu_H = region_2["nu"]
phi_H = region_2["phi"]
mu_H = region_2["mu"]
line_HI = flow.line(nu_H, phi_H, region_8["nu"], region_8["phi"])

nu_HJ = line_HI["v_minus"]
phi_HJ = np.zeros(n)
line_HJ = flow.line(nu_HJ, phi_HJ, region_8["nu"], region_8["phi"])

region_9 = flow.non_simple_region_1(line_HI["v_minus"], line_HJ["v_plus"], line_HJ["nu"], line_HJ["phi"], region=9)


phi_IJ = region_9["phi"][:, -1]
nu_IJ = region_9["nu"][:, -1]
M_IJ = flow.mach_number_nu(nu_IJ, uni=False)
mu_IJ = np.array([flow.mach_angle(M) for M in M_IJ])
region_10 = {"nu": nu_IJ, "phi": phi_IJ, "M": M_IJ, "mu": mu_IJ, "v_plus": flow.V_plus(nu_IJ, phi_IJ, uni=False), "v_minus": flow.V_minus(nu_IJ, phi_IJ, uni=False),
            "G_plus": region_9["G_plus"][:,-1], "G_minus": flow.Gamma_minus(mu_IJ, phi_IJ, uni=False)}
































#Mach distribution in region 0
x_OB = np.linspace(flow.x_A, x_B, n)
y_OB = 0
xa_0 = []
ya_0 = []
Ma_0 = []
for i in range(n):
    xa_min = flow.x_A
    xa_max = x_OB[i]
    xa_0.append(np.linspace(xa_min, xa_max, n))
    ya_0.append(np.ones(n)*(flow.y_A + region_0["G_minus"]*(xa_max - flow.x_A)))
    Ma_0.append(flow.M_e*np.ones(n))
Xa_0 = np.concatenate(xa_0).reshape(n, n)
Ya_0 = np.concatenate(ya_0).reshape(n, n)
Ma_0 = np.concatenate(Ma_0).reshape(n, n)
xa_4 = []
ya_4 = []
Ma_4 = []
for i in range(n):
    xa_min = flow.x_A
    xa_max = x_BC[i]
    xa_4.append(np.linspace(xa_min, xa_max, n))
    ya_4.append(region_4["G_minus"][i]*(xa_4[i] - flow.x_A) + flow.y_A)
    Ma_4.append(flow.mach_number_nu(region_4["nu"][i])*np.ones(n))
Xa_4 = np.concatenate(xa_4).reshape(n, n)
Ya_4 = np.concatenate(ya_4).reshape(n, n)
Ma_4 = np.concatenate(Ma_4).reshape(n, n)
x_AD = []
y_AD = []
xa_1 = []
ya_1 = []
Ma_1 = []
for i in range(n):
    x_AD.append(fsolve(lambda x: np.tan(region_1["phi"])*(x - flow.x_A) + flow.y_A - region_1["G_plus"]*(x - x_AC[i]) - y_AC[i], 0)[0])
    y_AD.append(region_1["G_plus"]*(x_AD[i] - x_AC[i]) + y_AC[i])
    xa_min = x_AC[i]
    xa_max = x_AD[i]
    xa_1.append(np.linspace(xa_min, xa_max, n))
    ya_1.append(region_1["G_plus"]*(xa_1[i] - x_AC[i]) + y_AC[i])
    Ma_1.append(region_1["M"]*np.ones(n))
Xa_1 = np.concatenate(xa_1).reshape(n, n)
Ya_1 = np.concatenate(ya_1).reshape(n, n)
Ma_1 = np.concatenate(Ma_1).reshape(n, n)
xa_5 = np.zeros((n, n))
ya_5 = np.zeros((n, n))
xa_5[0, :] = x_BC
ya_5[0, :] = y_BC
for i in range(1, n): #along gamma plus
    for j in range(n): #along gamma minus
        if j == i:
            xa_5[i, j] = x_BE[i]
            ya_5[i, j] = y_BE[i]
        if j > i:
            xa_5[i, j] = fsolve(lambda x: region_5["G_minus"][i,j]*(x - xa_5[i-1, j]) + ya_5[i-1, j] - region_5["G_plus"][i,j]*(x - xa_5[i, j-1]) - ya_5[i, j-1], 0)[0]
            ya_5[i, j] = region_5["G_minus"][i,j]*(xa_5[i, j] - xa_5[i-1, j]) + ya_5[i-1, j]
x_CE = np.transpose(xa_5[:, -1])
y_CE = np.transpose(ya_5[:, -1])
slope_DF = np.tan((region_6["phi"]-region_6["mu"]+phi_D-mu_D)/2)
x_DF = np.zeros(n)
y_DF = np.zeros(n)
for i in range(n):
    x_DF[i] = fsolve(lambda x: region_6["G_plus"][i]*(x - x_CE[i]) + y_CE[i] - slope_DF[i]*(x - x_D) - y_D, 0)[0]
    y_DF[i] = region_6["G_plus"][i]*(x_DF[i] - x_CE[i]) + y_CE[i]
x_F = x_DF[-1]
y_F = y_DF[-1]
xa_6 = []
ya_6 = []
Ma_6 = []
for i in range(n):
    xa_min = x_CE[i]
    xa_max = x_DF[i]
    xa_6.append(np.linspace(xa_min, xa_max, n))
    ya_6.append(region_6["G_plus"][i]*(xa_6[i] - x_CE[i]) + y_CE[i])
    Ma_6.append(region_6["M"][i]*np.ones(n))
Xa_6 = np.concatenate(xa_6).reshape(n, n)
Ya_6 = np.concatenate(ya_6).reshape(n, n)
Ma_6 = np.concatenate(Ma_6).reshape(n, n)
slope_DG = np.tan((phi_DG+phi_D)/2)
slope_DF_plus = np.tan((line_DF["phi"]+line_DF["mu"]+phi_DG+mu_DG))
x_DG = np.zeros(n)
y_DG = np.zeros(n)
x_ref = x_D
y_ref = y_D
for i in range(n):
    x_DG[i] = fsolve(lambda x: slope_DG[i]*(x - x_ref) + y_ref - slope_DF_plus[i]*(x - x_DF[i]) - y_DF[i], 0)[0]
    y_DG[i] = slope_DG[i]*(x_DG[i] - x_ref) + y_ref
xa_7 = np.zeros((n, n))
ya_7 = np.zeros((n, n))
xa_7[0, :] = x_DF
ya_7[0, :] = y_DF
for i in range(1, n): #along gamma minus
    for j in range(n): #along gamma plus
        if j == i:
            xa_7[i, j] = x_DG[j]
            ya_7[i, j] = y_DG[j]
        if j > i:
            # slope_1 = np.tan((region_7["phi"][i-1, j]+region_7["mu"][i-1,j]+region_7["phi"][i,j]+region_7["mu"][i,j])/2)
            # slope_2 = np.tan((region_7["phi"][i, j-1]-region_7["mu"][i,j-1]+region_7["phi"][i,j]-region_7["mu"][i,j])/2)
            # xa_7[i, j] = fsolve(lambda x:region_7["G_plus"][i,j]*(x - xa_7[i-1, j]) + ya_7[i-1, j] - region_7["G_minus"][i,j]*(x - xa_7[i, j-1]) - ya_7[i, j-1], 0)[0]
            # ya_7[i, j] = region_7["G_plus"][i,j]*(xa_7[i, j] - xa_7[i-1, j]) + ya_7[i-1, j]
            xa_7[i, j] = fsolve(lambda x:slope_DF_plus[j]*(x - xa_7[i-1, j]) + ya_7[i-1, j] - slope_DF[j]*(x - xa_7[i, i]) - ya_7[i, i], 0)[0]
            ya_7[i, j] =slope_DF_plus[j]*(xa_7[i, j]  - xa_7[i-1, j]) + ya_7[i-1, j]

#get properties of region 8 and mach distribution
x_FG = np.transpose(xa_7[:, -1])
y_FG = np.transpose(ya_7[:, -1])
x_G = x_FG[-1]
y_G = y_FG[-1]

y_H = 0
x_H = (y_H - y_F)/region_2["G_minus"] + x_F
xa_2 = []
ya_2 = []
Ma_2 = []
for i in range(n):
    xa_min = xa_6[-1][i]
    xa_max = (y_H - ya_6[-1][i])/region_2["G_minus"] + xa_6[-1][i]
    xa_2.append(np.linspace(xa_min, xa_max, n))
    ya_2.append(region_2["G_minus"]*(xa_2[i] - xa_min) + ya_6[-1][i])
    Ma_2.append(region_2["M"]*np.ones(n))
Xa_2 = np.concatenate(xa_2).reshape(n, n)
Ya_2 = np.concatenate(ya_2).reshape(n, n)
Ma_2 = np.concatenate(Ma_2).reshape(n, n)

slope_HI = np.tan((phi_H+mu_H+line_HI["phi"]+line_HI["mu"])/2)
x_HI = np.zeros(n)
y_HI = np.zeros(n)
for i in range(n):
    x_HI[i]=fsolve(lambda x: region_8["G_minus"][i]*(x - x_FG[i]) + y_FG[i] - slope_HI[i]*(x - x_H) - y_H, 0)[0]
    y_HI[i]=region_8["G_minus"][i]*(x_HI[i] - x_FG[i]) + y_FG[i]
x_I = x_HI[-1]
y_I = y_HI[-1]

xa_8 = []
ya_8 = []
Ma_8 = []
for i in range(n):
    xa_min = x_FG[i]
    xa_max = x_HI[i]
    xa_8.append(np.linspace(xa_min, xa_max, n))
    ya_8.append(region_8["G_minus"][i]*(xa_8[i] - xa_min) + y_FG[i])
    Ma_8.append(region_8["M"][i]*np.ones(n))
Xa_8 = np.concatenate(xa_8).reshape(n, n)
Ya_8 = np.concatenate(ya_8).reshape(n, n)
Ma_8 = np.concatenate(Ma_8).reshape(n, n)

x_K = fsolve(lambda x: region_3["G_plus"]*(x - x_I) + y_I - np.tan(line_DG["phi"][-1])*(x - x_G) - y_G, 0)[0]
y_K = np.tan(line_DG["phi"][-1])*(x_K - x_G) + y_G
x_GI = np.linspace(x_G, x_I, n)
y_GI = region_3["G_minus"]*(x_GI - x_G) + y_G

x_GK = []
y_GK = []
xa_3 = []
ya_3 = []
Ma_3 = []
for i in range(n):
    x_GK.append(fsolve(lambda x: np.tan(line_DG["phi"][-1])*(x - x_G) + y_G - region_3["G_plus"]*(x - x_GI[i]) - y_GI[i], 0)[0])
    y_GK.append(region_3["G_plus"]*(x_GK[i] - x_GI[i]) + y_GI[i])
    xa_min = x_GI[i]
    xa_max = x_GK[i]
    xa_3.append(np.linspace(xa_min, xa_max, n))
    ya_3.append(region_3["G_plus"]*(xa_3[i] - x_GI[i]) + y_GI[i])
    Ma_3.append(region_3["M"]*np.ones(n))
Xa_3 = np.concatenate(xa_3).reshape(n, n)
Ya_3 = np.concatenate(ya_3).reshape(n, n)
Ma_3 = np.concatenate(Ma_3).reshape(n, n)


locate_c = np.tan((line_HI["phi"]-line_HI["mu"]+line_HJ["phi"]-line_HJ["mu"])/2)
y_J = 0
y_HJ = np.zeros(n)
x_HJ = (y_HJ - y_HI)/locate_c + x_HI
x_J = x_HJ[-1]

xa_9 = np.zeros((n, n))
ya_9 = np.zeros((n, n))
xa_9[0, :] = x_HI
ya_9[0, :] = y_HI
for i in range(1, n): #along gamma plus
    for j in range(n): #along gamma minus
        if j == i:
            xa_9[i, j] = x_HJ[i]
            ya_9[i, j] = y_HJ[i]
        if j > i:
            xa_9[i, j] = fsolve(lambda x: region_9["G_minus"][i,j]*(x - xa_9[i-1, j]) + ya_9[i-1, j] - region_9["G_plus"][i,j]*(x - xa_9[i, j-1]) - ya_9[i, j-1], 0)[0]
            ya_9[i, j] = region_9["G_minus"][i,j]*(xa_9[i, j] - xa_9[i-1, j]) + ya_9[i-1, j]

#get properties of region 10 and mach distribution
x_IJ = np.transpose(xa_9[:, -1])
y_IJ = np.transpose(ya_9[:, -1])


x_L = fsolve(lambda x: region_10["G_plus"][-1]*(x - x_J) + y_J - region_3["G_plus"]*(x - x_I) - y_I, 0)[0]
y_L = region_10["G_plus"][-1]*(x_L - x_J) + y_J
slope_10 = (y_L-y_IJ)/(x_L-x_IJ)
xa_10 = []
ya_10 = []
Ma_10 = []
for i in range(n):
    xa_min = x_IJ[i]
    xa_max = x_L
    xa_10.append(np.linspace(xa_min, xa_max, n))
    ya_10.append(slope_10[i]*(xa_10[i] - xa_min) + y_IJ[i])
    Ma_10.append(region_10["M"][i]*np.ones(n))
Xa_10 = np.concatenate(xa_10).reshape(n, n)
Ya_10 = np.concatenate(ya_10).reshape(n, n)
Ma_10 = np.concatenate(Ma_10).reshape(n, n)














































pressure = True
if pressure:
    Pa_0 = flow.P_e*np.ones((n, n))
    Pa_1 = flow.P_a*np.ones((n, n))
    Pa_2 = flow.static_pressure(region_2["M"], flow.P_t, flow.gamma)*np.ones((n, n))
    Pa_3 = flow.P_a*np.ones((n, n))
    Pa_4, Pa_5, Pa_6, Pa_7, Pa_8, Pa_9, Pa_10 = [np.zeros((n, n)) for _ in range(7)]
    for i in range(n):
        for j in range(n):
            Pa_4[i, j] = flow.static_pressure(region_4["M"][i], flow.P_t, flow.gamma)
            Pa_5[i, j] = flow.static_pressure(region_5["M"][i, j], flow.P_t, flow.gamma)
            Pa_6[i, j] = flow.static_pressure(region_6["M"][i], flow.P_t, flow.gamma)
            Pa_7[i, j] = flow.static_pressure(region_7["M"][i, j], flow.P_t, flow.gamma)
            Pa_8[i, j] = flow.static_pressure(region_8["M"][i], flow.P_t, flow.gamma)
            Pa_9[i, j] = flow.static_pressure(region_9["M"][i, j], flow.P_t, flow.gamma)
            Pa_10[i, j] = flow.static_pressure(region_10["M"][i], flow.P_t, flow.gamma)
    fig, ax = plt.subplots(figsize=(14, 2))

    all_pressure = [Pa_0, Pa_1, Pa_2, Pa_3, Pa_4, Pa_5, Pa_6, Pa_7, Pa_8, Pa_9, Pa_10]
    min_pressure = min(np.min(p) for p in all_pressure)
    max_pressure = max(np.max(p) for p in all_pressure)
    norm = Normalize(vmin=min_pressure, vmax=max_pressure)
    c0 = ax.contourf(Xa_0, Ya_0, Pa_0, levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c1 = ax.contourf(Xa_1, Ya_1, Pa_1, levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c2 = ax.contourf(Xa_2, Ya_2, Pa_2, levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c3 = ax.contourf(Xa_3, Ya_3, Pa_3, levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c4 = ax.contourf(Xa_4, Ya_4, Pa_4, levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c5 = ax.contourf(xa_5, ya_5, flow.upper_filter(Pa_5), levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c6 = ax.contourf(Xa_6, Ya_6, Pa_6, levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c7 = ax.contourf(xa_7, ya_7, flow.upper_filter(Pa_7), levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c8 = ax.contourf(Xa_8, Ya_8, Pa_8, levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c9 = ax.contourf(xa_9, ya_9, flow.upper_filter(Pa_9), levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    c10 = ax.contourf(Xa_10, Ya_10, Pa_10, levels=np.linspace(min_pressure, max_pressure, n), cmap="jet")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="jet"), ax=ax, label="Pressure (Pa)")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# Plot the contours
charateristics = False
if plot:
    fig, ax = plt.subplots(figsize=(14, 2))

    #make a consistent color map
    all_mach = [region_0["M"], region_1["M"], region_2["M"], region_3["M"], region_4["M"], region_5["M"], region_6["M"], region_7["M"], region_8["M"], region_9["M"], region_10["M"]]
    min_mach = min(np.min(m) for m in all_mach)
    max_mach = max(np.max(m) for m in all_mach)
    norm = Normalize(vmin=min_mach, vmax=max_mach)
    
    c0 = ax.contourf(Xa_0, Ya_0, Ma_0, levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    
    c1 = ax.contourf(Xa_1, Ya_1, Ma_1, levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    c2 = ax.contourf(Xa_2, Ya_2, Ma_2, levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    c3 = ax.contourf(Xa_3, Ya_3, Ma_3, levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    c4 = ax.contourf(Xa_4, Ya_4, Ma_4, levels=np.linspace(min_mach, max_mach, n), cmap="jet")

    c5 = ax.contourf(xa_5, ya_5, flow.upper_filter(region_5["M"]), levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    c6 = ax.contourf(Xa_6, Ya_6, Ma_6, levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    c7 = ax.contourf(xa_7, ya_7, flow.upper_filter(region_7["M"]), levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    c8 = ax.contourf(Xa_8, Ya_8, Ma_8, levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    c9 = ax.contourf(xa_9, ya_9, flow.upper_filter(region_9["M"]), levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    c10 = ax.contourf(Xa_10, Ya_10, Ma_10, levels=np.linspace(min_mach, max_mach, n), cmap="jet")
    
    if charateristics:
        #plot simple wave characteristics
        for i in range(n):
            ax.plot([flow.x_A, x_BC[i]], [flow.y_A, y_BC[i]], color="black", linewidth=1)
            ax.plot([x_CE[i], x_DF[i]], [y_CE[i], y_DF[i]], color="black", linewidth=1)
            ax.plot([x_FG[i], x_HI[i]], [y_FG[i], y_HI[i]], color="black", linewidth=1)
            ax.plot([x_IJ[i], x_L], [y_IJ[i], y_L], color="black", linewidth=1)
        #plot non-simple wave characteristics
        for i in range(n-1):
            for j in range(n):
                if j > i:
                    ax.plot([xa_5[i,j], xa_5[i+1,j]], [ya_5[i, j], ya_5[i+1, j]], color="black", linewidth=1)
                    ax.plot([xa_7[i,j], xa_7[i+1,j]], [ya_7[i, j], ya_7[i+1, j]], color="black", linewidth=1)
                    ax.plot([xa_9[i,j], xa_9[i+1,j]], [ya_9[i, j], ya_9[i+1, j]], color="black", linewidth=1)
        for i in range(n-1):
            for j in range(n-1):
                if j >= i:
                    ax.plot([xa_5[i,j], xa_5[i,j+1]], [ya_5[i, j], ya_5[i, j+1]], color="black", linewidth=1)
                    ax.plot([xa_7[i,j], xa_7[i,j+1]], [ya_7[i, j], ya_7[i, j+1]], color="black", linewidth=1)
                    ax.plot([xa_9[i,j], xa_9[i,j+1]], [ya_9[i, j], ya_9[i, j+1]], color="black", linewidth=1)
        ax.plot([x_L, x_K], [y_L, y_K], color="black", linewidth=2)#shock wave
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="jet"), ax=ax, label="Mach Number")
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #ax.set_title("Underexpanded Jet Mach Number Distribution and Characteristics")
    plt.show()