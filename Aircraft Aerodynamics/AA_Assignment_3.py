# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:18:11 2025

@author: weiwe
"""
import numpy as np
import matplotlib.pyplot as plt

def parse_javafoil_pressure(filepath):
    data = []
    in_data_section = False
    current_element = None
    
    with open(filepath) as f:
        for line in f:
            
            if line.startswith(('NACA', 'Transition', 'Upper', 'Lower', 'Mach', 'Element:', 'Pressure', '\tAngle')):
                continue
                
            if line.startswith('Element\t'):
                current_element = line.split()[-1]
                in_data_section = False
                continue
                
            if line.strip() == 'x\ty':
                in_data_section = True
                continue
                
            if in_data_section:
                parts = list(filter(None, line.replace('\t', ' ').split()))
                try:
                    row = list(map(float, parts))
                    data.append(row)
                except ValueError:
                    continue
                    
    return np.array(data)

data = np.loadtxt('C:\\Python\\polar_flap.txt',skiprows=5)
aoa, cl = data[:,0], data[:,1]

alpha = np.array([-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 15])
cl_exp = np.array([-0.1, 0.139, 0.39, 0.62, 0.87, 1.01, 1.33, 1.535, 1.72, 1.9, 1.98, 1.99])
alpha1 = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 14, 14.2, 14.4, 14.6])
cl_exp1 = np.array([-0.08, 0.14, 0.35, 0.56, 0.77, 0.98, 1.17, 1.34, 1.48, 1.5, 1.52, 1.53])

# plt.plot(aoa, cl, label='JavaFoil')
# plt.plot(alpha, cl_exp, label='Experiment')
# plt.xlabel('$AoA$ [deg]')
# plt.ylabel('$c_l$ [-]')
# plt.title('flap deflection = 10 deg, $Re=3500000$')
# plt.grid()
# plt.legend()
# plt.show()

data1 = np.loadtxt('C:\\Python\\polar.txt',skiprows=5)
aoa, cl1 = data1[:,0], data1[:,1]

# plt.plot(aoa, cl, label='High-lift Airfoil')
# plt.plot(aoa, cl1, label='Retracted Airfoil')
# plt.plot(alpha1, cl_exp1, label='Experiment Retracted Airfoil')
# plt.xlabel('$AoA$ [deg]')
# plt.ylabel('$c_l$ [-]')
# plt.grid()
# plt.legend()
# plt.show()

data_retracted = parse_javafoil_pressure('retracted_pressure.txt')
x_retracted, cp_retracted = data_retracted[:76,0], data_retracted[:76,2]
x_retracted2, cp_retracted2 = data_retracted[76:,0], data_retracted[76:,2]

data_flap = parse_javafoil_pressure('flap_pressure.txt')
x_flap, cp_flap = data_flap[:76,0], data_flap[:76,4]
x_flap2, cp_flap2 = data_flap[76:,0], data_flap[76:,4]

plt.plot(x_retracted, cp_retracted, color='red', label='Retracted Airfoil, AoA=7 deg')
plt.plot(x_retracted2, cp_retracted2, color='red')
plt.plot(x_flap, cp_flap, color='blue', label='High-lift Airfoil, AoA=2 deg')
plt.plot(x_flap2, cp_flap2, color='blue')
plt.gca().invert_yaxis()
plt.legend()
plt.grid()
plt.show()

# data3 = np.loadtxt('C:\\Python\\flap_pressure.txt',skiprows=12)

# x2, cp7 = data2[:,0], data2[:,15]

# data_airfoil = np.loadtxt('C:\\Python\\naca_23012.txt',skiprows=1, max_rows=76)
# x, y = data_airfoil[:,0], data_airfoil[:,1]

# plt.plot(x,y)
# plt.show()