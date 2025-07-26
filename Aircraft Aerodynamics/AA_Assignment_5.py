# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# data = np.loadtxt(
#     'C:\\Python\\NR640ctcp.txt',
#     skiprows=2)

# J = data[:, 0]   # Advance ratio (v/(nD))
# Ct = data[:, 2]  # Thrust coefficient
# Cp = data[:, 3]  # Power coefficient
# eta = data[:, 7] # Efficiency (η)
# data_ = np.loadtxt(
#     'C:\\Python\\NR640ctcp10068.txt',
#     skiprows=2)

# J_ = data_[:, 0]   # Advance ratio (v/(nD))
# Ct_ = data_[:, 2]  # Thrust coefficient
# Cp_ = data_[:, 3]  # Power coefficient
# eta_ = data_[:, 7] # Efficiency (η)

# data1 = np.loadtxt(
#     'C:\\Python\\experiment 1.txt',
#     skiprows=1) 
# data2 = np.loadtxt(
#     'C:\\Python\\experiment 2.txt',
#     skiprows=1) 

# J1 = data1[:, 0]   # Advance ratio (v/(nD))
# Ct1 = data1[:, 1]  # Thrust coefficient
# Cp1 = data1[:, 2]  # Power coefficient
# eta1 = data1[:, 3]*100 # Efficiency (η)

# J2 = data2[:, 0]   # Advance ratio (v/(nD))
# Ct2 = data2[:, 1]  # Thrust coefficient
# Cp2 = data2[:, 2]  # Power coefficient
# eta2 = data2[:, 3]*100 # Efficiency (η)

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6))

# ax1.plot(J, Ct, 'b', label='JavaProp RPM=10000')
# ax1.plot(J1, Ct1, '-', color='skyblue', label='Experiment RPM=10035')
# ax1.plot(J2, Ct2, '-', color='dodgerblue', label='Experiment RPM=10068')
# ax1.set_title('Thrust Coefficient vs Advance Ratio')
# ax1.set_ylabel('$C_T$')
# ax1.grid(True)
# ax1.legend(loc='lower left')

# ax2.plot(J, Cp, 'r', label='JavaProp RPM=10000')
# ax2.plot(J1, Cp1, '-', color='lightcoral', label='Experiment RPM=10035')
# ax2.plot(J2, Cp2, '-', color='pink', label='Experiment RPM=10068')
# ax2.set_title('Power Coefficient vs Advance Ratio')
# ax2.set_ylabel('$C_P$')
# ax2.grid(True)
# ax2.legend(loc='lower left')

# ax3.plot(J, eta, 'g', label='JavaProp')
# ax3.plot(J1, eta1, '-', color='lightgreen', label='Experiment RPM=10035')
# ax3.plot(J2, eta2, '-', color='olive', label='Experiment RPM=10068')
# ax3.set_title('Efficiency vs Advance Ratio')
# ax3.set_xlabel('Advance Ratio (J) [-]')
# ax3.set_ylabel('η [%]')
# ax3.grid(True)
# ax3.legend(loc='lower left')

# plt.tight_layout()
# fig.suptitle('NR640 JavaProp simulation and experiment data at Re=24500')
# fig.subplots_adjust(top=0.9)
# plt.show()

data2 = np.loadtxt(
    'C:\\Python\\etaB2.txt',
    skiprows=2)

J2 = data2[:, 0]   
eta2 = data2[:, 7] 
data3 = np.loadtxt(
    'C:\\Python\\etaD150.txt',
    skiprows=2)

J3 = data3[:, 0]   
eta3 = data3[:, 7] 
data4 = np.loadtxt(
    'C:\\Python\\etaD50.txt',
    skiprows=2)

J4 = data4[:, 0]   
eta4 = data4[:, 7]
# data6 = np.loadtxt(
#     'C:\\Python\\etaB6.txt',
#     skiprows=2)

# J6 = data6[:, 0]   
# eta6 = data6[:, 7] 

plt.plot(J2, eta2, label='1xD')
plt.plot(J3, eta3, label='1.5xD')
plt.plot(J4, eta4, label='0.5xD')
#plt.plot(J6, eta6, label='6 Blades')
plt.xlabel('Advance Ratio (J) [-]')
plt.ylabel('η [%]')
plt.title('The Effect of Radius in Efficiency (T = 0.3N)')
plt.grid()
plt.legend()
plt.show()