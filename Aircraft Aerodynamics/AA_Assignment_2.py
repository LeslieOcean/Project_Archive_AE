# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:17:34 2025

@author: weiwe
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# data = np.loadtxt('C:\\Python\\naca_2412.txt',skiprows=1)
# x, y = data[:,0], data[:,1]

# plt.plot(x, y)
# plt.axis('equal')
# plt.xlabel('x/c[-]')
# plt.ylabel('z/c[-]')
# plt.grid()
# plt.show()

# data = np.loadtxt('C:\\Python\\lift_drag.txt',skiprows=12)
# aoa, cl, cd, top_xtr, bot_xtr = data[:,0], data[:,1], data[:,2], data[:,5], data[:,6]

# top_xtr = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# alpha_0 = np.array([0.00898, 0.00867, 0.00814, 0.00758, 0.00703, 0.0065, 0.00599, 0.00557, 0.00545, 0.00545, 0.00545])
# alpha_4 = np.array([0.0111, 0.01018, 0.00926, 0.00842, 0.00774, 0.00736, 0.00736, 0.00736, 0.00736, 0.00736, 0.00736])
xtr = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.68, 0.7, 0.715, 0.73, 0.75, 0.78, 0.8, 0.9, 1])
cd = np.array([0.01208, 0.01162, 0.01097, 0.01031, 0.00971, 0.00913, 0.00861, 0.0084, 0.00831, 0.00826, 
               0.00825, 0.00826, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083])
plt.plot(xtr, cd)
plt.xlabel('$x_{tr}/c$ [-]')
plt.ylabel('$c_d$ [-]')
plt.title('$alpha=1$ deg, $Re=3e5$')
plt.grid()
plt.show()
# data = np.loadtxt('C:\\Python\\alpha15.txt',skiprows=1)
# x, cf = data[:,1], data[:,6]
# data4 = np.loadtxt('C:\\Python\\alpha4.txt',skiprows=1)
# x4, cf4 = data4[:,1], data4[:,6]
# data_cruise = np.loadtxt('C:\\Python\\alpha_cruise.txt',skiprows=1)
# xc, cfc = data_cruise[:,1], data_cruise[:,6]

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)

# # Plot for AoA = 0°
# axes[0].plot(x[:82], cf[:82], label='AoA = 0°', color='b')
# axes[0].set_xlabel('$x/c$ [-]')
# axes[0].set_ylabel('$C_f$ [-]')
# axes[0].legend()
# axes[0].grid()

# laminar_idex0 = 40
# axes[0].annotate('laminar', 
#                   xy=(x[laminar_idex0], cf[laminar_idex0]), 
#                   xytext=(x[laminar_idex0] + 0.1, cf[laminar_idex0] + 0.0005),
#                   arrowprops=dict(arrowstyle="->", color='black'))

# tran_idex0 = 19
# axes[0].annotate('transition', 
#                   xy=(x[tran_idex0], cf[tran_idex0]), 
#                   xytext=(x[tran_idex0] + 0.1, cf[tran_idex0] + 0.0005),
#                   arrowprops=dict(arrowstyle="->", color='black'))

# tur_idex0 = 10
# axes[0].annotate('turbulent', 
#                   xy=(x[tur_idex0], cf[tur_idex0]), 
#                   xytext=(x[tur_idex0] + 0.01, cf[tur_idex0] + 0.001),
#                   arrowprops=dict(arrowstyle="->", color='black'))
# # Plot for AoA = 4°
# axes[1].plot(xc[:82], cfc[:82], label='AoA = 4°', color='r')
# axes[1].set_xlabel('$x/c$ [-]')
# #axes[1].set_ylabel('$C_f$ [-]')
# axes[1].legend()
# axes[1].grid()

# laminar_idex0 = 55
# axes[1].annotate('laminar', 
#                   xy=(x4[laminar_idex0], cf4[laminar_idex0]), 
#                   xytext=(x4[laminar_idex0] + 0.1, cf4[laminar_idex0] + 0.0005),
#                   arrowprops=dict(arrowstyle="->", color='black'))

# tran_idex0 = 35
# axes[1].annotate('transition', 
#                   xy=(x4[tran_idex0], cf4[tran_idex0]), 
#                   xytext=(x4[tran_idex0] + 0.1, cf4[tran_idex0] + 0.0005),
#                   arrowprops=dict(arrowstyle="->", color='black'))

# tur_idex0 = 15
# axes[1].annotate('turbulent', 
#                   xy=(x4[tur_idex0], cf4[tur_idex0]), 
#                   xytext=(x4[tur_idex0] + 0.01, cf4[tur_idex0] + 0.001),
#                   arrowprops=dict(arrowstyle="->", color='black'))
# # Show the plot
# plt.tight_layout()
# plt.show()

# data = np.loadtxt('C:\Python\cp_alpha1.txt', skiprows=3)
# x, cp = data[:,0], data[:,2]
# data1 = np.loadtxt('C:\Python\cp_alpha1_invisc.txt', skiprows=3)
# x1, cp1 = data1[:,0], data1[:,2]
# plt.plot(x,cp,label='viscid')
# plt.plot(x1,cp1,'--',color='grey',label='inviscid')
# plt.plot(x[15:30],cp[15:30],color='red')
# laminar_idex0 = 20
# plt.annotate('laminar separation bubble', 
#                   xy=(x[laminar_idex0], cp[laminar_idex0]-0.001), 
#                   xytext=(x[laminar_idex0] - 0.1, cp[laminar_idex0] - 0.09),
#                   arrowprops=dict(arrowstyle="<-", color='black'))
# # tur_idex0 = 60
# # plt.annotate('reduced peak', 
# #                   xy=(x[tur_idex0], cp[tur_idex0]), 
# #                   xytext=(x[tur_idex0] + 0.05, cp[tur_idex0] + 0.2),
# #                   arrowprops=dict(arrowstyle="<-", color='black'))
# # rect_width = 0.5  
# # rect_height = 0.001  
# # rect = patches.Rectangle((x[laminar_idex0] - rect_width / 2, cp[laminar_idex0] - rect_height / 2),
# #                          rect_width, rect_height, linewidth=2, edgecolor='red', facecolor='none')

# plt.legend()
# plt.xlabel('$x/c$ [-]')
# plt.ylabel('$c_p$ [-]')
# plt.title('$alpha=1$ deg, $Re=3e5$')
# plt.gca().invert_yaxis()
# plt.show()