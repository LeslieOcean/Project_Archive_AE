# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 00:04:17 2025

@author: weiwe
"""

import numpy as np
import matplotlib.pyplot as plt

cant_angle = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

CD_ind = np.array([0.00437, 0.00425, 0.00413, 0.00402, 0.00393, 0.00385, 0.00378, 0.00373, 0.0037, 0.00369])
CD_ind_wing = 0.00533*np.ones(len(cant_angle))

# plt.plot(cant_angle, CD_ind, color='blue', label='Wing-1w')
# plt.plot(cant_angle, CD_ind_wing, color='red', label='Wing-1')
# plt.xlabel('Cant Angle [deg]')
# plt.ylabel('Induced Drag Coefficient ($C_{D_{ind}}$)')
# #plt.title('Effect of Cant Angle on Induced Drag')
# plt.grid(True)
# plt.legend()
# plt.show()

cl = np.array([-0.4, 0, 0.4, 1, 2])

wing_1 = np.array([0.00522, 0.00001, 0.00533, 0.03305, 0.12955])
wing_1s = np.array([0.00606, 0.00004, 0.00613, 0.03791, 0.14833])
wing_1w = np.array([0.0045, 0.00019, 0.00437, 0.02723, 0.11074])
wing_2 = np.array([0.00522, 0.0, 0.00548, 0.03454, 0.13988])

plt.plot(cl,wing_1,'-o',label='Wing-1')
plt.plot(cl,wing_1s,'-o',label='Wing-1s')
plt.plot(cl,wing_1w,'-o',label='Wing-1w($phi$=0)')
plt.plot(cl,wing_2,'-o',label='Wing-2')
plt.xlabel('Lift Coefficient ($C_{l}$)')
plt.ylabel('Induced Drag Coefficient ($C_{D_{ind}}$)')
plt.grid()
plt.legend()
plt.show()