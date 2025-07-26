# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:51:23 2022

@author: weiwe
"""
#1

import matplotlib.pyplot as plt
import numpy as np
'''
x=np.arange(0,9,0.01)
y1=x**2
y2=x**3
plt.plot(x,y1,'r')
plt.plot(x,y2,'b')
13

plt.show()
'''


#2
NACA = open('NACA0012.dat','r')
lines=NACA.readlines()
NACA.close()

alpha = []
cl = []
for line in lines:
    col = line.strip().split('\t')
    alpha.append(col[0])
    cl.append(col[1])

'''
#3
b=0

from math import sqrt
n=float(input('Which number do you want to check:'))
if n%int(n)>=0.5:
    a=int(n)+1
else:
    a=int(n)
    
for i in range(1,a):
    b=a-i**2
    if b>0:
        if sqrt(b)==int(sqrt(b)):
            print( n, 'is the sum of the square of', i, 'and', sqrt(b))
            break
    else:
        print(n,'is not the sum of two squares')
        break


#4


from math import atan2,cos,sin
def graveforce(mass, x, y):
    G=6.674*10**(-11)
    n=len(mass)
    fx=[]
    fy=[]
    for j in range(n):
        fx1=0
        fy1=0
        for i in range(n):
            if i!=j:
                theta=atan2(y[i]-y[j],x[i]-x[j])
                r=(x[i]-x[j])**2+(y[i]-y[j])**2
                fx1=fx1+G*mass[j]*mass[i]/r*cos(theta)               
                fy1=fy1+G*mass[j]*mass[i]/r*sin(theta)
                   
        fx.append(fx1)   
        fy.append(fy1)      
    return fx,fy      
'''












