# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:37:21 2022

@author: weiwe
"""

h=int(input('Enter the hour:'))
m=int(input('Enter the minute:'))
if h>12:
    h=h-12
m=m/5
angle=abs((h+m/12)*30-m*30)
print('angle is:',angle)
    