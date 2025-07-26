# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 21:19:18 2022

@author: weiwe
"""

data=[0]*40
data1=[0]*40
data[0]=1
data[1]=2
c=[0]*40
j=2
for i in range(40):
    if data[j-1]+data[j-2]>4000000:
        break    
    data[j]=data[j-1]+data[j-2]
    if data[j]%2==0:
        c[j]=data[j]
    j+=1
c[1]=2
print(data)
print(c)
print(sum(c))