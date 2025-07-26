# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 08:47:15 2022

@author: weiwe
"""
pi=''
with open('C:\Python\pi-decimals.txt','r') as pidata:
    for line in pidata:
        pi1=line.strip()
        pi=pi+pi1
a=3.14

while a==3.14:
    print('3.14')
    input('Press Enter')
    import os
    os.system('cls')
    i=0
    pit=str(input('Enter the digits you remember about Pi:'))
    for j in range(len(pit)):
        if ord(pi[j])==ord(pit[j]):
            i+=1
        else:
            break
    if i==len(pit):
        print('Here is your point:',i-2)
        print('You next goal is:',pi[0:j+2])
    else:
        print('Here is your point:',i-2)
    a=float(input('Print "3.14" to start again'))
    

