# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:31:24 2022

@author: weiwe
"""
import math
import matplotlib.pyplot as plt

#plot exponential function
r=0.044
tau=1/(math.log(1+r))
'''x1=[] #year
y1=[] #unimpeded growth factor
for i in range(2005,2101):
    x1.append(i)
    y1.append(math.exp((i-2005)/tau))






#Evolutionary improvement
tau1=1/(math.log(1+(r-0.01)))
y2=[] #evolutionary growth factor
for i in range(2005,2101):
#y2.append(math.exp((i-2005)/tau)*(0.91/((1+0.01)**(i-2005))+0.09))
    y2.append(math.exp((i-2005)/tau1)*0.91+math.exp((i-2005)/tau)*0.09)



#Revolutionary improvement
tL=20 #the aircraft are replaced every 20 years

#Before 2030
y3=[] #e+r growth factor
for i in range(2005,2030):
    y3.append(math.exp((i-2005)/tau1)*0.91+math.exp((i-2005)/tau)*0.09)
    

#<500km
a=0
data1=[]
y3_500=[]

Nintro_1=math.exp((2030-2005-1)/tau) #/N_0

for j in range(1,5):
    a=a+math.exp(-j*20/tau)
    
for i in range(2030,2050):#t_cur
    m=0 
    for j in range(2030,i): 
        m=m+(1-math.exp(-1/tau))*math.exp((i-2005)/tau)*a #N_replace_sum
    data1.append(m)
    if Nintro_1-data1[i-2030]>0:
        y3_500.append(Nintro_1-data1[i-2030])


#500-1500km
b=0
data2=[]
y3_1500=[]

for k in range(1,5):
    b=b+math.exp(-k*20/tau1)

Nintro_2=math.exp((2030-2005-1)/tau1)

for i in range(2030,2050):
    n=0
    for j in range(2030,i):
        n=n+(1-math.exp(-1/tau1))*math.exp((i-2005)/tau1)*b
    data2.append(n)
    if Nintro_2-data2[i-2030]>0:
        y3_1500.append(Nintro_2-data2[i-2030]+(1-0.8)*(math.exp((i-2005)/tau1)-(Nintro_2-data2[i-2030])))


#1500-3000km


data3=[]
y3_3000=[]

Nintro_3=math.exp((2035-2005-1)/tau1)

for i in range(2035,2055):
    p=0
    for j in range(2035,i):
        p=p+(1-math.exp(-1/tau1))*math.exp((i-2005)/tau1)*b
    data3.append(-p)
    if Nintro_3+data3[i-2035]>0:
        y3_3000.append(Nintro_3+data3[i-2035]+(1-0.4)*(math.exp((i-2005)/tau1)-(Nintro_3+data3[i-2035])))

#<3000km

data4=[]
y3_4000=[]

Nintro_4=math.exp((2040-2005-1)/tau1)

for i in range(2040,2060):
    q=0
    for j in range(2040,i):
        q=q+(1-math.exp(-1/tau1))*math.exp((i-2005)/tau1)*b
    data4.append(-q)
    if Nintro_4+data4[i-2040]>0:
        y3_4000.append(Nintro_4+data4[i-2040]+(1-0.3)*(math.exp((i-2005)/tau1)-(Nintro_4+data4[i-2040])))



for i in range(2030,2101):
    if i<2035:
        y3.append(0.09*y3_500[i-2030]+0.24*y3_1500[i-2030]+0.67*math.exp((i-2005)/tau1))

    elif 2035<=i<2040:
        y3.append(0.09*y3_500[i-2030]+0.24*y3_1500[i-2030]+0.19*y3_3000[i-2035]+0.48*math.exp((i-2005)/tau1))
       
    elif 2040<=i<2047:
        y3.append(0.09*y3_500[i-2030]+0.24*y3_1500[i-2030]+0.19*y3_3000[i-2035]+0.48*y3_4000[i-2040])
        
    elif i==2047:
        y3.append(0.09*0+0.24*y3_1500[i-2030]+0.19*y3_3000[i-2035]+0.48*y3_4000[i-2040])
        
    elif 2048<=i<2053:
        y3.append(0.09*0+0.24*(1-0.8)*(math.exp((i-2005)/tau1))+0.19*y3_3000[i-2035]+0.48*y3_4000[i-2040])
    
    elif 2053<=i<2058:
        y3.append(0.09*0+0.24*(1-0.8)*(math.exp((i-2005)/tau1))+0.19*(1-0.4)*(math.exp((i-2005)/tau1))+0.48*y3_4000[i-2040])
    
    else:
        y3.append(0.09*0+0.24*(1-0.8)*(math.exp((i-2005)/tau1))+0.19*(1-0.4)*(math.exp((i-2005)/tau1))+0.48*(1-0.3)*(math.exp((i-2005)/tau1)))


plt.plot(x1,y1,'r',label='Unimpeded growth')
plt.xlabel('Year')
plt.ylabel('Growth factor')
plt.plot(x1,y2,'g',label='Evolutionary improvements')
plt.plot(x1,y3,'b',label='Evolutionary+Revolutionary')
plt.axis([2001,2100,0,8])
plt.legend()
plt.show()
'''


def improve(tL1:int,g:list,r1:float):
    '''tL1=int(input('Input the lifetime of the aircraft: '))
    t_intro=int(input('Input the year of introduction of 1500-3000 km new generation aircraft(between 2030 and 2040): '))
    F_reduction=float(input('Input the reduction of 1500-3000 km new generation aircraft: '))
    r1=float(input('Input the growth factor for the aircraft except shortest range aircraft: '))'''
    

    tau2=1/math.log(1+r1)    
    #Before 2030
    y4=[] #e+r growth factor
    for i in range(2005,g[0][0]):
        y4.append(math.exp((i-2005)/tau2)*0.91+math.exp((i-2005)/tau)*0.09)
        
    
    #<500km
    a=0
    data11=[]
    y4_500=[]
    c=0
    Nintro_11=math.exp((g[0][0]-2005-1)/tau) #/N_0
    
    for j in range(1,5):
        a=a+math.exp(-j*tL1/tau)
        
    for i in range(2030,2030+tL1):#t_cur
        m=0 
        for j in range(2030,i): 
            m=m+(1-math.exp(-1/tau))*math.exp((i-2005)/tau)*a #N_replace_sum
        data11.append(m)
        if Nintro_11-data11[i-2030]>0:
            c+=1
            y4_500.append(Nintro_11-data11[i-2030])
    

    #500-1500km
    b=0
    d=0
    data22=[]
    y4_1500=[]
    
    for k in range(1,5):
        b=b+math.exp(-k*tL1/tau2)
    
    Nintro_22=math.exp((2030-2005-1)/tau2)
    
    for i in range(2030,2030+tL1):
        n=0
        for j in range(2030,i):
            n=n+(1-math.exp(-1/tau2))*math.exp((i-2005)/tau2)*b
        data22.append(n)
        if Nintro_22-data22[i-2030]>0:
            d+=1
            y4_1500.append(Nintro_22-data22[i-2030]+(1-0.8)*(math.exp((i-2005)/tau2)-(Nintro_22-data22[i-2030])))
    
    
    #1500-3000km
    
    e=0
    data33=[]
    y4_3000=[]
    
    Nintro_33=math.exp((2035-2005-1)/tau2)
    
    for i in range(t_intro,t_intro+tL1):
        p=0
        for j in range(t_intro,i):
            p=p+(1-math.exp(-1/tau2))*math.exp((i-2005)/tau2)*b
        data33.append(-p)
        if Nintro_33+data33[i-t_intro]>0:
            e+=1
            y4_3000.append(Nintro_33+data33[i-t_intro]+(1-F_reduction)*(math.exp((i-2005)/tau2)-(Nintro_33+data33[i-t_intro])))
    
    #<3000km
    f=0
    data44=[]
    y4_4000=[]
    
    Nintro_44=math.exp((2040-2005-1)/tau2)
    
    for i in range(2040,2040+tL1):
        q=0
        for j in range(2040,i):
            q=q+(1-math.exp(-1/tau2))*math.exp((i-2005)/tau2)*b
        data44.append(-q)
        if Nintro_44+data44[i-2040]>0:
            f+=1
            y4_4000.append(Nintro_44+data44[i-2040]+(1-0.3)*(math.exp((i-2005)/tau2)-(Nintro_44+data44[i-2040])))
    
    
    
   ''' for i in range(2030,2101):
        if i<t_intro:
            y4.append(0.09*y4_500[i-2030]+0.24*y4_1500[i-2030]+0.67*math.exp((i-2005)/tau2))
    
        elif t_intro<=i<2040:
            y4.append(0.09*y4_500[i-2030]+0.24*y4_1500[i-2030]+0.19*y4_3000[i-t_intro]+0.48*math.exp((i-2005)/tau2))
           
        elif 2040<=i<2030+c:
            y4.append(0.09*y4_500[i-2030]+0.24*y4_1500[i-2030]+0.19*y4_3000[i-t_intro]+0.48*y4_4000[i-2040])
            
        elif 2030+c<=i<2030+d:
            y4.append(0.09*0+0.24*y4_1500[i-2030]+0.19*y4_3000[i-t_intro]+0.48*y4_4000[i-2040])
            
        elif 2030+d<=i<t_intro+e:
            y4.append(0.09*0+0.24*(1-0.8)*(math.exp((i-2005)/tau2))+0.19*y4_3000[i-t_intro]+0.48*y4_4000[i-2040])
        
        elif 2030+e<=i<2040+f:
            y4.append(0.09*0+0.24*(1-0.8)*(math.exp((i-2005)/tau2))+0.19*(1-F_reduction)*(math.exp((i-2005)/tau2))+0.48*y4_4000[i-2040])
        
        else:
            y4.append(0.09*0+0.24*(1-0.8)*(math.exp((i-2005)/tau2))+0.19*(1-F_reduction)*(math.exp((i-2005)/tau2))+0.48*(1-0.3)*(math.exp((i-2005)/tau2)))
    x11=[]
    for i in range(2005,2101):
        x11.append(i)'''
    
    #plt.plot(x11,y1,'r',label='Unimpeded growth')
    
    #plt.plot(x1,y2,'g',label='Evolutionary improvements')
    plt.plot(x11,y4,'y',label='Evolutionary+Revolutionary')
    plt.xlabel('Year')
    plt.ylabel('Growth factor')
    plt.axis([2001,2100,0,8])
    plt.legend()
    fig=plt.show()
    return fig

