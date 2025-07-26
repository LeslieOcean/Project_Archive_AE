import math
print('*** Welcome to ISA calculator ***')
print('Choose your input unit')
print('1. Calculate ISA for altitude in meters')
print('2. Calculate ISA for altitude in feet')
print('3. Calculate ISA for altitude in FL')
n=int(input('\nEnter your choice:'))
if n>3:
    print('Sorry, you have to choose agian in three options.')
#def p(a,b,c):
#        print('Temperature [K]:',round(a,3))
#        print('Pressure [Pa]:',round(b,3))
#        print('Density [kg/m^3]:',round(c,3))
if n==1:
    h=float(input('Enter altitude [m]:'))
    g0=9.80665
    R=287.0
    if h<=11000.0:
        h1=min(h,11000.0)
        T0=288.15
        p=101325.0
        T1=T0-0.0065*h1
        P1=p*(T1/T0)**(-g0/(-0.0065*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=20000.0:
        h1=min(h,20000.0)
        T0=216.65
        p=22625.791
        P1=p*math.exp(-g0/(R*T0)*(h-11000))
        rho1=P1/(R*T0)
        print('Temperature [K]:',round(T0,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=32000.0:
        h1=min(h,32000.0)
        T0=216.65
        p=5474.89
        T1=T0+0.001*(h1-20000)
        P1=p*(T1/T0)**(-g0/(0.001*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=47000.0:
        h1=min(h,47000.0)
        T0=228.65
        p=867.723
        T1=T0+0.0028*(h1-32000)
        P1=p*(T1/T0)**(-g0/(0.0028*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=51000.0:
        h1=min(h,51000.0)
        T0=270.65
        p=110.826
        P1=p*math.exp(-g0/(R*T0)*(h-47000))
        rho1=P1/(R*T0)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,5))
    elif h<=71000.0:
        h1=min(h,71000.0)
        T0=270.65
        p=66.884
        T1=T0-0.0028*(h1-51000)
        P1=p*(T1/T0)**(-g0/(-0.0028*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,7))
    elif h<=86000.0:
        h1=min(h,86000.0)
        T0=214.65
        p=3.951
        T1=T0-0.002*(h1-71000)
        P1=p*(T1/T0)**(-g0/(-0.002*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,8))
    else:
        print('Sorry, I can only do altitudes up to 86000 m.')
    dummy=input('Press enter to end the ISA calculator.')

elif n==2:
    ft=0.3048
    hinft=float(input('Enter altitude in ft:'))
    h=hinft*ft
    g0=9.80665
    R=287.0
    if h<=11000.0:
        h1=min(h,11000.0)
        T0=288.15
        p=101325.0
        T1=T0-0.0065*h1
        P1=p*(T1/T0)**(-g0/(-0.0065*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=20000.0:
        h1=min(h,20000.0)
        T0=216.65
        p=22625.791
        P1=p*math.exp(-g0/(R*T0)*(h-11000))
        rho1=P1/(R*T0)
        print('Temperature [K]:',round(T0,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=32000.0:
        h1=min(h,32000.0)
        T0=216.65
        p=5474.89
        T1=T0+0.001*(h1-20000)
        P1=p*(T1/T0)**(-g0/(0.001*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=47000.0:
        h1=min(h,47000.0)
        T0=228.65
        p=867.723
        T1=T0+0.0028*(h1-32000)
        P1=p*(T1/T0)**(-g0/(0.0028*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=51000.0:
        h1=min(h,51000.0)
        T0=270.65
        p=110.826
        P1=p*math.exp(-g0/(R*T0)*(h-47000))
        rho1=P1/(R*T0)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,5))
    elif h<=71000.0:
        h1=min(h,71000.0)
        T0=270.65
        p=66.884
        T1=T0-0.0028*(h1-51000)
        P1=p*(T1/T0)**(-g0/(-0.0028*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,7))
    elif h<=86000.0:
        h1=min(h,86000.0)
        T0=214.65
        p=3.951
        T1=T0-0.002*(h1-71000)
        P1=p*(T1/T0)**(-g0/(-0.002*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,8))
    else:
        print('Sorry, I can only do altitudes up to 86000 m.')
    dummy=input('Press enter to end the ISA calculator.')
    
else:
    ft=0.3048
    hinFL=float(input('Enter altitude in FL:'))
    h=hinFL*100*ft
    g0=9.80665
    R=287.0
    if h<=11000.0:
        h1=min(h,11000.0)
        T0=288.15
        p=101325.0
        T1=T0-0.0065*h1
        P1=p*(T1/T0)**(-g0/(-0.0065*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=20000.0:
        h1=min(h,20000.0)
        T0=216.65
        p=22625.791
        P1=p*math.exp(-g0/(R*T0)*(h-11000))
        rho1=P1/(R*T0)
        print('Temperature [K]:',round(T0,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=32000.0:
        h1=min(h,32000.0)
        T0=216.65
        p=5474.89
        T1=T0+0.001*(h1-20000)
        P1=p*(T1/T0)**(-g0/(0.001*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=47000.0:
        h1=min(h,47000.0)
        T0=228.65
        p=867.723
        T1=T0+0.0028*(h1-32000)
        P1=p*(T1/T0)**(-g0/(0.0028*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,3))
    elif h<=51000.0:
        h1=min(h,51000.0)
        T0=270.65
        p=110.826
        P1=p*math.exp(-g0/(R*T0)*(h-47000))
        rho1=P1/(R*T0)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,5))
    elif h<=71000.0:
        h1=min(h,71000.0)
        T0=270.65
        p=66.884
        T1=T0-0.0028*(h1-51000)
        P1=p*(T1/T0)**(-g0/(-0.0028*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,7))
    elif h<=86000.0:
        h1=min(h,86000.0)
        T0=214.65
        p=3.951
        T1=T0-0.002*(h1-71000)
        P1=p*(T1/T0)**(-g0/(-0.002*R))
        rho1=P1/(R*T1)
        print('Temperature [K]:',round(T1,3))
        print('Pressure [Pa]:',round(P1,3))
        print('Density [kg/m^3]:',round(rho1,8))
    else:
        print('Sorry, I can only do altitudes up to 86000 m.')
    dummy=input('Press enter to end the ISA calculator.')
