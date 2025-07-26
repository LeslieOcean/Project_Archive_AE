# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:33:23 2022

@author: weiwe
"""
import view      # file with your otwv drawing functions

from math import cos, sin, pi
import pygame as pg
# Size of window
xmax = 1000 #[pixels] window width (= x-coordinate runs of right side right+1)
ymax = 700  #[pixels] window height(= y-coordinate of lower side of window+1)

# Size of viewport (angles in degrees)
minelev = -14.0 #[deg] elevation angle lower side
maxelev = 14.0  #[deg] elevation angle top side
minazi = -20.0  #[deg] azimuth angle left side
maxazi = 20.0   #[deg] azimuth angle right side

#inital condition
gamar = -3*pi/180
alpha = 3*pi/180
V = 220
x = -3000
y = 200

#time
tsim = 0.0
tstart=pg.time.get_ticks()/1000.
dt = 0.1

# Model parameters
mzerofuel = 40000.0 #[kg] mass aircraft + payload excl fuel
mfuel = 5000.0 # [kg]
S = 102.0 #[m2]
Tmax = 200 * 1e3 #[N]
CT = 17.8 * 1e6 # [kg/Ns] kg/s per N thrustSpecific fuel consumption
rho = 1.225 # [kg/m3] air density (constant or use ISA)
g = 9.81 # [m/s2] gravitational constant
throttledot = 0.1 # [1/s] throttle speed (spool up/down speed)
alphadot = 1.0*pi/180 # [deg/s] alpha change due to control
flapsdot = 0.2 # [1/s] flap deflection speed
alphamin = -10.0*pi/180 # [deg]
alphamax = 20.0*pi/180 # [deg]


throttle = 0
flaps = 0
dbrake = 0
dgear = 0

# Set pitch angle
theta = float(input("Enter pitch angle[deg]:")) # pitch angle [deg]

# Set up window, scr is surface of screen
scr = view.openwindow(xmax, ymax)

running = True
while running:
    

    # Clear screen scr
    view.clr(scr)

    # Get user inputs by processing events
    dalpha, dthrottle, dflaps, gearpressed, brakepressed, userquit = view.processevents()
    
    alpha,throttle,flaps,dgear,dbrake=view.alpha(alpha, alphadot, throttle, throttledot, flaps, flapsdot, dt, alphamax, alphamin,gearpressed,brakepressed,dgear,dbrake)
    
    # Acceleration
    CL,CD = view.CLCD(alpha, dflaps, dbrake, dgear)
    L = CL*rho*V**2*S
    D = CD*rho*V**2*S
    m = mzerofuel+mfuel
    W = m*g
    T = dthrottle*Tmax
    mfueldot = -CT*T
    dVdt = (T*cos(alpha)-D-W*sin(gamar))/m
    dgamardt = (L-W*cos(gamar)+T*sin(alpha))/(m*V)
    
    theta=gamar+alpha
    Vx=V*cos(gamar)
    Vy=V*sin(gamar)
    
    
    # Control
    turn=pg.time.get_ticks()/1000.-tstart
    if turn + dt >= tsim:
        tsim = tsim +dt
        Vx = Vx + dVdt*dt
        Vy = Vy + dVdt*dt
        x = x+Vx*dt
        y = y+Vy*dt
        print(x,y)
    
    # Draw horizon on scr, using pitch angle theta, and screen dimensions
    view.drawhor(scr, theta, xmax, ymax, minelev, maxelev)
    
    # Draw the runway
    view.drawrunway(scr, theta, x, y, xmax, ymax, minazi, maxazi, minelev, maxelev)
    
    if userquit:
        running = False
    if x>=0.0:
        running = False
        
     # Update screen
    view.flip()
    
# Close window
view.closewindow()
