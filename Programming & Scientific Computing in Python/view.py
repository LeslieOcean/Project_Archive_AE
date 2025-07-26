# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:56:05 2022

@author: weiwe
"""
import pygame as pg

# Colours
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

def openwindow(xmax, ymax):
    """ Init pygame, set up window, return scr (window Surface) """
    pg.init()
    scr = pg.display.set_mode((xmax,ymax))
    scr.get_rect()
    return scr

def processevents():
    """ Let PyGame process events, and detect keypresses. """
    dalpha = 0            # -1 or 0 or 1
    dthrottle = 0         # -1 or 0 or 1
    dflaps = 0            # -1 or 0 or 1
    gearpressed = False   # True or False
    brakepressed = False  # True or False
    userquit = False      # True or False
    pg.event.pump()
    keys = pg.key.get_pressed()
    
    #gear and brake
    if keys[pg.K_g]:
        gearpressed = True
    if keys[pg.K_b]:
        brakepressed = True
     
    '''#dalpah
    if keys[pg.K_RIGHT]:
        dalpha=min(dalpha+1,1)
    if keys[pg.K_LEFT]:
        dalpha=max(dalpha-1,0)
            
    #dthrottle
    if keys[pg.K_UP]:
        dthrottle=min(dthrottle+1,1)
    if keys[pg.K_DOWN]:
        dthrottle=max(dthrottle-1,0)
            
    #dflag
    if keys[pg.K_RIGHTBRACKET]:
        dflaps=min(dflaps+1,1)
    if keys[pg.K_LEFTBRACKET]:
        dflaps=max(dflaps-1,0)'''
            
            
    if keys[pg.K_ESCAPE]:
        userquit = True
    for event in pg.event.get(): 
        if event.type == pg.QUIT:
            userquit = True
    return dalpha, dthrottle, dflaps, gearpressed, brakepressed, userquit

def clr(scr):
    """Clears surface, fill with black"""
    scr.fill(black)
    return

def flip():
    """Flip (update) display"""
    pg.display.flip()
    return

def closewindow():
    """Close window, quit pygame"""              
    pg.quit()
    return

def elev2y(elev, ymax, minelev, maxelev):
    """Scale an elevation angle to y-pixels"""
    y = ymax - ymax*(elev-minelev)/(maxelev-minelev)
    return y

def azi2x(azi, xmax, minazi, maxazi):
    """Scale an azimuth angle to x-pixels"""
    x = xmax*(azi-minazi)/(maxazi-minazi)
    return x

def drawhor(scr, theta, xmax, ymax, minelev, maxelev):
    """Draw horizon for pitch angle theta[deg]"""
    import view
    y=view.elev2y(-theta, ymax, minelev, maxelev)
    pg.draw.line(scr, white, (0,y), (xmax,y))
    return

def drawrunway(scr, theta, x, y, xmax, ymax,
minazi, maxazi, minelev, maxelev):
    from math import sqrt, pi
    import numpy as np
    runwaywidth=60
    runwaylength=3000
    dist0=sqrt(y**2+x**2)
    dist1=sqrt(y**2+(-x+runwaylength)**2)
    
    dazi0=np.arctan(runwaywidth/2/dist0)*180/pi
    dazi1=np.arctan(runwaywidth/2/dist1)*180/pi
    elev0=-theta+np.arctan(y/(x))*180/pi
    elev1=-theta-np.arctan(y/(-x+runwaylength))*180/pi
    dx0=xmax*dazi0/(maxazi-minazi)
    dx1=xmax*dazi1/(maxazi-minazi)
    
    import view
    xc=view.azi2x(0, xmax, minazi, maxazi)
    y0=view.elev2y(elev0, ymax, minelev, maxelev)
    y1=view.elev2y(elev1, ymax, minelev, maxelev)

    pg.draw.line(scr, white, (xc-dx1,y1), (xc+dx1,y1))#DC
    pg.draw.line(scr, white, (xc-dx0,y0), (xc+dx0,y0))#AB
    pg.draw.line(scr, white, (xc-dx1,y1),(xc-dx0,y0))#DA
    pg.draw.line(scr, white, (xc+dx1,y1), (xc+dx0,y0))#CB
    return

def CLCD(alpha, flaps, dgear, dbrake):
    CLmax = 1.5 + 1.4*flaps
    CL = min(0.1*(alpha+3)+flaps,CLmax)
    CD0 = 0.021+0.020*dgear+0.120*flaps+0.4*dbrake
    CD = CD0+0.0365*CL**2 
    return CL,CD

def alpha(alpha,alphadot,throttle,throttledot,flaps,flapsdot,dt,alphamax,alphamin,gearpressed,brakepressed,dbrake,dgear):
    #alpha
    pg.event.pump()
    keys = pg.key.get_pressed()
    
    if keys[pg.K_RIGHT]:
        alpha=min(alpha+alphadot*dt,alphamax)
    if keys[pg.K_LEFT]:
        alpha=max(alpha-alphadot*dt,alphamin)
            
    #throttle
    if keys[pg.K_UP]:
        throttle=min(throttle+throttledot*dt,1)
    if keys[pg.K_DOWN]:
        throttle=max(throttle-throttledot*dt,0)
            
    #flag
    if keys[pg.K_RIGHTBRACKET]:
        flaps=min(flaps+flapsdot*dt,1)
    if keys[pg.K_LEFTBRACKET]:
        flaps=max(flaps-flapsdot*dt,0)
    
    if gearpressed==True:
        dgear=1
    if brakepressed==True:
        dbrake=1
        
    return alpha, throttle, flaps, dgear, dbrake