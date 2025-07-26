# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:07:11 2022

@author: weiwei
"""
from math import sin, cos, sqrt

#setting up the window
import pygame as pg

pg.init()

screenwidth=800
screemheight=800
reso=(screenwidth,screemheight)

scr=pg.display.set_mode(reso)
scrret=scr.get_rect()
bgcolor=(251, 201, 113)
scr.fill(bgcolor)

#the state
paddle_pos=[380,773]
ball_pos=[370,400]
ball_spd=[600,600]
nrows=8
ncols=20
col=[1]*ncols
grid=[]
for i in range(nrows):
    grid.append(col[:])
#drawing

ballcolor=(0,0,0)
padcolor=(181, 5, 253)
gridcolor=(250, 116, 103)
a=0

#the game loop
gain = 5
escape = False
timestep = 30
maxangle = 15
m=0
angle=0
while not escape:
    time=pg.time.wait(timestep)
    
    scrret=scr.get_rect()
    scr.fill(bgcolor)
    gridrect=scr.get_rect()#grid
    for i in range(nrows):
        for j in range(ncols):
            if grid[i][j]==1:
                pg.draw.rect(scr, gridcolor, (0+40*j,100+15*i,40,15))
                pg.draw.rect(scr, (158, 116, 103), (0+40*j,100+15*i,40,15), 1)
            
    paddlerect=scr.get_rect()#paddle        
    paddlerect.centerx=paddle_pos[0]
    
    paddle=pg.draw.rect(scr, padcolor, (paddle_pos[0],paddle_pos[1],60,6))
    
    pg.draw.circle(scr, ballcolor, ball_pos, 4, 0)
    
    
            
    keys = pg.key.get_pressed()
    pg.event.pump()
    if keys[pg.K_LEFT]:        
        if paddle_pos[0]-gain < 0:
            paddlerect.centerx=0
        else:
            paddle_pos[0]=paddle_pos[0]-gain
            paddlerect.centerx=paddle_pos[0]

    if keys[pg.K_RIGHT]:

        if paddle_pos[0]+gain > 740:
            paddlerect.centerx=740            
        else:        
            paddle_pos[0]=paddle_pos[0]+gain
            paddlerect.centerx=paddle_pos[0]
            
    if keys[pg.K_ESCAPE]:
        escape = True
    for event in pg.event.get():
        if event.type == pg.QUIT:
            escape = True

    ball_pos[0]=ball_pos[0]+ball_spd[0]*timestep/2000
    ball_pos[1]=ball_pos[1]+ball_spd[1]*timestep/2000
    
    if ball_pos[0]>800-2 or ball_pos[0]<0+2:#side walls
        ball_spd[0]=-ball_spd[0]
    if ball_pos[1]>800-2 or ball_pos[1]<0+2:#top and bottom walls
        ball_spd[1]=-ball_spd[1]
    if ball_pos[1]>paddle_pos[1]-5 and paddle_pos[0]<ball_pos[0]<paddle_pos[0]+60:#paddel
        m=ball_pos[0]%paddle_pos[0]
        if m<30:
            angle=45+maxangle*m/30
            ball_spd[0]=600*sqrt(2)*cos(angle)
            ball_spd[1]=ball_spd[1]/abs(ball_spd[1])*600*sqrt(2)*sin(angle)
        elif m>30:
            angle=45-maxangle*m/30
            ball_spd[0]=600*sqrt(2)*cos(angle)
            ball_spd[1]=ball_spd[1]/abs(ball_spd[1])*600*sqrt(2)*sin(angle)
        else:
            ball_spd[1]=-ball_spd[1]
        
    for i in range(nrows):
        for j in range(ncols):  
            if 100+15*i-2<=ball_pos[1]<=100+15*(i+1)+2:
                a=int(ball_pos[0]/40)#record the colomn                
                if grid[i][a-1]==1: 
#                    if 0+40*(a-1)-2<ball_pos[0]<0+40*(a)+2:                                      
                    grid[i][a-1]=0
                    ball_spd[1]=-ball_spd[1]
#                    else:
                        

    pg.display.flip()
       
pg.quit()

