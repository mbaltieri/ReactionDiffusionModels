#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:42:32 2017

Grey-Scott model found online, details below
Introducing now two different types of autocatalysts (see Virgo's thesis)

@author: manuelbaltieri
"""

# Reaction-Diffusion Simulation Using Gray-Scott Model
# https://en.wikipedia.org/wiki/Reaction-diffusion_system
# http://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html#
# FB - 20160130
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Reaction Diffusion : Gray-Scott model

References:
----------
Complex Patterns in a Simple System
John E. Pearson, Science 261, 5118, 189-192, 1993.

Encode movie
------------

ffmpeg -r 30 -i "tmp-%03d.png" -c:v libx264 -crf 23 -pix_fmt yuv420p bacteria.mp4
'''

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
#np.random.seed(42)
dt = .5
dx = 1
T = 10000
iterations = int(T/dt)
n = 200
#n = int(size/dx)
size = 200
dx = size/n
#dt = .9 * dx**2/2
#iterations = int(T/dt)

dxx = dx**2

# Parameters from http://www.aliensaint.com/uo/java/rd/
# -----------------------------------------------------

#Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065 # Bacteria 1
#Du, Dv, F, k = 0.14, 0.06, 0.035, 0.065 # Bacteria 2
#Du, Dv, F, k = 0.16, 0.08, 0.060, 0.062 # Coral
#Du, Dv, F, k = 0.19, 0.05, 0.060, 0.062 # Fingerprint
#Du, Dv, F, k = 0.10, 0.10, 0.018, 0.050 # Spirals
#Du, Dv, F, k = 0.12, 0.08, 0.020, 0.050 # Spirals Dense
#Du, Dv, F, k = 0.10, 0.16, 0.020, 0.050 # Spirals Fast
#Du, Dv, F, k = 0.16, 0.08, 0.020, 0.055 # Unstable
#Du, Dv, F, k = 0.16, 0.08, 0.050, 0.065 # Worms 1
#Du, Dv, F, k = 0.16, 0.08, 0.054, 0.063 # Worms 2
#Du, Dv, F, k = 0.16, 0.08, 0.035, 0.060 # Zebrafish
#Du, Dv, F, k = 0.16, 0.08, 0.04, 0.065 #
Du, Dv, Ds, F, k_v, k_s, alpha_s = 0.2*dxx, 0.1*dxx, 0.0*dxx, 0.04, 0.105, 0.0, 0. # u,v
Du, Dv, Ds, F, k_v, k_s, alpha_s = 0.2*dxx, 0.1*dxx, 0.01*dxx, 0.04, 0.105, 0.005, 0.7 # u,v,s

#l_max = 1.0

#def bivariate_gaussian(x_coord):
#    x_light = np.array([9.,37.])
#    sigma_x = 30.
#    sigma_y = 30.
#    Sigma = np.array([[sigma_x ** 2, 0.], [0., sigma_y ** 2]])
#    mu = x_light
#    corr = Sigma[0, 1] / (sigma_x * sigma_y)
#    
#    return 5655 * l_max / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - corr ** 2)) * np.exp(
#            - 1 / (2 * (1 - corr ** 2)) * ((x_coord[0] - mu[0]) ** 2 / 
#            (sigma_x ** 2) + (x_coord[1] - mu[1]) ** 2 / (sigma_y ** 2) - 
#            2 * corr * (x_coord[0] - mu[0]) * (x_coord[1] - mu[1]) / (sigma_x * sigma_y)))



Z = np.zeros((n+2,n+2), [('U', np.double), ('V', np.double), ('S', np.double)])
U,V,S = Z['U'], Z['V'], Z['S']
u,v,s = U[1:-1,1:-1], V[1:-1,1:-1], S[1:-1,1:-1]

r = 10
r2 = 7
u[...] = 1.0
#l = int(n/4)
#for i in range(1,l-1):
#    for j in range(1,l-1):
#        U[i,j] = bivariate_gaussian(np.array([i,j]))
#U = 1 - U

m = int(n/2)
U[m-r:m+r,m-r:m+r] = 0.50
V[m-r:m+r,m-r:m+r] = 0.25
S[m-r2:m+r2,m+r+r2:m+r+3*r2] = 1.50
u += 0.05*np.random.random((n,n))
v += 0.05*np.random.random((n,n))
s += 0.05*np.random.random((n,n))


plt.ion()

size = np.array(Z.shape)
dpi = 72.0
figsize= 2*size[1]/float(dpi),6*size[0]/float(dpi)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, dpi=dpi, facecolor="white")

plt.subplot(311)
CC = -1/(1+np.exp(-100*(S-.2)))
im = plt.imshow(CC, interpolation='bicubic', cmap=plt.cm.gray)

plt.subplot(312)
im2 = plt.imshow(V+S, interpolation='bicubic', cmap=plt.cm.gray_r)

plt.subplot(313)
im3 = plt.imshow(U, interpolation='bicubic', cmap=plt.cm.gray_r)

u_res = 1.0


for i in range(iterations):
#    if i == 200:
#        dt *= 2
    
    if i == 4000:
#        alpha_s = .8
#        F = .036                # stop division, leave one survivor
        F = .038                # stop division, leave more than one survivor, these ones live quite long
#        u_res = 1.02
    if i == 6000:
        S[m-r:m+r,m-r:m+r] = 0.0
    
#    if i == 2000:
#        u_res = 1.02
#        l = int(n/4)
#        for k in range(1,l-1):
#            for j in range(1,l-1):
#                U[k,j] = bivariate_gaussian(np.array([k,j]))
#        u_res = 1.002
#        u[...] = 0.0
#        U[:,m-r:m+r] = 0.0
        
    print(i)
    Lu = (                 U[0:-2,1:-1] +
          U[1:-1,0:-2] - 4*U[1:-1,1:-1] + U[1:-1,2:] +
                           U[2:  ,1:-1] ) / dxx
    Lv = (                 V[0:-2,1:-1] +
          V[1:-1,0:-2] - 4*V[1:-1,1:-1] + V[1:-1,2:] +
                           V[2:  ,1:-1] ) / dxx
    Ls = (                 S[0:-2,1:-1] +
          S[1:-1,0:-2] - 4*S[1:-1,1:-1] + S[1:-1,2:] +
                           S[2:  ,1:-1] ) / dxx

    uvv = u*v*v
    vss = v*s*s
    u +=  dt * (Du*Lu - uvv +  F*(u_res-u))
    v +=  dt * (Dv*Lv + uvv - alpha_s*vss - k_v*v)
    s +=  dt * (Ds*Ls + alpha_s*vss - k_s*s)

    if i % 100 == 0:
        CC = -1/(1+np.exp(-100*(S-.2)))
        im.set_data(CC)
        im.set_clim(vmin=CC.min(), vmax=CC.max())
        
        A = (V+S)
        im2.set_data(A)
        im2.set_clim(vmin=(A).min(), vmax=A.max())
        
        im3.set_data(U)
        im3.set_clim(vmin=U.min(), vmax=U.max())

        plt.draw()
        plt.pause(.0001)

plt.ioff()
plt.show()
