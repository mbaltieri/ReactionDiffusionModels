#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:42:32 2017

Grey-Scott model found online, details below
Introducing now input limitation (u_res changes over time) (see Virgo's thesis)

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
dt = 1
T = 100000
iterations = int(T/dt)


# Parameters from http://www.aliensaint.com/uo/java/rd/
# -----------------------------------------------------
n  = 200
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
Du, Dv, F, k = 0.2, 0.1, 0.04, 0.065 #
Du, Dv, F, k = 0.2, 0.1, 0.04, 0.1 #



Z = np.zeros((n+2,n+2), [('U', np.double), ('V', np.double)])
U,V = Z['U'], Z['V']
u,v = U[1:-1,1:-1], V[1:-1,1:-1]

r = 1
u[...] = 1.0
m = int(n/2)
#U[m-r:m+r,m-r:m+r] = 0.50
V[m-r:m+r,m-r:m+r] = 0.25
u += 0.05*np.random.random((n,n))
v += 0.05*np.random.random((n,n))


plt.ion()

size = np.array(Z.shape)
dpi = 36.0
figsize= size[1]/float(dpi),size[0]/float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
im = plt.imshow(V, interpolation='bicubic', cmap=plt.cm.gray_r)
plt.xticks([]), plt.yticks([])

u_res_h = np.zeros((iterations))
u_res = 1.0

lam = 50
w = 10000000

for i in range(iterations):
    print(i)
    Lu = (                 U[0:-2,1:-1] +
          U[1:-1,0:-2] - 4*U[1:-1,1:-1] + U[1:-1,2:] +
                           U[2:  ,1:-1] )
    Lv = (                 V[0:-2,1:-1] +
          V[1:-1,0:-2] - 4*V[1:-1,1:-1] + V[1:-1,2:] +
                           V[2:  ,1:-1] )

    uvv = u*v*v
    u += dt * (Du*Lu - uvv +  F   *(u_res-u))
    v += dt * (Dv*Lv + uvv - (k)*v    )
    
#    print(np.sum(0.01**2*U))
    u_res += dt * ((lam - F * (n**2*u_res - np.sum(U))) / w)

    if i % 100 == 0:
        im.set_data(V)
        im.set_clim(vmin=V.min(), vmax=V.max())
        plt.draw()
        plt.pause(.01)
#         To make movie
#        plt.savefig("./tmp-%03d.png" % (i/10) ,dpi=dpi)
    u_res_h[i] = u_res

plt.ioff()
# plt.savefig("../figures/zebra.png",dpi=dpi)
# plt.savefig("../figures/bacteria.png",dpi=dpi)
# plt.savefig("../figures/fingerprint.png",dpi=dpi)

plt.figure()
plt.plot(u_res_h)
plt.show()


